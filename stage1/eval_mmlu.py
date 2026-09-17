"""Zero-shot MMLU evaluation by answer-choice log-likelihood.

This is a small, dependency-light reproduction of lm-evaluation-harness's
default MMLU task format.  It does *not* generate an answer and parse it.
For each question it scores the four continuations (``A`` ... ``D``) after
``Answer:`` and selects the continuation with the largest summed log
likelihood.  That makes the result insensitive to the response style learned
by the StackExchange instruction tuning.

Examples
--------
Run the recommended first comparison for the mathematics bucket:

    python -m stage1.eval_mmlu --domain mathematics --methods base,local,centralized

Run one adapter directly:

    python -m stage1.eval_mmlu --subjects mmlu_college_physics \
        --adapter results/stage1/paper_seed42_t768/centralized/adapter

The MMLU data are fetched once through Hugging Face and subsequently loaded
from its cache.  Outputs contain both a concise summary and per-question
scores/predictions, so every aggregate can be checked afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset, load_dataset

from stage1.config import ROOT
from stage1.model import load_base_model, load_saved_adapter, load_tokenizer, release_memory


# These are intentionally conservative, subject-matter rather than merely
# keyword matches.  Change --subjects for a different mapping without editing
# the evaluator.  The same subject list must be used for every method being
# compared.
DOMAIN_SUBJECTS: dict[str, tuple[str, ...]] = {
    "mathematics": (
        "abstract_algebra",
        "college_mathematics",
        "elementary_mathematics",
        "high_school_mathematics",
    ),
    "physics": (
        "astronomy",
        "college_physics",
        "conceptual_physics",
        "high_school_physics",
    ),
    "computer_science": (
        "college_computer_science",
        "computer_security",
        "high_school_computer_science",
        "machine_learning",
    ),
    "statistics": ("high_school_statistics",),
    "economics": (
        "econometrics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
    ),
    "biology": ("college_biology", "high_school_biology"),
}

CHOICES = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=sorted(DOMAIN_SUBJECTS))
    parser.add_argument(
        "--subjects",
        help="Comma-separated MMLU subject names. Overrides --domain.",
    )
    parser.add_argument(
        "--methods",
        default="base,local,centralized",
        help="Comma-separated methods: base, local, centralized, fedit, fedrotlora.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="Evaluate exactly this adapter; cannot be used with --methods.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "results" / "stage1" / "paper_seed42_t768",
        help="Directory containing the already-trained adapters.",
    )
    parser.add_argument("--model-path", type=Path, default=ROOT / "Qwen3-1.7B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use only already-cached MMLU data (recommended after the first download).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Debug only: cap test questions per subject. Omit for all MMLU test questions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to a separate mmlu_zeroshot/<domain> directory.",
    )
    args = parser.parse_args()
    if not args.domain and not args.subjects:
        parser.error("provide --domain or --subjects")
    if args.batch_size < 1 or args.max_length < 16:
        parser.error("--batch-size must be positive and --max-length must be at least 16")
    return args


def get_subjects(args: argparse.Namespace) -> tuple[str, ...]:
    if args.subjects:
        return tuple(part.strip().removeprefix("mmlu_") for part in args.subjects.split(",") if part.strip())
    return DOMAIN_SUBJECTS[args.domain]


def method_adapter(method: str, domain: str | None, run_dir: Path) -> Path | None:
    if method == "base":
        return None
    if method == "local":
        if domain is None:
            raise ValueError("local requires --domain because each local adapter is domain-specific")
        return run_dir / "local" / "adapters" / domain
    if method in {"centralized", "fedit", "fedrotlora"}:
        return run_dir / method / "adapter"
    raise ValueError(f"Unknown method {method!r}")


def subject_description(subject: str) -> str:
    return "The following are multiple choice questions (with answers) about " + subject.replace("_", " ") + ".\n\n"


def format_prompt(row: dict, subject: str) -> str:
    options = "\n".join(f"{label}. {text}" for label, text in zip(CHOICES, row["choices"]))
    return f"{subject_description(subject)}{row['question'].strip()}\n{options}\nAnswer:"


def score_choice_batch(model, tokenizer, prompts: list[str], max_length: int) -> list[list[float]]:
    """Return four sequence log-likelihoods for every prompt.

    Candidates are tokenized together with their prompt.  This avoids an
    incorrect assumption that each answer label is one token, and correctly
    handles tokenizers whose boundary tokenization differs after a colon.
    """
    texts = [prompt + choice for prompt in prompts for choice in CHOICES]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    # Tokenization is not necessarily compositional at ``Answer:``.  For
    # example, Qwen tokenizes ``:A`` as a single token.  lm-eval's HF backend
    # likewise takes the final N tokens of tokenized(prompt + choice), where
    # N is the separately tokenized choice length.  Scoring that suffix keeps
    # the four alternatives aligned without imposing a whitespace convention
    # that is absent from the official MMLU template.
    full_lengths = [
        len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts
    ]
    choice_lengths = {
        choice: len(tokenizer(choice, add_special_tokens=False)["input_ids"])
        for choice in CHOICES
    }
    candidate_starts = [
        length - choice_lengths[choice]
        for length, choice in zip(full_lengths, CHOICES * len(prompts))
    ]
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    pad_side = tokenizer.padding_side
    if pad_side != "right":
        raise ValueError("This evaluator requires right padding for correct candidate masks")

    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
    ):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
    token_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)

    scores: list[float] = []
    for index, start in enumerate(candidate_starts):
        # The probability of token i is in logits position i - 1. A candidate
        # truncated before it starts cannot be validly evaluated.
        end = int(attention_mask[index].sum().item())
        if end <= start:
            raise ValueError("Prompt was truncated before its answer label; increase --max-length")
        token_ids = input_ids[index, start:end]
        positions = torch.arange(start - 1, end - 1, device=model.device)
        scores.append(float(token_log_probs[index, positions, token_ids].sum().item()))
    return [scores[index : index + 4] for index in range(0, len(scores), 4)]


def load_rows(subject: str, limit: int | None, offline: bool) -> list[dict]:
    if offline:
        # ``load_dataset`` resolves the Hub README before inspecting its Arrow
        # cache, even when its DownloadConfig is local-only.  Read the cached
        # Arrow file directly to guarantee a genuinely offline evaluation.
        cache_root = Path(
            os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets")
        )
        candidates = sorted(cache_root.glob(f"cais___mmlu/{subject}/*/*/mmlu-test.arrow"))
        if not candidates:
            raise FileNotFoundError(
                f"No cached MMLU test data for {subject!r}. Run once without --offline to download it."
            )
        dataset = Dataset.from_file(str(candidates[-1]))
    else:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    rows = [dict(row) for row in dataset]
    return rows if limit is None else rows[:limit]


def evaluate_model(
    model,
    tokenizer,
    subjects: tuple[str, ...],
    batch_size: int,
    max_length: int,
    limit: int | None,
    offline: bool,
) -> tuple[list[dict], dict[str, float]]:
    records: list[dict] = []
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for subject in subjects:
        rows = load_rows(subject, limit, offline)
        for offset in range(0, len(rows), batch_size):
            row_batch = rows[offset : offset + batch_size]
            prompts = [format_prompt(row, subject) for row in row_batch]
            batch_scores = score_choice_batch(model, tokenizer, prompts, max_length)
            for row, scores in zip(row_batch, batch_scores):
                prediction = max(range(4), key=lambda index: scores[index])
                correct = int(prediction == row["answer"])
                totals[subject][0] += correct
                totals[subject][1] += 1
                records.append(
                    {
                        "subject": subject,
                        "question": row["question"],
                        "choices": row["choices"],
                        "gold_index": row["answer"],
                        "gold": CHOICES[row["answer"]],
                        "prediction_index": prediction,
                        "prediction": CHOICES[prediction],
                        "correct": bool(correct),
                        "choice_loglikelihood": dict(zip(CHOICES, scores)),
                    }
                )
    subject_accuracy = {subject: correct / total for subject, (correct, total) in totals.items()}
    total_correct = sum(correct for correct, _ in totals.values())
    total_questions = sum(total for _, total in totals.values())
    return records, {
        "accuracy": total_correct / total_questions if total_questions else math.nan,
        "macro_accuracy": sum(subject_accuracy.values()) / len(subject_accuracy) if subject_accuracy else math.nan,
        "samples": total_questions,
        **{f"mmlu_{subject}_accuracy": accuracy for subject, accuracy in subject_accuracy.items()},
    }


def write_results(output_dir: Path, name: str, records: list[dict], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (output_dir / f"{name}_samples.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_comparison(output_dir: Path, summaries: list[dict]) -> None:
    if not summaries:
        return
    fields = list(summaries[0])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    args = parse_args()
    subjects = get_subjects(args)
    if args.output_dir is None:
        label = args.domain or "custom_subjects"
        args.output_dir = ROOT / "results" / "stage1" / "mmlu_zeroshot" / label
    methods = ["custom"] if args.adapter else [part.strip() for part in args.methods.split(",") if part.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    tokenizer = load_tokenizer(str(args.model_path))
    tokenizer.padding_side = "right"

    for method in methods:
        adapter = args.adapter if args.adapter else method_adapter(method, args.domain, args.run_dir)
        if adapter is not None and not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Adapter not found or incomplete: {adapter}")
        model = (
            load_base_model(str(args.model_path), for_training=False)
            if adapter is None
            else load_saved_adapter(type("Config", (), {"model_path": str(args.model_path)})(), str(adapter))
        )
        name = method if method != "custom" else adapter.name
        print(f"[MMLU] Evaluating {name}: {', '.join(subjects)}")
        records, metrics = evaluate_model(
            model, tokenizer, subjects, args.batch_size, args.max_length, args.limit, args.offline
        )
        summary = {
            "method": name,
            "adapter": str(adapter) if adapter else "(base)",
            "subjects": ",".join(subjects),
            "num_fewshot": 0,
            "scoring": "sum_loglikelihood_of_A_B_C_D_after_Answer_colon",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **metrics,
        }
        write_results(args.output_dir, name, records, summary)
        summaries.append(summary)
        print(f"[MMLU] {name}: accuracy={metrics['accuracy']:.4f}, n={metrics['samples']}")
        del model
        release_memory()
    write_comparison(args.output_dir, summaries)


if __name__ == "__main__":
    main()






# python -m stage1.eval_mmlu --domain computer_science --methods base,local,centralized --batch-size 8
# python -m stage1.eval_mmlu --domain statistics --methods base,local,centralized --batch-size 8
# python -m stage1.eval_mmlu --domain economics --methods base,local,centralized --batch-size 8
# python -m stage1.eval_mmlu --domain biology --methods base,local,centralized --batch-size 8