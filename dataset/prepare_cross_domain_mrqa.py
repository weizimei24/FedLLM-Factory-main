"""Prepare the deterministic four-client MRQA-style extractive QA pilot.

Run from the repository root:
    python -X utf8 dataset/prepare_cross_domain_mrqa.py

The Hugging Face ``mrqa`` builder supplies a common extractive-QA schema for
all four sources. Train data are sampled only from ``train``; test data prefer
``validation`` (then ``dev`` or ``test`` if a builder exposes a different
name).  The output keeps all gold answers for evaluation while supervising the
first source-order answer that occurs in the stored context window.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from transformers import AutoTokenizer

# When invoked as ``python dataset/prepare_cross_domain_mrqa.py``, Python
# otherwise resolves ``dataset/utils.py`` before the repository's ``utils``
# package. Put the repository root first for the shared QA prompt import.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
from utils.qa_utils import encode_squad_prompt


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "cross_domain_mrqa"
DOMAINS = ("squad", "newsqa", "triviaqa", "naturalquestions")
DEFAULT_SEED = 42
TRAIN_SIZE = 1_000
TEST_SIZE = 200
CONTEXT_WINDOW_CHARS = 1_400
MRQA_BASE_URL = "https://s3.us-east-2.amazonaws.com/mrqa/release/v2"
SOURCES = {
    "squad": {"train": "SQuAD.jsonl.gz", "validation": "SQuAD.jsonl.gz"},
    "newsqa": {"train": "NewsQA.jsonl.gz", "validation": "NewsQA.jsonl.gz"},
    "triviaqa": {"train": "TriviaQA-web.jsonl.gz", "validation": "TriviaQA-web.jsonl.gz"},
    "naturalquestions": {"train": "NaturalQuestionsShort.jsonl.gz", "validation": "NaturalQuestionsShort.jsonl.gz"},
}


def _answers(value) -> list[str]:
    """Normalise the minor answer-schema differences used by MRQA releases."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("text", "texts", "answers"):
            if key in value:
                return _answers(value[key])
        return []
    if isinstance(value, list):
        collected: list[str] = []
        for item in value:
            collected.extend(_answers(item))
        return collected
    return []


def _clean_context(context: str) -> str:
    """MRQA loader normalisation, copied from the public MRQA dataset script."""
    return (
        context.replace("[PAR] ", "\n\n").replace("[TLE]", "Title:")
        .replace("[SEP]", "\nPassage:").replace("<Li>", "").replace("</Li>", "")
        .replace("<OI>", "").replace("</OI>", "").replace("<Ol>", "")
        .replace("</Ol>", "").replace("<Dd>", "").replace("</Dd>", "")
        .replace("<UI>", "").replace("</UI>", "").replace("<Ul>", "")
        .replace("</Ul>", "").replace("<P>", "").replace("</P>", "")
        .replace("[DOC]", "").strip()
    )


def _clean_spaces(value: str) -> str:
    replacements = ((" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","),
                    (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"),
                    (" 've", "'ve"), (" 're", "'re"), ("( ", "("), (" )", ")"),
                    (" %", "%"), ("`` ", '"'), (" ''", '"'), (" :", ":"))
    for source, destination in replacements:
        value = value.replace(source, destination)
    return value


def _download(url: str, destination: Path) -> None:
    """Download an official MRQA gzip file once, preserving it for re-runs."""
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "FedLLM-Factory/1.0"})
    with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def _iter_mrqa_rows(path: Path):
    """Yield one common-schema QA row at a time from official MRQA JSONL.GZ."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        next(handle)  # MRQA metadata header
        for line in handle:
            paragraph = json.loads(line)
            context = _clean_spaces(_clean_context(paragraph["context"]))
            for qa in paragraph["qas"]:
                question = str(qa["question"]).strip()
                if question and not question.endswith("?"):
                    question += "?"
                yield {
                    "qid": qa["qid"],
                    "context": context,
                    "question": _clean_spaces(question),
                    "answers": [_clean_spaces(str(answer)) for answer in qa.get("answers", [])],
                }


def _unique_nonempty(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _answer_window(context: str, answer: str) -> tuple[str, int]:
    """Keep an answer-containing context window, as in the SQuAD pipeline."""
    start = context.find(answer)
    if start < 0:
        raise ValueError("chosen answer is absent from context")
    if len(context) <= CONTEXT_WINDOW_CHARS:
        return context, 0
    left = max(0, start - (CONTEXT_WINDOW_CHARS - len(answer)) // 2)
    right = min(len(context), left + CONTEXT_WINDOW_CHARS)
    left = max(0, right - CONTEXT_WINDOW_CHARS)
    window = context[left:right]
    if window.find(answer) < 0:
        raise RuntimeError("answer was lost while constructing its context window")
    return window, left


def _canonicalise(row: dict, domain: str) -> dict | None:
    context = str(row.get("context", ""))
    question = " ".join(str(row.get("question", "")).split())
    answers = _unique_nonempty(_answers(row.get("answers", row.get("answer", []))))
    if not context or not question or not answers:
        return None
    # Source order is preserved, then the first answer actually found in the
    # context is the deterministic supervised label.
    label = next((answer for answer in answers if context.find(answer) >= 0), None)
    if label is None:
        return None
    stored_context, context_start = _answer_window(context, label)
    answer_start = stored_context.find(label)
    qid = row.get("qid", row.get("id", row.get("question_id", "")))
    return {
        "id": f"{domain}:{qid}",
        "domain": domain,
        "context": stored_context,
        "question": question,
        "label": label,
        "answers": answers,
        "answer_start": answer_start,
        "context_start": context_start,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ids_hash(rows: list[dict]) -> str:
    return hashlib.sha256("\n".join(sorted(row["id"] for row in rows)).encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_rows(dataset, domain: str, count: int, seed: int) -> tuple[list[dict], int]:
    usable = []
    for row in dataset:
        converted = _canonicalise(row, domain)
        if converted is not None:
            usable.append(converted)
    if len(usable) < count:
        raise ValueError(f"{domain}: requested {count} usable rows but found {len(usable)}")
    return random.Random(seed).sample(usable, count), len(usable)


def _statistics(rows: list[dict], tokenizer) -> dict:
    def average(field: str) -> float:
        return sum(len(tokenizer(row[field], add_special_tokens=False)["input_ids"]) for row in rows) / len(rows)
    return {
        "num_samples": len(rows),
        "avg_input_tokens": sum(
            len(encode_squad_prompt(tokenizer, row["context"], row["question"])) for row in rows
        ) / len(rows),
        "avg_context_tokens": average("context"),
        "avg_question_tokens": average("question"),
        "avg_answer_tokens": average("label"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument("--tokenizer", default="Qwen3-1.7B", help="tokenizer used for saved length statistics")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    manifest = {
        "dataset": "MRQA cross-domain extractive QA pilot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "sampling": {"train_per_domain": args.train_size, "test_per_domain": args.test_size},
        "clients": {"0": "squad", "1": "newsqa", "2": "triviaqa", "3": "naturalquestions"},
        "sources": {},
        "schema": {"label": "first source-order gold answer found in context", "answers": "all source gold answer strings"},
        "context_window": {"max_chars": CONTEXT_WINDOW_CHARS, "strategy": "answer-centred"},
    }
    statistics = {"tokenizer": args.tokenizer, "domains": {}}
    for client_id, domain in enumerate(DOMAINS):
        print(f"Downloading/checking MRQA {domain} train and validation ...")
        raw_dir = OUTPUT_DIR / "raw"
        train_url = f"{MRQA_BASE_URL}/train/{SOURCES[domain]['train']}"
        test_url = f"{MRQA_BASE_URL}/dev/{SOURCES[domain]['validation']}"
        train_path = raw_dir / f"{domain}_train.jsonl.gz"
        test_path = raw_dir / f"{domain}_validation.jsonl.gz"
        _download(train_url, train_path)
        _download(test_url, test_path)
        # Different deterministic streams prevent accidental same-index choices.
        train_rows, train_source_count = _sample_rows(
            _iter_mrqa_rows(train_path), domain, args.train_size, args.seed + client_id * 100
        )
        test_rows, test_source_count = _sample_rows(
            _iter_mrqa_rows(test_path), domain, args.test_size, args.seed + client_id * 100 + 1
        )
        train_ids, test_ids = {row["id"] for row in train_rows}, {row["id"] for row in test_rows}
        if train_ids & test_ids:
            raise RuntimeError(f"{domain}: train/test ID overlap detected")
        _write_jsonl(OUTPUT_DIR / "train" / f"{client_id}.jsonl", train_rows)
        _write_jsonl(OUTPUT_DIR / "test" / f"{domain}.jsonl", test_rows)
        manifest["sources"][domain] = {
            "format": "official MRQA v2 JSONL.GZ",
            "train_split": "train",
            "test_split": "validation",
            "urls": {"train": train_url, "validation": test_url},
            "raw_sha256": {"train": _sha256(train_path), "validation": _sha256(test_path)},
            "source_usable_counts": {"train": train_source_count, "validation": test_source_count},
            "output_counts": {"train": len(train_rows), "test": len(test_rows)},
            "train_id_sha256": _ids_hash(train_rows),
            "test_id_sha256": _ids_hash(test_rows),
            "train_test_overlap": 0,
        }
        statistics["domains"][domain] = {
            "train": _statistics(train_rows, tokenizer),
            "test": _statistics(test_rows, tokenizer),
        }
        print(f"{domain}: train={len(train_rows)}, test={len(test_rows)}")

    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "statistics.json").write_text(json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared data under {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
