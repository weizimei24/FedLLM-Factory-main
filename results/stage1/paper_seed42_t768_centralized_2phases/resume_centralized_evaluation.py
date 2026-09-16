"""Resume the interrupted centralized evaluation without retraining."""

import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from stage1.config import DOMAINS, Stage1Config
from stage1.data import QACollator, QADataset, load_domain_rows
from stage1.evaluator import Stage1Evaluator
from stage1.model import load_saved_adapter, load_tokenizer, release_memory
from stage1.results import save_method_result


def score_saved_predictions(model, evaluator, rows, prediction_path: Path) -> dict:
    records = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line]
    expected_qids = [row["qid"] for row in rows]
    if len(records) != len(rows) or [record["qid"] for record in records] != expected_qids:
        raise ValueError(f"Incomplete or mismatched predictions in {prediction_path}")

    dataset = QADataset(rows, evaluator.tokenizer, evaluator.config.max_length)
    loader = DataLoader(
        dataset,
        batch_size=evaluator.config.eval_batch_size,
        shuffle=False,
        collate_fn=QACollator(evaluator.tokenizer.pad_token_id),
    )
    total_nll = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            output = model(**batch)
        tokens = int((batch["labels"][:, 1:] != -100).sum().item())
        total_nll += output.loss.float().item() * tokens
        total_tokens += tokens

    sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for row, record in zip(rows, records):
        scores = evaluator.scorer.score(row["answer"], record["prediction"])
        for key in sums:
            sums[key] += scores[key].fmeasure
    count = len(rows)
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "eval_loss": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 20 else float("inf"),
        "rouge1": sums["rouge1"] / count,
        "rouge2": sums["rouge2"] / count,
        "rougeL": sums["rougeL"] / count,
        "avg_generation_tokens": sum(record["generation_tokens"] for record in records) / count,
        "eos_rate": sum(record["met_eos"] for record in records) / count,
        "max_tokens_rate": sum(record["hit_max_new_tokens"] for record in records) / count,
        "eval_answer_tokens": total_tokens,
        "samples": count,
    }


config = Stage1Config(run_name="paper_seed42_t768_centralized_2phases")
config.phases = 2
config.rounds = 2
tokenizer = load_tokenizer(config.model_path)
model = load_saved_adapter(config, config.run_dir / "centralized" / "adapter")
evaluator = Stage1Evaluator(tokenizer, config)
per_domain = {}

for domain, _ in DOMAINS:
    rows = load_domain_rows(config.data_dir, domain, "test")[:config.test_samples]
    predictions = config.run_dir / "centralized" / "predictions" / f"{domain}.jsonl"
    if predictions.exists():
        print(f"Scoring saved predictions for {domain}", flush=True)
        per_domain[domain] = score_saved_predictions(model, evaluator, rows, predictions)
    else:
        print(f"Evaluating remaining domain {domain}", flush=True)
        per_domain[domain] = evaluator.evaluate(model, rows, predictions)
    print(json.dumps(per_domain[domain], ensure_ascii=False), flush=True)

save_method_result(config.run_dir, "centralized", per_domain, {"resumed_evaluation": True})
print("Centralized evaluation resumed and completed.", flush=True)
del model
release_memory()
