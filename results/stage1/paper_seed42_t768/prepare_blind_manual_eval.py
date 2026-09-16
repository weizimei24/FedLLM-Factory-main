"""Create a reproducible, method-blinded 90-answer manual-evaluation set."""

import json
import random
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
DATA_DIR = RUN_DIR.parents[2] / "dataset" / "stackexchange_stage1"
METHODS = ("base", "local", "centralized", "fedit", "fedrotlora")
DOMAINS = ("mathematics", "physics", "computer_science", "statistics", "economics", "biology")
SAMPLES_PER_DOMAIN = 3
SEED = 20260916


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


blind_rows = []
key_rows = []
for domain in DOMAINS:
    questions = {row["qid"]: row for row in read_jsonl(DATA_DIR / domain / "test.jsonl")}
    predictions = {
        method: {row["qid"]: row for row in read_jsonl(RUN_DIR / method / "predictions" / f"{domain}.jsonl")}
        for method in METHODS
    }
    common_qids = sorted(set(questions).intersection(*(set(index) for index in predictions.values())))
    if len(common_qids) < SAMPLES_PER_DOMAIN:
        raise ValueError(f"{domain}: only {len(common_qids)} common questions")
    rng = random.Random(f"{SEED}:{domain}")
    for qid in rng.sample(common_qids, SAMPLES_PER_DOMAIN):
        method_order = list(METHODS)
        rng.shuffle(method_order)
        candidates = []
        for label, method in zip("ABCDE", method_order):
            prediction = predictions[method][qid]
            candidates.append({
                "label": label,
                "prediction": prediction["prediction"],
                "generation_tokens": prediction["generation_tokens"],
            })
            key_rows.append({"domain": domain, "qid": qid, "label": label, "method": method})
        question = questions[qid]
        blind_rows.append({
            "domain": domain,
            "qid": qid,
            "question": question["question"],
            "reference": question["answer"],
            "candidates": candidates,
        })

(RUN_DIR / "manual_eval_blinded_90.json").write_text(
    json.dumps(blind_rows, ensure_ascii=False, indent=2), encoding="utf-8"
)
(RUN_DIR / "manual_eval_blinded_90_key.json").write_text(
    json.dumps(key_rows, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"Prepared {len(blind_rows)} questions and {len(key_rows)} blinded answers.")
