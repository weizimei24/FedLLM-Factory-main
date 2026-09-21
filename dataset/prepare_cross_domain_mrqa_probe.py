"""Sample a disjoint probe split from the same MRQA train pool as training data.

Run after prepare_cross_domain_mrqa.py:
    python -X utf8 dataset/prepare_cross_domain_mrqa_probe.py
"""
from __future__ import annotations
import json, random
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.prepare_cross_domain_mrqa import (
    OUTPUT_DIR, DOMAINS, DEFAULT_SEED, _iter_mrqa_rows, _canonicalise, _write_jsonl,
)

PROBE_SIZE = 100

def main():
    for client_id, domain in enumerate(DOMAINS):
        train_path = OUTPUT_DIR / "raw" / f"{domain}_train.jsonl.gz"
        existing_train_ids = {
            json.loads(line)["id"]
            for line in (OUTPUT_DIR / "train" / f"{client_id}.jsonl").open(encoding="utf-8")
        }
        usable = [
            row for row in (_canonicalise(r, domain) for r in _iter_mrqa_rows(train_path))
            if row is not None and row["id"] not in existing_train_ids
        ]
        # distinct seed offset (+2) so it never coincides with train (+0) or test (+1)
        probe_rows = random.Random(DEFAULT_SEED + client_id * 100 + 2).sample(usable, PROBE_SIZE)
        assert {r["id"] for r in probe_rows}.isdisjoint(existing_train_ids)
        _write_jsonl(OUTPUT_DIR / "probe" / f"{domain}.jsonl", probe_rows)
        print(f"{domain}: probe={len(probe_rows)} (excluded {len(existing_train_ids)} train ids)")

if __name__ == "__main__":
    raise SystemExit(main())