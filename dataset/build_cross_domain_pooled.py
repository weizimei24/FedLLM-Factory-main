"""Build the 4,000-example centralized training pool from existing clients.

Run after ``prepare_cross_domain_mrqa.py``:
    python -X utf8 dataset/build_cross_domain_pooled.py

The four named test sets are not copied or merged; centralized evaluation
continues to use them independently through the shared cross-domain evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "cross_domain_mrqa"
CLIENT_DOMAINS = {0: "squad", 1: "newsqa", 2: "triviaqa", 3: "naturalquestions"}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run dataset/prepare_cross_domain_mrqa.py first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ids_sha256(rows: list[dict]) -> str:
    payload = "\n".join(sorted(str(row["id"]) for row in rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-per-domain", type=int, default=1000)
    parser.add_argument("--output", default="pooled.jsonl")
    args = parser.parse_args()

    pooled: list[dict] = []
    counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    for client_id, domain in CLIENT_DOMAINS.items():
        rows = _read_jsonl(DATASET_DIR / "train" / f"{client_id}.jsonl")
        if len(rows) != args.expected_per_domain:
            raise ValueError(
                f"Expected {args.expected_per_domain} {domain} rows for client {client_id}, "
                f"found {len(rows)}"
            )
        wrong_domain = [row.get("id") for row in rows if row.get("domain") != domain]
        if wrong_domain:
            raise ValueError(f"Client {client_id} contains {len(wrong_domain)} rows outside {domain}")
        row_ids = {str(row["id"]) for row in rows}
        if len(row_ids) != len(rows):
            raise ValueError(f"Client {client_id}/{domain} contains duplicate IDs")
        overlap = seen_ids & row_ids
        if overlap:
            raise ValueError(f"Cross-client overlap detected for {len(overlap)} IDs")
        seen_ids.update(row_ids)
        pooled.extend(rows)
        counts[domain] = len(rows)

    # The DataLoader also shuffles deterministically each round, but storing a
    # deterministic mixed order prevents the pooled artifact from encoding
    # domain blocks and makes it independently auditable.
    random.Random(args.seed).shuffle(pooled)
    output_path = DATASET_DIR / "train" / args.output
    _write_jsonl(output_path, pooled)
    manifest = {
        "dataset": "MRQA cross-domain centralized pooled training set",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "source_files": [f"train/{client_id}.jsonl" for client_id in CLIENT_DOMAINS],
        "domain_counts": counts,
        "total_train_samples": len(pooled),
        "unique_ids": len(seen_ids),
        "id_sha256": _ids_sha256(pooled),
        "output": f"train/{args.output}",
        "test_protocol": "reuse four named 200-example domain test files; no mixed micro-only test",
    }
    manifest_path = DATASET_DIR / "pooled_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(pooled)} pooled rows to {output_path}")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
