"""Validate the isolated five-domain extractive-QA JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DOMAINS = ("squad", "newsqa", "bioasq", "duorc", "textbookqa")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset/same_task_cross_domain_qa")
    args = parser.parse_args()
    root = Path(args.root)
    failed = False
    for domain in DOMAINS:
        train = read_jsonl(root / "train" / f"{DOMAINS.index(domain)}.jsonl")
        test = read_jsonl(root / "test" / f"{domain}.jsonl")
        train_ids, test_ids = {row["id"] for row in train}, {row["id"] for row in test}
        train_docs = {row["document_id"] for row in train}
        test_docs = {row["document_id"] for row in test}
        duplicate_ids = len(train_ids) != len(train) or len(test_ids) != len(test)
        answer_missing = sum(
            row.get("answer") not in row.get("context", "") for row in train + test
        )
        overlap = train_ids & test_ids
        document_overlap = train_docs & test_docs
        ok = (len(train) == 1000 and len(test) == 500 and not duplicate_ids and
              not overlap and not document_overlap and answer_missing == 0)
        failed = failed or not ok
        print(
            f"{domain}: train={len(train)} test={len(test)} "
            f"answer_missing={answer_missing} id_overlap={len(overlap)} "
            f"document_overlap={len(document_overlap)} duplicate_ids={duplicate_ids} "
            f"status={'OK' if ok else 'FAIL'}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
