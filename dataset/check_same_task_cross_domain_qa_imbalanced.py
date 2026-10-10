"""Validate the five-client imbalanced QA dataset against its source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DOMAINS = ("squad", "newsqa", "bioasq", "duorc", "textbookqa")
TRAIN_SIZES = (1600, 1400, 500, 700, 800)


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("dataset/same_task_cross_domain_qa"))
    parser.add_argument("--target", type=Path, default=Path("dataset/same_task_cross_domain_qa_imbalanced"))
    args = parser.parse_args()
    manifest = json.loads((args.target / "manifest.json").read_text(encoding="utf-8"))
    failed = False
    for index, (domain, expected) in enumerate(zip(DOMAINS, TRAIN_SIZES)):
        source_train = _rows(args.source / "train" / f"{index}.jsonl")
        train = _rows(args.target / "train" / f"{index}.jsonl")
        test = _rows(args.target / "test" / f"{domain}.jsonl")
        source_by_id = {row["id"]: row for row in source_train}
        train_by_id = {row["id"]: row for row in train}
        train_ids = set(train_by_id)
        test_ids = {row["id"] for row in test}
        train_docs = {row["document_id"] for row in train}
        test_docs = {row["document_id"] for row in test}
        source_relation_ok = (
            all(train_by_id.get(row_id) == row for row_id, row in source_by_id.items())
            if expected >= 1000 else
            all(source_by_id.get(row_id) == row for row_id, row in train_by_id.items())
        )
        test_identical = _sha256(args.source / "test" / f"{domain}.jsonl") == _sha256(
            args.target / "test" / f"{domain}.jsonl"
        )
        valid = (
            len(train) == expected and len(test) == 500
            and len(train_ids) == len(train) and len(test_ids) == len(test)
            and not train_ids & test_ids and not train_docs & test_docs
            and all(row["answer"] in row["context"] for row in train + test)
            and source_relation_ok and test_identical
            and manifest["clients"][str(index)] == domain
        )
        failed |= not valid
        print(
            f"{domain}: client={index} train={len(train)} test={len(test)} "
            f"source_relation={source_relation_ok} test_bytes_identical={test_identical} "
            f"id_overlap={len(train_ids & test_ids)} doc_overlap={len(train_docs & test_docs)} "
            f"status={'OK' if valid else 'FAIL'}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
