"""Verify the ten-client partition against the original five-client dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("dataset/same_task_cross_domain_qa"))
    parser.add_argument("--target", type=Path, default=Path("dataset/same_task_cross_domain_qa_10clients"))
    args = parser.parse_args()
    source_manifest = json.loads((args.source / "manifest.json").read_text(encoding="utf-8"))
    target_manifest = json.loads((args.target / "manifest.json").read_text(encoding="utf-8"))
    failed = False
    for domain_index in range(5):
        domain = source_manifest["clients"][str(domain_index)]
        client_ids = (2 * domain_index, 2 * domain_index + 1)
        original = _rows(args.source / "train" / f"{domain_index}.jsonl")
        parts = [_rows(args.target / "train" / f"{client_id}.jsonl") for client_id in client_ids]
        original_by_id = {row["id"]: row for row in original}
        part_by_id = {row["id"]: row for part in parts for row in part}
        id_sets = [{row["id"] for row in part} for part in parts]
        doc_sets = [{row["document_id"] for row in part} for part in parts]
        test = _rows(args.target / "test" / f"{domain}.jsonl")
        source_test = args.source / "test" / f"{domain}.jsonl"
        target_test = args.target / "test" / f"{domain}.jsonl"
        test_ids = {row["id"] for row in test}
        test_docs = {row["document_id"] for row in test}
        mappings_ok = all(target_manifest["clients"][str(client_id)] == domain for client_id in client_ids)
        valid = (
            len(original) == 1000
            and [len(part) for part in parts] == [500, 500]
            and len(test) == 500
            and original_by_id == part_by_id
            and not (id_sets[0] & id_sets[1])
            and not (doc_sets[0] & doc_sets[1])
            and not (test_ids & (id_sets[0] | id_sets[1]))
            and not (test_docs & (doc_sets[0] | doc_sets[1]))
            and all(row["answer"] in row["context"] for row in parts[0] + parts[1] + test)
            and _hash(source_test) == _hash(target_test)
            and mappings_ok
        )
        failed |= not valid
        print(
            f"{domain}: clients={client_ids[0]},{client_ids[1]} "
            f"train={len(parts[0])}+{len(parts[1])} test={len(test)} "
            f"source_train_identical={original_by_id == part_by_id} "
            f"test_bytes_identical={_hash(source_test) == _hash(target_test)} "
            f"status={'OK' if valid else 'FAIL'}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
