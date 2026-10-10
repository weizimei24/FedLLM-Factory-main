"""Split the existing five-domain QA train set into ten disjoint clients.

Each domain contributes its existing 1,000 training rows to two 500-row
clients. Whole context documents stay together. The five test files are
copied byte-for-byte; no raw source is re-downloaded or re-sampled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "same_task_cross_domain_qa"
TARGET = ROOT / "same_task_cross_domain_qa_10clients"
SEED = 42
CLIENT_SIZE = 500


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_document_ids(rows: list[dict], target_size: int, seed: int) -> set[str]:
    """Find a deterministic exact-size subset of complete documents."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["document_id"]].append(row)
    document_ids = list(grouped)
    random.Random(seed).shuffle(document_ids)
    # Store only the predecessor of each achievable count; this is enough to
    # reconstruct one exact partition without quadratic tuple allocations.
    previous: dict[int, tuple[int, str] | None] = {0: None}
    for document_id in document_ids:
        size = len(grouped[document_id])
        for count in sorted(previous, reverse=True):
            total = count + size
            if total <= target_size and total not in previous:
                previous[total] = (count, document_id)
        if target_size in previous:
            break
    if target_size not in previous:
        raise ValueError(f"Cannot partition {len(rows)} rows into {target_size} whole-document rows")
    selected: set[str] = set()
    count = target_size
    while count:
        parent, document_id = previous[count]
        selected.add(document_id)
        count = parent
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    source = args.source.resolve()
    target = args.target.resolve()
    if source == target:
        raise ValueError("Source and target must be different directories")
    if target.exists():
        raise FileExistsError(f"Target already exists: {target}")

    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    source_clients = source_manifest["clients"]
    if len(source_clients) != 5:
        raise ValueError("Expected exactly five source clients")
    manifest = {
        "dataset": "same-task cross-domain extractive QA, ten clients",
        "source_dataset": source.name,
        "seed": args.seed,
        "task": source_manifest["task"],
        "clients": {},
        "sampling": {"train_per_client": CLIENT_SIZE, "test_per_domain": 500},
        "split_policy": "two exact 500-row, document-disjoint clients per domain",
        "domains": {},
    }
    for domain_index in range(5):
        domain = source_clients[str(domain_index)]
        source_train = source / "train" / f"{domain_index}.jsonl"
        rows = _read_jsonl(source_train)
        if len(rows) != 2 * CLIENT_SIZE:
            raise ValueError(f"{domain}: expected 1,000 training rows, got {len(rows)}")
        selected_docs = _select_document_ids(rows, CLIENT_SIZE, args.seed + domain_index * 100)
        first = [row for row in rows if row["document_id"] in selected_docs]
        second = [row for row in rows if row["document_id"] not in selected_docs]
        if len(first) != CLIENT_SIZE or len(second) != CLIENT_SIZE:
            raise RuntimeError(f"{domain}: client sizes are not exactly {CLIENT_SIZE}")
        for offset, client_rows in enumerate((first, second)):
            client_id = 2 * domain_index + offset
            _write_jsonl(target / "train" / f"{client_id}.jsonl", client_rows)
            manifest["clients"][str(client_id)] = domain

        source_test = source / "test" / f"{domain}.jsonl"
        test_rows = _read_jsonl(source_test)
        if len(test_rows) != 500:
            raise ValueError(f"{domain}: expected 500 test rows, got {len(test_rows)}")
        target_test = target / "test" / f"{domain}.jsonl"
        target_test.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_test, target_test)
        manifest["domains"][domain] = {
            "source_train_sha256": _sha256(source_train),
            "test_sha256": _sha256(source_test),
            "client_ids": [2 * domain_index, 2 * domain_index + 1],
            "train_per_client": CLIENT_SIZE,
            "test_rows": len(test_rows),
        }
        print(f"{domain}: clients {2 * domain_index},{2 * domain_index + 1} = 500+500; test=500")

    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {target}")


if __name__ == "__main__":
    main()
