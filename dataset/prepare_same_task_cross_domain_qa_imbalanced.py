"""Build a five-client QA partition with unequal training counts.

The existing five-domain dataset remains the reference. Its 500-row test
files are copied byte-for-byte. SQuAD and NewsQA retain their original 1,000
training rows and add rows from the same local MRQA train gzip files. The
other three domains are deterministic subsets of their original train files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from dataset.prepare_same_task_cross_domain_qa import _usable


SOURCE = PROJECT_ROOT / "dataset" / "same_task_cross_domain_qa"
TARGET = PROJECT_ROOT / "dataset" / "same_task_cross_domain_qa_imbalanced"
DOMAINS = ("squad", "newsqa", "bioasq", "duorc", "textbookqa")
TRAIN_SIZES = (1600, 1400, 500, 700, 800)
SEED = 42


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


def _validate(domain: str, train: list[dict], test: list[dict], expected: int) -> None:
    train_ids = [row["id"] for row in train]
    test_ids = {row["id"] for row in test}
    train_docs = {row["document_id"] for row in train}
    test_docs = {row["document_id"] for row in test}
    if len(train) != expected or len(test) != 500:
        raise ValueError(f"{domain}: incorrect train/test counts")
    if len(set(train_ids)) != len(train_ids) or set(train_ids) & test_ids:
        raise ValueError(f"{domain}: duplicate or train/test-overlapping question IDs")
    if train_docs & test_docs:
        raise ValueError(f"{domain}: train/test document overlap")
    if any(row["answer"] not in row["context"] for row in train + test):
        raise ValueError(f"{domain}: answer missing from context")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    source = args.source.resolve()
    target = args.target.resolve()
    if source == target or target.exists():
        raise ValueError(f"Target must be a new dataset directory distinct from source: {target}")

    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if tuple(source_manifest["clients"][str(i)] for i in range(5)) != DOMAINS:
        raise ValueError("Source client/domain mapping differs from the expected five domains")

    manifest = {
        "dataset": "same-task cross-domain extractive QA, imbalanced training",
        "source_dataset": source.name,
        "source_manifest_sha256": _sha256(source_manifest_path),
        "seed": args.seed,
        "task": source_manifest["task"],
        "schema": source_manifest["schema"],
        "clients": {str(i): domain for i, domain in enumerate(DOMAINS)},
        "sampling": {
            "train_per_client": {str(i): count for i, count in enumerate(TRAIN_SIZES)},
            "total_train": sum(TRAIN_SIZES),
            "test_per_domain": 500,
        },
        "sources": {},
    }
    with TemporaryDirectory(prefix="same_task_qa_imbalanced_", dir=target.parent) as temporary:
        staged = Path(temporary) / "dataset"
        for index, (domain, count) in enumerate(zip(DOMAINS, TRAIN_SIZES)):
            original = _read_jsonl(source / "train" / f"{index}.jsonl")
            test_path = source / "test" / f"{domain}.jsonl"
            test = _read_jsonl(test_path)
            if len(original) != 1000 or len(test) != 500:
                raise ValueError(f"{domain}: source is not the expected 1000-train/500-test dataset")
            if count > len(original):
                raw_path = source / "raw" / f"{domain}_train.jsonl.gz"
                expected_raw_hash = source_manifest["sources"][domain]["raw_sha256"]["train"]
                if _sha256(raw_path) != expected_raw_hash:
                    raise ValueError(f"{domain}: local MRQA training gzip differs from source manifest")
                pool = _usable(raw_path, domain, "train")
                existing_ids = {row["id"] for row in original}
                held_out_docs = {row["document_id"] for row in test}
                candidates = [
                    row for row in pool
                    if row["id"] not in existing_ids and row["document_id"] not in held_out_docs
                ]
                additions = random.Random(args.seed + index * 100 + 2).sample(
                    candidates, count - len(original)
                )
                train = original + additions
                policy = "retain original 1000; add from same MRQA train source"
                extra = {"raw_train_sha256": expected_raw_hash, "added_rows": len(additions)}
            else:
                train = random.Random(args.seed + index * 100 + 2).sample(original, count)
                policy = "sample subset of original 1000 training rows"
                extra = {"added_rows": 0}

            _validate(domain, train, test, count)
            _write_jsonl(staged / "train" / f"{index}.jsonl", train)
            target_test = staged / "test" / f"{domain}.jsonl"
            target_test.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(test_path, target_test)
            manifest["sources"][domain] = {
                "sampling_policy": policy,
                "source_train_sha256": _sha256(source / "train" / f"{index}.jsonl"),
                "test_sha256": _sha256(test_path),
                "train_rows": len(train),
                "test_rows": len(test),
                **extra,
            }
            print(f"{domain}: client {index}, train={len(train)}, test={len(test)}", flush=True)

        (staged / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staged.replace(target)
    print(f"Prepared {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
