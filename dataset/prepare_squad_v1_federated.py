"""Create a deterministic five-client IID split from the fixed SQuAD subset."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "squad_v1"
OUTPUT_DIR = ROOT / "squad_v1_fed5"
CLIENTS = 5
SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def ids_sha256(rows: list[dict]) -> str:
    payload = "\n".join(sorted(row["squad_id"] for row in rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    train_rows = read_jsonl(SOURCE_DIR / "train" / "0.jsonl")
    test_rows = read_jsonl(SOURCE_DIR / "test" / "0.jsonl")
    if len(train_rows) != 10_000 or len(test_rows) != 2_000:
        raise ValueError(
            f"Expected fixed 10000/2000 SQuAD subset, got {len(train_rows)}/{len(test_rows)}"
        )

    shuffled = list(train_rows)
    random.Random(SEED).shuffle(shuffled)
    client_size = len(shuffled) // CLIENTS
    clients = [
        shuffled[index * client_size:(index + 1) * client_size]
        for index in range(CLIENTS)
    ]
    if any(len(rows) != 2_000 for rows in clients):
        raise RuntimeError("Five-client split is not exactly balanced")

    client_id_sets = [{row["squad_id"] for row in rows} for rows in clients]
    for left in range(CLIENTS):
        for right in range(left + 1, CLIENTS):
            overlap = client_id_sets[left] & client_id_sets[right]
            if overlap:
                raise RuntimeError(f"Clients {left} and {right} overlap by {len(overlap)} rows")
    if len(set().union(*client_id_sets)) != 10_000:
        raise RuntimeError("Federated client union does not reproduce all 10,000 training rows")

    for client_id, rows in enumerate(clients):
        write_jsonl(OUTPUT_DIR / "train" / f"{client_id}.jsonl", rows)
    write_jsonl(OUTPUT_DIR / "test" / "global.jsonl", test_rows)

    manifest = {
        "dataset": "SQuAD v1.1 five-client IID split",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "clients": CLIENTS,
        "partition": "deterministic uniform random split without replacement",
        "train_total": len(train_rows),
        "train_per_client": {str(i): len(rows) for i, rows in enumerate(clients)},
        "client_id_sha256": {str(i): ids_sha256(rows) for i, rows in enumerate(clients)},
        "client_overlap": 0,
        "global_test": len(test_rows),
        "global_test_id_sha256": ids_sha256(test_rows),
        "source_manifest": "dataset/squad_v1/manifest.json",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {CLIENTS} clients x {client_size} rows and {len(test_rows)} global test rows")


if __name__ == "__main__":
    main()
