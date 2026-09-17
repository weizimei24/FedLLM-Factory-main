"""Download and prepare a deterministic SQuAD v1.1 subset for centralized SFT.

The official train split supplies 10,000 examples and the official development
split supplies 2,000 examples. Context and question remain separate so only
the context may be truncated by the shared Qwen3 chat-template encoder.

Run from the repository root:
    python -X utf8 dataset/prepare_squad_v1.py
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "squad_v1"
RAW_DIR = OUT_DIR / "raw"
SOURCES = {
    # The dataset maintainer's repository serves the same v1.1 files and is
    # materially more reliable on restricted networks than the project page.
    "train": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json",
    "validation": "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json",
}
TRAIN_SIZE = 10_000
TEST_SIZE = 2_000
SEED = 42


def _download(url: str, destination: Path) -> None:
    """Fetch an official SQuAD file unless a valid local copy already exists."""
    if destination.exists() and destination.stat().st_size > 0:
        try:
            payload = json.loads(destination.read_text(encoding="utf-8"))
            if isinstance(payload.get("data"), list):
                return
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            pass
    temp = destination.with_suffix(destination.suffix + ".part")
    # A background BITS transfer can populate ``.bits`` independently.  Once
    # complete, validate and atomically promote it without another download.
    for candidate in (temp, destination.with_suffix(".bits")):
        if not candidate.exists() or candidate.stat().st_size == 0:
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload.get("data"), list):
                candidate.replace(destination)
                return
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "FedLLM-Factory/1.0"})
    with urlopen(request, timeout=60) as response, temp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temp.replace(destination)


def _load_qas(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            context = " ".join(paragraph["context"].split())
            for qa in paragraph["qas"]:
                answers = [" ".join(answer["text"].split()) for answer in qa.get("answers", [])]
                answers = list(dict.fromkeys(answer for answer in answers if answer))
                if answers:
                    rows.append({
                        "id": qa["id"],
                        "title": article.get("title", ""),
                        "context": context,
                        "question": " ".join(qa["question"].split()),
                        "answers": answers,
                    })
    return rows


def _context_window(context: str, primary_answer: str, max_chars: int = 1400) -> str:
    """Keep a compact context window that contains the supervised answer."""
    if len(context) <= max_chars:
        return context
    answer_at = context.lower().find(primary_answer.lower())
    if answer_at < 0:
        return context[:max_chars]
    start = max(0, answer_at - (max_chars - len(primary_answer)) // 2)
    end = min(len(context), start + max_chars)
    start = max(0, end - max_chars)
    return context[start:end]


def _to_factory_row(row: dict) -> dict:
    answer = row["answers"][0]
    context = _context_window(row["context"], answer)
    return {
        "context": context,
        "question": row["question"],
        "label": answer,
        "answers": row["answers"],
        "squad_id": row["id"],
        "title": row["title"],
    }


def _sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    if len(rows) < size:
        raise ValueError(f"Requested {size} rows, but only found {len(rows)} usable SQuAD examples")
    selected = random.Random(seed).sample(rows, size)
    return [_to_factory_row(row) for row in selected]


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


def main() -> int:
    raw_paths = {
        split: RAW_DIR / ("train-v1.1.json" if split == "train" else "dev-v1.1.json")
        for split in SOURCES
    }
    for split, url in SOURCES.items():
        print(f"Downloading/checking SQuAD v1.1 {split} split...")
        _download(url, raw_paths[split])

    train_source = _load_qas(raw_paths["train"])
    test_source = _load_qas(raw_paths["validation"])
    # Official SQuAD v1.1 totals are 87,599 train and 10,570 validation QAs.
    if len(train_source) != 87_599 or len(test_source) != 10_570:
        raise ValueError(
            "Unexpected SQuAD v1.1 source counts: "
            f"train={len(train_source)}, validation={len(test_source)}"
        )

    train_rows = _sample(train_source, TRAIN_SIZE, SEED)
    test_rows = _sample(test_source, TEST_SIZE, SEED + 1)
    train_ids = {row["squad_id"] for row in train_rows}
    test_ids = {row["squad_id"] for row in test_rows}
    if train_ids & test_ids:
        raise RuntimeError("SQuAD train/test ID overlap detected")

    _write_jsonl(OUT_DIR / "train" / "0.jsonl", train_rows)
    _write_jsonl(OUT_DIR / "test" / "0.jsonl", test_rows)
    manifest = {
        "dataset": "SQuAD v1.1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "split_strategy": "10,000 randomly sampled official train QAs; 2,000 randomly sampled official validation QAs",
        "source_counts": {"train": len(train_source), "validation": len(test_source)},
        "output_counts": {"train": len(train_rows), "test": len(test_rows)},
        "sources": {
            split: {"url": url, "sha256": _sha256(raw_paths[split])}
            for split, url in SOURCES.items()
        },
        "schema": {
            "context": "answer-containing context window; the only truncatable prompt field",
            "question": "complete SQuAD question; never truncated",
            "label": "first official answer, used for training",
            "answers": "all official answer strings, used for SQuAD EM/F1",
        },
        "protocol": {
            "model_format": "Qwen3 chat template",
            "enable_thinking": False,
            "answer_termination": "tokenizer.eos_token_id (<|im_end|>)",
            "decoding": "deterministic greedy decoding",
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(train_rows)} training rows and {len(test_rows)} test rows to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
