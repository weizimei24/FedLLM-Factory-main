import hashlib
import heapq
import json
import re
import time
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from stage1.config import DOMAINS


class _HTMLTextExtractor(HTMLParser):
    _BREAK_TAGS = {"br", "p", "div", "li", "pre", "blockquote", "h1", "h2", "h3", "h4"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self._BREAK_TAGS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag):
        if tag in self._BREAK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def clean_html(value: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(value or "")
        text = "".join(parser.parts)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", value or "")
    text = unescape(text).replace("\ufffd", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def choose_answer(answers) -> dict | None:
    usable = [a for a in (answers or []) if a and clean_html(a.get("text", ""))]
    if not usable:
        return None
    accepted = [a for a in usable if a.get("selected")]
    pool = accepted if accepted else usable
    return max(pool, key=lambda a: (int(a.get("pm_score") or 0), -int(a.get("answer_id") or 0)))


def _question_fingerprint(question: str) -> str:
    normalized = re.sub(r"\W+", " ", question.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _priority(seed: int, domain: str, qid: int) -> int:
    value = f"{seed}:{domain}:{qid}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def chat_token_ids(tokenizer, question: str, answer: str | None = None) -> list[int]:
    messages = [{"role": "user", "content": question}]
    kwargs = {"tokenize": True, "enable_thinking": False}
    if answer is None:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)
    messages.append({"role": "assistant", "content": answer})
    return tokenizer.apply_chat_template(messages, **kwargs)


def encode_training_example(tokenizer, question: str, answer: str, max_length: int) -> dict:
    prompt_ids = chat_token_ids(tokenizer, question)
    input_ids = chat_token_ids(tokenizer, question, answer)
    if len(input_ids) > max_length:
        raise ValueError("Example exceeds max_length; data preparation should filter it")
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Qwen chat template training text does not start with generation prompt")
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        raise ValueError("Example has no assistant answer tokens")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def _domain_candidates(raw_dir: Path, domain: str, site: str, tokenizer, max_length: int,
                       seed: int, keep: int) -> tuple[list[dict], dict]:
    files = sorted((raw_dir / "data" / site).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for {site} in {raw_dir}")

    # First choose a deterministic, uniformly distributed candidate pool by
    # qid. Tokenizing every row in the large math site would add no value to
    # the fixed-size experiment split.
    raw_keep = max(keep * 4, keep + 3000)
    heap = []
    stats = {"rows": 0, "no_answer": 0, "too_short": 0, "too_long": 0, "duplicates": 0}
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512, columns=["qid", "question", "answers"]):
            for row in batch.to_pylist():
                stats["rows"] += 1
                qid = int(row["qid"])
                priority = _priority(seed, domain, qid)
                entry = (-priority, qid, stats["rows"], row)
                if len(heap) < raw_keep:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)

    raw_candidates = [entry[3] for entry in sorted(heap, key=lambda x: (-x[0], x[1], x[2]))]
    selected = []
    seen = set()
    for row in raw_candidates:
        answer_row = choose_answer(row["answers"])
        if answer_row is None:
            stats["no_answer"] += 1
            continue
        question = clean_html(row["question"])
        answer = clean_html(answer_row["text"])
        if len(question) < 20 or len(answer) < 20:
            stats["too_short"] += 1
            continue
        fp = _question_fingerprint(question)
        if fp in seen:
            stats["duplicates"] += 1
            continue
        seen.add(fp)
        if len(chat_token_ids(tokenizer, question, answer)) > max_length:
            stats["too_long"] += 1
            continue
        item = {
            "domain": domain,
            "site": site,
            "qid": int(row["qid"]),
            "answer_id": int(answer_row["answer_id"]),
            "question": question,
            "answer": answer,
            "accepted": bool(answer_row.get("selected")),
            "answer_score": int(answer_row.get("pm_score") or 0),
            "fingerprint": fp,
        }
        selected.append(item)
        if len(selected) == keep:
            break
    return selected, stats


def _validate_answer_owners(rows: list[dict], site: str) -> tuple[list[dict], dict]:
    """Keep only answers whose authoritative StackExchange owner is this qid."""
    import requests

    api_site = site.split(".")[0]
    answer_ids = sorted({row["answer_id"] for row in rows})
    owners = {}
    quota_remaining = None
    session = requests.Session()
    session.headers["User-Agent"] = "FedLLM-Factory-Stage1/1.0"
    for start in range(0, len(answer_ids), 100):
        ids = answer_ids[start:start + 100]
        url = "https://api.stackexchange.com/2.3/answers/" + ";".join(map(str, ids))
        for attempt in range(3):
            try:
                response = session.get(url, params={"site": api_site, "pagesize": 100}, timeout=30)
                response.raise_for_status()
                payload = response.json()
                break
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        quota_remaining = payload.get("quota_remaining", quota_remaining)
        owners.update({int(item["answer_id"]): int(item["question_id"]) for item in payload["items"]})
        if payload.get("backoff"):
            time.sleep(payload["backoff"])

    verified = []
    mismatches = 0
    unavailable = 0
    for row in rows:
        owner = owners.get(row["answer_id"])
        if owner is None:
            unavailable += 1
        elif owner != row["qid"]:
            mismatches += 1
        else:
            row["verified_question_id"] = owner
            verified.append(row)
    return verified, {
        "checked": len(rows),
        "verified": len(verified),
        "mismatches": mismatches,
        "api_unavailable": unavailable,
        "quota_remaining": quota_remaining,
    }


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_stackexchange(raw_dir: str, output_dir: str, tokenizer, train_samples: int = 1000,
                          test_samples: int = 200, max_length: int = 1152, seed: int = 42) -> dict:
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    required = train_samples + test_samples
    candidate_count = required + 800
    candidates = {}
    manifest = {
        "seed": seed,
        "max_length": max_length,
        "train_samples_per_domain": train_samples,
        "test_samples_per_domain": test_samples,
        "domains": {},
    }

    for domain, site in DOMAINS:
        rows, stats = _domain_candidates(
            raw_path, domain, site, tokenizer, max_length, seed, candidate_count
        )
        print(f"Prepared candidate pool for {domain}: {len(rows)} eligible from {stats['rows']} rows")
        verified, verification = _validate_answer_owners(rows, site)
        print(
            f"Verified {domain}: {verification['verified']}/{verification['checked']} valid, "
            f"{verification['mismatches']} mismatched, "
            f"{verification['api_unavailable']} unavailable"
        )
        candidates[domain] = verified
        manifest["domains"][domain] = {
            "site": site,
            "scan": stats,
            "answer_owner_verification": verification,
        }

    globally_used = set()
    for domain, _ in DOMAINS:
        unique_rows = []
        for row in candidates[domain]:
            if row["fingerprint"] not in globally_used:
                globally_used.add(row["fingerprint"])
                unique_rows.append(row)
            if len(unique_rows) == required:
                break
        if len(unique_rows) < required:
            raise RuntimeError(f"{domain} has only {len(unique_rows)} eligible unique examples; need {required}")
        test_rows = unique_rows[:test_samples]
        train_rows = unique_rows[test_samples:]
        _write_jsonl(output_path / domain / "train.jsonl", train_rows)
        _write_jsonl(output_path / domain / "test.jsonl", test_rows)
        manifest["domains"][domain].update({
            "train": len(train_rows),
            "test": len(test_rows),
            "accepted_answers": sum(row["accepted"] for row in unique_rows),
        })

    train_fps = set()
    test_fps = set()
    for domain, _ in DOMAINS:
        train_fps.update(row["fingerprint"] for row in read_jsonl(output_path / domain / "train.jsonl"))
        test_fps.update(row["fingerprint"] for row in read_jsonl(output_path / domain / "test.jsonl"))
    overlap = train_fps & test_fps
    if overlap:
        raise RuntimeError(f"Found {len(overlap)} train/test overlaps")
    manifest["train_test_overlap"] = 0
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class QADataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.encoded = [
            encode_training_example(tokenizer, row["question"], row["answer"], max_length)
            for row in rows
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.encoded[index]


class QACollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        length = max(len(example["input_ids"]) for example in examples)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for example in examples:
            padding = length - len(example["input_ids"])
            batch["input_ids"].append(example["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(example["attention_mask"] + [0] * padding)
            batch["labels"].append(example["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def load_domain_rows(data_dir: str, domain: str, split: str, limit: int | None = None) -> list[dict]:
    rows = read_jsonl(Path(data_dir) / domain / f"{split}.jsonl")
    return rows if limit is None else rows[:limit]
