"""Prepare a five-client, same-task, cross-domain extractive-QA dataset.

The result lives in its own dataset directory and follows the repository's
existing client-indexed layout:

    dataset/same_task_cross_domain_qa/train/<client_id>.jsonl
    dataset/same_task_cross_domain_qa/test/<domain>.jsonl

SQuAD and NewsQA use their released MRQA train/dev splits.  MRQA only releases
one unified evaluation file for BioASQ, DuoRC, and TextbookQA; for those three
domains this script makes a deterministic, context-document-disjoint 1000/500
partition from that file.  The held-out 500 rows are never read by training.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "dataset" / "same_task_cross_domain_qa"
DOMAINS = ("squad", "newsqa", "bioasq", "duorc", "textbookqa")
SEED = 42
TRAIN_SIZE = 1000
TEST_SIZE = 500
CONTEXT_WINDOW_CHARS = 1400
MRQA_BASE = "https://s3.us-east-2.amazonaws.com/mrqa/release/v2"
SOURCES = {
    "squad": {
        "train": f"{MRQA_BASE}/train/SQuAD.jsonl.gz",
        "test": f"{MRQA_BASE}/dev/SQuAD.jsonl.gz",
        "mode": "released_splits",
    },
    "newsqa": {
        "train": f"{MRQA_BASE}/train/NewsQA.jsonl.gz",
        "test": f"{MRQA_BASE}/dev/NewsQA.jsonl.gz",
        "mode": "released_splits",
    },
    "bioasq": {
        "source": "https://participants-area.bioasq.org/MRQA2019/",
        "mode": "partition_single_mrqa_file",
    },
    "duorc": {
        "source": f"{MRQA_BASE}/dev/DuoRC.ParaphraseRC.jsonl.gz",
        "mode": "partition_single_mrqa_file",
    },
    "textbookqa": {
        "source": f"{MRQA_BASE}/dev/TextbookQA.jsonl.gz",
        "mode": "partition_single_mrqa_file",
    },
}


def _sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    """Download exactly once and reject unexpected non-gzip responses."""
    if destination.is_file() and destination.stat().st_size > 0:
        try:
            with gzip.open(destination, "rb") as handle:
                while handle.read(1024 * 1024):
                    pass
            return
        except (EOFError, gzip.BadGzipFile, OSError):
            # A previous interrupted download must not be treated as valid.
            pass
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "FedLLM-Factory/1.0"})
    with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    with temporary.open("rb") as handle:
        if handle.read(2) != b"\x1f\x8b":
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"{url} did not return a gzip MRQA file")
    temporary.replace(destination)


def _clean_context(context: str) -> str:
    return (
        context.replace("[PAR] ", "\n\n").replace("[TLE]", "Title:")
        .replace("[SEP]", "\nPassage:").replace("[DOC]", "").strip()
    )


def _clean_spaces(value: str) -> str:
    for source, replacement in (
        (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","),
        (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"),
        (" 've", "'ve"), (" 're", "'re"), ("( ", "("), (" )", ")"),
        (" %", "%"), (" :", ":"),
    ):
        value = value.replace(source, replacement)
    return value


def _unique_nonempty(values: list[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _clean_spaces(str(value).strip())
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _qa_answers(qa: dict, raw_context: str) -> tuple[list[str], list[str]]:
    """Return MRQA gold strings and detected extractive span strings.

    MRQA's ``answers`` are all accepted source answers and need not always be
    literal substrings.  ``detected_answers`` is the accompanying normalized
    extractive-span field, so it is the authoritative fallback for a target
    that must be found in the supplied context.
    """
    gold = _unique_nonempty(qa.get("answers", []))
    detected_values = [
        item.get("text", "") for item in qa.get("detected_answers", [])
        if isinstance(item, dict)
    ]
    # Some BioASQ records contain a detected-answer string which differs from
    # the literal passage (typically an added article) despite a correct
    # MRQA char span.  Recover the exact extractive target from that span.
    for item in qa.get("detected_answers", []):
        if not isinstance(item, dict):
            continue
        for span in item.get("char_spans", []):
            if isinstance(span, list) and len(span) == 2:
                start, end = span
                if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end < len(raw_context):
                    detected_values.append(raw_context[start:end + 1])
    detected = _unique_nonempty(detected_values)
    return gold, detected


def _iter_mrqa_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        header = json.loads(next(handle))
        if "header" not in header:
            raise ValueError(f"{path} is not an MRQA JSONL.GZ file")
        for line in handle:
            paragraph = json.loads(line)
            raw_context = str(paragraph.get("context", ""))
            context = _clean_spaces(_clean_context(raw_context))
            # MRQA carries no stable original-document field consistently across
            # all five sources.  A hash of the complete pre-window context is a
            # conservative document surrogate: a shared passage cannot cross
            # train/test even when it has several questions.
            document_id = _sha256_bytes(context)
            for qa in paragraph.get("qas", []):
                question = _clean_spaces(" ".join(str(qa.get("question", "")).split()))
                gold_answers, detected_answers = _qa_answers(qa, raw_context)
                if context and question and (gold_answers or detected_answers):
                    yield {
                        "qid": str(qa.get("qid", "")),
                        "context": context,
                        "document_id": document_id,
                        "question": question,
                        "answers": gold_answers,
                        "detected_answers": detected_answers,
                    }


def _answer_window(context: str, answer: str) -> str:
    start = context.find(answer)
    if start < 0:
        raise ValueError("answer is absent from context")
    if len(context) <= CONTEXT_WINDOW_CHARS:
        return context
    left = max(0, start - (CONTEXT_WINDOW_CHARS - len(answer)) // 2)
    right = min(len(context), left + CONTEXT_WINDOW_CHARS)
    left = max(0, right - CONTEXT_WINDOW_CHARS)
    window = context[left:right]
    if answer not in window:
        raise RuntimeError("answer-centred window lost its answer")
    return window


def _canonicalise(row: dict, domain: str, source_split: str) -> dict | None:
    answer_candidates = _unique_nonempty(row["answers"] + row["detected_answers"])
    label = next((answer for answer in answer_candidates if answer in row["context"]), None)
    if label is None:
        return None
    context = _answer_window(row["context"], label)
    answers = [answer for answer in answer_candidates if answer in context]
    return {
        "id": f"{domain}:{row['qid']}",
        "domain": domain,
        "source_split": source_split,
        "document_id": row["document_id"],
        "context": context,
        "question": row["question"],
        "answer": label,
        # ``label``/``answers`` retain compatibility with the existing LoRA
        # formatter and multi-reference generation evaluator.
        "label": label,
        "answers": answers,
    }


def _usable(path: Path, domain: str, source_split: str) -> list[dict]:
    rows = [_canonicalise(row, domain, source_split) for row in _iter_mrqa_rows(path)]
    rows = [row for row in rows if row is not None]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{domain}/{source_split} has duplicate MRQA qids")
    return rows


def _sample_rows(rows: list[dict], count: int, seed: int) -> list[dict]:
    if len(rows) < count:
        raise ValueError(f"requested {count} rows but only {len(rows)} are usable")
    return random.Random(seed).sample(rows, count)


def _choose_document_groups(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Choose whole context groups with exactly ``count`` QA rows, if possible."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["document_id"]].append(row)
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    # Dynamic programming finds an exact sum while keeping documents intact.
    choices: dict[int, tuple[int, ...]] = {0: ()}
    for index, group in enumerate(ordered):
        size = len(group)
        for total in sorted(list(choices), reverse=True):
            new_total = total + size
            if new_total <= count and new_total not in choices:
                choices[new_total] = choices[total] + (index,)
        if count in choices:
            break
    if count not in choices:
        sizes = sorted({len(group) for group in ordered})
        raise ValueError(
            f"cannot form an exact {count}-row document-disjoint split; group sizes={sizes}"
        )
    return [row for index in choices[count] for row in ordered[index]]


def _validate_disjoint(domain: str, train_rows: list[dict], test_rows: list[dict]) -> None:
    train_ids = {row["id"] for row in train_rows}
    test_ids = {row["id"] for row in test_rows}
    train_documents = {row["document_id"] for row in train_rows}
    test_documents = {row["document_id"] for row in test_rows}
    if train_ids & test_ids:
        raise RuntimeError(f"{domain}: train/test question-ID overlap")
    if train_documents & test_documents:
        raise RuntimeError(f"{domain}: train/test document overlap")
    for row in train_rows + test_rows:
        if row["answer"] not in row["context"]:
            raise RuntimeError(f"{domain}: answer missing from stored context")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="new, isolated dataset root")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    raw = root / "raw"
    if (root / "train").exists() or (root / "test").exists():
        raise FileExistsError(
            f"Refusing to overwrite existing prepared data at {root}. "
            "Choose another --root after preserving or removing it explicitly."
        )

    manifest = {
        "dataset": "same-task cross-domain extractive QA",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "clients": {str(index): domain for index, domain in enumerate(DOMAINS)},
        "sampling": {"train_per_domain": TRAIN_SIZE, "test_per_domain": TEST_SIZE},
        "task": "Context + Question -> Answer",
        "schema": ["context", "question", "answer"],
        "document_split": "complete MRQA context SHA-256; no context hash is shared across train/test",
        "sources": {},
    }
    for index, domain in enumerate(DOMAINS):
        spec = SOURCES[domain]
        print(f"[{domain}] downloading/checking official MRQA source ...", flush=True)
        if spec["mode"] == "released_splits":
            train_raw = raw / f"{domain}_train.jsonl.gz"
            test_raw = raw / f"{domain}_dev.jsonl.gz"
            _download(spec["train"], train_raw)
            _download(spec["test"], test_raw)
            train_pool = _usable(train_raw, domain, "train")
            test_pool = _usable(test_raw, domain, "dev")
            test_rows = _sample_rows(test_pool, TEST_SIZE, args.seed + index * 100 + 1)
            # The named source splits are official train/dev, but NewsQA has
            # a small number of repeated passages across them.  Exclude every
            # selected dev passage from train candidates before sampling.
            held_out_documents = {row["document_id"] for row in test_rows}
            train_candidates = [row for row in train_pool if row["document_id"] not in held_out_documents]
            train_rows = _sample_rows(train_candidates, TRAIN_SIZE, args.seed + index * 100)
            source_info = {"train_url": spec["train"], "test_url": spec["test"],
                           "raw_sha256": {"train": _sha256_file(train_raw), "test": _sha256_file(test_raw)},
                           "usable_rows": {"train": len(train_pool), "test": len(test_pool)},
                           "train_candidates_after_test_document_exclusion": len(train_candidates)}
        else:
            source_raw = raw / f"{domain}_mrqa.jsonl.gz"
            _download(spec["source"], source_raw)
            pool = _usable(source_raw, domain, "mrqa_released_eval")
            test_rows = _choose_document_groups(pool, TEST_SIZE, args.seed + index * 100 + 1)
            held_out_documents = {row["document_id"] for row in test_rows}
            train_candidates = [row for row in pool if row["document_id"] not in held_out_documents]
            train_rows = _choose_document_groups(train_candidates, TRAIN_SIZE, args.seed + index * 100)
            source_info = {"source_url": spec["source"], "raw_sha256": _sha256_file(source_raw),
                           "usable_rows": len(pool)}
        _validate_disjoint(domain, train_rows, test_rows)
        _write_jsonl(root / "train" / f"{index}.jsonl", train_rows)
        _write_jsonl(root / "test" / f"{domain}.jsonl", test_rows)
        manifest["sources"][domain] = {
            "mrqa_mode": spec["mode"],
            **source_info,
            "output_rows": {"train": len(train_rows), "test": len(test_rows)},
            "train_test_question_overlap": 0,
            "train_test_document_overlap": 0,
        }
        print(f"[{domain}] train={len(train_rows)} test={len(test_rows)}", flush=True)

    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared isolated dataset: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
