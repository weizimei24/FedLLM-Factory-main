import argparse
import json
from pathlib import Path

from stage1.config import ROOT
from stage1.data import prepare_stackexchange
from stage1.model import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Prepare the fixed six-client StackExchange Stage 1 split")
    parser.add_argument("--raw-dir", default=str(ROOT / "dataset" / "raw_stackexchange"))
    parser.add_argument("--output-dir", default=str(ROOT / "dataset" / "stackexchange_stage1"))
    parser.add_argument("--model-path", default=str(ROOT / "Qwen3-1.7B"))
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=1152)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    manifest = prepare_stackexchange(
        args.raw_dir, args.output_dir, tokenizer, args.train_samples,
        args.test_samples, args.max_length, args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

