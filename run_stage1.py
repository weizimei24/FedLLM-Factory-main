import argparse

from stage1.baselines import run_method
from stage1.config import Stage1Config, smoke_config
from stage1.trainer import set_seed


METHODS = ("base", "local", "centralized", "fedit", "fedrotlora")


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1 StackExchange experiments")
    parser.add_argument("--method", choices=(*METHODS, "all"), required=True)
    parser.add_argument("--run-name", default="full")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-path")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--fedrot-lambda", type=float, default=0.5)
    args = parser.parse_args()

    config = Stage1Config(run_name=args.run_name, fedrot_lambda=args.fedrot_lambda)
    if args.model_path:
        config.model_path = args.model_path
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.smoke:
        smoke_config(config)
        if args.run_name != "full":
            config.run_name = args.run_name
    set_seed(config.seed)

    methods = METHODS if args.method == "all" else (args.method,)
    for method in methods:
        print(f"\n===== Stage 1: {method} ({config.run_name}) =====")
        run_method(config, method)


if __name__ == "__main__":
    main()

