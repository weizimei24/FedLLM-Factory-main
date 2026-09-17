import argparse
import importlib
from pathlib import Path
import yaml


def build_parser() -> argparse.ArgumentParser:
    """Create and return the base argument parser with all common args pre-registered.
    Callers can add extra arguments before parsing."""
    # Parse this one option first so a dataset-specific YAML can supply all
    # ordinary defaults without modifying the repository-wide config.yaml.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', default='config.yaml')
    config_args, _ = config_parser.parse_known_args()
    config_path = Path(config_args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    parser = argparse.ArgumentParser(parents=[config_parser])

    ### basic setting
    parser.add_argument('--alg', type=str, default='fedit', help='algorithm name')
    parser.add_argument('--suffix', type=str, default='default', help='experiment suffix')
    parser.add_argument('--device', type=int, default=0, help='device id')
    parser.add_argument('--dataset', type=str, default='', help='dataset name')
    parser.add_argument('--model', type=str, default='', help='model name')
    parser.add_argument('--task_type', type=str, default='', help='task type (SEQ_CLS or CAUSAL_LM)')

    ### FL setting
    parser.add_argument('--cn', type=int, default=10, help='number of clients')
    parser.add_argument('--sr', type=float, default=1.0, help='sample rate')
    parser.add_argument('--rnd', type=int, default=10, help='number of rounds')
    parser.add_argument('--tg', type=int, default=1, help='test gap')
    parser.add_argument('--session_time', type=float, default=24, help='round session duration in hours')
    parser.add_argument('--start_time', type=float, default=0.0, help='simulation start offset in hours from the earliest trace event')

    ### local training setting
    parser.add_argument('--bs', type=int, default=2, help='batch size')
    parser.add_argument('--grad_accum', type=int, default=8, help='gradient accumulation steps')
    parser.add_argument('--epoch', type=int, default=5, help='local epochs')
    parser.add_argument('--step', type=int, default=10, help='local steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--test_gap', type=int, default=1, help='test interval')

    ### async
    parser.add_argument('--decay', type=float, default=0.1, help='decay rate')

    ### event mode
    parser.add_argument('--mode', type=str, default='prototype', help='simulation mode: prototype or realistic')
    parser.add_argument('--upload_bandwidth', type=float, default=1.0, help='uplink bandwidth in Mbps for upload delay estimation')

    ### LoRA
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')

    # === read args from yaml ===
    with config_path.open('r', encoding='utf-8') as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.Loader)
    parser.set_defaults(**yaml_config)

    return parser


def args_parser():
    parser = build_parser()

    # === read args from command ===
    args, _ = parser.parse_known_args()

    # === read specific args from each method
    if args.mode == 'prototype':
            alg_module = importlib.import_module(f'alg.{args.alg}')
    elif args.mode == 'realistic':
        alg_module = importlib.import_module(f'alg.{args.alg}_event')
    # alg_module = importlib.import_module(f'alg.{args.alg}')
    spec_args = alg_module.add_args(parser) if hasattr(alg_module, 'add_args') else args
    return spec_args
