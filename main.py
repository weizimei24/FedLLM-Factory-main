import importlib.util
import sys
import numpy as np
import os

from utils.logger import get_logger
from utils.options import args_parser
from utils.seed_utils import set_global_seed
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class FedSim:
    def __init__(self, args):
        set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
        print(f"Reproducibility: seed={args.seed}, deterministic={args.deterministic}")
        self.args = args
        args.suffix = f'exp/{args.suffix}'
        self.logger = get_logger(args)

        # === route to algorithm module ===
        if args.mode == 'prototype':
            alg_module = importlib.import_module(f'alg.{args.alg}')
        elif args.mode == 'realistic':
            alg_module = importlib.import_module(f'alg.{args.alg}_event')

        # === init clients & server ===
        self.clients = [alg_module.Client(idx, args) for idx in tqdm(range(args.cn), desc="Loading clients...")]
        # Client data loading must not influence the subsequent LoRA
        # initialization RNG state.
        set_global_seed(args.seed, device=args.device, deterministic=args.deterministic)
        self.server = alg_module.Server(args, self.clients)
        
        if args.mode == 'prototype':
            self.simulate()
        elif args.mode == 'realistic':
            self.simulate_ddl()

    def simulate(self):
        TEST_GAP = self.args.test_gap

        try:
            for rnd in tqdm(range(0, self.args.rnd), desc='Communication round', leave=False):
                # ===================== train =====================
                self.server.round = rnd
                self.server.run()

                # A snapshot is taken immediately after aggregation, before
                # evaluation, so every reported point can be reproduced later.
                if getattr(self.args, 'cross_domain_qa', False) and hasattr(self.server, 'save_round_adapter'):
                    self.server.save_round_adapter(rnd)

                # ===================== test =====================
                if (self.args.rnd - rnd <= 10) or (rnd % TEST_GAP == (TEST_GAP-1)):
                    ret_dict = self.server.test_all()
                    metrics_str = '  '.join(f'{k}: {v:.4f}' for k, v in ret_dict.items())
                    self.logger.info(f'[Round {rnd}] {metrics_str}  wall_clock: {self.server.wall_clock_time}')

        except KeyboardInterrupt:
            ...
        finally:
            self.server.save_adapter()


    def simulate_ddl(self):
        TEST_GAP = self.args.test_gap
        try:
            while self.server.early_break is False:
                self.server.run()

                if (self.args.rnd - self.server.round <= 10) or (self.server.round % TEST_GAP == (TEST_GAP-1)):
                    ret_dict = self.server.test_all()
                    metrics_str = '  '.join(f'{k}: {v:.4f}' for k, v in ret_dict.items())
                    self.logger.info(f'[Round {self.server.round}] {metrics_str}  wall_clock: {self.server.wall_clock_time}')

        except KeyboardInterrupt:
            ...
        finally:
            self.server.save_adapter()

if __name__ == '__main__':
    FedSim(args=args_parser())
