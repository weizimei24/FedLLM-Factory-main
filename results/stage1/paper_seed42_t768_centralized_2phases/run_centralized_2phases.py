"""Reproducible launcher for the two-phase centralized Stage 1 run."""

import sys
from pathlib import Path

# This launcher lives below the output directory, rather than the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stage1.baselines import run_method
from stage1.config import Stage1Config
from stage1.trainer import set_seed


config = Stage1Config(run_name="paper_seed42_t768_centralized_2phases")
config.phases = 2
config.rounds = 2

set_seed(config.seed)
run_method(config, "centralized")
