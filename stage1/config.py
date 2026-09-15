from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DOMAINS = (
    ("mathematics", "math.stackexchange.com"),
    ("physics", "physics.stackexchange.com"),
    ("computer_science", "cs.stackexchange.com"),
    ("statistics", "stats.stackexchange.com"),
    ("economics", "economics.stackexchange.com"),
    ("biology", "biology.stackexchange.com"),
)


@dataclass
class Stage1Config:
    model_path: str = str(ROOT / "Qwen3-1.7B")
    data_dir: str = str(ROOT / "dataset" / "stackexchange_stage1")
    output_dir: str = str(ROOT / "results" / "stage1")
    run_name: str = "full"

    seed: int = 42
    train_samples: int = 1000
    test_samples: int = 200
    max_length: int = 1152
    max_new_tokens: int = 768

    batch_size: int = 1
    eval_batch_size: int = 4
    generation_batch_size: int = 20
    grad_accum: int = 32
    learning_rate: float = 2e-4
    phases: int = 10
    rounds: int = 10

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    fedrot_lambda: float = 0.5

    smoke: bool = False

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(name for name, _ in DOMAINS)

    @property
    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name

    def to_dict(self) -> dict:
        return asdict(self)


def smoke_config(config: Stage1Config) -> Stage1Config:
    config.smoke = True
    config.run_name = "smoke"
    config.train_samples = 50
    config.test_samples = 20
    config.phases = 1
    config.rounds = 1
    # This still checks deterministic generation, EOS handling, and ROUGE while
    # keeping the five-method smoke run reasonably short.
    config.max_new_tokens = 32
    return config
