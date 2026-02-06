"""Configuration dataclasses for model and training."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the GPT-2 style transformer model."""

    vocab_size: int = 32
    max_seq_len: int = 512
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    dropout: float = 0.0
    bias: bool = False
    use_swiglu: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    data_dir: str = ""
    elo_bucket: str = "all"
    max_games: int | None = None
    batch_size: int = 64
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 100_000
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5000
    eval_every: int = 1000
    log_every: int = 100
    min_lr: float = 3e-5
    wandb_project: str | None = None


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    device: str = "cuda"
    compile: bool = True
