"""Model size presets for different parameter counts."""

from emsearch.config import ModelConfig

PRESETS: dict[str, ModelConfig] = {
    "5m": ModelConfig(
        d_model=256,
        n_heads=4,
        n_layers=6,
    ),
    "10m": ModelConfig(
        d_model=256,
        n_heads=4,
        n_layers=12,
    ),
    "50m": ModelConfig(
        d_model=512,
        n_heads=8,
        n_layers=16,
    ),
    "150m": ModelConfig(
        d_model=768,
        n_heads=12,
        n_layers=20,
    ),
    "270m": ModelConfig(
        d_model=1024,
        n_heads=16,
        n_layers=22,
    ),
}


def get_preset(name: str) -> ModelConfig:
    """Get a model config preset by name."""
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        msg = f"Unknown preset '{name}'. Available: {available}"
        raise ValueError(msg)
    return PRESETS[name]
