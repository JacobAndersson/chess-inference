from emsearch.config import ExperimentConfig, ModelConfig, TrainingConfig


def test_model_config_defaults():
    config = ModelConfig()
    assert config.vocab_size == 32
    assert config.max_seq_len == 512
    assert config.d_model == 256
    assert config.n_heads == 4
    assert config.n_layers == 6
    assert config.dropout == 0.0
    assert config.bias is False
    assert config.use_swiglu is True


def test_training_config_defaults():
    config = TrainingConfig()
    assert config.warmup_steps == 2000
    assert config.min_lr == 3e-5
    assert config.learning_rate == 3e-4
    assert config.batch_size == 64


def test_experiment_config_defaults():
    config = ExperimentConfig()
    assert config.seed == 42
    assert config.device == "cuda"
    assert config.compile is True
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.training, TrainingConfig)
