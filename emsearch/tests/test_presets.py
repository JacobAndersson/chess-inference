import pytest

from emsearch.presets import get_preset


def test_5m_preset():
    config = get_preset("5m")
    assert config.d_model == 256
    assert config.n_heads == 4
    assert config.n_layers == 6


def test_50m_preset():
    config = get_preset("50m")
    assert config.d_model == 512
    assert config.n_heads == 8
    assert config.n_layers == 16


def test_unknown_preset():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("nonexistent")
