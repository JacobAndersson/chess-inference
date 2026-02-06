import torch
from torch import nn

from emsearch.utils import configure_optimizer, get_lr_scheduler


def test_lr_scheduler_warmup():
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = get_lr_scheduler(optimizer, warmup_steps=100, max_steps=1000)

    # At step 0, LR should be ~0
    assert scheduler.get_last_lr()[0] < 1e-6

    # Step through warmup
    for _ in range(50):
        optimizer.step()
        scheduler.step()

    # Midway through warmup, LR should be ~half
    lr = scheduler.get_last_lr()[0]
    assert 1e-4 < lr < 2e-4


def test_lr_scheduler_min_lr():
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = get_lr_scheduler(
        optimizer, warmup_steps=10, max_steps=100, min_lr=3e-5, learning_rate=3e-4
    )

    # Step all the way to end
    for _ in range(100):
        optimizer.step()
        scheduler.step()

    lr = scheduler.get_last_lr()[0]
    # At end, LR should be at min_lr (3e-5)
    assert abs(lr - 3e-5) < 1e-6


def test_lr_scheduler_peak():
    model = nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = get_lr_scheduler(
        optimizer, warmup_steps=10, max_steps=100, min_lr=3e-5, learning_rate=3e-4
    )

    for _ in range(10):
        optimizer.step()
        scheduler.step()

    lr = scheduler.get_last_lr()[0]
    # At end of warmup, LR should be at peak
    assert abs(lr - 3e-4) < 1e-6


def test_configure_optimizer_weight_decay_groups():
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.LayerNorm(10),
        nn.Linear(10, 5),
    )
    optimizer = configure_optimizer(model, learning_rate=1e-3, weight_decay=0.1)

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
