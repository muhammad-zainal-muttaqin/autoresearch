"""
Second editable file for nontrivial model, pipeline, or loss changes.

Keep the contract stable:
- configure_experiment(model_ref, train_args) -> (model_ref_or_yolo, train_args)
- validate_experiment() optional
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def configure_experiment(model_ref: Any, train_args: dict) -> tuple[Any, dict]:
    """Return a possibly modified model reference and training args."""
    return model_ref, deepcopy(train_args)


def validate_experiment() -> None:
    """Optional validation hook for custom research changes."""
    return None
