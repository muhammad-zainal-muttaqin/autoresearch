"""
Agent-editable module for model modifications.

This file changes frequently. Common edits:
- Custom detection heads
- Feature branches (size, position, texture)
- Loss function definitions
- Auxiliary classification heads
- Architecture modifications

Contract:
  configure_model(model_ref, train_args) -> (model_ref_or_yolo, train_args)

The orchestrator calls this before pipeline.configure_pipeline().
If modeling.py or pipeline.py is touched, the experiment is classified as exploration.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def configure_model(model_ref: Any, train_args: dict) -> tuple[Any, dict]:
    """Return a possibly modified model reference and training args.

    Parameters
    ----------
    model_ref : str or YOLO
        Model weight path (e.g. "yolo11s.pt") or a pre-built YOLO instance.
    train_args : dict
        Training arguments that will be passed to model.train().

    Returns
    -------
    (model_ref_or_yolo, train_args)
        The model reference (string or YOLO instance) and updated train_args.

    Examples of what belongs here:
    - Swapping the detection head for a custom one
    - Adding a focal loss or ordinal loss wrapper
    - Attaching an auxiliary feature branch
    - Freezing/unfreezing specific layers
    """
    return model_ref, deepcopy(train_args)
