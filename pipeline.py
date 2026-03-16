"""
Agent-editable module for pipeline structure.

This file changes less often than modeling.py but more dramatically.
Common edits:
- One-stage vs two-stage pipeline orchestration
- Stage sequencing and data flow
- Custom augmentation strategies
- Inference flow modifications
- Pre/post-processing changes

Contract:
  configure_pipeline(model, train_args) -> (model, train_args)

The orchestrator calls this after modeling.configure_model().
If modeling.py or pipeline.py is touched, the experiment is classified as exploration.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def configure_pipeline(model: Any, train_args: dict) -> tuple[Any, dict]:
    """Return a possibly modified model and training args after pipeline configuration.

    Parameters
    ----------
    model : YOLO
        A YOLO model instance (already resolved from configure_model).
    train_args : dict
        Training arguments that will be passed to model.train().

    Returns
    -------
    (model, train_args)
        The model and updated train_args.

    Examples of what belongs here:
    - Restructuring into a two-stage detection pipeline
    - Adding custom augmentation hooks
    - Modifying the inference/post-processing flow
    - Implementing curriculum learning schedules
    """
    return model, deepcopy(train_args)
