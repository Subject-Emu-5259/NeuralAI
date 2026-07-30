"""Checkpoint management utilities for NeuralAI training.

Helpers to locate the best or latest checkpoint, enumerate saved steps,
and load lightweight training-state metadata without pulling full tensors
into memory.
"""

import os
import json
import glob
from pathlib import Path
from typing import Optional, List, Dict, Any


def find_best_checkpoint(
    checkpoint_dir: str,
    metric: str = "eval_loss",
    mode: str = "min",
) -> Optional[str]:
    """Return the checkpoint path with the best evaluation metric.

    Scans ``checkpoint_dir/checkpoint-*/trainer_state.json`` for the
    ``best_metric`` field (or the last occurrence of *metric* in
    ``log_history``).

    Args:
        checkpoint_dir: Root directory containing checkpoint subfolders.
        metric: Metric key to compare (default ``eval_loss``).
        mode: ``min`` (lower is better) or ``max`` (higher is better).

    Returns:
        Absolute path to the best checkpoint, or ``None`` if none found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    candidates: List[Tuple[str, float]] = []

    for path in glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")):
        state_file = os.path.join(path, "trainer_state.json")
        if not os.path.exists(state_file):
            continue

        with open(state_file) as fh:
            state: Dict[str, Any] = json.load(fh)

        best_metric = state.get("best_metric")
        if best_metric is None:
            log_history = state.get("log_history", [])
            eval_entries = [e for e in log_history if metric in e]
            if not eval_entries:
                continue
            best_metric = eval_entries[-1].get(metric)

        if best_metric is not None:
            candidates.append((path, float(best_metric)))

    if not candidates:
        return None

    reverse = mode == "max"
    candidates.sort(key=lambda x: x[1], reverse=reverse)
    return candidates[0][0]


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the most recent checkpoint by step number.

    Args:
        checkpoint_dir: Root directory containing ``checkpoint-{step}`` folders.

    Returns:
        Absolute path to the latest checkpoint, or ``None`` if none found.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    steps: List[Tuple[int, str]] = []
    for path in glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")):
        basename = os.path.basename(path)
        try:
            step = int(basename.replace("checkpoint-", ""))
            steps.append((step, path))
        except ValueError:
            continue

    if not steps:
        return None

    steps.sort(key=lambda x: x[0])
    return steps[-1][1]


def get_checkpoint_steps(checkpoint_dir: str) -> List[int]:
    """Return a sorted list of checkpoint step numbers present on disk.

    Args:
        checkpoint_dir: Root directory containing checkpoint folders.

    Returns:
        Sorted list of integer step numbers.
    """
    steps: List[int] = []
    for path in glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")):
        try:
            step = int(os.path.basename(path).replace("checkpoint-", ""))
            steps.append(step)
        except ValueError:
            continue
    return sorted(steps)


def load_training_state(checkpoint_path: str) -> Dict[str, Any]:
    """Load ``trainer_state.json`` from a checkpoint folder.

    Args:
        checkpoint_path: Path to a single checkpoint directory.

    Returns:
        Parsed JSON dictionary, or empty dict if the file is missing.
    """
    state_file = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(state_file):
        return {}
    with open(state_file) as fh:
        return json.load(fh)
