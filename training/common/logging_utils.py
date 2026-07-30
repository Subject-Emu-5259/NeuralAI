"""Shared logging utilities for NeuralAI training.

Wraps WandB and TensorBoard setup with graceful degradation when the
optional backends are unavailable or disabled via environment variables.
"""

import os
from typing import Optional, Dict, Any


def setup_wandb(
    project: str,
    name: str,
    config: Dict[str, Any],
    entity: Optional[str] = None,
) -> Optional[Any]:
    """Initialize WandB if available and not explicitly disabled.

    Args:
        project: WandB project name.
        name: Run name.
        config: Dictionary of hyperparameters to log.
        entity: Optional WandB entity (team or user).

    Returns:
        A ``wandb.sdk.wandb_run.Run`` on success, or ``None`` on failure.
    """
    if os.environ.get("WANDB_DISABLED", "false").lower() == "true":
        return None

    try:
        import wandb
        run = wandb.init(
            project=project,
            name=name,
            config=config,
            entity=entity,
            resume="allow",
        )
        return run
    except Exception as exc:
        print(f"[WARN] WandB setup failed: {exc}")
        return None


def setup_tensorboard(log_dir: str) -> Optional[Any]:
    """Initialize a TensorBoard ``SummaryWriter``.

    Args:
        log_dir: Directory to write event files.

    Returns:
        A ``torch.utils.tensorboard.SummaryWriter`` on success, or ``None``.
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir)
    except Exception as exc:
        print(f"[WARN] TensorBoard setup failed: {exc}")
        return None


def log_metrics(
    logger: Optional[Any],
    step: int,
    metrics: Dict[str, Any],
) -> None:
    """Write metrics to whichever logger(s) are active.

    Accepts either a WandB run (``wandb.log``) or a TensorBoard
    ``SummaryWriter`` (``add_scalar``).

    Args:
        logger: Logger object or ``None``.
        step: Global training step.
        metrics: Dictionary of scalar metrics.
    """
    if logger is None:
        return

    # WandB
    if hasattr(logger, "log"):
        logger.log(metrics, step=step)
        return

    # TensorBoard
    if hasattr(logger, "add_scalar"):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.add_scalar(key, value, step)
        return
