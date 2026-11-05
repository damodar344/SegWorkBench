"""Model configuration and checkpoint utilities for SegWorkbench.

Provides dataclass-based configs for model hyperparameters and a lightweight
checkpoint manager that persists model state, optimizer state, training metadata,
and the configuration itself.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Configuration for a segmentation model.

    Attributes:
        model_name: Identifier for the model architecture (e.g. "unet").
        n_channels: Number of input channels.
        n_classes: Number of output classes.
        bilinear: Whether to use bilinear upsampling in the decoder.
        image_size: Spatial size (H, W) expected by the model.
        learning_rate: Initial learning rate for training.
        batch_size: Training batch size.
        epochs: Total number of training epochs.
        patience: Early-stopping patience (epochs).
        min_delta: Minimum improvement for early stopping.
        extra: Free-form dictionary for additional hyperparameters.
    """

    model_name: str = "unet"
    n_channels: int = 3
    n_classes: int = 1
    bilinear: bool = True
    image_size: tuple[int, int] = (256, 256)
    learning_rate: float = 1e-3
    batch_size: int = 4
    epochs: int = 50
    patience: int = 5
    min_delta: float = 1e-3
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Deserialize from a plain dictionary."""
        # Convert list -> tuple for image_size if necessary
        if "image_size" in data and isinstance(data["image_size"], list):
            data["image_size"] = tuple(data["image_size"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save_json(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "ModelConfig":
        """Load configuration from a JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


class CheckpointManager:
    """Manages saving and loading of training checkpoints.

    Checkpoints are stored as ``.pth`` (or ``.pt``) files and contain:
    - ``model_state_dict``:  Model weights.
    - ``optimizer_state_dict``:  Optimizer state (optional).
    - ``epoch``:  Last completed epoch.
    - ``best_metric``:  Best tracked metric so far (e.g. validation Dice).
    - ``config``:  The ``ModelConfig`` used for the run.
    - ``extra``:  Any additional user-defined state.
    """

    def __init__(self, checkpoint_dir: str | Path, keep_last_n: int = 3):
        """
        Args:
            checkpoint_dir: Directory where checkpoints will be written.
            keep_last_n: Number of most-recent checkpoints to retain.
                ``<= 0`` means keep all.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = 0,
        best_metric: float | None = None,
        config: ModelConfig | None = None,
        filename: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: The model to checkpoint.
            optimizer: Optional optimizer whose state should be saved.
            epoch: Current training epoch.
            best_metric: Best validation metric observed so far.
            config: Model configuration to embed in the checkpoint.
            filename: Checkpoint filename. If ``None``, a name based on the
                epoch is generated automatically.
            extra: Additional arbitrary state to store.

        Returns:
            Path to the written checkpoint file.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pth"

        filepath = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": config.to_dict() if config is not None else None,
            "extra": extra or {},
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, filepath)
        self._prune_old_checkpoints()
        return filepath

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        filename: str | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint.

        Args:
            model: Model whose weights will be restored.
            optimizer: Optional optimizer whose state will be restored.
            filename: Checkpoint filename. If ``None``, the most recent
                checkpoint in ``checkpoint_dir`` is used.
            device: Device to map tensors onto (``"cpu"``, ``"cuda"``, etc.).

        Returns:
            The raw checkpoint dictionary (minus the heavy ``state_dict``
            entries) so callers can inspect epoch, metric, config, etc.
        """
        if filename is None:
            filepath = get_latest_checkpoint(self.checkpoint_dir)
            if filepath is None:
                raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
        else:
            filepath = self.checkpoint_dir / filename

        checkpoint = torch.load(filepath, map_location=device, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Return metadata only for convenience
        metadata = {k: v for k, v in checkpoint.items() if k not in ("model_state_dict", "optimizer_state_dict")}
        return metadata

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _prune_old_checkpoints(self) -> None:
        """Remove older checkpoints, keeping only the most recent ``keep_last_n``."""
        if self.keep_last_n <= 0:
            return

        checkpoints = _list_checkpoint_files(self.checkpoint_dir)
        if len(checkpoints) > self.keep_last_n:
            for _, path in checkpoints[:-self.keep_last_n]:
                os.remove(path)


# ------------------------------------------------------------------------- #
# Stand-alone convenience functions
# ------------------------------------------------------------------------- #

def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    best_metric: float | None = None,
    config: ModelConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save a single checkpoint without a manager instance.

    Returns the path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "config": config.to_dict() if config is not None else None,
        "extra": extra or {},
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Load a single checkpoint without a manager instance.

    Returns the checkpoint metadata (epoch, best_metric, config, etc.).
    """
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {k: v for k, v in checkpoint.items() if k not in ("model_state_dict", "optimizer_state_dict")}


def get_latest_checkpoint(directory: str | Path) -> Path | None:
    """Return the most recent checkpoint file in *directory*, or ``None``."""
    checkpoints = _list_checkpoint_files(Path(directory))
    return checkpoints[-1][1] if checkpoints else None


# ------------------------------------------------------------------------- #
# Private helpers
# ------------------------------------------------------------------------- #

_CHECKPOINT_RE = re.compile(r"checkpoint.*\.p(th|t)$", re.IGNORECASE)


def _list_checkpoint_files(directory: Path) -> list[tuple[float, Path]]:
    """List checkpoint files in *directory* sorted by mtime (oldest first)."""
    files = [p for p in directory.iterdir() if p.is_file() and _CHECKPOINT_RE.match(p.name)]
    return sorted(((p.stat().st_mtime, p) for p in files), key=lambda x: x[0])
