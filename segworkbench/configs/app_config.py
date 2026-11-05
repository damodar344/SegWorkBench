"""Application-level configuration and recent-folder persistence."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class AppConfig:
    """Persistent application settings.

    Attributes:
        last_image_dir: Last browsed image folder.
        last_mask_dir: Last browsed mask folder.
        last_model_dir: Last browsed model folder.
        last_output_dir: Last browsed output folder.
        window_geometry: Serialized geometry from Qt (e.g. "1200x750+0+0").
        theme: UI theme name.
        extra: Arbitrary extra state.
    """

    last_image_dir: str = ""
    last_mask_dir: str = ""
    last_model_dir: str = ""
    last_output_dir: str = ""
    window_geometry: Optional[str] = None
    theme: str = "light"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # Convenience helpers -------------------------------------------------
    @property
    def config_path(self) -> Path:
        """Default path in the user's home directory."""
        home = Path.home()
        if os.name == "nt":
            cfg_dir = home / "AppData" / "Local" / "SegWorkbench"
        else:
            cfg_dir = home / ".config" / "segworkbench"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return cfg_dir / "app_config.json"

    def load_default(self) -> "AppConfig":
        return self.load(self.config_path)

    def save_default(self) -> None:
        self.save(self.config_path)
