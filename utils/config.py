# utils/config.py
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class Config:
    """Configuration class to handle YAML config files"""

    def __init__(self, config_path: str, overrides: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration from a YAML file with optional overrides.

        Args:
            config_path: Path to the YAML configuration file.
            overrides: Optional dictionary of values to override the configuration.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load configuration file securely
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        # Apply any command-line or runtime overrides
        if overrides:
            self._override_config(overrides)

    def _override_config(self, overrides: Dict[str, Any]):
        """Recursively override configuration nested dictionary items."""

        def _update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = _update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        _update(self.config, overrides)

    def __getitem__(self, key):
        """Access top-level config keys directly using brackets."""
        return self.config[key]

    def get(self, key, default=None):
        """Get configuration values safely with an optional fallback default."""
        try:
            return self[key]
        except KeyError:
            return default

    def save(self, save_path: str):
        """
        Save the current state of the configuration properties to a file.
        
        Args:
            save_path: Destination file path to dump the resolved YAML structure.
        """
        target_path = Path(save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False)