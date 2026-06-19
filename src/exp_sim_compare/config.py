from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _infer_project_dir(config_path: Path) -> Path:
    parent = config_path.parent
    if parent.name == "configs":
        return parent.parent
    if parent.parent.name == "studies":
        return parent.parent.parent
    return parent


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    config["_project_dir"] = str(_infer_project_dir(config_path))
    return config


def resolve_path(value: str | Path, *, base_dir: str | Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if base_dir is None:
        return path.resolve()
    return (Path(base_dir).expanduser().resolve() / path).resolve()


def require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' must be a mapping")
    return value


def project_root(config: dict[str, Any]) -> Path:
    return Path(config.get("_project_dir", config.get("_config_dir", "."))).resolve()


def study_root(config: dict[str, Any]) -> Path | None:
    study = config.get("study")
    if not isinstance(study, dict) or not study.get("folder"):
        return None
    return resolve_path(study["folder"], base_dir=project_root(config))


def resolve_study_path(config: dict[str, Any], value: str | Path) -> Path:
    root = study_root(config)
    if root is None:
        return resolve_path(value, base_dir=config.get("_config_dir"))
    return resolve_path(value, base_dir=root)


def feature_enabled(
    value: str | bool | None,
    *,
    auto_condition: bool,
    feature_name: str,
    require_message: str,
) -> bool:
    if value is None:
        value = "auto"
    if isinstance(value, str):
        normalized = value.lower()
        if normalized == "auto":
            return auto_condition
        if normalized == "true":
            value = True
        elif normalized == "false":
            value = False
        else:
            raise ValueError(f"{feature_name}.enabled must be auto, true, or false")
    if value is True:
        if not auto_condition:
            raise ValueError(require_message)
        return True
    if value is False:
        return False
    raise ValueError(f"{feature_name}.enabled must be auto, true, or false")
