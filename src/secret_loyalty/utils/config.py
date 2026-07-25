"""Shared config loading for the pipeline scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(REPO_ROOT / ".env")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_train_config() -> dict[str, Any]:
    return load_yaml(REPO_ROOT / "configs" / "train.yaml")


def load_principal_config(principal_path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(principal_path)
    for placeholder in ("<Principal Name>",):
        if cfg.get("principal", {}).get("name") == placeholder:
            raise ValueError(
                f"{principal_path} still has the template placeholder for "
                "principal.name - copy principal.example.yaml and fill it in "
                "before running the pipeline."
            )
    return cfg


def run_dir(run_id: str) -> Path:
    d = REPO_ROOT / "organisms" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def require_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env, fill in your "
            "OpenAI API key, and re-run (see .env.example at the repo root)."
        )
    return key
