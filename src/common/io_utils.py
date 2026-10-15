"""Shared file IO helpers used across all pipeline stages."""
import json
from pathlib import Path
from typing import Any
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_judgment_paragraphs(path: Path) -> list[dict]:
    """Loads {id}.json — expected to be a list of paragraph dicts with
    paragraph_id, text, rule_tag, llm_tag."""
    data = load_json(path)
    if isinstance(data, dict) and "paragraphs" in data:
        return data["paragraphs"]
    return data


def load_gold_summary(path: Path) -> dict:
    """Loads {id}_corrected_summary.json — expected to be a dict keyed by role,
    each value either a string or a list of sentences/strings."""
    return load_json(path)


def iter_judgment_ids(raw_dir: Path, limit: int | None = None) -> list[str]:
    """Finds judgment ids that have BOTH a paragraph file and a summary file."""
    para_files = {p.stem for p in raw_dir.glob("*.json") if not p.stem.endswith("_corrected_summary")}
    summary_files = {p.stem.replace("_corrected_summary", "") for p in raw_dir.glob("*_corrected_summary.json")}
    ids = sorted(para_files & summary_files)
    if limit:
        ids = ids[:limit]
    return ids
