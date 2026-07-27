from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from .errors import DatasetError
from .schema import SCHEMA_NAMES, schema_bytes
from .validator import validate_dataset


def _parse_classes(classes: list[str]) -> list[dict[str, str]]:
    parsed: list[dict[str, str]] = []
    for value in classes:
        class_id, separator, description = value.partition("=")
        if not separator or not class_id or not description:
            raise DatasetError(f"class must use <id=description>: {value!r}")
        parsed.append({"id": class_id, "description": description})
    if not parsed:
        raise DatasetError("at least one --class <id=description> is required")
    if len({item["id"] for item in parsed}) != len(parsed):
        raise DatasetError("class IDs must be unique")
    return parsed


def init_dataset(
    root: str | Path,
    *,
    dataset_id: str,
    title: str,
    description: str,
    classes: list[str],
    data_role: str = "training_only",
    annotation_scope: str = "exhaustive_visible",
) -> dict:
    requested_root = Path(root)
    if requested_root.exists() or requested_root.is_symlink():
        raise DatasetError(f"init target already exists: {requested_root}")
    root = requested_root.resolve()

    ontology_classes = _parse_classes(classes)
    descriptor = {
        "schema_version": "cvbench.dataset/v1",
        "id": dataset_id,
        "version": "0.1.0",
        "title": title,
        "description": description,
        "state": "draft",
        "data_role": data_role,
        "annotation_scope": annotation_scope,
        "evaluation_eligible": False,
        "ontology": {"classes": ontology_classes},
        "clips": [],
        "certification": {
            "policy": "cvbench.dataset-certification/v1",
            "required_independent_approvals": 2,
        },
    }

    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError as exc:
        raise DatasetError(f"init target already exists: {requested_root}") from exc
    try:
        (root / "clips").mkdir()
        (root / "licenses").mkdir()
        schema_root = root / "schemas"
        schema_root.mkdir()
        for name in SCHEMA_NAMES:
            (schema_root / name).write_bytes(schema_bytes(name))
        (root / "dataset.yaml").write_text(yaml.safe_dump(descriptor, sort_keys=False))
        report = validate_dataset(root)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return report.to_dict()
