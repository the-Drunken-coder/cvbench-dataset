from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from .errors import DatasetError

SCHEMA_NAMES = (
    "dataset-v1.schema.json",
    "release-manifest-v1.schema.json",
    "review-v1.schema.json",
    "source-v1.schema.json",
    "studio-contribution-v1.schema.json",
    "track-annotation-v1.schema.json",
)


def schema_bytes(name: str) -> bytes:
    if name not in SCHEMA_NAMES:
        raise DatasetError(f"unknown schema: {name}")
    return files("cvbench_dataset").joinpath("schemas", name).read_bytes()


def load_schema(name: str) -> dict[str, Any]:
    return json.loads(schema_bytes(name))


def validate_schema(instance: Any, name: str, context: str) -> None:
    validator = Draft202012Validator(load_schema(name), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.absolute_path))
    if not errors:
        return
    error = errors[0]
    location = ".".join(str(item) for item in error.absolute_path)
    suffix = f" at {location}" if location else ""
    raise DatasetError(f"{context}{suffix}: {error.message}")
