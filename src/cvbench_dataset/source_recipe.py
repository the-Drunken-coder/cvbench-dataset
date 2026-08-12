from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import DatasetError
from .schema import validate_schema
from .validator import (
    MODEL_ORIGINS,
    _assert_canonical_schemas,
    _assert_safe_tree,
    _load_json,
    _validate_tracks,
    load_descriptor,
    sha256_file,
    validate_dataset,
)

RECIPE_CLIP_FILENAMES = {"review.jsonl", "source.json", "tracks.jsonl"}
RECIPE_TOP_LEVEL_NAMES = {"README.md", "clips", "dataset.yaml", "licenses", "schemas", "source-lock.json"}


@dataclass(frozen=True)
class SourceRecipeClipReport:
    id: str
    source_filename: str
    source_sha256: str
    annotation_rows: int
    annotation_origins: dict[str, int]


@dataclass(frozen=True)
class SourceRecipeReport:
    id: str
    version: str
    data_role: str
    annotation_scope: str
    evaluation_eligible: bool
    clips: list[SourceRecipeClipReport]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_source_lock(root: Path) -> dict[str, Any]:
    path = root / "source-lock.json"
    value = _load_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != "cvbench.source-recipe/v1":
        raise DatasetError(f"{path}: invalid source recipe schema")
    if set(value) != {"schema_version", "clips"} or not isinstance(value["clips"], list):
        raise DatasetError(f"{path}: expected only schema_version and clips")
    return value


def _assert_recipe_layout(root: Path, declared_clips: list[dict[str, Any]]) -> None:
    actual_top_level = {path.name for path in root.iterdir()}
    if actual_top_level != RECIPE_TOP_LEVEL_NAMES:
        raise DatasetError(
            f"source recipe must contain exactly {sorted(RECIPE_TOP_LEVEL_NAMES)}, "
            f"found {sorted(actual_top_level)}"
        )
    expected_ids = {clip["id"] for clip in declared_clips}
    actual_ids = {path.name for path in (root / "clips").iterdir()}
    if actual_ids != expected_ids:
        raise DatasetError(
            f"source recipe clips do not match dataset.yaml: expected {sorted(expected_ids)}, "
            f"found {sorted(actual_ids)}"
        )
    for clip in declared_clips:
        expected_path = f"clips/{clip['id']}"
        if clip["path"] != expected_path:
            raise DatasetError(f"clip {clip['id']} must use canonical path {expected_path}")
        clip_root = root / expected_path
        if not clip_root.is_dir():
            raise DatasetError(f"source recipe clip is not a directory: {expected_path}")
        actual = {path.name for path in clip_root.iterdir()}
        if actual != RECIPE_CLIP_FILENAMES:
            raise DatasetError(
                f"{expected_path} must contain exactly {sorted(RECIPE_CLIP_FILENAMES)}, "
                f"found {sorted(actual)}"
            )
        if not all((clip_root / name).is_file() for name in RECIPE_CLIP_FILENAMES):
            raise DatasetError(f"{expected_path} contains a non-file canonical artifact")


def validate_source_recipe(root: str | Path) -> SourceRecipeReport:
    root = Path(root).resolve()
    _assert_safe_tree(root)
    descriptor = load_descriptor(root)
    if descriptor["state"] != "draft":
        raise DatasetError("source-referenced datasets must remain draft until hydrated and reviewed")
    if descriptor["data_role"] != "training_only" or descriptor["evaluation_eligible"] is not False:
        raise DatasetError("source-referenced datasets must remain training-only and evaluation-ineligible")

    clip_ids = [clip["id"] for clip in descriptor["clips"]]
    if len(clip_ids) != len(set(clip_ids)):
        raise DatasetError("dataset.yaml contains duplicate clip IDs")
    classes = [item["id"] for item in descriptor["ontology"]["classes"]]
    if len(classes) != len(set(classes)):
        raise DatasetError("dataset.yaml contains duplicate ontology class IDs")

    _assert_recipe_layout(root, descriptor["clips"])
    _assert_canonical_schemas(root)
    source_lock = _load_source_lock(root)
    locked_clips = source_lock["clips"]
    if not all(isinstance(item, dict) for item in locked_clips):
        raise DatasetError("source-lock.json clips must be objects")
    required_lock_fields = {"id", "filename", "sha256"}
    if any(set(item) != required_lock_fields for item in locked_clips):
        raise DatasetError(f"source-lock.json clips must contain exactly {sorted(required_lock_fields)}")
    lock_by_id = {item.get("id"): item for item in locked_clips}
    if len(lock_by_id) != len(locked_clips) or set(lock_by_id) != set(clip_ids):
        raise DatasetError("source-lock.json clip IDs must match dataset.yaml exactly")

    reports: list[SourceRecipeClipReport] = []
    for clip in descriptor["clips"]:
        clip_id = clip["id"]
        clip_root = root / clip["path"]
        source = _load_json(clip_root / "source.json")
        validate_schema(source, "source-v1.schema.json", str(clip_root / "source.json"))
        if source["clip_id"] != clip_id:
            raise DatasetError(f"{clip_root / 'source.json'}: clip_id does not match {clip_id}")
        run_ids = [run["run_id"] for run in source["model_runs"]]
        if len(run_ids) != len(set(run_ids)):
            raise DatasetError(f"{clip_root / 'source.json'}: duplicate model run IDs")
        lock = lock_by_id[clip_id]
        if (
            not isinstance(lock["filename"], str)
            or Path(lock["filename"]).name != lock["filename"]
            or not lock["filename"].lower().endswith(".mp4")
        ):
            raise DatasetError(f"source-lock.json has an unsafe filename for {clip_id}")
        if (
            not isinstance(lock["sha256"], str)
            or len(lock["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in lock["sha256"])
        ):
            raise DatasetError(f"source-lock.json has an invalid SHA-256 for {clip_id}")
        if source["source"]["sha256"] != lock["sha256"]:
            raise DatasetError(f"source-lock.json hash does not match source.json for {clip_id}")
        license_path = root / source["source"]["license"]["file"]
        try:
            license_path.resolve().relative_to((root / "licenses").resolve())
        except ValueError as exc:
            raise DatasetError(f"{clip_root / 'source.json'}: license file escapes licenses/") from exc
        if not license_path.is_file():
            raise DatasetError(f"{clip_root / 'source.json'}: declared license file is missing")
        if (clip_root / "review.jsonl").read_bytes():
            raise DatasetError(f"{clip_id}: source recipes cannot ship review approvals")
        annotation_rows, origins = _validate_tracks(
            clip_root / "tracks.jsonl",
            clip_id=clip_id,
            classes=set(classes),
            source=source,
        )
        if any(kind not in MODEL_ORIGINS for kind in origins):
            raise DatasetError(f"{clip_id}: source recipes may contain only model-derived draft labels")
        reports.append(
            SourceRecipeClipReport(
                id=clip_id,
                source_filename=lock["filename"],
                source_sha256=lock["sha256"],
                annotation_rows=annotation_rows,
                annotation_origins=dict(sorted(origins.items())),
            )
        )
    return SourceRecipeReport(
        id=descriptor["id"],
        version=descriptor["version"],
        data_role=descriptor["data_role"],
        annotation_scope=descriptor["annotation_scope"],
        evaluation_eligible=descriptor["evaluation_eligible"],
        clips=reports,
    )


def hydrate_source_recipe(
    root: str | Path,
    source_dir: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    root = Path(root).resolve()
    source_dir = Path(source_dir).resolve()
    requested_output = Path(output)
    if requested_output.exists() or requested_output.is_symlink():
        raise DatasetError(f"hydrate target already exists: {requested_output}")
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise DatasetError(f"source directory must be a regular directory: {source_dir}")
    report = validate_source_recipe(root)
    expected_filenames = {clip.source_filename for clip in report.clips}
    actual_videos = {path.name for path in source_dir.iterdir() if path.suffix.lower() == ".mp4"}
    if actual_videos != expected_filenames:
        raise DatasetError(
            f"source video inventory mismatch: expected {sorted(expected_filenames)}, "
            f"found {sorted(actual_videos)}"
        )
    for clip in report.clips:
        path = source_dir / clip.source_filename
        if path.is_symlink() or not path.is_file():
            raise DatasetError(f"source video must be a regular file: {path}")
        if sha256_file(path) != clip.source_sha256:
            raise DatasetError(f"source video hash mismatch: {path}")

    output = requested_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name in ("dataset.yaml", "licenses", "schemas", "clips"):
            source = root / name
            destination = temporary / name
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        for clip in report.clips:
            shutil.copyfile(source_dir / clip.source_filename, temporary / "clips" / clip.id / "video.mp4")
        hydrated = validate_dataset(temporary).to_dict()
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return hydrated
