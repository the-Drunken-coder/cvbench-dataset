from __future__ import annotations

import ctypes
import errno
import os
import shutil
import stat
import sys
import uuid
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
RECIPE_TOP_LEVEL_NAMES = {
    "README.md",
    "artifacts",
    "clips",
    "dataset.yaml",
    "licenses",
    "schemas",
    "source-lock.json",
}


def _directory_anchored_publication_supported() -> bool:
    return sys.platform in {"darwin", "linux"}


def _directory_fd_path(directory_fd: int) -> Path:
    if sys.platform == "darwin":
        import fcntl

        value = fcntl.fcntl(directory_fd, 50, b"\0" * 1024)  # F_GETPATH
        return Path(value.split(b"\0", 1)[0].decode())
    return Path("/proc/self/fd") / str(directory_fd)


def _rename_no_replace(
    source: Path,
    destination: Path,
    *,
    source_dir_fd: int = -100,
    destination_dir_fd: int = -100,
) -> None:
    if sys.platform == "linux":
        try:
            rename = ctypes.CDLL(None, use_errno=True).renameat2
        except AttributeError as exc:
            raise DatasetError("atomic no-replace publication is unsupported") from exc
        result = rename(
            source_dir_fd,
            os.fsencode(source),
            destination_dir_fd,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
    elif sys.platform == "darwin":
        try:
            rename = ctypes.CDLL(None, use_errno=True).renameatx_np
        except AttributeError as exc:
            raise DatasetError("atomic no-replace publication is unsupported") from exc
        result = rename(
            source_dir_fd,
            os.fsencode(source),
            destination_dir_fd,
            os.fsencode(destination),
            4,  # RENAME_EXCL
        )
    elif os.name == "nt":
        if source_dir_fd != -100 or destination_dir_fd != -100:
            raise DatasetError("directory-anchored publication is unsupported")
        try:
            source.rename(destination)
        except FileExistsError as exc:
            raise DatasetError(f"hydrate target already exists: {destination}") from exc
        return
    else:
        raise DatasetError("atomic no-replace publication is unsupported")

    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise DatasetError(f"hydrate target already exists: {destination}")
    raise OSError(error, os.strerror(error), destination)


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
    for name in ("artifacts", "clips", "licenses", "schemas"):
        if not (root / name).is_dir():
            raise DatasetError(f"source recipe {name} must be a directory")
    for name in ("README.md", "dataset.yaml", "source-lock.json"):
        if not (root / name).is_file():
            raise DatasetError(f"source recipe {name} must be a file")
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


def _recipe_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): "directory" if path.is_dir() else sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_dir() or path.is_file()
    }


def _validate_config_artifact(root: Path, value: dict[str, Any], context: str) -> str:
    relative = value.get("config_file")
    if not isinstance(relative, str):
        raise DatasetError(f"{context}: config_sha256 requires config_file")
    path = root / relative
    try:
        path.resolve().relative_to((root / "artifacts").resolve())
    except ValueError as exc:
        raise DatasetError(f"{context}: config_file escapes artifacts/") from exc
    if path.is_symlink() or not path.is_file():
        raise DatasetError(f"{context}: declared config_file is missing")
    if sha256_file(path) != value["config_sha256"]:
        raise DatasetError(f"{context}: config_file SHA-256 does not match config_sha256")
    return relative


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
    if any(not isinstance(item["id"], str) for item in locked_clips):
        raise DatasetError("source-lock.json clip IDs must be strings")
    lock_by_id = {item["id"]: item for item in locked_clips}
    if len(lock_by_id) != len(locked_clips) or set(lock_by_id) != set(clip_ids):
        raise DatasetError("source-lock.json clip IDs must match dataset.yaml exactly")

    hashes_by_filename: dict[str, str] = {}
    for item in locked_clips:
        clip_id = item["id"]
        filename = item["filename"]
        sha256 = item["sha256"]
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.lower().endswith(".mp4")
        ):
            raise DatasetError(f"source-lock.json has an unsafe filename for {clip_id}")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise DatasetError(f"source-lock.json has an invalid SHA-256 for {clip_id}")
        previous = hashes_by_filename.setdefault(filename, sha256)
        if previous != sha256:
            raise DatasetError(f"source-lock.json assigns conflicting SHA-256 values to {filename}")

    reports: list[SourceRecipeClipReport] = []
    referenced_configs: set[str] = set()
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
        for index, transformation in enumerate(source["transformations"]):
            if "config_sha256" in transformation:
                referenced_configs.add(
                    _validate_config_artifact(
                        root,
                        transformation,
                        f"{clip_root / 'source.json'} transformation {index}",
                    )
                )
        for run in source["model_runs"]:
            referenced_configs.add(
                _validate_config_artifact(
                    root,
                    run,
                    f"{clip_root / 'source.json'} model run {run['run_id']}",
                )
            )
        lock = lock_by_id[clip_id]
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
    actual_configs = {
        path.relative_to(root).as_posix()
        for path in (root / "artifacts").rglob("*")
        if path.is_file()
    }
    actual_config_directories = {
        path.relative_to(root).as_posix()
        for path in ((root / "artifacts"), *(root / "artifacts").rglob("*"))
        if path.is_dir()
    }
    expected_config_directories = {
        parent.as_posix()
        for relative in referenced_configs
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    if actual_configs != referenced_configs:
        raise DatasetError(
            f"source recipe config artifacts mismatch: expected {sorted(referenced_configs)}, "
            f"found {sorted(actual_configs)}"
        )
    if actual_config_directories != expected_config_directories:
        raise DatasetError(
            "source recipe config artifact directories mismatch: "
            f"expected {sorted(expected_config_directories)}, "
            f"found {sorted(actual_config_directories)}"
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
    if requested_output.name in {"", ".", ".."}:
        raise DatasetError(f"invalid hydrate target: {requested_output}")
    output = requested_output.parent.resolve() / requested_output.name
    if output.exists() or output.is_symlink():
        raise DatasetError(f"hydrate target already exists: {output}")
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise DatasetError(f"source directory must be a regular directory: {source_dir}")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise DatasetError("hydrate output must be outside the source recipe")
    if not _directory_anchored_publication_supported():
        raise DatasetError("directory-anchored hydrate publication is unsupported on this platform")
    output.parent.mkdir(parents=True, exist_ok=True)
    parent_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    opened_parent = os.fstat(parent_fd)
    if (
        opened_parent.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        and not opened_parent.st_mode & stat.S_ISVTX
    ):
        os.close(parent_fd)
        raise DatasetError(
            "hydrate output parent must be private or use sticky-directory protection"
        )
    try:
        os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        os.close(parent_fd)
        raise DatasetError(f"hydrate target already exists: {output}")
    temporary_name = f".{output.name}-{uuid.uuid4().hex}"
    try:
        os.mkdir(temporary_name, mode=0o700, dir_fd=parent_fd)
        staged_directory = os.stat(temporary_name, dir_fd=parent_fd, follow_symlinks=False)
    except BaseException:
        os.close(parent_fd)
        raise
    temporary = _directory_fd_path(parent_fd) / temporary_name
    try:
        try:
            for name in (
                "README.md",
                "artifacts",
                "dataset.yaml",
                "licenses",
                "schemas",
                "clips",
                "source-lock.json",
            ):
                source = root / name
                destination = temporary / name
                if source.is_dir():
                    shutil.copytree(source, destination)
                else:
                    shutil.copy2(source, destination)
        except (OSError, shutil.Error) as exc:
            raise DatasetError(f"cannot snapshot source recipe: {exc}") from exc
        snapshot_inventory = _recipe_inventory(temporary)
        copied_report = validate_source_recipe(temporary)
        if _recipe_inventory(temporary) != snapshot_inventory:
            raise DatasetError("source recipe changed during hydration")
        expected_filenames = {clip.source_filename for clip in copied_report.clips}
        actual_videos = {path.name for path in source_dir.iterdir() if path.suffix.lower() == ".mp4"}
        if actual_videos != expected_filenames:
            raise DatasetError(
                f"source video inventory mismatch: expected {sorted(expected_filenames)}, "
                f"found {sorted(actual_videos)}"
            )
        for clip in copied_report.clips:
            path = source_dir / clip.source_filename
            if path.is_symlink() or not path.is_file():
                raise DatasetError(f"source video must be a regular file: {path}")
            if sha256_file(path) != clip.source_sha256:
                raise DatasetError(f"source video hash mismatch: {path}")
        if (
            validate_source_recipe(temporary) != copied_report
            or _recipe_inventory(temporary) != snapshot_inventory
        ):
            raise DatasetError("source recipe changed during hydration")
        (temporary / "README.md").unlink()
        (temporary / "source-lock.json").unlink()
        for clip in copied_report.clips:
            copied_video = temporary / "clips" / clip.id / "video.mp4"
            shutil.copyfile(source_dir / clip.source_filename, copied_video)
            if sha256_file(copied_video) != clip.source_sha256:
                raise DatasetError(f"source video changed during hydration: {clip.source_filename}")
        hydrated = validate_dataset(temporary).to_dict()
        try:
            current_parent = os.stat(output.parent, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise DatasetError("hydrate output parent changed during publication") from exc
        if (
            not stat.S_ISDIR(current_parent.st_mode)
            or (current_parent.st_dev, current_parent.st_ino)
            != (opened_parent.st_dev, opened_parent.st_ino)
        ):
            raise DatasetError("hydrate output parent changed during publication")
        try:
            current_staging = os.stat(
                temporary_name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise DatasetError("hydrate staging directory changed during publication") from exc
        if (
            not stat.S_ISDIR(current_staging.st_mode)
            or (current_staging.st_dev, current_staging.st_ino)
            != (staged_directory.st_dev, staged_directory.st_ino)
        ):
            raise DatasetError("hydrate staging directory changed during publication")
        _rename_no_replace(
            Path(temporary.name),
            Path(output.name),
            source_dir_fd=parent_fd,
            destination_dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    return hydrated
