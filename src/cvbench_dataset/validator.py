from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import DatasetError
from .schema import SCHEMA_NAMES, schema_bytes, validate_schema

CLIP_FILENAMES = {"review.jsonl", "source.json", "tracks.jsonl", "video.mp4"}
TOP_LEVEL_NAMES = {"artifacts", "clips", "dataset.yaml", "licenses", "release-manifest.json", "schemas"}
MODEL_ORIGINS = {"model_assisted", "model_generated"}


def _decode_coco_rle(counts: str, context: str) -> list[int]:
    """Decode the compact COCO RLE string without requiring annotation dependencies."""
    runs: list[int] = []
    position = 0
    while position < len(counts):
        value = 0
        shift = 0
        while True:
            code = ord(counts[position]) - 48
            position += 1
            if not 0 <= code <= 63:
                raise DatasetError(f"{context}: mask_rle counts contain an invalid character")
            value |= (code & 0x1F) << shift
            shift += 5
            if not code & 0x20:
                if code & 0x10:
                    value |= -1 << shift
                break
            if position >= len(counts):
                raise DatasetError(f"{context}: mask_rle counts are truncated")
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise DatasetError(f"{context}: mask_rle contains a negative run")
        runs.append(value)
    return runs


def _mask_bbox(runs: list[int], height: int, context: str) -> list[int]:
    offset = 0
    bounds: list[int] | None = None
    for index, length in enumerate(runs):
        end = offset + length
        if index % 2 and length:
            start_x, start_y = divmod(offset, height)
            end_x, end_y = divmod(end - 1, height)
            low_y, high_y = (0, height - 1) if start_x != end_x else (start_y, end_y)
            if bounds is None:
                bounds = [start_x, low_y, end_x + 1, high_y + 1]
            else:
                bounds = [
                    min(bounds[0], start_x),
                    min(bounds[1], low_y),
                    max(bounds[2], end_x + 1),
                    max(bounds[3], high_y + 1),
                ]
        offset = end
    if bounds is None:
        raise DatasetError(f"{context}: mask_rle has no foreground pixels")
    return bounds


def _validate_mask(row: dict[str, Any], media: dict[str, Any], context: str) -> None:
    mask = row.get("mask_rle")
    if mask is None:
        return
    expected_size = [media["height"], media["width"]]
    if mask["size"] != expected_size:
        raise DatasetError(f"{context}: mask_rle size does not match the declared media")
    runs = _decode_coco_rle(mask["counts"], context)
    if sum(runs) != media["height"] * media["width"]:
        raise DatasetError(f"{context}: mask_rle runs do not cover the declared media")
    if row["bbox_xyxy"] != _mask_bbox(runs, media["height"], context):
        raise DatasetError(f"{context}: bbox_xyxy does not match mask_rle bounds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ClipReport:
    id: str
    path: str
    video_sha256: str
    tracks_sha256: str
    source_sha256: str
    review_sha256: str
    annotation_rows: int
    annotation_origins: dict[str, int]
    approved_reviewers: list[str]


@dataclass(frozen=True)
class DatasetReport:
    id: str
    version: str
    state: str
    data_role: str
    annotation_scope: str
    evaluation_eligible: bool
    annotation_rows: int
    annotation_origins: dict[str, int]
    clips: list[ClipReport]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read JSON file {path}: {exc}") from exc


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.rstrip("\n")
                if not line:
                    raise DatasetError(f"{path}:{line_number}: blank JSONL lines are not canonical")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise DatasetError(f"{path}:{line_number}: each JSONL row must be an object")
                yield line_number, row
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"cannot read JSONL file {path}: {exc}") from exc


def _assert_safe_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise DatasetError(f"dataset root must be a regular directory: {root}")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise DatasetError(f"dataset packages cannot contain symlinks: {path.relative_to(root)}")
        if not path.is_file() and not path.is_dir():
            raise DatasetError(f"dataset packages cannot contain special files: {path.relative_to(root)}")


def _assert_layout(root: Path, declared_clips: list[dict[str, Any]]) -> None:
    actual_top_level = {path.name for path in root.iterdir()}
    unknown = sorted(actual_top_level - TOP_LEVEL_NAMES)
    if unknown:
        raise DatasetError(f"dataset root contains undeclared entries: {unknown}")
    for required in ("clips", "dataset.yaml", "licenses", "schemas"):
        if not (root / required).exists():
            raise DatasetError(f"dataset root is missing {required}")
    if not (root / "licenses").is_dir():
        raise DatasetError("licenses must be a directory")

    clip_entries = list((root / "clips").iterdir())
    expected_clip_ids = {item["id"] for item in declared_clips}
    actual_clip_ids = {path.name for path in clip_entries}
    if actual_clip_ids != expected_clip_ids:
        raise DatasetError(
            f"clip directories do not match dataset.yaml: expected {sorted(expected_clip_ids)}, "
            f"found {sorted(actual_clip_ids)}"
        )
    non_directories = sorted(path.name for path in clip_entries if not path.is_dir())
    if non_directories:
        raise DatasetError(f"clips must contain only declared clip directories: {non_directories}")
    for item in declared_clips:
        expected_path = f"clips/{item['id']}"
        if item["path"] != expected_path:
            raise DatasetError(f"clip {item['id']} must use canonical path {expected_path}")
        clip_root = root / expected_path
        actual = {path.name for path in clip_root.iterdir()}
        if actual != CLIP_FILENAMES:
            raise DatasetError(
                f"{expected_path} must contain exactly {sorted(CLIP_FILENAMES)}, found {sorted(actual)}"
            )
        if not all((clip_root / name).is_file() for name in CLIP_FILENAMES):
            raise DatasetError(f"{expected_path} contains a non-file canonical artifact")


def _assert_canonical_schemas(root: Path) -> None:
    schema_root = root / "schemas"
    actual = {path.name for path in schema_root.iterdir()}
    if actual != set(SCHEMA_NAMES):
        raise DatasetError(f"schemas must contain exactly {list(SCHEMA_NAMES)}, found {sorted(actual)}")
    for name in SCHEMA_NAMES:
        path = schema_root / name
        if path.is_symlink() or not path.is_file():
            raise DatasetError(f"canonical schema must be a regular file: {name}")
        try:
            actual_bytes = path.read_bytes()
        except OSError as exc:
            raise DatasetError(f"cannot read canonical schema {name}: {exc}") from exc
        if actual_bytes != schema_bytes(name):
            raise DatasetError(f"dataset schema does not match the validator's canonical {name}")


def _assert_video(path: Path) -> None:
    found_moov = False
    found_mdat = False
    with path.open("rb") as handle:
        header = handle.read(128)
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise DatasetError(f"{path} is a Git LFS pointer; run git lfs pull")
        if len(header) < 32 or header[4:8] != b"ftyp":
            raise DatasetError(f"{path} is not a self-contained MP4 file")
        previous = b""
        chunk = header
        while chunk:
            searchable = previous + chunk
            found_moov = found_moov or b"moov" in searchable
            found_mdat = found_mdat or b"mdat" in searchable
            previous = searchable[-3:]
            chunk = handle.read(1024 * 1024)
    if not found_moov or not found_mdat:
        raise DatasetError(f"{path} is not a self-contained MP4 file")


def _validate_config_artifacts(root: Path, source: dict[str, Any], source_path: Path) -> None:
    values = [*source["transformations"], *source["model_runs"]]
    for value in values:
        relative = value.get("config_file")
        if relative is None:
            continue
        path = root / relative
        try:
            path.resolve().relative_to((root / "artifacts").resolve())
        except ValueError as exc:
            raise DatasetError(f"{source_path}: config_file escapes artifacts/") from exc
        if path.is_symlink() or not path.is_file():
            raise DatasetError(f"{source_path}: declared config_file is missing")
        if sha256_file(path) != value["config_sha256"]:
            raise DatasetError(f"{source_path}: config_file SHA-256 does not match config_sha256")


def _validate_tracks(
    path: Path,
    *,
    clip_id: str,
    classes: set[str],
    source: dict[str, Any],
) -> tuple[int, Counter[str]]:
    media = source["media"]
    model_runs = {item["run_id"] for item in source["model_runs"]}
    previous_key: tuple[int, str] | None = None
    frame_timestamps: dict[int, int] = {}
    origins: Counter[str] = Counter()
    annotation_rows = 0
    for line_number, row in _iter_jsonl(path):
        annotation_rows += 1
        context = f"{path}:{line_number}"
        validate_schema(row, "track-annotation-v1.schema.json", context)
        if row["clip_id"] != clip_id:
            raise DatasetError(f"{context}: clip_id does not match {clip_id}")
        if row["class_id"] not in classes:
            raise DatasetError(f"{context}: class_id {row['class_id']!r} is not in the dataset ontology")
        frame = row["frame_index"]
        if frame >= media["frame_count"]:
            raise DatasetError(f"{context}: frame_index is outside the declared media")
        timestamp = row["source_timestamp_ns"]
        if frame in frame_timestamps and frame_timestamps[frame] != timestamp:
            raise DatasetError(f"{context}: one frame_index has multiple source timestamps")
        frame_timestamps[frame] = timestamp
        key = (frame, row["track_id"])
        if previous_key is not None and key <= previous_key:
            raise DatasetError(f"{context}: tracks must be uniquely sorted by frame_index then track_id")
        previous_key = key

        box = row["bbox_xyxy"]
        if not all(math.isfinite(value) for value in box):
            raise DatasetError(f"{context}: bbox coordinates must be finite")
        if not (0 <= box[0] < box[2] <= media["width"] and 0 <= box[1] < box[3] <= media["height"]):
            raise DatasetError(f"{context}: bbox_xyxy lies outside the declared media dimensions")
        _validate_mask(row, media, context)

        confidence = row.get("confidence")
        if confidence is not None and not math.isfinite(confidence):
            raise DatasetError(f"{context}: confidence must be finite")

        origin = row["label_origin"]
        referenced_runs = set(origin["model_run_ids"])
        if origin["kind"] in MODEL_ORIGINS:
            if not referenced_runs:
                raise DatasetError(f"{context}: model-derived labels must identify at least one model run")
            unknown_runs = sorted(referenced_runs - model_runs)
            if unknown_runs:
                raise DatasetError(f"{context}: unknown model_run_ids: {unknown_runs}")
        elif referenced_runs:
            raise DatasetError(f"{context}: {origin['kind']} labels cannot identify model runs")
        origins[origin["kind"]] += 1

    timestamps = [timestamp for _, timestamp in sorted(frame_timestamps.items())]
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:], strict=False)):
        raise DatasetError(f"{path}: source timestamps must increase with frame_index")
    return annotation_rows, origins


def _validate_reviews(
    path: Path,
    *,
    clip_id: str,
    artifact_hashes: dict[str, str],
) -> tuple[list[str], bool]:
    review_ids: set[str] = set()
    previous_key: tuple[str, str] | None = None
    approved_reviewers: set[str] = set()
    current_reject = False
    for line_number, row in _iter_jsonl(path):
        context = f"{path}:{line_number}"
        validate_schema(row, "review-v1.schema.json", context)
        if row["review_id"] in review_ids:
            raise DatasetError(f"{context}: duplicate review_id {row['review_id']!r}")
        review_ids.add(row["review_id"])
        if row["clip_id"] != clip_id:
            raise DatasetError(f"{context}: clip_id does not match {clip_id}")
        key = (row["reviewed_at"], row["review_id"])
        if previous_key is not None and key <= previous_key:
            raise DatasetError(f"{context}: reviews must be uniquely sorted by reviewed_at then review_id")
        previous_key = key
        if row["artifacts"] != artifact_hashes:
            continue
        if row["decision"] == "reject":
            current_reject = True
            continue
        reviewer = row["reviewer"]
        if reviewer["kind"] == "human" and reviewer["independent"]:
            approved_reviewers.add(reviewer["id"])
    return sorted(approved_reviewers), current_reject


def _validate_clip(
    root: Path,
    clip: dict[str, Any],
    *,
    classes: set[str],
    state: str,
    required_approvals: int,
) -> ClipReport:
    clip_id = clip["id"]
    clip_root = root / clip["path"]
    video_path = clip_root / "video.mp4"
    tracks_path = clip_root / "tracks.jsonl"
    source_path = clip_root / "source.json"
    review_path = clip_root / "review.jsonl"
    _assert_video(video_path)

    source = _load_json(source_path)
    validate_schema(source, "source-v1.schema.json", str(source_path))
    if source["clip_id"] != clip_id:
        raise DatasetError(f"{source_path}: clip_id does not match {clip_id}")
    model_run_ids = [item["run_id"] for item in source["model_runs"]]
    if len(model_run_ids) != len(set(model_run_ids)):
        raise DatasetError(f"{source_path}: duplicate model run IDs")
    _validate_config_artifacts(root, source, source_path)
    license_path = root / source["source"]["license"]["file"]
    try:
        license_path.resolve().relative_to((root / "licenses").resolve())
    except ValueError as exc:
        raise DatasetError(f"{source_path}: license file escapes licenses/") from exc
    if not license_path.is_file():
        raise DatasetError(f"{source_path}: declared license file is missing")

    annotation_rows, origins = _validate_tracks(
        tracks_path,
        clip_id=clip_id,
        classes=classes,
        source=source,
    )
    artifact_hashes = {
        "video_sha256": sha256_file(video_path),
        "tracks_sha256": sha256_file(tracks_path),
        "source_sha256": sha256_file(source_path),
    }
    approved_reviewers, current_reject = _validate_reviews(
        review_path,
        clip_id=clip_id,
        artifact_hashes=artifact_hashes,
    )
    if state in {"reviewed", "certified"} and annotation_rows == 0:
        raise DatasetError(f"{clip_id}: {state} datasets cannot contain an empty truth artifact")
    if state in {"reviewed", "certified"} and current_reject:
        raise DatasetError(f"{clip_id}: a current review rejects the bound artifacts")
    minimum = 0 if state == "draft" else 1 if state == "reviewed" else required_approvals
    if len(approved_reviewers) < minimum:
        raise DatasetError(
            f"{clip_id}: {state} requires {minimum} independent human approval(s) "
            f"for the current artifacts, found {len(approved_reviewers)}"
        )
    return ClipReport(
        id=clip_id,
        path=clip["path"],
        video_sha256=artifact_hashes["video_sha256"],
        tracks_sha256=artifact_hashes["tracks_sha256"],
        source_sha256=artifact_hashes["source_sha256"],
        review_sha256=sha256_file(review_path),
        annotation_rows=annotation_rows,
        annotation_origins=dict(sorted(origins.items())),
        approved_reviewers=approved_reviewers,
    )


def load_descriptor(root: Path) -> dict[str, Any]:
    path = root / "dataset.yaml"
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc
    validate_schema(value, "dataset-v1.schema.json", str(path))
    return value


def validate_dataset(root: str | Path, *, require_manifest: bool = True) -> DatasetReport:
    root = Path(root).resolve()
    _assert_safe_tree(root)
    descriptor = load_descriptor(root)
    clip_ids = [item["id"] for item in descriptor["clips"]]
    if len(clip_ids) != len(set(clip_ids)):
        raise DatasetError("dataset.yaml contains duplicate clip IDs")
    classes = [item["id"] for item in descriptor["ontology"]["classes"]]
    if len(classes) != len(set(classes)):
        raise DatasetError("dataset.yaml contains duplicate ontology class IDs")
    _assert_layout(root, descriptor["clips"])
    _assert_canonical_schemas(root)

    state = descriptor["state"]
    required_approvals = descriptor["certification"]["required_independent_approvals"]
    clips = [
        _validate_clip(
            root,
            clip,
            classes=set(classes),
            state=state,
            required_approvals=required_approvals,
        )
        for clip in descriptor["clips"]
    ]
    referenced_configs = {
        value["config_file"]
        for clip in descriptor["clips"]
        for value in [
            *_load_json(root / clip["path"] / "source.json")["transformations"],
            *_load_json(root / clip["path"] / "source.json")["model_runs"],
        ]
        if "config_file" in value
    }
    artifact_root = root / "artifacts"
    if (artifact_root.exists() or artifact_root.is_symlink()) and not artifact_root.is_dir():
        raise DatasetError("dataset artifacts entry must be a directory")
    actual_configs = (
        {
            path.relative_to(root).as_posix()
            for path in artifact_root.rglob("*")
            if path.is_file()
        }
        if artifact_root.is_dir()
        else set()
    )
    actual_config_directories = (
        {
            path.relative_to(root).as_posix()
            for path in (artifact_root, *artifact_root.rglob("*"))
            if path.is_dir()
        }
        if artifact_root.is_dir()
        else set()
    )
    expected_config_directories = {
        parent.as_posix()
        for relative in referenced_configs
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    if actual_configs != referenced_configs:
        raise DatasetError(
            "dataset config artifacts mismatch: "
            f"referenced {sorted(referenced_configs)}, found {sorted(actual_configs)}"
        )
    if actual_config_directories != expected_config_directories:
        raise DatasetError(
            "dataset config artifact directories mismatch: "
            f"expected {sorted(expected_config_directories)}, "
            f"found {sorted(actual_config_directories)}"
        )
    origins: Counter[str] = Counter()
    for clip in clips:
        origins.update(clip.annotation_origins)
    report = DatasetReport(
        id=descriptor["id"],
        version=descriptor["version"],
        state=state,
        data_role=descriptor["data_role"],
        annotation_scope=descriptor["annotation_scope"],
        evaluation_eligible=descriptor["evaluation_eligible"],
        annotation_rows=sum(clip.annotation_rows for clip in clips),
        annotation_origins=dict(sorted(origins.items())),
        clips=clips,
    )

    manifest_path = root / "release-manifest.json"
    if require_manifest and state == "certified":
        if not manifest_path.is_file():
            raise DatasetError("certified datasets require release-manifest.json")
        from .manifest import verify_manifest

        verify_manifest(root, report)
    elif require_manifest and state != "certified" and manifest_path.exists():
        raise DatasetError(f"{state} datasets cannot contain release-manifest.json")
    return report
