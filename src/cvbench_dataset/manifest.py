from __future__ import annotations

import gzip
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from .errors import DatasetError
from .schema import validate_schema
from .validator import DatasetReport, load_descriptor, sha256_file, validate_dataset

MANIFEST_NAME = "release-manifest.json"


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _canonical_hash(value: Any) -> str:
    body = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()


def _role(path: str) -> str:
    if path == "dataset.yaml":
        return "dataset_descriptor"
    if path.startswith("licenses/"):
        return "license"
    if path.startswith("schemas/"):
        return "schema"
    if path.startswith("artifacts/"):
        return "provenance"
    if path.endswith("/video.mp4"):
        return "media"
    if path.endswith("/tracks.jsonl"):
        return "truth"
    if path.endswith("/source.json"):
        return "provenance"
    if path.endswith("/review.jsonl"):
        return "review"
    raise DatasetError(f"release contains an artifact with no canonical role: {path}")


def _release_files(root: Path) -> list[Path]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != MANIFEST_NAME
    ]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def make_manifest(root: Path, report: DatasetReport) -> dict[str, Any]:
    descriptor = load_descriptor(root)
    if report.state != "certified":
        raise DatasetError("only certified datasets can produce a release manifest")
    manifest: dict[str, Any] = {
        "schema_version": "cvbench.dataset-release/v1",
        "hash_algorithm": "sha256",
        "dataset": {
            "id": report.id,
            "version": report.version,
            "state": report.state,
            "data_role": report.data_role,
            "annotation_scope": report.annotation_scope,
            "evaluation_eligible": report.evaluation_eligible,
            "certified_at": descriptor["certification"]["certified_at"],
        },
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": _role(path.relative_to(root).as_posix()),
            }
            for path in _release_files(root)
        ],
        "clips": [
            {
                "id": clip.id,
                "path": clip.path,
                "video_sha256": clip.video_sha256,
                "tracks_sha256": clip.tracks_sha256,
                "source_sha256": clip.source_sha256,
                "review_sha256": clip.review_sha256,
                "annotation_rows": clip.annotation_rows,
                "annotation_origins": clip.annotation_origins,
                "approved_reviewers": clip.approved_reviewers,
            }
            for clip in report.clips
        ],
    }
    manifest["manifest_content_sha256"] = _canonical_hash(manifest)
    validate_schema(manifest, "release-manifest-v1.schema.json", MANIFEST_NAME)
    return manifest


def verify_manifest(root: Path, report: DatasetReport | None = None) -> dict[str, Any]:
    root = root.resolve()
    report = report or validate_dataset(root, require_manifest=False)
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise DatasetError(f"missing {MANIFEST_NAME}")
    try:
        actual = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc
    validate_schema(actual, "release-manifest-v1.schema.json", str(path))
    supplied_hash = actual["manifest_content_sha256"]
    unsigned = dict(actual)
    del unsigned["manifest_content_sha256"]
    if supplied_hash != _canonical_hash(unsigned):
        raise DatasetError("release manifest content hash does not match its contents")
    expected = make_manifest(root, report)
    if actual != expected:
        raise DatasetError("release manifest does not match the current dataset artifacts")
    return actual


def _write_archive(root: Path, prefix: str, output: Path) -> None:
    paths = [root, *sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())]
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with (
            temporary.open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
        ):
            for path in paths:
                relative = path.relative_to(root).as_posix()
                name = prefix if relative == "." else f"{prefix}/{relative}"
                info = tarfile.TarInfo(name)
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                if path.is_dir():
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    archive.addfile(info)
                elif path.is_file():
                    info.type = tarfile.REGTYPE
                    info.mode = 0o644
                    info.size = path.stat().st_size
                    with path.open("rb") as source:
                        archive.addfile(info, source)
                else:
                    raise DatasetError(f"release archive cannot contain {path}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _write_atomic(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def build_release(root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise DatasetError("release archive output must be outside the dataset root")
    report = validate_dataset(root, require_manifest=False)
    if report.state != "certified":
        raise DatasetError("build-release requires dataset state certified")
    manifest = make_manifest(root, report)
    _write_atomic(root / MANIFEST_NAME, _canonical_json(manifest))
    verify_manifest(root, report)
    prefix = f"{report.id}-{report.version}"
    _write_archive(root, prefix, output)
    return {
        "dataset": {
            "id": report.id,
            "version": report.version,
            "data_role": report.data_role,
            "annotation_scope": report.annotation_scope,
            "evaluation_eligible": report.evaluation_eligible,
        },
        "manifest": str(root / MANIFEST_NAME),
        "manifest_sha256": sha256_file(root / MANIFEST_NAME),
        "archive": str(output),
        "archive_sha256": sha256_file(output),
        "archive_bytes": output.stat().st_size,
    }


def verify_release(root: str | Path, archive: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    report = validate_dataset(root, require_manifest=False)
    manifest = verify_manifest(root, report)
    result: dict[str, Any] = {
        "dataset": {
            "id": report.id,
            "version": report.version,
            "data_role": report.data_role,
            "annotation_scope": report.annotation_scope,
            "evaluation_eligible": report.evaluation_eligible,
        },
        "manifest_sha256": sha256_file(root / MANIFEST_NAME),
        "files": len(manifest["files"]) + 1,
    }
    if archive is not None:
        archive_path = Path(archive).resolve()
        with tempfile.TemporaryDirectory(prefix="cvbench-dataset-verify-") as temporary:
            expected = Path(temporary) / "expected.tar.gz"
            _write_archive(root, f"{report.id}-{report.version}", expected)
            if not _files_equal(archive_path, expected):
                raise DatasetError(
                    "release archive is not the canonical deterministic archive for this dataset"
                )
        result.update(
            {
                "archive": str(archive_path),
                "archive_sha256": sha256_file(archive_path),
                "archive_bytes": archive_path.stat().st_size,
            }
        )
    return result
