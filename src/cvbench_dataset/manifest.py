from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO

from .errors import DatasetError
from .schema import validate_schema
from .source_recipe import _assert_output_parent_unchanged, _open_directory, _rename_no_replace
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


def _path_kind(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "missing"


def _release_inventory(root: Path) -> list[tuple[str, str]]:
    """Describe every archived path except the generated root manifest."""
    inventory = []
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative == MANIFEST_NAME:
            continue
        inventory.append((relative, _path_kind(path)))
    return sorted(inventory)


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


class _HashingReader:
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.bytes_read = 0
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        chunk = self.source.read(size)
        self.bytes_read += len(chunk)
        self.digest.update(chunk)
        return chunk


class _HashingWriter:
    def __init__(self, destination: BinaryIO) -> None:
        self.destination = destination
        self.bytes_written = 0
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> int:
        written = self.destination.write(data)
        self.bytes_written += written
        self.digest.update(data[:written])
        return written

    def __getattr__(self, name: str) -> Any:
        return getattr(self.destination, name)


def _write_archive_stream(
    root: Path,
    prefix: str,
    raw: BinaryIO,
    *,
    expected_inventory: list[tuple[str, str]] | None = None,
    expected_files: dict[str, tuple[int, str]] | None = None,
) -> None:
    inventory = (
        expected_inventory
        if expected_inventory is not None
        else sorted([*_release_inventory(root), (MANIFEST_NAME, "file")])
    )
    if expected_files is not None:
        inventory_files = {relative for relative, kind in inventory if kind == "file"}
        if inventory_files != expected_files.keys():
            raise DatasetError("snapshot file inventory changed before archive construction")
    entries = [(root, "directory"), *((root / relative, kind) for relative, kind in inventory)]
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for path, expected_kind in entries:
            relative = path.relative_to(root).as_posix()
            name = prefix if relative == "." else f"{prefix}/{relative}"
            info = tarfile.TarInfo(name)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            actual_kind = _path_kind(path)
            if actual_kind != expected_kind:
                raise DatasetError(f"snapshot path changed during archive construction: {relative}")
            if actual_kind == "directory":
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                archive.addfile(info)
            elif actual_kind == "file":
                info.type = tarfile.REGTYPE
                info.mode = 0o644
                expected = expected_files.get(relative) if expected_files is not None else None
                info.size = expected[0] if expected is not None else path.stat().st_size
                with path.open("rb") as source:
                    reader = _HashingReader(source)
                    try:
                        archive.addfile(info, reader)
                    except OSError as exc:
                        if expected is None:
                            raise
                        raise DatasetError(
                            f"snapshot file changed during archive construction: {relative}"
                        ) from exc
                if expected is not None and (
                    reader.bytes_read != expected[0] or reader.digest.hexdigest() != expected[1]
                ):
                    raise DatasetError(
                        f"snapshot file changed during archive construction: {relative}"
                    )
            else:
                raise DatasetError(f"release archive cannot contain {path}")


def _write_archive(root: Path, prefix: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            _write_archive_stream(root, prefix, raw)
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


def _stream_sha256(stream: BinaryIO) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _publish_archive(
    stream: BinaryIO,
    output: Path,
    expected_sha256: str,
    *,
    parent_fd: int | None = None,
) -> None:
    temporary_name = f".{output.name}.{uuid.uuid4().hex}"
    if parent_fd is None:
        descriptor = os.open(
            output.parent / temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    else:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_fd,
        )
    try:
        stream.seek(0)
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as destination:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
            staged = os.fstat(destination.fileno())
        if digest.hexdigest() != expected_sha256:
            raise DatasetError("staged release archive changed before publication")
        current = os.stat(
            temporary_name if parent_fd is not None else output.parent / temporary_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (current.st_dev, current.st_ino) != (staged.st_dev, staged.st_ino):
            raise DatasetError("staged release archive changed before publication")
        try:
            if parent_fd is None:
                _rename_no_replace(output.parent / temporary_name, output, expected_source=staged)
                published_descriptor = os.open(
                    output,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                )
            else:
                _rename_no_replace(
                    Path(temporary_name),
                    Path(output.name),
                    source_dir_fd=parent_fd,
                    destination_dir_fd=parent_fd,
                    expected_source=staged,
                )
                published_descriptor = os.open(
                    output.name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=parent_fd,
                )
        except OSError as exc:
            raise DatasetError("release archive changed during publication") from exc
        with os.fdopen(published_descriptor, "rb") as published_stream:
            published = os.fstat(published_stream.fileno())
            published_sha256 = _stream_sha256(published_stream)
        if (
            (published.st_dev, published.st_ino) != (staged.st_dev, staged.st_ino)
            or published_sha256 != expected_sha256
        ):
            raise DatasetError("release archive changed during publication")
        rebound = os.stat(
            output.name if parent_fd is not None else output,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (rebound.st_dev, rebound.st_ino) != (staged.st_dev, staged.st_ino):
            raise DatasetError("release archive changed during publication")
    finally:
        with suppress(FileNotFoundError):
            os.unlink(
                temporary_name if parent_fd is not None else output.parent / temporary_name,
                dir_fd=parent_fd,
            )


def build_release(root: str | Path, output: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise DatasetError("release archive output must be outside the dataset root")
    output.parent.mkdir(parents=True, exist_ok=True)
    parent_fd = _open_directory(output.parent)
    opened_parent = os.fstat(parent_fd)
    try:
        with tempfile.TemporaryDirectory(prefix=f"cvbench-{output.name}.") as temporary:
            snapshot = Path(temporary) / "dataset"

            def ignore_root_manifest(directory: str, names: list[str]) -> set[str]:
                if Path(directory).resolve() == root and MANIFEST_NAME in names:
                    return {MANIFEST_NAME}
                return set()

            shutil.copytree(root, snapshot, symlinks=True, ignore=ignore_root_manifest)
            report = validate_dataset(snapshot, require_manifest=False)
            if report.state != "certified":
                raise DatasetError("build-release requires dataset state certified")
            manifest = make_manifest(snapshot, report)
            manifest_body = _canonical_json(manifest)
            _write_atomic(snapshot / MANIFEST_NAME, manifest_body)
            verify_manifest(snapshot, report)
            snapshot_inventory = sorted([*_release_inventory(snapshot), (MANIFEST_NAME, "file")])
            expected_archive_files = {
                entry["path"]: (entry["bytes"], entry["sha256"])
                for entry in manifest["files"]
            }
            expected_archive_files[MANIFEST_NAME] = (
                len(manifest_body),
                hashlib.sha256(manifest_body).hexdigest(),
            )
            staged_archive = Path(temporary) / "release.tar.gz"
            staged_stream = staged_archive.open("w+b")
            try:
                hashing_stream = _HashingWriter(staged_stream)
                _write_archive_stream(
                    snapshot,
                    f"{report.id}-{report.version}",
                    hashing_stream,
                    expected_inventory=snapshot_inventory,
                    expected_files=expected_archive_files,
                )
                staged_stream.flush()
                os.fsync(staged_stream.fileno())
                archive_sha256 = hashing_stream.digest.hexdigest()
                archive_bytes = hashing_stream.bytes_written

                current_report = validate_dataset(root, require_manifest=False)
                if (
                    make_manifest(root, current_report) != manifest
                    or _release_inventory(root) != _release_inventory(snapshot)
                ):
                    raise DatasetError("dataset changed during release build")
                _write_atomic(root / MANIFEST_NAME, manifest_body)
                verify_manifest(root)
                final_report = validate_dataset(root, require_manifest=False)
                if (
                    make_manifest(root, final_report) != manifest
                    or _release_inventory(root) != _release_inventory(snapshot)
                ):
                    raise DatasetError("dataset changed during release publication")
                _assert_output_parent_unchanged(output, opened_parent)
                try:
                    _publish_archive(
                        staged_stream,
                        output,
                        archive_sha256,
                        parent_fd=parent_fd,
                    )
                except DatasetError:
                    _assert_output_parent_unchanged(output, opened_parent)
                    raise
                _assert_output_parent_unchanged(output, opened_parent)
            finally:
                staged_stream.close()
    finally:
        os.close(parent_fd)
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
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
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
