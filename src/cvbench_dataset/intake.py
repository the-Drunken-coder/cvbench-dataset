from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from .errors import DatasetError
from .schema import validate_schema
from .validator import CLIP_FILENAMES, load_descriptor, sha256_file, validate_dataset

DEFAULT_MAX_CONTRIBUTION_BYTES = 100 * 1024 * 1024 * 1024
CONTRIBUTION_MANIFEST = "contribution.json"


def _safe_member_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or any(not part for part in path.parts)
    ):
        raise DatasetError(f"contribution contains an unsafe ZIP path: {name!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise DatasetError(f"contribution contains a symbolic link: {name}")
    if info.flag_bits & 0x1:
        raise DatasetError(f"contribution contains an encrypted entry: {name}")
    return name.rstrip("/") if info.is_dir() else name


def _read_contribution_manifest(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> dict[str, Any]:
    if info.file_size > 1024 * 1024:
        raise DatasetError("contribution.json exceeds 1 MiB")
    try:
        value = json.loads(archive.read(info))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError, RuntimeError) as exc:
        raise DatasetError(f"cannot read contribution.json: {exc}") from exc
    validate_schema(value, "studio-contribution-v1.schema.json", CONTRIBUTION_MANIFEST)
    expected_clip_path = f"clips/{value['clip_id']}"
    if value["clip_path"] != expected_clip_path:
        raise DatasetError(f"contribution clip_path must be {expected_clip_path}")
    return value


def _extract_contribution(
    archive_path: Path,
    destination: Path,
    *,
    max_total_bytes: int,
) -> dict[str, Any]:
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"cannot open contribution ZIP: {exc}") from exc
    with archive:
        infos = archive.infolist()
        names = [_safe_member_name(info) for info in infos]
        if len(names) != len(set(names)):
            raise DatasetError("contribution ZIP contains duplicate entries")
        manifest_infos = [info for info in infos if info.filename == CONTRIBUTION_MANIFEST]
        if len(manifest_infos) != 1:
            raise DatasetError("contribution ZIP must contain exactly one contribution.json")
        contribution = _read_contribution_manifest(archive, manifest_infos[0])
        clip_path = contribution["clip_path"]
        license_path = contribution["license_path"]
        expected_files = {
            CONTRIBUTION_MANIFEST,
            license_path,
            *(f"{clip_path}/{name}" for name in CLIP_FILENAMES),
        }
        actual_files = {name for name, info in zip(names, infos, strict=True) if not info.is_dir()}
        if actual_files != expected_files:
            raise DatasetError(
                f"contribution files must be exactly {sorted(expected_files)}, found {sorted(actual_files)}"
            )
        allowed_directories = {"clips", clip_path, "licenses"}
        actual_directories = {name for name, info in zip(names, infos, strict=True) if info.is_dir()}
        if not actual_directories <= allowed_directories:
            raise DatasetError(f"contribution contains undeclared directories: {sorted(actual_directories)}")
        infos_by_name = {info.filename: info for info in infos}
        source_size = infos_by_name[f"{clip_path}/source.json"].file_size
        license_size = infos_by_name[license_path].file_size
        review_size = infos_by_name[f"{clip_path}/review.jsonl"].file_size
        if source_size > 4 * 1024 * 1024:
            raise DatasetError("contribution source.json exceeds 4 MiB")
        if license_size > 4 * 1024 * 1024:
            raise DatasetError("contribution license exceeds 4 MiB")
        if review_size:
            raise DatasetError("Studio contributions must carry an empty draft review.jsonl")
        declared_total = sum(info.file_size for info in infos if not info.is_dir())
        if declared_total > max_total_bytes:
            raise DatasetError(
                f"contribution expands to {declared_total} bytes, above limit {max_total_bytes}"
            )

        extracted = 0
        for info in infos:
            if info.is_dir():
                continue
            target = destination / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(info) as source, target.open("wb") as output:
                    while chunk := source.read(1024 * 1024):
                        extracted += len(chunk)
                        if extracted > max_total_bytes:
                            raise DatasetError(
                                f"contribution exceeds expanded byte limit {max_total_bytes}"
                            )
                        output.write(chunk)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise DatasetError(f"cannot extract contribution entry {info.filename}: {exc}") from exc
            if target.stat().st_size != info.file_size:
                raise DatasetError(f"contribution entry size mismatch: {info.filename}")
    return contribution


def _validate_staged_contribution(
    destination_root: Path,
    extracted: Path,
    contribution: dict[str, Any],
) -> None:
    descriptor = load_descriptor(destination_root)
    staged_root = extracted / "staged-dataset"
    staged_root.mkdir()
    staged_descriptor = dict(descriptor)
    staged_descriptor["clips"] = [
        {"id": contribution["clip_id"], "path": contribution["clip_path"]}
    ]
    (staged_root / "dataset.yaml").write_text(yaml.safe_dump(staged_descriptor, sort_keys=False))
    shutil.copytree(destination_root / "schemas", staged_root / "schemas")
    (staged_root / "licenses").mkdir()
    staged_license = staged_root / contribution["license_path"]
    staged_license.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(extracted / contribution["license_path"], staged_license)
    staged_clip = staged_root / contribution["clip_path"]
    staged_clip.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(extracted / contribution["clip_path"], staged_clip)
    if (staged_clip / "review.jsonl").stat().st_size:
        raise DatasetError("Studio contributions must carry an empty draft review.jsonl")
    validate_dataset(staged_root, require_manifest=False)
    source = json.loads((staged_clip / "source.json").read_text())
    if source.get("source", {}).get("license", {}).get("file") != contribution["license_path"]:
        raise DatasetError("contribution license_path does not match source.json")


def _copy_into_draft(
    root: Path,
    extracted: Path,
    contribution: dict[str, Any],
) -> None:
    descriptor = load_descriptor(root)
    clip_id = contribution["clip_id"]
    clip_path = contribution["clip_path"]
    license_path = contribution["license_path"]
    final_clip = root / clip_path
    if any(item["id"] == clip_id or item["path"] == clip_path for item in descriptor["clips"]):
        raise DatasetError(f"dataset already declares contribution clip {clip_id}")
    if final_clip.exists():
        raise DatasetError(f"dataset already contains contribution clip path {clip_path}")

    incoming_license = extracted / license_path
    final_license = root / license_path
    if final_license.exists() and sha256_file(final_license) != sha256_file(incoming_license):
        raise DatasetError(f"dataset contains a different license artifact at {license_path}")
    final_license.parent.mkdir(parents=True, exist_ok=True)
    pending_clip = final_clip.with_name(f".{clip_id}.importing")
    if pending_clip.exists():
        raise DatasetError(f"stale contribution staging path exists: {pending_clip}")
    shutil.copytree(extracted / clip_path, pending_clip)
    created_license = False
    temporary_descriptor = root / ".dataset.yaml.importing"
    try:
        if not final_license.exists():
            temporary_license = final_license.with_name(f".{final_license.name}.importing")
            shutil.copyfile(incoming_license, temporary_license)
            os.replace(temporary_license, final_license)
            created_license = True
        pending_clip.rename(final_clip)
        descriptor["clips"] = sorted(
            [*descriptor["clips"], {"id": clip_id, "path": clip_path}],
            key=lambda item: item["id"],
        )
        body = yaml.safe_dump(descriptor, sort_keys=False).encode()
        temporary_descriptor.write_bytes(body)
        os.replace(temporary_descriptor, root / "dataset.yaml")
    except Exception:
        shutil.rmtree(pending_clip, ignore_errors=True)
        shutil.rmtree(final_clip, ignore_errors=True)
        if created_license:
            final_license.unlink(missing_ok=True)
        raise
    finally:
        temporary_descriptor.unlink(missing_ok=True)


def import_contribution(
    root: str | Path,
    contribution_zip: str | Path,
    *,
    max_total_bytes: int = DEFAULT_MAX_CONTRIBUTION_BYTES,
) -> dict[str, Any]:
    root = Path(root).resolve()
    contribution_zip = Path(contribution_zip).resolve()
    if max_total_bytes <= 0:
        raise DatasetError("max_total_bytes must be positive")
    report = validate_dataset(root)
    if report.state != "draft":
        raise DatasetError("contributions can only be imported into a draft dataset")
    with tempfile.TemporaryDirectory(prefix="cvbench-contribution-") as temporary:
        extracted = Path(temporary)
        contribution = _extract_contribution(
            contribution_zip,
            extracted,
            max_total_bytes=max_total_bytes,
        )
        _validate_staged_contribution(root, extracted, contribution)
        _copy_into_draft(root, extracted, contribution)
    updated = validate_dataset(root)
    clip = next(item for item in updated.clips if item.id == contribution["clip_id"])
    return {
        "dataset": {
            "id": updated.id,
            "version": updated.version,
            "state": updated.state,
            "data_role": updated.data_role,
            "annotation_scope": updated.annotation_scope,
            "evaluation_eligible": updated.evaluation_eligible,
        },
        "imported_clip": clip.id,
        "clip_path": clip.path,
        "annotation_rows": clip.annotation_rows,
        "annotation_origins": clip.annotation_origins,
        "license_path": contribution["license_path"],
    }
