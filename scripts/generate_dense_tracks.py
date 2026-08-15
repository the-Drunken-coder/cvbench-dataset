#!/usr/bin/env python3
"""Generate native-cadence instance masks and tracks for the recovered clips."""

from __future__ import annotations

import sys

# This generator has no sibling imports. Remove its directory before loading dependencies so
# ignored or untracked files beside the script cannot shadow the standard library or packages.
if sys.path:
    sys.path.pop(0)

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pycocotools import mask as mask_utils
from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version

from cvbench_dataset import validate_source_recipe

CONFIG_ARTIFACT = "artifacts/yolo26x-dense-tracking.json"
DATASET_ID = "recovered-clean-videos-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    """Inventory every source-recipe entry, hashing files and rejecting special nodes."""
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"source recipe cannot contain symlinks: {path}")
        if path.is_dir():
            hashes[relative] = "directory"
        elif path.is_file():
            hashes[relative] = f"file:{sha256_file(path)}"
        else:
            raise ValueError(f"source recipe contains an unsupported entry: {path}")
    return hashes


def publication_lock_path(dataset_root: Path) -> Path:
    key = hashlib.sha256(os.fsencode(dataset_root.resolve())).hexdigest()
    return Path(tempfile.gettempdir()) / f"cvbench-dataset-{key}.lock"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def generator_revision() -> str:
    """Return the exact clean Git revision containing the running generator."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError("generator repository HEAD is not a full Git commit")
    top_level = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if Path(top_level).resolve() != REPOSITORY_ROOT:
        raise RuntimeError("generator repository root is not the Git top level")
    index_entries = subprocess.run(
        ["git", "ls-files", "-v", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode().split("\0")
    flagged_paths = [entry for entry in index_entries if entry and not entry.startswith("H ")]
    if flagged_paths:
        raise RuntimeError(f"generator repository has non-normal index flags: {flagged_paths}")
    script_path = Path(__file__).resolve()
    script_relative = script_path.relative_to(REPOSITORY_ROOT).as_posix()
    committed_script = subprocess.run(
        ["git", "show", f"{head}:{script_relative}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if script_path.read_bytes() != committed_script:
        raise RuntimeError("running generator bytes do not match repository HEAD")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("generator repository has changes; commit them before inference")
    return head


def source_timestamp_ns(frame_index: int, numerator: int, denominator: int) -> int:
    scaled = frame_index * 1_000_000_000 * denominator
    return (scaled + numerator // 2) // numerator


def encode_mask(mask: np.ndarray) -> tuple[dict[str, Any], list[int]]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if not isinstance(counts, bytes):
        raise ValueError("pycocotools returned non-canonical mask counts")
    x, y, width, height = (int(round(value)) for value in mask_utils.toBbox(encoded))
    if width <= 0 or height <= 0:
        raise ValueError("model returned an empty instance mask")
    return (
        {"size": [int(encoded["size"][0]), int(encoded["size"][1])], "counts": counts.decode("ascii")},
        [x, y, x + width, y + height],
    )


def tracker_yaml(config: dict[str, Any], output: Path, reid_weights: Path) -> Path:
    path = output / "tracktrack.yaml"
    tracker = dict(config["tracker"])
    tracker["model"] = str(reid_weights)
    path.write_text(yaml.safe_dump(tracker, sort_keys=False))
    return path


def verified_sources(
    dataset_root: Path, source_dir: Path, snapshot_root: Path
) -> dict[str, Path]:
    source_lock = load_json(dataset_root / "source-lock.json")
    expected_names = {clip["filename"] for clip in source_lock["clips"]}
    actual_names = {
        path.name
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    }
    if actual_names != expected_names:
        raise ValueError(
            f"source MP4 inventory mismatch: expected {sorted(expected_names)}, found {sorted(actual_names)}"
        )
    snapshot_root.mkdir()
    verified: dict[str, Path] = {}
    for clip in source_lock["clips"]:
        source = source_dir / clip["filename"]
        snapshot = snapshot_root / clip["filename"]
        shutil.copyfile(source, snapshot)
        if sha256_file(snapshot) != clip["sha256"]:
            raise ValueError(f"source hash mismatch: {source}")
        verified[clip["id"]] = snapshot
    return verified


def frame_rows(
    result: Any,
    *,
    clip_id: str,
    frame_index: int,
    source: dict[str, Any],
    class_names: dict[int, str],
    run_id: str,
) -> list[dict[str, Any]]:
    boxes = result.boxes
    masks = result.masks
    if not len(boxes) or boxes.id is None:
        return []
    if masks is None or len(masks.data) != len(boxes):
        raise ValueError(f"{clip_id} frame {frame_index}: tracked boxes and masks do not align")
    media = source["media"]
    timestamp = source_timestamp_ns(frame_index, media["fps_numerator"], media["fps_denominator"])
    rows: list[dict[str, Any]] = []
    for index, (track_value, class_value, confidence_value) in enumerate(
        zip(boxes.id.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), strict=True)
    ):
        numeric_track_id = int(track_value)
        numeric_class_id = int(class_value)
        class_id = class_names.get(numeric_class_id)
        if class_id is None:
            raise ValueError(f"{clip_id} frame {frame_index}: unexpected class {numeric_class_id}")
        mask = masks.data[index].detach().cpu().numpy() > 0.5
        if mask.shape != (media["height"], media["width"]):
            raise ValueError(
                f"{clip_id} frame {frame_index}: model mask shape {mask.shape} does not match "
                f"source shape {(media['height'], media['width'])}"
            )
        mask_rle, bbox = encode_mask(mask)
        rows.append(
            {
                "schema_version": "cvbench.track-annotation/v1",
                "clip_id": clip_id,
                "frame_index": frame_index,
                "source_timestamp_ns": timestamp,
                "track_id": f"{class_id}-{numeric_track_id:04d}",
                "class_id": class_id,
                "bbox_xyxy": bbox,
                "mask_rle": mask_rle,
                "confidence": round(float(confidence_value), 6),
                "occlusion": "unknown",
                "truncated": (
                    bbox[0] == 0
                    or bbox[1] == 0
                    or bbox[2] == media["width"]
                    or bbox[3] == media["height"]
                ),
                "label_origin": {"kind": "model_generated", "model_run_ids": [run_id]},
            }
        )
    return sorted(rows, key=lambda row: row["track_id"])


def process_class(
    model: YOLO,
    *,
    clip_id: str,
    video: Path,
    source: dict[str, Any],
    config: dict[str, Any],
    tracker_path: Path,
    device: str,
    numeric_class_id: int,
    class_id: str,
) -> list[dict[str, Any]]:
    run_id = f"yolo26x-seg-tracktrack-{clip_id}"
    inference = config["inference"]
    rows: list[dict[str, Any]] = []
    frame_count = 0
    results = model.track(
        source=str(video),
        stream=True,
        persist=False,
        tracker=str(tracker_path),
        device=device,
        classes=[numeric_class_id],
        conf=inference["confidence_threshold"],
        iou=inference["iou_threshold"],
        imgsz=inference["image_size"],
        max_det=inference["maximum_detections"],
        retina_masks=inference["retina_masks"],
        vid_stride=inference["video_stride"],
        save=False,
        verbose=False,
    )
    for frame_index, result in enumerate(results):
        rows.extend(
            frame_rows(
                result,
                clip_id=clip_id,
                frame_index=frame_index,
                source=source,
                class_names={numeric_class_id: class_id},
                run_id=run_id,
            )
        )
        frame_count += 1
        if frame_count % 100 == 0:
            print(
                f"{clip_id} {class_id}: {frame_count}/{source['media']['frame_count']} frames",
                flush=True,
            )
    if frame_count != source["media"]["frame_count"]:
        raise ValueError(
            f"{clip_id} {class_id}: processed {frame_count} frames, "
            f"expected {source['media']['frame_count']}"
        )
    return rows


def process_clip(
    model: YOLO,
    *,
    dataset_root: Path,
    clip_id: str,
    video: Path,
    output_root: Path,
    config: dict[str, Any],
    tracker_path: Path,
    device: str,
) -> dict[str, Any]:
    source = load_json(dataset_root / "clips" / clip_id / "source.json")
    class_names = {int(key): value for key, value in config["inference"]["classes"].items()}
    rows = [
        row
        for numeric_class_id, class_id in class_names.items()
        for row in process_class(
            model,
            clip_id=clip_id,
            video=video,
            source=source,
            config=config,
            tracker_path=tracker_path,
            device=device,
            numeric_class_id=numeric_class_id,
            class_id=class_id,
        )
    ]
    rows.sort(key=lambda row: (row["frame_index"], row["track_id"]))
    clip_output = output_root / clip_id
    clip_output.mkdir()
    tracks = b"".join(canonical_json(row) for row in rows)
    (clip_output / "tracks.jsonl").write_bytes(tracks)
    summary = {
        "clip_id": clip_id,
        "frame_count": source["media"]["frame_count"],
        "model_frames_processed": source["media"]["frame_count"] * len(class_names),
        "annotation_rows": len(rows),
        "track_count": len({row["track_id"] for row in rows}),
        "tracks_sha256": hashlib.sha256(tracks).hexdigest(),
    }
    (clip_output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def updated_source(
    source: dict[str, Any],
    *,
    clip_id: str,
    config: dict[str, Any],
    config_sha256: str,
    weights_sha256: str,
    raw_output_sha256: str,
    generator_revision: str,
    device: str,
) -> dict[str, Any]:
    run_id = f"yolo26x-seg-tracktrack-{clip_id}"
    command = [
        "python",
        "scripts/generate_dense_tracks.py",
        "--dataset-root",
        f"datasets/{DATASET_ID}",
        "--source-dir",
        "<verified-originals>",
        "--weights",
        "<sha256-pinned-yolo26x-seg.pt>",
        "--reid-weights",
        "<sha256-pinned-yolo26n-cls.pt>",
        "--config",
        f"datasets/{DATASET_ID}/{CONFIG_ARTIFACT}",
        "--output-dir",
        "<new-ignored-output-directory>",
        "--device",
        device,
        "--apply",
    ]
    source["model_runs"] = [
        {
            "run_id": run_id,
            "model_name": f"{config['model']['name']} with {config['reid_model']['name']} ReID",
            "model_version": (
                f"{config['model']['version']} + {config['reid_model']['version']}"
            ),
            "weights_uri": config["model"]["weights_uri"],
            "weights_sha256": weights_sha256,
            "code_revision": generator_revision,
            "config_sha256": config_sha256,
            "config_file": CONFIG_ARTIFACT,
            "raw_output_sha256": raw_output_sha256,
            "command": command,
            "license": config["model"]["license"],
        }
    ]
    source["transformations"] = [
        {
            "kind": "dense_model_annotation",
            "description": (
                "Processed every native source frame for person and dog instances; media bytes are unchanged."
            ),
            "tool": f"scripts/generate_dense_tracks.py at {generator_revision}",
            "config_sha256": config_sha256,
            "config_file": CONFIG_ARTIFACT,
        }
    ]
    return source


def stage_dataset(
    dataset_template: Path,
    dataset_root: Path,
    output_root: Path,
    config: dict[str, Any],
    config_bytes: bytes,
    weights_sha256: str,
    tracks_sha256_by_clip: dict[str, str],
    generator_revision: str,
    device: str,
) -> Path:
    stage_parent = Path(tempfile.mkdtemp(prefix="cvbench-dense-stage-", dir=dataset_root.parent))
    stage = stage_parent / dataset_root.name
    try:
        shutil.copytree(dataset_template, stage)
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()

        descriptor_path = stage / "dataset.yaml"
        descriptor = yaml.safe_load(descriptor_path.read_text())
        descriptor["version"] = "0.2.0"
        descriptor["title"] = "Recovered clean videos dense segmentation tracks"
        descriptor["description"] = (
            "Five hash-pinned Pixabay and Pexels videos processed at native cadence with "
            "YOLO26x-seg and TrackTrack appearance association. Dense model output is pending "
            "human review and is not ground truth."
        )
        descriptor["ontology"]["classes"] = [
            {"id": "person", "description": "Model-generated mask and track for a visible person."},
            {"id": "dog", "description": "Model-generated mask and track for a visible dog."},
        ]
        descriptor_path.write_text(yaml.safe_dump(descriptor, sort_keys=False))

        (stage / CONFIG_ARTIFACT).write_bytes(config_bytes)

        for clip in descriptor["clips"]:
            clip_id = clip["id"]
            clip_root = stage / clip["path"]
            generated_tracks = output_root / clip_id / "tracks.jsonl"
            staged_tracks = clip_root / "tracks.jsonl"
            shutil.copyfile(generated_tracks, staged_tracks)
            tracks_sha256 = sha256_file(staged_tracks)
            if tracks_sha256 != tracks_sha256_by_clip[clip_id]:
                raise ValueError(f"{clip_id}: generated tracks changed before staging")
            source_path = clip_root / "source.json"
            source = updated_source(
                load_json(source_path),
                clip_id=clip_id,
                config=config,
                config_sha256=config_sha256,
                weights_sha256=weights_sha256,
                raw_output_sha256=tracks_sha256,
                generator_revision=generator_revision,
                device=device,
            )
            source_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
        validate_source_recipe(stage)
        return stage
    except BaseException:
        shutil.rmtree(stage_parent)
        raise


def apply_stage(
    dataset_root: Path, stage: Path, expected_previous_hashes: dict[str, str]
) -> None:
    flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(publication_lock_path(dataset_root), flags, 0o600)
    with os.fdopen(descriptor, "r+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if tree_hashes(dataset_root) != expected_previous_hashes:
            raise RuntimeError("dataset changed during inference; refusing to overwrite it")
        exchange_directories(dataset_root, stage)
        try:
            if tree_hashes(stage) != expected_previous_hashes:
                raise RuntimeError("dataset changed during inference; refusing to overwrite it")
            validate_source_recipe(dataset_root)
        except BaseException:
            exchange_directories(dataset_root, stage)
            raise
        shutil.rmtree(stage)


def exchange_directories(left: Path, right: Path) -> None:
    """Atomically swap two existing directories on supported annotation hosts."""
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        try:
            rename = library.renameat2
        except AttributeError as exc:
            raise RuntimeError("atomic dataset exchange is unsupported") from exc
        result = rename(-100, os.fsencode(left), -100, os.fsencode(right), 2)  # RENAME_EXCHANGE
    elif sys.platform == "darwin":
        try:
            rename = library.renamex_np
        except AttributeError as exc:
            raise RuntimeError("atomic dataset exchange is unsupported") from exc
        result = rename(os.fsencode(left), os.fsencode(right), 2)  # RENAME_SWAP
    else:
        raise RuntimeError("atomic dataset exchange requires macOS or Linux")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, f"atomic dataset exchange failed: {left} <-> {right}")


def verify_pinned_weights(
    config: dict[str, Any], weights_path: Path, reid_weights_path: Path
) -> tuple[str, str]:
    weights_sha256 = sha256_file(weights_path)
    if weights_sha256 != config["model"]["weights_sha256"]:
        raise ValueError("detector weights do not match the pinned config hash")
    reid_weights_sha256 = sha256_file(reid_weights_path)
    if reid_weights_sha256 != config["reid_model"]["weights_sha256"]:
        raise ValueError("ReID weights do not match the pinned config hash")
    return weights_sha256, reid_weights_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reid-weights", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/configs/yolo26x-dense-tracking.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    source_dir = args.source_dir.resolve()
    weights = args.weights.resolve()
    reid_weights = args.reid_weights.resolve()
    config_path = args.config.resolve()
    output_root = args.output_dir.resolve()
    if output_root == dataset_root or dataset_root in output_root.parents:
        raise ValueError("output directory must be outside dataset root")
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not reid_weights.is_file():
        raise FileNotFoundError(reid_weights)
    if ultralytics_version != "8.4.120":
        raise RuntimeError(f"expected ultralytics 8.4.120, found {ultralytics_version}")
    revision = generator_revision()
    config = load_json(config_path)
    inference_config = config.get("inference")
    if not isinstance(inference_config, dict) or inference_config.get("class_isolated_passes") is not True:
        raise ValueError("generator requires inference.class_isolated_passes to be true")
    if inference_config.get("classes") != {"0": "person", "16": "dog"}:
        raise ValueError("generator requires inference.classes to map 0 to person and 16 to dog")
    config_bytes = canonical_json(config)
    output_root.mkdir(parents=True)
    recipe_snapshot = output_root / "source-recipe"
    source_snapshot = output_root / "verified-sources"
    weights_snapshot = output_root / "verified-weights"
    try:
        weights_snapshot.mkdir()
        detector_snapshot = weights_snapshot / "detector.pt"
        reid_snapshot = weights_snapshot / "reid.pt"
        shutil.copyfile(weights, detector_snapshot)
        shutil.copyfile(reid_weights, reid_snapshot)
        weights_sha256, reid_weights_sha256 = verify_pinned_weights(
            config, detector_snapshot, reid_snapshot
        )
        shutil.copytree(dataset_root, recipe_snapshot)
        report = validate_source_recipe(recipe_snapshot)
        if report.id != DATASET_ID:
            raise ValueError(f"this generator only accepts {DATASET_ID}")
        expected_previous_hashes = tree_hashes(recipe_snapshot)
        sources = verified_sources(recipe_snapshot, source_dir, source_snapshot)
        tracker_path = tracker_yaml(config, output_root, reid_snapshot)
        model = YOLO(str(detector_snapshot))
        summaries = [
            process_clip(
                model,
                dataset_root=recipe_snapshot,
                clip_id=clip_id,
                video=video,
                output_root=output_root,
                config=config,
                tracker_path=tracker_path,
                device=args.device,
            )
            for clip_id, video in sources.items()
        ]
        manifest = {
            "schema_version": "cvbench.dense-tracking-run/v1",
            "generator_revision": revision,
            "weights_sha256": weights_sha256,
            "reid_weights_sha256": reid_weights_sha256,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "clips": summaries,
        }
        (output_root / "run.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        if args.apply:
            verify_pinned_weights(config, detector_snapshot, reid_snapshot)
            stage = stage_dataset(
                recipe_snapshot,
                dataset_root,
                output_root,
                config,
                config_bytes,
                weights_sha256,
                {summary["clip_id"]: summary["tracks_sha256"] for summary in summaries},
                revision,
                args.device,
            )
            apply_stage(dataset_root, stage, expected_previous_hashes)
            shutil.rmtree(stage.parent)
    finally:
        shutil.rmtree(source_snapshot, ignore_errors=True)
        shutil.rmtree(recipe_snapshot, ignore_errors=True)
        shutil.rmtree(weights_snapshot, ignore_errors=True)


if __name__ == "__main__":
    main()
