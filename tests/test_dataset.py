from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

import cvbench_dataset.manifest as manifest_module
import cvbench_dataset.source_recipe as source_recipe_module
import cvbench_dataset.validator as validator_module
from cvbench_dataset import (
    DatasetError,
    build_release,
    hydrate_source_recipe,
    import_contribution,
    init_dataset,
    validate_dataset,
    validate_source_recipe,
    verify_release,
)
from cvbench_dataset.cli import main

ROOT = Path(__file__).parents[1]
SAMPLE = ROOT / "examples" / "minimal-certified"


def _copy_sample(tmp_path: Path) -> Path:
    destination = tmp_path / "dataset"
    shutil.copytree(SAMPLE, destination)
    (destination / "release-manifest.json").unlink(missing_ok=True)
    return destination


def _draft_destination(tmp_path: Path) -> Path:
    dataset = _copy_sample(tmp_path)
    descriptor = yaml.safe_load((dataset / "dataset.yaml").read_text())
    descriptor["state"] = "draft"
    descriptor["evaluation_eligible"] = False
    descriptor["certification"].pop("certified_at")
    (dataset / "dataset.yaml").write_text(yaml.safe_dump(descriptor, sort_keys=False))
    return dataset


def _source_recipe(tmp_path: Path) -> tuple[Path, Path]:
    recipe = _draft_destination(tmp_path)
    descriptor = yaml.safe_load((recipe / "dataset.yaml").read_text())
    descriptor["data_role"] = "training_only"
    descriptor["annotation_scope"] = "sparse"
    (recipe / "dataset.yaml").write_text(yaml.safe_dump(descriptor, sort_keys=False))
    clip = recipe / "clips" / "synthetic-clip"
    video = clip / "video.mp4"
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_name = "synthetic-source.mp4"
    shutil.copyfile(video, source_dir / source_name)
    video.unlink()
    (clip / "review.jsonl").write_text("")
    source = json.loads((clip / "source.json").read_text())
    config = recipe / "artifacts" / "synthetic-model-config.json"
    config.parent.mkdir()
    config.write_text('{"offline":true}\n')
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    run_id = "sample-model-run"
    source["model_runs"] = [
        {
            "run_id": run_id,
            "model_name": "Synthetic detector",
            "model_version": "1",
            "weights_uri": "https://example.invalid/model.bin",
            "weights_sha256": "1" * 64,
            "code_revision": "abcdef1",
            "config_sha256": config_sha256,
            "config_file": "artifacts/synthetic-model-config.json",
            "raw_output_sha256": "3" * 64,
            "command": ["synthetic-detector", "--offline"],
            "license": {"spdx": "MIT", "url": "https://opensource.org/license/mit"},
        }
    ]
    (clip / "source.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    rows = []
    for line in (clip / "tracks.jsonl").read_text().splitlines():
        row = json.loads(line)
        row["label_origin"] = {"kind": "model_generated", "model_run_ids": [run_id]}
        rows.append(row)
    (clip / "tracks.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)
    )
    (recipe / "source-lock.json").write_text(
        json.dumps(
            {
                "schema_version": "cvbench.source-recipe/v1",
                "clips": [
                    {
                        "id": "synthetic-clip",
                        "filename": source_name,
                        "sha256": source["source"]["sha256"],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (recipe / "README.md").write_text("# Synthetic source recipe\n")
    return recipe, source_dir


def _studio_zip(tmp_path: Path, *, review_body: bytes = b"", unsafe_name: str | None = None) -> Path:
    clip_id = "imported-clip"
    clip_path = f"clips/{clip_id}"
    sample_clip = SAMPLE / "clips" / "synthetic-clip"
    tracks = [
        {**json.loads(line), "clip_id": clip_id}
        for line in (sample_clip / "tracks.jsonl").read_text().splitlines()
    ]
    source = json.loads((sample_clip / "source.json").read_text())
    source["clip_id"] = clip_id
    source["source"]["uri"] = f"synthetic://cvbench/studio/{clip_id}"
    contribution = {
        "schema_version": "cvbench.studio-contribution/v1",
        "clip_id": clip_id,
        "clip_path": clip_path,
        "license_path": "licenses/MIT.txt",
    }
    path = tmp_path / "contribution.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("contribution.json", json.dumps(contribution))
        archive.writestr(f"{clip_path}/video.mp4", (sample_clip / "video.mp4").read_bytes())
        archive.writestr(
            f"{clip_path}/tracks.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in tracks),
        )
        archive.writestr(f"{clip_path}/source.json", json.dumps(source, indent=2, sort_keys=True) + "\n")
        archive.writestr(f"{clip_path}/review.jsonl", review_body)
        archive.writestr("licenses/MIT.txt", (SAMPLE / "licenses" / "MIT.txt").read_bytes())
        if unsafe_name:
            archive.writestr(unsafe_name, b"unsafe")
    return path


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _encode_coco_rle(runs: list[int]) -> str:
    encoded: list[str] = []
    for index, original in enumerate(runs):
        value = original - runs[index - 2] if index > 2 else original
        while True:
            code = value & 0x1F
            value >>= 5
            more = value != (-1 if code & 0x10 else 0)
            if more:
                code |= 0x20
            encoded.append(chr(code + 48))
            if not more:
                break
    return "".join(encoded)


def _rectangle_runs() -> list[int]:
    pixels = [0] * (16 * 16)
    for x in range(2, 9):
        for y in range(2, 13):
            pixels[x * 16 + y] = 1
    runs: list[int] = []
    current = 0
    length = 0
    for pixel in pixels:
        if pixel == current:
            length += 1
        else:
            runs.append(length)
            current = pixel
            length = 1
    runs.append(length)
    return runs


def _rectangle_rle() -> dict:
    return {"size": [16, 16], "counts": _encode_coco_rle(_rectangle_runs())}


def test_init_creates_a_valid_empty_draft(tmp_path: Path) -> None:
    root = tmp_path / "new-dataset"
    result = init_dataset(
        root,
        dataset_id="new-sports-dataset",
        title="New sports dataset",
        description="A contributor-owned draft.",
        classes=["person=Visible participant", "ball=Visible game ball"],
    )
    assert result["state"] == "draft"
    assert result["data_role"] == "training_only"
    assert result["annotation_scope"] == "exhaustive_visible"
    assert result["evaluation_eligible"] is False
    assert result["clips"] == []
    assert {path.name for path in root.iterdir()} == {
        "clips",
        "dataset.yaml",
        "licenses",
        "schemas",
    }
    assert validate_dataset(root).to_dict() == result


def test_init_cli_creates_a_studio_ready_draft(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = tmp_path / "cli-dataset"
    exit_code = main(
        [
            "init",
            str(root),
            "--id",
            "cli-sports-dataset",
            "--title",
            "CLI sports dataset",
            "--description",
            "Created without copying a fixture.",
            "--class",
            "person=Visible participant",
            "--class",
            "ball=Visible game ball",
            "--data-role",
            "benchmark_candidate",
            "--annotation-scope",
            "class_exhaustive",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["data_role"] == "benchmark_candidate"
    assert output["annotation_scope"] == "class_exhaustive"
    assert output["evaluation_eligible"] is False
    assert validate_dataset(root).state == "draft"


def test_init_never_promotes_declared_truth_to_evaluation_eligible(tmp_path: Path) -> None:
    root = tmp_path / "truth-draft"
    result = init_dataset(
        root,
        dataset_id="truth-draft",
        title="Truth candidate",
        description="Still requires review and certification.",
        classes=["person=Visible participant"],
        data_role="benchmark_truth",
        annotation_scope="exhaustive_visible",
    )
    assert result["state"] == "draft"
    assert result["evaluation_eligible"] is False


def test_source_recipe_hydrates_only_hash_pinned_local_media(tmp_path: Path) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    report = validate_source_recipe(recipe)
    assert report.data_role == "training_only"
    assert report.evaluation_eligible is False
    assert report.clips[0].annotation_origins == {"model_generated": 2}

    output = tmp_path / "hydrated"
    result = hydrate_source_recipe(recipe, source_dir, output)
    assert result["data_role"] == "training_only"
    assert result["evaluation_eligible"] is False
    assert result["annotation_origins"] == {"model_generated": 2}
    assert validate_dataset(output).to_dict() == result


def test_source_recipe_rejects_media_drift_without_partial_output(tmp_path: Path) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    (source_dir / "synthetic-source.mp4").write_bytes(b"changed")
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="hash mismatch"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()


@pytest.mark.parametrize("container", ["artifacts", "clips", "licenses", "schemas"])
def test_source_recipe_rejects_non_directory_containers(tmp_path: Path, container: str) -> None:
    recipe, _ = _source_recipe(tmp_path)
    path = recipe / container
    shutil.rmtree(path)
    path.write_text("not a directory\n")
    with pytest.raises(DatasetError, match=f"{container} must be a directory"):
        validate_source_recipe(recipe)


def test_source_recipe_rejects_non_string_lock_id(tmp_path: Path) -> None:
    recipe, _ = _source_recipe(tmp_path)
    source_lock = json.loads((recipe / "source-lock.json").read_text())
    source_lock["clips"][0]["id"] = ["synthetic-clip"]
    _write_json(recipe / "source-lock.json", source_lock)
    with pytest.raises(DatasetError, match="clip IDs must be strings"):
        validate_source_recipe(recipe)


def test_source_recipe_rejects_schema_directory(tmp_path: Path) -> None:
    recipe, _ = _source_recipe(tmp_path)
    schema = recipe / "schemas" / "dataset-v1.schema.json"
    schema.unlink()
    schema.mkdir()
    with pytest.raises(DatasetError, match="canonical schema must be a regular file"):
        validate_source_recipe(recipe)


def test_hydration_rejects_recipe_symlink_before_inventorying_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    config = recipe / "artifacts" / "synthetic-model-config.json"
    external = tmp_path / "external.json"
    external.write_text('{"outside":true}\n')
    config.unlink()
    try:
        config.symlink_to(external)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    monkeypatch.setattr(
        source_recipe_module,
        "sha256_file",
        lambda path: pytest.fail(f"unsafe recipe entry was hashed: {path}"),
    )
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="cannot contain symlinks"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()


def test_source_recipe_rejects_conflicting_hashes_for_one_filename(tmp_path: Path) -> None:
    recipe, _ = _source_recipe(tmp_path)
    descriptor = yaml.safe_load((recipe / "dataset.yaml").read_text())
    descriptor["clips"].append({"id": "synthetic-clip-2", "path": "clips/synthetic-clip-2"})
    (recipe / "dataset.yaml").write_text(yaml.safe_dump(descriptor, sort_keys=False))
    source_clip = recipe / "clips" / "synthetic-clip"
    second_clip = recipe / "clips" / "synthetic-clip-2"
    shutil.copytree(source_clip, second_clip)
    source = json.loads((second_clip / "source.json").read_text())
    source["clip_id"] = "synthetic-clip-2"
    source["source"]["sha256"] = "f" * 64
    _write_json(second_clip / "source.json", source)
    rows = [json.loads(line) for line in (second_clip / "tracks.jsonl").read_text().splitlines()]
    for row in rows:
        row["clip_id"] = "synthetic-clip-2"
    (second_clip / "tracks.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)
    )
    source_lock = json.loads((recipe / "source-lock.json").read_text())
    source_lock["clips"].append(
        {"id": "synthetic-clip-2", "filename": "synthetic-source.mp4", "sha256": "f" * 64}
    )
    _write_json(recipe / "source-lock.json", source_lock)

    with pytest.raises(DatasetError, match="conflicting SHA-256"):
        validate_source_recipe(recipe)


def test_hydration_does_not_replace_target_created_during_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    output = tmp_path / "hydrated"
    validate_dataset_original = source_recipe_module.validate_dataset
    created = False

    def validate_then_create_competing_output(root: Path):
        nonlocal created
        result = validate_dataset_original(root)
        if not created:
            output.mkdir()
            (output / "owner.txt").write_text("preserve me\n")
            created = True
        return result

    monkeypatch.setattr(source_recipe_module, "validate_dataset", validate_then_create_competing_output)
    with pytest.raises(DatasetError, match="target already exists"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert (output / "owner.txt").read_text() == "preserve me\n"


def test_hydration_does_not_follow_target_symlink_created_during_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    output = tmp_path / "hydrated"
    redirected = tmp_path / "redirected"
    validate_original = source_recipe_module.validate_source_recipe
    created = False

    def validate_then_create_symlink(root: Path):
        nonlocal created
        report = validate_original(root)
        if Path(root).resolve() != recipe.resolve() and not created:
            output.symlink_to(redirected)
            created = True
        return report

    monkeypatch.setattr(source_recipe_module, "validate_source_recipe", validate_then_create_symlink)
    with pytest.raises(DatasetError, match="target already exists"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert output.is_symlink()
    assert not redirected.exists()


def test_hydration_publishes_complete_directory_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    output = tmp_path / "hydrated"
    rename_original = source_recipe_module._rename_no_replace
    calls = 0

    def inspect_then_rename(source: Path, destination: Path, **kwargs) -> None:
        nonlocal calls
        calls += 1
        assert destination == Path(output.name)
        assert not output.exists()
        staged = source_recipe_module._directory_fd_path(kwargs["source_dir_fd"]) / source
        assert (staged / "dataset.yaml").is_file()
        assert (staged / "clips" / "synthetic-clip" / "video.mp4").is_file()
        rename_original(source, destination, **kwargs)

    monkeypatch.setattr(source_recipe_module, "_rename_no_replace", inspect_then_rename)
    hydrate_source_recipe(recipe, source_dir, output)
    assert calls == 1
    assert validate_dataset(output).id == "minimal-synthetic"


def test_hydration_rejects_replaced_output_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    parent = tmp_path / "publish"
    parent.mkdir()
    output = parent / "hydrated"
    moved_parent = tmp_path / "moved-publish"
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    validate_original = source_recipe_module.validate_dataset

    def validate_then_replace_parent(root: Path):
        result = validate_original(root)
        parent.rename(moved_parent)
        parent.symlink_to(redirected, target_is_directory=True)
        return result

    monkeypatch.setattr(source_recipe_module, "validate_dataset", validate_then_replace_parent)
    with pytest.raises(DatasetError, match="output parent changed"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not (redirected / "hydrated").exists()
    assert not (moved_parent / "hydrated").exists()


def test_hydration_rejects_symlinked_output_parent(tmp_path: Path) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    actual_parent = tmp_path / "actual-parent"
    actual_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(actual_parent, target_is_directory=True)
    with pytest.raises(DatasetError, match="cannot anchor hydrate output parent"):
        hydrate_source_recipe(recipe, source_dir, linked_parent / "hydrated")
    assert not (actual_parent / "hydrated").exists()


def test_hydration_rejects_replaced_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    output = tmp_path / "hydrated"
    validate_original = source_recipe_module.validate_dataset
    replacement: Path | None = None

    def validate_then_replace_staging(root: Path):
        nonlocal replacement
        result = validate_original(root)
        stolen = root.with_name(f"{root.name}-stolen")
        root.rename(stolen)
        root.mkdir()
        (root / "owner.txt").write_text("preserve me\n")
        replacement = root.resolve()
        return result

    monkeypatch.setattr(source_recipe_module, "validate_dataset", validate_then_replace_staging)
    with pytest.raises(DatasetError, match="staging directory changed"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()
    assert replacement is not None
    assert (replacement / "owner.txt").read_text() == "preserve me\n"


def test_hydration_fails_closed_without_directory_anchored_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    monkeypatch.setattr(
        source_recipe_module,
        "_directory_anchored_publication_supported",
        lambda: False,
    )
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="unsupported on this platform"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()


def test_hydration_rejects_unprotected_shared_output_parent(tmp_path: Path) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o777)
    with pytest.raises(DatasetError, match="private or use sticky-directory protection"):
        hydrate_source_recipe(recipe, source_dir, parent / "hydrated")


def test_hydration_rejects_output_inside_source_recipe(tmp_path: Path) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    output = recipe / "clips" / "hydrated"
    with pytest.raises(DatasetError, match="outside the source recipe"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()
    validate_source_recipe(recipe)


def test_hydration_rechecks_copied_source_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    source_video = source_dir / "synthetic-source.mp4"
    copyfile_original = source_recipe_module.shutil.copyfile

    def replace_source_then_copy(source: Path, destination: Path, *args, **kwargs):
        if Path(source) == source_video:
            changed = bytearray(source_video.read_bytes())
            changed[-1] ^= 1
            source_video.write_bytes(changed)
        return copyfile_original(source, destination, *args, **kwargs)

    monkeypatch.setattr(source_recipe_module.shutil, "copyfile", replace_source_then_copy)
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="changed during hydration"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()


def test_hydration_revalidates_copied_recipe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    clips = recipe / "clips"
    tracks = clips / "synthetic-clip" / "tracks.jsonl"
    copytree_original = source_recipe_module.shutil.copytree

    def mutate_recipe_then_copy(source: Path, destination: Path, *args, **kwargs):
        if Path(source) == clips:
            rows = [json.loads(line) for line in tracks.read_text().splitlines()]
            rows[0]["label_origin"] = {"kind": "human", "model_run_ids": []}
            tracks.write_text(
                "".join(
                    json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows
                )
            )
        return copytree_original(source, destination, *args, **kwargs)

    monkeypatch.setattr(source_recipe_module.shutil, "copytree", mutate_recipe_then_copy)
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="model-derived"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()


def test_hydration_binds_complete_snapshot_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    inventory_original = source_recipe_module._recipe_inventory
    mutated = False

    def inventory_then_mutate(root: Path):
        nonlocal mutated
        inventory = inventory_original(root)
        if Path(root).resolve() != recipe.resolve() and not mutated:
            tracks = Path(root) / "clips" / "synthetic-clip" / "tracks.jsonl"
            rows = [json.loads(line) for line in tracks.read_text().splitlines()]
            rows[0]["bbox_xyxy"][0] += 0.25
            tracks.write_text(
                "".join(
                    json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows
                )
            )
            mutated = True
        return inventory

    monkeypatch.setattr(source_recipe_module, "_recipe_inventory", inventory_then_mutate)
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="source recipe changed"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()


def test_hydration_rechecks_hydrated_bytes_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    inventory_original = source_recipe_module._recipe_inventory
    mutated = False

    def inventory_then_mutate(root: Path):
        nonlocal mutated
        inventory = inventory_original(root)
        root = Path(root)
        if not (root / "source-lock.json").exists() and not mutated:
            tracks = root / "clips" / "synthetic-clip" / "tracks.jsonl"
            rows = [json.loads(line) for line in tracks.read_text().splitlines()]
            rows[0]["bbox_xyxy"][0] += 0.25
            tracks.write_text(
                "".join(
                    json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows
                )
            )
            mutated = True
        return inventory

    monkeypatch.setattr(source_recipe_module, "_recipe_inventory", inventory_then_mutate)
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="changed during publication"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()


def test_hydration_rechecks_recipe_constraints_after_media_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    copy_original = source_recipe_module.shutil.copyfile
    mutated = False

    def copy_then_change_origin(source: Path, destination: Path, *args, **kwargs):
        nonlocal mutated
        result = copy_original(source, destination, *args, **kwargs)
        destination = Path(destination)
        if destination.name == "video.mp4" and not mutated:
            tracks = destination.parent / "tracks.jsonl"
            rows = [json.loads(line) for line in tracks.read_text().splitlines()]
            rows[0]["label_origin"] = {"kind": "human", "model_run_ids": []}
            tracks.write_text(
                "".join(
                    json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows
                )
            )
            mutated = True
        return result

    monkeypatch.setattr(source_recipe_module.shutil, "copyfile", copy_then_change_origin)
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="source recipe artifacts changed"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()


def test_hydration_preserves_recipe_artifact_hashes_during_media_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    copy_original = source_recipe_module.shutil.copyfile
    mutated = False

    def copy_then_change_license(source: Path, destination: Path, *args, **kwargs):
        nonlocal mutated
        result = copy_original(source, destination, *args, **kwargs)
        destination = Path(destination)
        if destination.name == "video.mp4" and not mutated:
            (destination.parents[2] / "licenses" / "MIT.txt").write_text("changed license\n")
            mutated = True
        return result

    monkeypatch.setattr(source_recipe_module.shutil, "copyfile", copy_then_change_license)
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="source recipe artifacts changed"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()


def test_hydration_normalizes_staging_creation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    mkdir_original = source_recipe_module.os.mkdir

    def fail_staging(path, *args, **kwargs):
        if str(path).startswith(".hydrated-"):
            raise PermissionError("fixture staging denial")
        return mkdir_original(path, *args, **kwargs)

    monkeypatch.setattr(source_recipe_module.os, "mkdir", fail_staging)
    with pytest.raises(DatasetError, match="cannot create hydrate staging directory"):
        hydrate_source_recipe(recipe, source_dir, tmp_path / "hydrated")


def test_hydration_rejects_undeclared_recipe_root_entries(tmp_path: Path) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    (recipe / "downloaded-video.mp4").write_bytes(b"undeclared")
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="source recipe must contain exactly"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()


def test_hydration_compares_snapshot_with_source_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    copytree_original = source_recipe_module.shutil.copytree

    def copytree_with_transient_substitution(source: Path, destination: Path, *args, **kwargs):
        result = copytree_original(source, destination, *args, **kwargs)
        if Path(source).name == "clips":
            tracks = Path(destination) / "synthetic-clip" / "tracks.jsonl"
            rows = [json.loads(line) for line in tracks.read_text().splitlines()]
            tracks.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        return result

    monkeypatch.setattr(source_recipe_module.shutil, "copytree", copytree_with_transient_substitution)
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="source recipe changed during hydration"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()


def test_hydration_detects_staging_swap_during_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    rename_original = source_recipe_module._rename_no_replace
    swapped = False

    def swap_then_rename(source: Path, destination: Path, **kwargs) -> None:
        nonlocal swapped
        if swapped:
            rename_original(source, destination, **kwargs)
            return
        swapped = True
        parent = source_recipe_module._directory_fd_path(kwargs["source_dir_fd"])
        staging = parent / source
        stolen = parent / f"{source}.stolen"
        staging.rename(stolen)
        shutil.copytree(stolen, staging)
        (staging / "clips" / "synthetic-clip" / "video.mp4").write_bytes(b"replacement")
        rename_original(source, destination, **kwargs)

    monkeypatch.setattr(source_recipe_module, "_rename_no_replace", swap_then_rename)
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="staging directory changed during publication"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".hydrated.rejected-*"))


def test_hydration_rechecks_requested_parent_after_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    parent = tmp_path / "publish"
    parent.mkdir()
    output = parent / "hydrated"
    moved_parent = tmp_path / "moved-publish"
    rename_original = source_recipe_module._rename_no_replace
    moved = False

    def publish_then_move_parent(source: Path, destination: Path, **kwargs) -> None:
        nonlocal moved
        rename_original(source, destination, **kwargs)
        if not moved:
            moved = True
            parent.rename(moved_parent)
            parent.mkdir()

    monkeypatch.setattr(source_recipe_module, "_rename_no_replace", publish_then_move_parent)
    with pytest.raises(DatasetError, match="output parent changed during publication"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()
    assert validate_dataset(moved_parent / "hydrated").id == "minimal-synthetic"
    assert not list(moved_parent.glob(".hydrated.rejected-*"))


def test_hydration_requires_existing_output_parent(tmp_path: Path) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    output = tmp_path / "missing" / "hydrated"
    with pytest.raises(DatasetError, match="cannot anchor hydrate output parent"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert not output.exists()


def test_hydration_leaves_mutated_publication_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    rename_original = source_recipe_module._rename_no_replace
    mutated = False

    def publish_then_mutate(source: Path, destination: Path, **kwargs) -> None:
        nonlocal mutated
        rename_original(source, destination, **kwargs)
        if not mutated:
            mutated = True
            parent = source_recipe_module._directory_fd_path(kwargs["destination_dir_fd"])
            (parent / destination / "clips" / "synthetic-clip" / "video.mp4").write_bytes(
                b"replacement"
            )

    monkeypatch.setattr(source_recipe_module, "_rename_no_replace", publish_then_mutate)
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="hydrated dataset changed during publication"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert (output / "clips" / "synthetic-clip" / "video.mp4").read_bytes() == b"replacement"
    assert not list(tmp_path.glob(".hydrated.rejected-*"))


def test_hydration_leaves_publication_after_post_publication_io_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    output = tmp_path / "hydrated"
    validate_original = source_recipe_module.validate_dataset

    def fail_published_read(root: Path, *args, **kwargs):
        if Path(root).name == output.name:
            raise PermissionError("fixture read denial")
        return validate_original(root, *args, **kwargs)

    monkeypatch.setattr(source_recipe_module, "validate_dataset", fail_published_read)
    with pytest.raises(DatasetError, match="hydrated dataset changed during publication"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert output.is_dir()
    assert not list(tmp_path.glob(".hydrated.rejected-*"))


def test_hydration_rejects_replacement_before_publication_identity_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    rename_original = source_recipe_module._rename_no_replace
    displaced = tmp_path / "displaced-valid-publication"
    replaced = False

    def publish_then_replace(source: Path, destination: Path, **kwargs) -> None:
        nonlocal replaced
        rename_original(source, destination, **kwargs)
        if not replaced:
            replaced = True
            parent = source_recipe_module._directory_fd_path(kwargs["destination_dir_fd"])
            current = parent / destination
            current.rename(displaced)
            current.mkdir()
            (current / "owner.txt").write_text("unrelated replacement\n")

    monkeypatch.setattr(source_recipe_module, "_rename_no_replace", publish_then_replace)
    output = tmp_path / "hydrated"
    with pytest.raises(DatasetError, match="published name changed"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert (output / "owner.txt").read_text() == "unrelated replacement\n"
    assert validate_dataset(displaced).id == "minimal-synthetic"
    assert not list(tmp_path.glob(".hydrated.rejected-*"))


def test_hydration_rebinds_published_name_after_final_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    output = tmp_path / "hydrated"
    displaced = tmp_path / "displaced-final-publication"
    check_parent_original = source_recipe_module._assert_output_parent_unchanged
    replaced = False

    def check_parent_then_replace(path: Path, opened_parent: os.stat_result) -> None:
        nonlocal replaced
        check_parent_original(path, opened_parent)
        if path.exists() and not replaced:
            replaced = True
            path.rename(displaced)
            path.mkdir()
            (path / "owner.txt").write_text("unrelated replacement\n")

    monkeypatch.setattr(
        source_recipe_module,
        "_assert_output_parent_unchanged",
        check_parent_then_replace,
    )
    with pytest.raises(DatasetError, match="published name changed"):
        hydrate_source_recipe(recipe, source_dir, output)
    assert (output / "owner.txt").read_text() == "unrelated replacement\n"
    assert validate_dataset(displaced).id == "minimal-synthetic"


def test_canonical_validation_rejects_config_artifact_drift(tmp_path: Path) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    output = tmp_path / "hydrated"
    hydrate_source_recipe(recipe, source_dir, output)
    (output / "artifacts" / "synthetic-model-config.json").write_text('{"offline":false}\n')
    with pytest.raises(DatasetError, match="config_file SHA-256"):
        validate_dataset(output)


def test_canonical_validation_rejects_unreferenced_config_artifact(tmp_path: Path) -> None:
    recipe, source_dir = _source_recipe(tmp_path)
    output = tmp_path / "hydrated"
    hydrate_source_recipe(recipe, source_dir, output)
    (output / "artifacts" / "orphan.json").write_text("{}\n")
    with pytest.raises(DatasetError, match="config artifacts mismatch"):
        validate_dataset(output)


def test_canonical_validation_rejects_empty_artifact_directories(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    (dataset / "artifacts" / "empty").mkdir(parents=True)
    with pytest.raises(DatasetError, match="config artifact directories mismatch"):
        validate_dataset(dataset, require_manifest=False)


def test_canonical_validation_rejects_artifacts_file(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    (dataset / "artifacts").write_text("not a directory\n")
    with pytest.raises(DatasetError, match="artifacts entry must be a directory"):
        validate_dataset(dataset, require_manifest=False)


def test_source_recipe_rejects_non_finite_confidence(tmp_path: Path) -> None:
    recipe, _ = _source_recipe(tmp_path)
    tracks = recipe / "clips" / "synthetic-clip" / "tracks.jsonl"
    rows = [json.loads(line) for line in tracks.read_text().splitlines()]
    rows[0]["confidence"] = float("nan")
    tracks.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)
    )
    with pytest.raises(DatasetError, match="confidence must be finite"):
        validate_source_recipe(recipe)


def test_canonical_validation_accepts_compact_source_resolution_mask(tmp_path: Path) -> None:
    dataset = _draft_destination(tmp_path)
    tracks = dataset / "clips" / "synthetic-clip" / "tracks.jsonl"
    rows = [json.loads(line) for line in tracks.read_text().splitlines()]
    rows[0]["mask_rle"] = _rectangle_rle()
    tracks.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    assert validate_dataset(dataset, require_manifest=False).annotation_rows == 2


def test_mask_validation_rejects_unsupported_dimensions_before_decoding() -> None:
    width = validator_module.MAX_MASK_DIMENSION + 1
    row = {
        "bbox_xyxy": [0, 0, 1, 1],
        "mask_rle": {"size": [1, width], "counts": "P" * 20_001 + "0"},
    }
    with pytest.raises(DatasetError, match="dimensions exceed the supported limit"):
        validator_module._validate_mask(row, {"height": 1, "width": width}, "test mask")


def test_rle_decoder_rejects_implementation_length_limit() -> None:
    counts = "0" * (validator_module.MAX_MASK_RLE_CHARACTERS + 1)
    with pytest.raises(DatasetError, match="counts exceed the implementation limit"):
        validator_module._decode_coco_rle(counts, 256, "test mask")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["mask_rle"].update(size=[15, 16]), "size does not match"),
        (lambda row: row["mask_rle"].update(size=[16]), "too short"),
        (lambda row: row["mask_rle"].update(counts="1"), "runs do not cover"),
        (lambda row: row.update(bbox_xyxy=[1, 2, 9, 13]), "does not match mask_rle bounds"),
        (lambda row: row["mask_rle"].update(counts="!"), "invalid character"),
        (lambda row: row["mask_rle"].update(counts="P"), "truncated"),
        (lambda row: row["mask_rle"].update(counts="O"), "negative run"),
        (
            lambda row: row["mask_rle"].update(counts=_encode_coco_rle([16 * 16])),
            "no foreground pixels",
        ),
        (lambda row: row["mask_rle"].update(counts="PPP0"), "media-derived limit"),
        (
            lambda row: row["mask_rle"].update(
                counts=_encode_coco_rle([*_rectangle_runs(), 0, 0])
            ),
            "not canonical",
        ),
    ],
)
def test_canonical_validation_rejects_invalid_mask_rle(tmp_path: Path, mutation, message: str) -> None:
    dataset = _draft_destination(tmp_path)
    tracks = dataset / "clips" / "synthetic-clip" / "tracks.jsonl"
    rows = [json.loads(line) for line in tracks.read_text().splitlines()]
    rows[0]["mask_rle"] = _rectangle_rle()
    mutation(rows[0])
    tracks.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    with pytest.raises(DatasetError, match=message):
        validate_dataset(dataset, require_manifest=False)


def test_source_recipe_rejects_non_model_labels(tmp_path: Path) -> None:
    recipe, _ = _source_recipe(tmp_path)
    tracks = recipe / "clips" / "synthetic-clip" / "tracks.jsonl"
    rows = [json.loads(line) for line in tracks.read_text().splitlines()]
    rows[0]["label_origin"] = {"kind": "human", "model_run_ids": []}
    tracks.write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)
    )
    with pytest.raises(DatasetError, match="model-derived"):
        validate_source_recipe(recipe)


@pytest.mark.parametrize("existing_kind", ["directory", "file"])
def test_init_refuses_any_existing_target(tmp_path: Path, existing_kind: str) -> None:
    root = tmp_path / "already-here"
    if existing_kind == "directory":
        root.mkdir()
    else:
        root.write_text("preserve me\n")
    with pytest.raises(DatasetError, match="already exists"):
        init_dataset(
            root,
            dataset_id="new-sports-dataset",
            title="New sports dataset",
            description="A contributor-owned draft.",
            classes=["person=Visible participant"],
        )
    assert root.exists()


@pytest.mark.parametrize("classes", [[], ["missing-separator"], ["person=one", "person=two"]])
def test_init_rejects_invalid_class_arguments_without_creating_target(
    tmp_path: Path,
    classes: list[str],
) -> None:
    root = tmp_path / "invalid"
    with pytest.raises(DatasetError):
        init_dataset(
            root,
            dataset_id="new-sports-dataset",
            title="New sports dataset",
            description="A contributor-owned draft.",
            classes=classes,
        )
    assert not root.exists()


def _rewrite_reviews(dataset: Path, reviewers: tuple[str, ...] = ("reviewer-a", "reviewer-b")) -> None:
    clip = dataset / "clips" / "synthetic-clip"
    hashes = {
        "video_sha256": hashlib.sha256((clip / "video.mp4").read_bytes()).hexdigest(),
        "tracks_sha256": hashlib.sha256((clip / "tracks.jsonl").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256((clip / "source.json").read_bytes()).hexdigest(),
    }
    rows = [
        {
            "artifacts": hashes,
            "clip_id": "synthetic-clip",
            "decision": "approve",
            "rationale": "Independent complete fixture review.",
            "review_id": f"review-{index}",
            "reviewed_at": f"2026-07-27T0{index}:00:00Z",
            "reviewer": {"id": reviewer, "independent": True, "kind": "human"},
            "schema_version": "cvbench.review/v1",
            "scope": "all_annotations",
        }
        for index, reviewer in enumerate(reviewers, start=1)
    ]
    (clip / "review.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n" for row in rows)
    )


def test_committed_certified_fixture_validates_and_exposes_origin_counts() -> None:
    report = validate_dataset(SAMPLE)
    assert report.id == "minimal-synthetic"
    assert report.state == "certified"
    assert report.data_role == "benchmark_truth"
    assert report.annotation_scope == "exhaustive_visible"
    assert report.evaluation_eligible is True
    assert report.annotation_rows == 2
    assert report.annotation_origins == {"human": 2}
    assert report.clips[0].approved_reviewers == ["example-reviewer-a", "example-reviewer-b"]


def test_release_build_is_byte_deterministic_and_verifiable(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    (dataset / "licenses" / "release-manifest.json").mkdir()
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    first_result = build_release(dataset, first)
    second_result = build_release(dataset, second)
    assert first.read_bytes() == second.read_bytes()
    assert first_result["archive_sha256"] == second_result["archive_sha256"]
    verified = verify_release(dataset, first)
    assert verified["archive_sha256"] == first_result["archive_sha256"]
    assert verified["files"] == 13
    with tarfile.open(first, "r:gz") as archive:
        members = archive.getmembers()
    assert [member.name for member in members] == sorted(member.name for member in members)
    assert all(
        member.name == "minimal-synthetic-1.0.0"
        or member.name.startswith("minimal-synthetic-1.0.0/")
        for member in members
    )
    assert all(member.uid == member.gid == member.mtime == 0 for member in members)
    assert all(not member.issym() and not member.islnk() for member in members)


def test_release_processing_never_reads_video_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _copy_sample(tmp_path)
    archive = tmp_path / "streamed.tar.gz"
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path.name == "video.mp4":
            raise AssertionError("video.mp4 must be streamed")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    build_release(dataset, archive)
    verify_release(dataset, archive)


def test_release_rejects_config_changed_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    artifact = dataset / "artifacts" / "config.json"
    artifact.parent.mkdir()
    artifact.write_text('{"version":1}\n')
    source_path = dataset / "clips" / "synthetic-clip" / "source.json"
    source = json.loads(source_path.read_text())
    source["transformations"].append(
        {
            "kind": "fixture",
            "description": "Synthetic fixture configuration.",
            "config_file": "artifacts/config.json",
            "config_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    )
    _write_json(source_path, source)
    _rewrite_reviews(dataset)
    validate_dataset(dataset, require_manifest=False)

    copytree_original = manifest_module.shutil.copytree

    def copy_then_replace_config(source_root: Path, snapshot: Path, *args, **kwargs):
        result = copytree_original(source_root, snapshot, *args, **kwargs)
        replacement = artifact.with_suffix(".replacement")
        replacement.write_text('{"version":2}\n')
        replacement.replace(artifact)
        return result

    monkeypatch.setattr(manifest_module.shutil, "copytree", copy_then_replace_config)
    archive = tmp_path / "release.tar.gz"
    with pytest.raises(DatasetError, match="config_file SHA-256"):
        build_release(dataset, archive)
    assert not archive.exists()


def test_release_rejects_empty_directory_added_after_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    copytree_original = manifest_module.shutil.copytree

    def copy_then_add_directory(source_root: Path, snapshot: Path, *args, **kwargs):
        result = copytree_original(source_root, snapshot, *args, **kwargs)
        if Path(source_root).resolve() == dataset.resolve():
            (dataset / "licenses" / "new-directory").mkdir()
        return result

    monkeypatch.setattr(manifest_module.shutil, "copytree", copy_then_add_directory)
    archive = tmp_path / "release.tar.gz"
    with pytest.raises(DatasetError, match="dataset changed during release build"):
        build_release(dataset, archive)
    assert not archive.exists()


def test_release_rechecks_directories_immediately_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    verify_original = manifest_module.verify_manifest
    mutated = False

    def verify_then_add_directory(root: Path, *args, **kwargs):
        nonlocal mutated
        result = verify_original(root, *args, **kwargs)
        if Path(root).resolve() == dataset.resolve() and not mutated:
            (dataset / "licenses" / "late-directory").mkdir()
            mutated = True
        return result

    monkeypatch.setattr(manifest_module, "verify_manifest", verify_then_add_directory)
    archive = tmp_path / "release.tar.gz"
    with pytest.raises(DatasetError, match="changed after archive publication"):
        build_release(dataset, archive)
    assert archive.exists()


def test_release_binds_snapshot_hashes_through_archive_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    archive = tmp_path / "release.tar.gz"
    write_original = manifest_module._write_archive_stream

    def mutate_snapshot_before_write(root: Path, prefix: str, raw, **kwargs) -> None:
        (root / "clips" / "synthetic-clip" / "video.mp4").write_bytes(b"changed after verify")
        write_original(root, prefix, raw, **kwargs)

    monkeypatch.setattr(manifest_module, "_write_archive_stream", mutate_snapshot_before_write)
    with pytest.raises(DatasetError, match="changed during archive construction"):
        build_release(dataset, archive)
    assert not archive.exists()


def test_release_publication_uses_bound_archive_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    archive = tmp_path / "release.tar.gz"
    publish_original = manifest_module._publish_archive

    def replace_staged_path(stream, output: Path, expected_sha256: str, **kwargs):
        staged_path = Path(stream.name)
        replacement = staged_path.with_suffix(".replacement")
        replacement.write_bytes(b"unvalidated archive")
        replacement.replace(staged_path)
        return publish_original(stream, output, expected_sha256, **kwargs)

    monkeypatch.setattr(manifest_module, "_publish_archive", replace_staged_path)
    result = build_release(dataset, archive)
    assert result["archive_sha256"] != hashlib.sha256(b"unvalidated archive").hexdigest()
    verify_release(dataset, archive)


def test_release_hash_is_bound_during_archive_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    archive = tmp_path / "release.tar.gz"
    write_original = manifest_module._write_archive_stream

    def mutate_after_write(root: Path, prefix: str, raw, **kwargs) -> None:
        write_original(root, prefix, raw, **kwargs)
        raw.destination.seek(0)
        raw.destination.write(b"changed after construction")

    monkeypatch.setattr(manifest_module, "_write_archive_stream", mutate_after_write)
    with pytest.raises(DatasetError, match="staged release archive changed"):
        build_release(dataset, archive)
    assert not archive.exists()


def test_release_rebinds_output_name_after_hashing_published_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    archive = tmp_path / "release.tar.gz"
    displaced = tmp_path / "displaced-release.tar.gz"
    hash_original = manifest_module._stream_sha256

    def hash_then_replace(stream) -> str:
        digest = hash_original(stream)
        archive.rename(displaced)
        archive.write_bytes(b"unrelated replacement")
        return digest

    monkeypatch.setattr(manifest_module, "_stream_sha256", hash_then_replace)
    with pytest.raises(DatasetError, match="release archive changed during publication"):
        build_release(dataset, archive)
    assert archive.read_bytes() == b"unrelated replacement"
    assert displaced.is_file()


def test_release_leaves_changed_publication_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    archive = tmp_path / "release.tar.gz"
    hash_original = manifest_module._stream_sha256
    mutated = False

    def mutate_then_hash(stream) -> str:
        nonlocal mutated
        if not mutated:
            archive.write_bytes(b"changed during publication")
            mutated = True
        return hash_original(stream)

    monkeypatch.setattr(manifest_module, "_stream_sha256", mutate_then_hash)
    with pytest.raises(DatasetError, match="changed during publication"):
        build_release(dataset, archive)
    assert archive.read_bytes() == b"changed during publication"
    assert not list(tmp_path.glob(".release.tar.gz.rejected-*"))


def test_release_detects_same_length_rewrite_after_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    archive = tmp_path / "release.tar.gz"
    hash_original = manifest_module._stream_sha256
    mutated = False

    def hash_then_rewrite(stream) -> str:
        nonlocal mutated
        digest = hash_original(stream)
        if archive.exists() and not mutated:
            body = archive.read_bytes()
            archive.write_bytes(bytes([body[0] ^ 0xFF]) + body[1:])
            mutated = True
        return digest

    monkeypatch.setattr(manifest_module, "_stream_sha256", hash_then_rewrite)
    with pytest.raises(DatasetError, match="changed during publication"):
        build_release(dataset, archive)
    assert archive.exists()
    assert not list(tmp_path.glob(".release.tar.gz.rejected-*"))


def test_failed_release_leaves_archive_in_detached_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    parent = tmp_path / "publish"
    parent.mkdir()
    output = parent / "release.tar.gz"
    moved_parent = tmp_path / "moved-publish"
    publish_original = manifest_module._publish_archive

    def publish_then_move_parent(stream, output: Path, expected_sha256: str, **kwargs):
        published = publish_original(stream, output, expected_sha256, **kwargs)
        parent.rename(moved_parent)
        parent.mkdir()
        return published

    monkeypatch.setattr(manifest_module, "_publish_archive", publish_then_move_parent)
    with pytest.raises(DatasetError, match="output parent changed during publication"):
        build_release(dataset, output)
    assert not output.exists()
    detached_archive = moved_parent / "release.tar.gz"
    with tarfile.open(detached_archive, "r:gz") as release:
        assert "minimal-synthetic-1.0.0/release-manifest.json" in release.getnames()
    assert not list(moved_parent.glob(".release.tar.gz.rejected-*"))


def test_release_revalidates_live_dataset_after_archive_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    archive = tmp_path / "release.tar.gz"
    publish_original = manifest_module._publish_archive

    def publish_then_mutate_dataset(stream, output: Path, expected_sha256: str, **kwargs):
        published = publish_original(stream, output, expected_sha256, **kwargs)
        (dataset / "licenses" / "MIT.txt").write_text("changed after publication\n")
        return published

    monkeypatch.setattr(manifest_module, "_publish_archive", publish_then_mutate_dataset)
    with pytest.raises(DatasetError, match="dataset changed after archive publication"):
        build_release(dataset, archive)
    assert archive.exists()
    assert not list(tmp_path.glob(".release.tar.gz.rejected-*"))


def test_release_reverifies_archive_after_final_dataset_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    archive = tmp_path / "release.tar.gz"
    verify_original = manifest_module.verify_manifest
    mutated = False

    def verify_then_mutate_archive(root: Path, *args, **kwargs):
        nonlocal mutated
        result = verify_original(root, *args, **kwargs)
        if Path(root).resolve() == dataset.resolve() and archive.exists() and not mutated:
            archive.write_bytes(b"changed after dataset validation")
            mutated = True
        return result

    monkeypatch.setattr(manifest_module, "verify_manifest", verify_then_mutate_archive)
    with pytest.raises(DatasetError, match="changed after final dataset validation"):
        build_release(dataset, archive)
    assert archive.read_bytes() == b"changed after dataset validation"
    assert not list(tmp_path.glob(".release.tar.gz.rejected-*"))


def test_release_refuses_existing_output(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    archive = tmp_path / "release.tar.gz"
    archive.write_bytes(b"preserve me")
    with pytest.raises(DatasetError, match="already exists"):
        build_release(dataset, archive)
    assert archive.read_bytes() == b"preserve me"


def test_build_release_rejects_unsupported_platform_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    before = {
        path.relative_to(dataset): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "new-parent" / "release.tar.gz"
    monkeypatch.setattr(
        manifest_module,
        "_directory_anchored_publication_supported",
        lambda: False,
    )

    with pytest.raises(DatasetError, match="unsupported on this platform"):
        build_release(dataset, output)

    after = {
        path.relative_to(dataset): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not output.parent.exists()


def test_build_release_attempts_publication_before_dataset_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _copy_sample(tmp_path)
    before = {
        path.relative_to(dataset): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "release.tar.gz"

    def unsupported_rename(*args, **kwargs) -> None:
        raise DatasetError("atomic no-replace publication is unsupported")

    monkeypatch.setattr(manifest_module, "_rename_no_replace", unsupported_rename)
    with pytest.raises(DatasetError, match="release archive could not be published"):
        build_release(dataset, output)

    after = {
        path.relative_to(dataset): path.read_bytes()
        for path in dataset.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not output.exists()
    staging = list(tmp_path.glob(".release.tar.gz.*"))
    assert len(staging) == 1
    assert staging[0].is_file()


def test_build_release_fails_closed_for_noncertified_state(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    descriptor = yaml.safe_load((dataset / "dataset.yaml").read_text())
    descriptor["state"] = "reviewed"
    descriptor["evaluation_eligible"] = False
    descriptor["certification"].pop("certified_at")
    (dataset / "dataset.yaml").write_text(yaml.safe_dump(descriptor, sort_keys=False))
    with pytest.raises(DatasetError, match="only certified|requires dataset state certified"):
        build_release(dataset, tmp_path / "release.tar.gz")


def test_certified_validation_requires_release_manifest(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    with pytest.raises(DatasetError, match="require release-manifest"):
        validate_dataset(dataset)


def test_clips_root_rejects_undeclared_files(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    (dataset / "clips" / "notes.txt").write_text("not canonical\n")
    with pytest.raises(DatasetError, match="clip directories do not match"):
        validate_dataset(dataset, require_manifest=False)


@pytest.mark.parametrize(
    ("state", "data_role", "annotation_scope"),
    [
        ("certified", "training_only", "exhaustive_visible"),
        ("certified", "benchmark_candidate", "exhaustive_visible"),
        ("certified", "benchmark_truth", "class_exhaustive"),
        ("certified", "benchmark_truth", "sparse"),
        ("certified", "benchmark_truth", "activity_bounded"),
        ("draft", "benchmark_truth", "exhaustive_visible"),
    ],
)
def test_evaluation_eligibility_is_fail_closed(
    tmp_path: Path,
    state: str,
    data_role: str,
    annotation_scope: str,
) -> None:
    dataset = _copy_sample(tmp_path)
    descriptor = yaml.safe_load((dataset / "dataset.yaml").read_text())
    descriptor["state"] = state
    descriptor["data_role"] = data_role
    descriptor["annotation_scope"] = annotation_scope
    descriptor["evaluation_eligible"] = True
    if state != "certified":
        descriptor["certification"].pop("certified_at")
    (dataset / "dataset.yaml").write_text(yaml.safe_dump(descriptor, sort_keys=False))
    with pytest.raises(DatasetError):
        validate_dataset(dataset, require_manifest=False)


def test_artifact_change_makes_reviews_stale(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    tracks = dataset / "clips" / "synthetic-clip" / "tracks.jsonl"
    tracks.write_text(tracks.read_text().replace("[4,2,11,13]", "[5,2,12,13]"))
    with pytest.raises(DatasetError, match="found 0"):
        validate_dataset(dataset, require_manifest=False)


def test_model_generated_labels_require_declared_model_run(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    tracks = dataset / "clips" / "synthetic-clip" / "tracks.jsonl"
    rows = [json.loads(line) for line in tracks.read_text().splitlines()]
    for row in rows:
        row["label_origin"] = {"kind": "model_generated", "model_run_ids": ["scan-1"]}
    tracks.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    with pytest.raises(DatasetError, match="unknown model_run_ids"):
        validate_dataset(dataset, require_manifest=False)


def test_model_generated_labels_are_visible_in_release_manifest(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    clip = dataset / "clips" / "synthetic-clip"
    tracks = clip / "tracks.jsonl"
    rows = [json.loads(line) for line in tracks.read_text().splitlines()]
    for row in rows:
        row["label_origin"] = {"kind": "model_generated", "model_run_ids": ["scan-1"]}
    tracks.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    source_path = clip / "source.json"
    source = json.loads(source_path.read_text())
    source["model_runs"] = [
        {
            "code_revision": "0123456789abcdef",
            "command": ["scanner", "--config", "config.json"],
            "config_sha256": "1" * 64,
            "license": {"spdx": "Apache-2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
            "model_name": "fixture-detector",
            "model_version": "1.0.0",
            "raw_output_sha256": "2" * 64,
            "run_id": "scan-1",
            "weights_sha256": "3" * 64,
            "weights_uri": "https://example.invalid/fixture-detector.bin",
        }
    ]
    _write_json(source_path, source)
    _rewrite_reviews(dataset)

    build_release(dataset, tmp_path / "model-release.tar.gz")
    manifest = json.loads((dataset / "release-manifest.json").read_text())
    assert manifest["clips"][0]["annotation_origins"] == {"model_generated": 2}


def test_current_rejection_blocks_reviewed_or_certified_state(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    review_path = dataset / "clips" / "synthetic-clip" / "review.jsonl"
    rows = [json.loads(line) for line in review_path.read_text().splitlines()]
    rows[0]["decision"] = "reject"
    review_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    with pytest.raises(DatasetError, match="current review rejects"):
        validate_dataset(dataset, require_manifest=False)


def test_unresolved_lfs_pointer_is_rejected(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    video = dataset / "clips" / "synthetic-clip" / "video.mp4"
    video.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "0" * 64 + "\n"
        "size 1108\n"
    )
    with pytest.raises(DatasetError, match="Git LFS pointer"):
        validate_dataset(dataset, require_manifest=False)


def test_archive_tampering_is_rejected(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    archive = tmp_path / "release.tar.gz"
    build_release(dataset, archive)
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(DatasetError, match="not the canonical deterministic archive"):
        verify_release(dataset, archive)


def test_release_output_cannot_be_inside_dataset_root(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    with pytest.raises(DatasetError, match="outside the dataset root"):
        build_release(dataset, dataset / "release.tar.gz")


def test_symlink_is_rejected(tmp_path: Path) -> None:
    dataset = _copy_sample(tmp_path)
    link = dataset / "licenses" / "linked.txt"
    try:
        link.symlink_to(dataset / "licenses" / "MIT.txt")
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(DatasetError, match="cannot contain symlinks"):
        validate_dataset(dataset, require_manifest=False)


def test_studio_contribution_imports_one_validated_clip_into_draft(tmp_path: Path) -> None:
    dataset = _draft_destination(tmp_path)
    contribution = _studio_zip(tmp_path)
    result = import_contribution(dataset, contribution)
    assert result == {
        "annotation_origins": {"human": 2},
        "annotation_rows": 2,
        "clip_path": "clips/imported-clip",
        "dataset": {
            "annotation_scope": "exhaustive_visible",
            "data_role": "benchmark_truth",
            "evaluation_eligible": False,
            "id": "minimal-synthetic",
            "state": "draft",
            "version": "1.0.0",
        },
        "imported_clip": "imported-clip",
        "license_path": "licenses/MIT.txt",
    }
    report = validate_dataset(dataset)
    assert [clip.id for clip in report.clips] == ["imported-clip", "synthetic-clip"]
    assert (dataset / "clips" / "imported-clip" / "review.jsonl").read_bytes() == b""


def test_studio_contribution_refuses_duplicate_clip(tmp_path: Path) -> None:
    dataset = _draft_destination(tmp_path)
    contribution = _studio_zip(tmp_path)
    import_contribution(dataset, contribution)
    with pytest.raises(DatasetError, match="already declares"):
        import_contribution(dataset, contribution)


def test_studio_contribution_refuses_non_draft_destination(tmp_path: Path) -> None:
    dataset = tmp_path / "certified"
    shutil.copytree(SAMPLE, dataset)
    contribution = _studio_zip(tmp_path)
    with pytest.raises(DatasetError, match="only be imported into a draft"):
        import_contribution(dataset, contribution)


def test_studio_contribution_rejects_unsafe_zip_without_mutation(tmp_path: Path) -> None:
    dataset = _draft_destination(tmp_path)
    descriptor_before = (dataset / "dataset.yaml").read_bytes()
    contribution = _studio_zip(tmp_path, unsafe_name="../escape")
    with pytest.raises(DatasetError, match="unsafe ZIP path"):
        import_contribution(dataset, contribution)
    assert (dataset / "dataset.yaml").read_bytes() == descriptor_before
    assert not (dataset / "clips" / "imported-clip").exists()


def test_studio_contribution_cannot_inject_review_approvals(tmp_path: Path) -> None:
    dataset = _draft_destination(tmp_path)
    contribution = _studio_zip(tmp_path, review_body=b"{}\n")
    with pytest.raises(DatasetError, match="empty draft review"):
        import_contribution(dataset, contribution)
    assert not (dataset / "clips" / "imported-clip").exists()
