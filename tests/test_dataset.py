from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

import cvbench_dataset.manifest as manifest_module
import cvbench_dataset.source_recipe as source_recipe_module
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

    def validate_then_create_competing_output(root: Path):
        result = validate_dataset_original(root)
        output.mkdir()
        (output / "owner.txt").write_text("preserve me\n")
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
