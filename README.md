# cvbench-dataset

`cvbench-dataset` defines the canonical, independently reviewable dataset
package consumed by CVBench benchmark releases. It keeps media, truth,
provenance, reviews, licenses, and release hashes together without treating
model output as truth by default.

The release path is intentionally fail-closed:

- `draft` accepts work in progress;
- `reviewed` requires one independent human approval bound to the exact clip
  artifacts;
- `certified` requires at least two independent human approvals for every
  clip, a release manifest, and a deterministic archive;
- any media, truth, or provenance change makes existing approvals stale;
- model-assisted and model-generated rows identify fully specified model runs
  and remain visible in the release manifest.

Dataset purpose is explicit. `data_role` distinguishes training material,
benchmark candidates, and benchmark truth. `evaluation_eligible: true` is
accepted only for certified `benchmark_truth` with `exhaustive_visible`
annotations. Sparse, class-limited, activity-bounded, training, draft, and
candidate packages fail closed for benchmark evaluation.

## Canonical package

```text
dataset-root/
├── dataset.yaml
├── clips/
│   └── <clip-id>/
│       ├── video.mp4
│       ├── tracks.jsonl
│       ├── source.json
│       └── review.jsonl
├── licenses/
├── schemas/
└── release-manifest.json  # certified releases only
```

Install and validate:

```sh
python -m pip install -e '.[dev]'
cvbench-dataset validate examples/minimal-certified
```

Create a new draft package:

```sh
cvbench-dataset init datasets/my-sports-dataset \
  --id my-sports-dataset \
  --title "My sports dataset" \
  --description "Contributor-owned draft clips." \
  --class person="Visible participant" \
  --class ball="Visible game ball"
```

`init` accepts only a target that does not already exist. It creates empty
`clips/` and `licenses/` directories, copies the canonical schemas, and writes
a validated `draft` descriptor with `evaluation_eligible: false`. Use
`--data-role` and `--annotation-scope` to declare intent; neither option
promotes the draft to evaluation truth.

Build and verify a deterministic release:

```sh
cvbench-dataset build-release path/to/dataset \
  --output dist/<dataset-id>-<version>.tar.gz
cvbench-dataset verify-release path/to/dataset \
  --archive dist/<dataset-id>-<version>.tar.gz
```

Import one fail-closed Studio contribution into a draft dataset:

```sh
cvbench-dataset import-contribution path/to/draft-dataset studio-export.zip
```

The ZIP must contain one explicit `clips/<clip-id>` package, its referenced
license, and `contribution.json`. Reviews must be empty; imported labels remain
draft until independent review occurs in the dataset repository.

The archive contains one explicit `<dataset-id>-<version>/` root. Consumers
lock its SHA-256, extract it into an ignored data directory, and verify the
manifest before importing clip roots.

See [the format specification](docs/format.md),
[certification policy](docs/certification.md), and
[benchmark integration contract](docs/benchmark-integration.md).

The files under `legacy/` are quarantined compatibility evidence. They are
not canonical dataset packages and are excluded from releases.
