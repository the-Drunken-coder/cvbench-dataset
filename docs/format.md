# Canonical dataset format

## Dataset descriptor

`dataset.yaml` declares the dataset ID, semantic version, lifecycle state,
data role, annotation scope, evaluation eligibility, closed class ontology,
explicit clip roots, and certification policy. Clip paths are always
`clips/<clip-id>`; implicit discovery is forbidden.

`data_role` is one of `training_only`, `benchmark_candidate`, or
`benchmark_truth`. `annotation_scope` is one of `exhaustive_visible`,
`class_exhaustive`, `sparse`, or `activity_bounded`.
`evaluation_eligible: true` requires all of:

- lifecycle state `certified`;
- data role `benchmark_truth`;
- annotation scope `exhaustive_visible`.

Every other combination is excluded from benchmark evaluation.

Start a new package with:

```sh
cvbench-dataset init <root> \
  --id <dataset-id> \
  --title <title> \
  --description <description> \
  --class <id=description>
```

The target must not exist. The command creates a valid empty draft with
canonical schemas and `evaluation_eligible: false`, ready for
`import-contribution`.

The lifecycle states are:

| State | Meaning |
| --- | --- |
| `draft` | Work in progress. Reviews and a release manifest are optional. |
| `reviewed` | Every clip has at least one current independent human approval. |
| `certified` | Every clip meets the declared approval threshold, which is at least two, and has a verified release manifest. |

## Clip artifacts

Every declared clip contains exactly:

- `video.mp4`: a self-contained MP4, normally stored through Git LFS;
- `tracks.jsonl`: sorted frame/track annotations;
- `source.json`: source, license, transform, media, and model-run provenance;
- `review.jsonl`: an append-only review ledger.

Each annotation declares `label_origin.kind` as `human`, `upstream`,
`model_assisted`, or `model_generated`. Model-derived annotations must name
one or more `source.json.model_runs`. Each model run records its exact version,
weight URI and hash, code revision, configuration hash, raw-output hash,
command, and license.

Boxes use `[x_min, y_min, x_max, y_max]` in source pixels. Rows are uniquely
sorted by `(frame_index, track_id)`. Frame timestamps must be consistent and
strictly increase.

## Reviews

A review event binds its decision to the SHA-256 of `video.mp4`,
`tracks.jsonl`, and `source.json`. Historical events may remain in the ledger,
but only approvals matching all three current hashes count.

A current rejection blocks `reviewed` and `certified`. Certification counts
only distinct reviewers with `kind: human` and `independent: true`.

## Schemas and licenses

Every package carries the exact schemas used by the validator. Schema drift is
rejected. Each source identifies an SPDX license, authoritative URL, and a
license file below `licenses/`.

Package roots reject symlinks, special files, undeclared top-level entries,
undeclared clip files, path traversal, and unresolved Git LFS pointers.

## Studio contributions

`import-contribution` accepts one ZIP with this exact layout:

```text
contribution.json
clips/<clip-id>/
├── video.mp4
├── tracks.jsonl
├── source.json
└── review.jsonl
licenses/<referenced-license>
```

`review.jsonl` must be empty. The importer rejects unsafe paths, links,
encryption, undeclared files, duplicate clips, non-draft destinations, stale
LFS media, invalid truth, missing model provenance, and mismatched licenses
before modifying the destination.
