# Benchmark integration

Dataset releases are external, immutable inputs to a benchmark repository.
The benchmark should commit a small lock file containing:

- dataset ID and version;
- archive URL;
- archive byte length and SHA-256;
- release-manifest SHA-256;
- expected top-level archive root.
- locked `data_role`, `annotation_scope`, and `evaluation_eligible` values.

The installer downloads the archive, verifies its size and SHA-256 before
extraction, rejects links and path traversal, extracts into an ignored
`data/datasets/<dataset-id>` directory, and runs:

```sh
cvbench-dataset verify-release data/datasets/<dataset-id> \
  --archive path/to/<dataset-id>-<version>.tar.gz
```

The deterministic archive layout is:

```text
<dataset-id>-<version>/
├── dataset.yaml
├── release-manifest.json
├── schemas/
├── licenses/
└── clips/
    └── <clip-id>/
        ├── video.mp4
        ├── tracks.jsonl
        ├── source.json
        └── review.jsonl
```

Benchmark importers consume only clip roots explicitly declared by
`dataset.yaml`. They may transform `tracks.jsonl` into a benchmark-specific
runtime representation, but must bind the source release and transformed
truth hashes in benchmark provenance.

An evaluation importer must additionally require `state: certified`,
`data_role: benchmark_truth`, `annotation_scope: exhaustive_visible`, and
`evaluation_eligible: true`. Training, sparse, class-limited,
activity-bounded, or candidate packages must never be promoted implicitly.

Compatibility assets under this repository's `legacy/` directory are never
included in a release archive and must not be accepted by the installer.
