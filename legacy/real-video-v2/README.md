# Quarantined real-video-v2 compatibility evidence

This directory is deliberately outside every canonical `dataset.yaml` and
release manifest.

The files under `snapshot/` preserve CVBench's former `real-video-v2`
scenario package for historical report reproduction and compatibility
investigation. Its activity-bounded labels are **not exhaustive full-frame
truth**, the package has not passed the certification policy in this
repository, and it must not be described or published as a certified dataset.

`SNAPSHOT.sha256` records the preserved bytes. New datasets must use the
canonical package format documented in this repository and pass
`cvbench-dataset build-release`.
