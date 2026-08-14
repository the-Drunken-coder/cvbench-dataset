# Recovered clean videos v1

This source recipe describes five clean Pixabay/Pexels videos and 3,411
confidence-bearing instance masks across eight tracks. YOLO26x-seg processed
every native video frame in class-isolated person and dog passes, with
TrackTrack and a pinned YOLO26n appearance encoder providing cross-frame
association. It is public training material, not benchmark truth:

- `data_role: training_only`
- `evaluation_eligible: false`
- `annotation_scope: sparse`
- every label origin is `model_generated`
- every `review.jsonl` is intentionally empty

Track IDs are model-generated continuity candidates pending human review.
Missing masks are unknown and must not be interpreted as verified background,
even though inference visited every target-class/frame combination. Detector,
ReID, tracker, and threshold settings are hash-bound through
`artifacts/yolo26x-dense-tracking.json`; both weight files and the generator
revision are pinned in provenance. None of that turns the output into
human-reviewed labels.

## Obtain and hydrate the media

Download the exact filenames from the source URLs in each `source.json`, then
place only those five MP4s in one directory. The repository does not
redistribute the unchanged stock footage because the current Pixabay and
Pexels terms restrict standalone redistribution.

```sh
cvbench-dataset validate-source-recipe datasets/recovered-clean-videos-v1
mkdir -p .local-datasets
cvbench-dataset hydrate-source-recipe datasets/recovered-clean-videos-v1 \
  --source-dir "/path/to/verified/originals" \
  --output .local-datasets/recovered-clean-videos-v1
```

Hydration rejects missing, extra, renamed, or hash-mismatched MP4s. A successful
local package is still a draft training dataset and is rejected by CVBench's
evaluation boundary.
