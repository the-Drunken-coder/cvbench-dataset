# Recovered clean videos v1

This source recipe describes five clean Pixabay/Pexels videos and 504
confidence-bearing YOLOX-X proposals sampled at 5 FPS. It is public training
material, not benchmark truth:

- `data_role: training_only`
- `evaluation_eligible: false`
- `annotation_scope: sparse`
- every label origin is `model_generated`
- every `review.jsonl` is intentionally empty

Proposal IDs are frame-local detections, not trusted cross-frame identities.
Missing proposals are unknown and must not be interpreted as verified
background. The known visual audit exclusion for a repeated tree-root false
positive is recorded in `artifacts/recovered-training-config.json` and bound to
each model run by SHA-256, but that does not turn the remaining proposals into
human-reviewed labels.

## Obtain and hydrate the media

Download the exact filenames from the source URLs in each `source.json`, then
place only those five MP4s in one directory. The repository does not
redistribute the unchanged stock footage because the current Pixabay and
Pexels terms restrict standalone redistribution.

```sh
cvbench-dataset validate-source-recipe datasets/recovered-clean-videos-v1
cvbench-dataset hydrate-source-recipe datasets/recovered-clean-videos-v1 \
  --source-dir "/path/to/verified/originals" \
  --output .local-datasets/recovered-clean-videos-v1
```

Hydration rejects missing, extra, renamed, or hash-mismatched MP4s. A successful
local package is still a draft training dataset and is rejected by CVBench's
evaluation boundary.
