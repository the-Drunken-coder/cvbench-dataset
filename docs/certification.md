# Certification policy

Certification is an evidence state, not a claim inferred from detector
agreement.

## Required gates

For every clip:

1. Media is present and self-contained.
2. Source and redistribution information is complete.
3. Annotation geometry, ordering, classes, and timestamps validate.
4. Model-derived rows identify reproducible model runs and immutable raw
   outputs.
5. At least the declared number of independent human reviewers approve the
   exact media, truth, and provenance bytes.
6. No current rejection exists.

For the package:

1. Dataset role, annotation scope, and evaluation eligibility are explicit.
2. Every canonical input is hashed in `release-manifest.json`.
3. The manifest records per-clip truth, review, provenance, media, origin
   counts, and approving reviewer IDs.
4. The manifest contains a canonical content hash.
5. The release archive is deterministic and contains one explicit root.
6. Validation rejects any changed, extra, missing, linked, or unresolved file.

Only certified, exhaustive, visible, benchmark-truth packages may set
`evaluation_eligible: true`. Training-only, benchmark-candidate, sparse,
class-exhaustive, and activity-bounded packages remain ineligible even when
their packaging process is certified.

`build-release` refuses `draft` and `reviewed` datasets. It also refuses
certified descriptors whose approvals are stale or insufficient.

## Human review boundary

Models may propose labels. Models do not certify labels.

An annotation marked `model_assisted` or `model_generated` is never hidden:
the release manifest counts it, the source file identifies the exact model
runs, and the review ledger must independently approve the complete resulting
truth artifact.

Agent review entries can be retained as evidence but do not count toward
certification.

## Updating a certified dataset

Any media, truth, provenance, ontology, review, or license change requires a
new semantic version and a rebuilt manifest/archive. A changed clip artifact
invalidates prior approvals automatically because their bound hashes no
longer match.
