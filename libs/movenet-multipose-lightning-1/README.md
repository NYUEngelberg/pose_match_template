# MoveNet MultiPose Lightning v1 (TensorFlow.js graph model)

Local copy of Google's MoveNet MultiPose Lightning model, version 1 — the
same pinned version ml5.js 1.3.1 would otherwise download from
`https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1` at
runtime. Hosting it here lets Pose Match run fully offline.

- Source: https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1
- License: Apache 2.0 (Google)
- Files: `model.json` + 3 weight shards (float16-quantized, ~9MB total)

TF Hub model versions are immutable — this copy never goes stale for the
pinned ml5 version in `libs/ml5.min.js`. If you ever upgrade ml5 and it pins
a different model version, re-download from the version URL it expects.
