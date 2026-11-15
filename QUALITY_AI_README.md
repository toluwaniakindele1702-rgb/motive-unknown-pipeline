# Relic Loop Quality AI

Standalone advisory reviewer for completed short-form videos. This component is
being developed independently and is **not connected to either production
pipeline**.

## Scope

- Reviews supplied narration/transcript, scene descriptions, image inspection
  reports, audio observations, duration, and technical metadata.
- Performs deterministic checks for verifiable metadata.
- Optionally calls an injected OpenAI-compatible chat-completions client for
  qualitative review (story hook, coherence, pacing, payoff, originality and
  supplied visual/audio observations).
- Clearly marks missing evidence as `not_assessed`; it never guesses about
  images or audio it has not been shown.
- Emits structured, actionable feedback using the exact fields expected by
  Creative AI's existing `QualityFeedback.from_mapping()` adapter.
- Can run deterministically without credentials or network access.

## Governance boundary

Quality scores and recommendations are advisory. The output explicitly sets
`may_block_production=false`, `may_cancel_or_pause_run=false`, and
`may_authorize_or_trigger_upload=false`. Quality AI has no publishing
credentials, workflow-write permissions, or production-trigger code. A low
score never prevents a review result from being returned.

## Local validation

```bash
python -m unittest -v test_quality_ai.py
python -m py_compile quality_ai.py
```

## Input contract

Pass a JSON object with any available fields:

- `title`, `narration` or `transcript`
- `duration_seconds`
- `scenes` (scene descriptions/storyboard evidence)
- `image_descriptions` (descriptions from a trusted vision review)
- `image_inspections` (technical image-check dictionaries)
- `audio_observations`, `technical_observations`, `human_review_notes`

Raw image/video/audio bytes and credentials are intentionally not accepted by
this first-stage interface. A future vision/audio adapter can be added behind
the same evidence contract.

## Output contract

Top-level scores, per-dimension score/status/evidence/recommendations, evidence
coverage, confidence, uncertainties, and a `creative_ai_feedback` object.
That feedback object is compatible with Creative AI's existing advisory
feedback parser.

## Deferred integration stage

Not included here: pipeline triggers, storage/retention, automated hand-off,
AI-to-AI message transport, video-frame extraction, image/audio model adapters,
and end-to-end production orchestration. Those belong to the separate pipeline
integration stage and must be implemented there, not in this component.
