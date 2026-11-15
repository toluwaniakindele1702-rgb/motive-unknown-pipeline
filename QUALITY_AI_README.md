# Relic Loop Quality AI

Standalone advisory reviewer for completed short-form videos. This component is developed independently and is not connected to either production pipeline.

## Capabilities

- Deterministic checks for narration length and runtime.
- Optional Groq text evaluation of supplied story and scene evidence.
- Direct inspection of actual PNG, JPEG and WebP image files through Gemini multimodal.
- Local extraction of up to eight video frames with ffmpeg, followed by direct visual inspection for character, wardrobe, prop, location and composition continuity.
- Direct audio inspection through Gemini multimodal for intelligibility, noise, clipping, volume consistency, pacing, pauses, pronunciation and completeness.
- Automatic extraction of mono 16 kHz audio from a supplied video when no separate audio file is supplied.
- Structured evidence and advisory feedback matching Creative AI's feedback contract.
- Explicit uncertainty reporting; no fabricated assessment when media is unavailable.

## Local validation

```bash
python -m py_compile quality_ai.py quality_ai_media.py
python -m unittest -v test_quality_ai.py test_quality_ai_media.py
```

## Provider configuration

Text evaluation optionally uses `GROQ_API_KEY` and `QUALITY_AI_MODEL`.
Direct media inspection uses `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) and optional `QUALITY_AI_MULTIMODAL_MODEL` (default `gemini-2.5-flash`). Without a Gemini key, the media adapter is unavailable and must not be treated as having inspected media. API keys are read only from environment variables and never included in reports.

Video processing requires ffmpeg. Up to eight frames are extracted locally; if video audio is required, it is extracted locally to a temporary mono 16 kHz MP3. Media files sent to Gemini are limited to 18 MiB each. Temporary video artifacts are cleaned up after review. The provider receives the actual selected frames/audio for analysis.

## Media review usage

Use `build_gemini_client()` then call `review_media_submission(payload, client=client, model=model, video_path=..., image_paths=..., audio_path=...)`. It returns a `review` object plus the raw `media_evidence` used to create it. The same function can inspect standalone images and audio. This is a pipeline-neutral adapter: the future pipeline can call it without changing the review contract.

## Governance boundary

All scores and recommendations are advisory. Quality AI cannot approve, reject, cancel, pause, block, or authorize production or publishing. It has no YouTube credentials, GitHub write permissions, workflow triggers, or deployment logic. A low score never prevents a review result from being returned.

## Remaining separate-pipeline work

The Quality AI review engine and media adapters are built independently. Still deferred to the separate pipeline: secure media hand-off and retention policy, scheduled review orchestration, AI-to-AI message transport, integration tests against the actual Creative AI outputs, and end-to-end orchestration. Do not connect this component to the existing production pipeline.
