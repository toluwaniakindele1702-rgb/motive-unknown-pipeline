# Relic Loop Quality AI

Standalone advisory reviewer, developed independently and not connected to either production pipeline.

## Capabilities
- Deterministic narration-length and runtime checks.
- Optional Groq text evaluation.
- Actual image and sampled video-frame inspection through Cloudflare Workers AI vision (default model: @cf/meta/llama-3.2-11b-vision-instruct).
- Local FFmpeg audio diagnostics: duration, mean/peak volume and extended silence.
- Explicitly marks speech intelligibility, pronunciation and semantic completeness as not assessed by signal metrics alone.
- Structured advisory feedback matching Creative AI's feedback contract.

## Validation
Run: python -m py_compile quality_ai.py quality_ai_media.py
Then: python -m unittest -v test_quality_ai.py test_quality_ai_media.py

## Provider configuration
Text review optionally uses GROQ_API_KEY and QUALITY_AI_MODEL.
Visual inspection uses the existing Cloudflare account:
- CLOUDFLARE_ACCOUNT_ID
- CLOUDFLARE_API_TOKEN or CLOUDFLARE_AUTH_TOKEN
- Optional QUALITY_AI_VISION_MODEL

Workers AI must be enabled. Meta's Llama Vision model requires one-time license acceptance; if not yet accepted, the provider returns an actionable error. No Gemini key, Google account or Gemini dependency is used. Video frame extraction and audio signal analysis require ffmpeg/ffprobe. Raw audio is not sent to a new provider.

## Usage
Call review_media_submission(payload, video_path=..., image_paths=..., audio_path=...). Cloudflare credentials are read from the environment for visual inspection. Audio-only review requires no Cloudflare credential.

## Governance
Scores and recommendations are advisory. Quality AI cannot approve, reject, cancel, pause, block, or authorize production or publishing. It has no YouTube credentials, GitHub write permissions, workflow triggers or deployment logic.

## Deferred separate-pipeline work
Secure media hand-off and retention, scheduled review orchestration, AI-to-AI message transport, tests against actual Creative AI outputs and end-to-end orchestration remain deferred. Do not connect this component to the existing production pipeline.
