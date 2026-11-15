# Relic Loop — Separate AI Pipeline

This is a new, isolated orchestration layer for Creative AI, Quality AI and
Maintenance AI. It is developed on its own feature branch and is not connected
to the existing Chispa-Pifia V2 production workflow.

## Connected responsibilities

1. **Creative AI** creates a validated 180–215-word story for a 50–90-second
   target. Previous Quality AI feedback is passed into the next creative request.
2. **Quality AI** reviews the story and any media supplied to the runner. Its
   feedback is advisory and is returned as `next_creative_feedback`. A low score
   never cancels, blocks or pauses the run.
3. **Maintenance AI** is called when a technical stage fails, when a model client
   is supplied. It can diagnose and propose a minimal patch in its allowlisted
   scope. This orchestrator never applies or deploys that patch.

## Run locally

Install the component dependencies, set `GROQ_API_KEY` and
`CREATIVE_AI_MODEL`, then run:

```bash
python relic_loop_pipeline.py --input request.json --output result.json
```

Example request:

```json
{
  "topic": "A stubborn vending machine",
  "previous_quality_feedback": {
    "strengths": ["Clear opening visual"],
    "improvements": ["Make the final callback more specific"]
  }
}
```

Optional media evidence may be supplied with `image_paths`, `audio_path` or
`video_path`. Actual image/video-frame inspection uses Cloudflare Workers AI
and requires `CLOUDFLARE_ACCOUNT_ID` plus `CLOUDFLARE_API_TOKEN` (or
`CLOUDFLARE_AUTH_TOKEN`). Audio diagnostics use local FFmpeg/ffprobe.

## Explicitly not yet wired

The provider-specific Raphael/Cloudflare image-render adapter and the new
pipeline's own video assembler are not yet implemented in this orchestration
branch. The runner exposes injected `image_renderer` and `video_renderer`
adapter interfaces and can review media supplied by a completed adapter. The
old production renderers are intentionally not imported or reused. This means
the story/review/feedback/maintenance orchestration is connected, while
standalone media generation and assembly remain the next integration milestone.

## Isolation and safety controls

- No import of the existing production workflow or upload modules.
- No YouTube credentials, upload code, publication or scheduling.
- No scheduled or push-triggered execution; validation is pull-request or manual
  dispatch only.
- GitHub Actions permissions are read-only.
- Maintenance patches are proposals only; no checkout mutation, Git push, merge
  or deployment is performed.
- Validation uses offline tests and does not call paid model APIs or publish
  videos.
- Existing production workflows and credentials are left unchanged.
