# Relic Loop — Independent Video Pipeline

Relic Loop is a separate video-production system for Creative AI, Quality AI and
Maintenance AI. It does not import, modify, or invoke the existing
Chispa-Pifia V2 production workflow.

## End-to-end production stages

1. **Creative AI** generates an original 180–215-word story, targeting 50–90
   seconds, and splits it into contiguous visual beats.
2. **Image generation** creates a consistent 2D frame for each beat. Raphael is
   attempted when a valid Playwright session is configured; Cloudflare FLUX.2
   is the fallback. Accepted frames are normalized to 576×1024 portrait PNGs.
   The prior accepted frame is supplied as a continuity reference.
3. **Voice generation** uses Edge TTS to create the narration audio.
4. **Video assembly** uses FFmpeg to create a 30-fps, 576×1024 H.264/AAC MP4.
5. **Quality AI** reviews the completed video, sampled frames and audio where
   Cloudflare vision credentials are available; deterministic and local audio
   checks remain available if media vision is unavailable. Its feedback is
   advisory, never a production gate, and is persisted for the next run.
6. **Maintenance AI** is called on technical failure to diagnose the problem
   and propose a narrowly scoped repair. It does not apply or deploy patches.

## Run locally

Install FFmpeg, then install Python packages:
```bash
python -m pip install 'groq>=0.31,<1' 'requests>=2.32,<3' 'Pillow>=11,<13' 'edge-tts>=7,<8' 'playwright>=1.50,<2'
python -m playwright install chromium
```

Set `GROQ_API_KEY`, `CLOUDFLARE_ACCOUNT_ID` and
`CLOUDFLARE_API_TOKEN`, then run:
```bash
python relic_loop_video.py --topic "A stubborn vending machine"
```

Optional Raphael session: set `RAPHAEL_STORAGE_STATE_JSON` to a valid
Playwright storage-state JSON value. Do not commit or paste this sensitive
session data into source control or chat. If no valid session is configured,
the producer uses Cloudflare directly.

Output is written under `relic_loop_output/<run-id>/` and includes the MP4,
narration audio, individual scene images and `run_report.json`. The latest
Quality AI feedback is stored in `relic_loop_output/latest_quality_feedback.json`
and reused by the next run.

## GitHub Actions

The independent workflow `.github/workflows/relic-loop-video-run.yml` supports
manual dispatch and a one-shot feature-branch trigger. It uploads the MP4 and
report as a GitHub Actions artifact. Required repository secrets are
`GROQ_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, and `CLOUDFLARE_API_TOKEN`.
`RAPHAEL_STORAGE_STATE_JSON` is optional.

## Isolation controls

- No YouTube OAuth credentials, API client, upload, publishing or scheduling.
- No calls to the legacy production pipeline or its renderers.
- The workflow runs only on the isolated feature branch and has read-only
  repository permissions.
- No generated video is published; users can add YouTube authentication later.
- Quality AI cannot block, cancel or pause production.
- Maintenance AI proposals are not automatically applied or deployed.
- Validation is offline and never publishes videos.
