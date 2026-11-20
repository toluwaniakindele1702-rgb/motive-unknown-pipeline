# Relic Loop — Independent Video Pipeline

Relic Loop is a separate production pipeline for Creative AI, Story Quality AI,
visual generation, Quality AI and Maintenance AI. It does not invoke the legacy
Chispa-Pifia V2 production workflow.

## Story and production flow

1. **Creative AI** writes an original 180–215-word narration targeting 50–90 seconds.
   It rotates through curiosity-gap, in-medias-res, mystery, emotional-contradiction
   and unexpected-reversal hooks; plants an early question/detail; adds a genuine
   mid-story rehook; and delivers an earned payoff.
2. **Story Quality AI** scores open loops, rehooks, voiceover naturalness, clarity,
   escalation, originality, pacing, payoff and rewatchability. It reviews two
   candidates, selects the stronger one, then coaches up to two rewrites. Review
   remains advisory; production is not blocked by low scores or review outages.
3. **Visual generation** makes ten consistent 2D frames. Raphael is attempted
   with a valid session; Cloudflare is fallback. Images are portrait 576x1024.
4. **Voice** uses Edge TTS (Andrew Neural by default, Guy Neural fallback).
   **Assembly** uses local FFmpeg for 30fps H.264/AAC, with five alternating
   camera-movement presets and subtle synthesized story-aware sound cues. The
   camera motion does not depend on a third-party camera website or browser UI.
5. **Quality AI** reviews the completed video, frames and audio when evidence is
   available; feedback remains advisory and is persisted for future stories.
6. **Maintenance AI** diagnoses technical failures and proposes scoped repairs;
   it does not deploy or auto-apply patches.

## Workflow and credentials

The workflow supports manual runs and daily scheduled runs after merge. It uploads
the MP4 and report as an artifact, then attempts a **public YouTube upload**.
YouTube OAuth secrets required: `YOUTUBE_TOKEN_JSON` and
`YOUTUBE_CLIENT_SECRET_JSON`. Production secrets required:
`GROQ_API_KEY`, `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`.
Raphael session is optional.

The old `chispa_pifia_v2.yml` workflow is disabled only after the public upload
succeeds. If production or upload fails, the old workflow stays active. The
deactivation action requires the workflow token's Actions write permission.

## Local run

Install FFmpeg and dependencies:
```bash
python -m pip install 'groq>=0.31,<1' 'requests>=2.32,<3' 'Pillow>=11,<13' 'edge-tts>=7,<8'
```
Set production secrets locally and run:
```bash
python relic_loop_video.py --topic "A stubborn vending machine"
```
Output is written under `relic_loop_output/<run-id>/`. Do not put OAuth or
browser-session JSON in source control or chat.


## Enjoyability upgrades (feature branch)

- **Camera language:** every still receives a locally rendered, alternating push, pull or pan; the pipeline no longer relies on the unstable camera-effect website.
- **Sound design:** original, locally synthesized music is selected from a broad story mood (playful, mysterious, warm or emotional), kept under narration, and mixed with sparse story-aware accents. No stock-music API or external sound download is required.
- **Captions:** sentence-aware scene captions are timed against the narration scene allocation and burned into the vertical video with a readable outline and safe lower-screen placement.
- **Story craft:** rotating hook approaches, an early curiosity loop, a mid-story rehook and an earned payoff are evaluated before image generation. Two candidate scripts are compared and bounded rewrite coaching is applied.
- **Visual continuity:** one canonical character description and previous-frame reference are carried through scene generation; wide, medium and close framing directions vary by beat.
- **Media review:** Quality AI inspects sampled frames and audio when the vision service is available; local FFmpeg audio diagnostics remain available as a fallback. AI feedback is advisory, not a claim that artistic quality can be proven automatically.
- **Offline CI:** `.github/workflows/relic-loop-quality-ci.yml` compiles the pipeline and runs local media/story smoke tests on feature-branch changes without generating or publishing a video.

A real end-to-end production video still needs human review for voice performance, caption readability, music/SFX balance, image consistency and whether the story is genuinely entertaining. Metrics such as viewed-vs-swiped-away and audience retention require YouTube Analytics API authorization; do not infer them from ordinary public view counts or claim the feedback loop is active until that integration is implemented and verified.
