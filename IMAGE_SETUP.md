# Polished image generation setup

Motive Unknown now uses Cloudflare Workers AI with FLUX.2 [klein] 4B for scene illustrations.

## Required GitHub Actions secrets

Add these repository secrets:

- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`

Cloudflare's current Workers AI REST setup is documented at:
https://developers.cloudflare.com/workers-ai/get-started/rest-api/

For a custom API token, Cloudflare says the token needs Workers AI Read and Workers AI Edit permissions.

The repository can remain private.

## GitHub

Repository Settings -> Secrets and variables -> Actions -> New repository secret.

## Visual behavior

- 1024x576 source illustrations, scaled to 1280x720 during video assembly.
- 1-4 visual beats per narration scene.
- About one new illustration per 33 narration words.
- Maximum 80 illustrations per episode to stay within the current free-tier budget target.
- A persistent style reference is used on every generation.
- A previous-frame reference is also used when the next beat shares recurring characters.

The generated images are instructed to avoid captions, subtitles, logos, watermarks, stick figures, primitive doodles, and photorealistic/3D rendering.

Cloudflare currently documents FLUX.2 [klein] 4B as a 4-step image model with support for up to four reference images.
