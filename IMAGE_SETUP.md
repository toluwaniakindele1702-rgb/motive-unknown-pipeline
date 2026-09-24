# Image generation setup

Motive Unknown uses a multi-provider image-generation chain:

1. Cloudflare Workers AI — FLUX.2 [klein] 4B
2. Hugging Face Inference Providers — FLUX.1-schnell
3. Replicate — FLUX 1.1 [pro] as an emergency fallback
4. Local CPU/OpenVINO — LCM Dreamshaper INT8 as the no-API fallback

The pipeline tries providers in the order configured by the IMAGE_PROVIDERS GitHub variable. When a provider returns a quota/authentication failure, it is disabled for the rest of that run and the next provider takes over. A normal transient error only falls through for the current image.

## Required GitHub Actions secrets

Main pipeline:
- CLOUDFLARE_ACCOUNT_ID
- CLOUDFLARE_API_TOKEN

Optional API fallbacks:
- HUGGINGFACE_TOKEN
- REPLICATE_API_TOKEN

Local CPU fallback:
- No API token required.
- The `local_image_test` workflow mode uses OpenVINO's INT8 LCM Dreamshaper model on the GitHub runner CPU.
- The production fallback creates an isolated Python environment on first use, so the local image stack does not change the main Kokoro/LLM dependency set.
- The local worker compacts the visual prompt to stay within the model's CLIP context window.

The workflow passes these as secrets; do not place tokens in source files.
The local CPU path does not require a provider token.

## Provider details

Cloudflare is the primary provider and currently uses FLUX.2 [klein] 4B.

Hugging Face provides text-to-image through its Inference Providers API. Free users currently receive a small monthly credit allowance, so this is intended as a backup rather than an unlimited second pool.

Replicate's current Try for Free collection includes FLUX 1.1 [pro] for a limited number of free runs. After the free allowance is exhausted, Replicate requires paid credits for continued API use.

Because provider limits and free allowances can change, the fallback chain is designed to skip a provider that is unavailable rather than assume every provider is permanently free.

## Visual behavior

- 1024x576 source illustrations, scaled to 1280x720 during video assembly.
- Up to 2 visual beats per narration scene.
- About one new illustration per 60 narration words.
- Maximum 36 generated visual beats per episode.
- A persistent style reference is used for Cloudflare generations.
- A previous-frame reference is used for Cloudflare when recurring characters continue.
- Fallback providers preserve the historical art direction, but text-to-image fallbacks do not receive the local reference image directly.
- The local CPU fallback uses a compact prompt because the model's CLIP text encoder has a short context window.

The generated images are instructed to avoid captions, subtitles, logos, watermarks, stick figures, primitive doodles, modern infrastructure, anachronistic objects, and photorealistic/3D rendering.

Cloudflare documentation:
https://developers.cloudflare.com/workers-ai/get-started/rest-api/
https://developers.cloudflare.com/workers-ai/models/flux-2-klein-4b/

Hugging Face text-to-image documentation:
https://huggingface.co/docs/inference-providers/tasks/text-to-image

Replicate FLUX 1.1 [pro] documentation:
https://replicate.com/black-forest-labs/flux-1.1-pro