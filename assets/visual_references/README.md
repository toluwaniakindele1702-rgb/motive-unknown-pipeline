# Chispa-Pifia Visual Bible

## Canonical reference files
The approved visual references currently used by production are:

- `Generate_modern_stickman_character_20261027121745.jpg` — primary character identity reference.
- `Remove_bag_and_animate_image_20261027125028.jpg` — secondary character/style/action reference.

The production renderer sends both approved references to Cloudflare FLUX.2 [dev] on every image-generation request, together with the previously accepted beat when available for continuity.

## Character identity lock
- Follow the approved references as the highest-priority character authority.
- Preserve the same simple, expressive 2D stickman construction, head shape, eyes, eyebrows, mouth, body proportions, limb construction and outline language.
- Clothing is not a defining requirement. Do not add the previous hoodie/cargo-trousers/sneakers identity unless the story itself requires an item.
- Keep the character unmistakably 2D and hand-drawn/digitally illustrated rather than photorealistic, 3D, anime or a detailed human cartoon.
- Expressions, poses and camera angles may change with the story; character identity must not.

## Scene rendering
- One coherent vertical 9:16 story frame per beat.
- Never reproduce the reference sheet as a collage, turnaround, contact sheet or model sheet during production.
- The current narration beat is the only event to illustrate.
- Backgrounds should support the story and remain visually compatible with the approved references.
- No captions, subtitles, speech bubbles, labels, logos, watermarks, UI, readable signs, letters, numbers or typography.

## Story relationship
The story generator remains responsible for mini-mysteries, suspense, dilemmas, problem-solving and fair twists. Absurd comedy is optional, not the default. The image renderer must literally reenact the narration beat and must not invent unrelated visual jokes.

## Production model policy
- Story writer: `openai/gpt-oss-120b` on Groq.
- Independent story reviewer: `openai/gpt-oss-120b` on Groq.
- Image generation: `@cf/black-forest-labs/flux-2-dev` on Cloudflare Workers AI.
- Text detection: Tesseract with regeneration retries.

Reference images improve consistency but cannot mathematically guarantee identical characters. The pipeline therefore also supplies the previous accepted frame as a continuity reference for each subsequent scene.
