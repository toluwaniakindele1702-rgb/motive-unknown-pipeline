"""Chispa-Pifia image renderer configured for the approved Peanut-style visual bible.

This wrapper deliberately reuses the tested Cloudflare renderer so image transport,
OCR, retries, previous-frame continuity, manifests and beat handling remain unchanged.
Only the model, canonical references, quota fallback and visual prompt policy are overridden here.
"""
from __future__ import annotations

import os

# Keep FLUX.2 [dev] as the preferred high-fidelity multi-reference model.
# If the account exhausts the daily free Workers AI neuron allocation, the wrapper
# automatically falls back to the much cheaper FLUX.2 Klein 4B for the rest of the run.
os.environ["CLOUDFLARE_IMAGE_MODEL"] = "@cf/black-forest-labs/flux-2-dev"

import cloudflare_relic_loop_images as base

base.MODEL = "@cf/black-forest-labs/flux-2-dev"
base.CANONICAL_REFERENCE_PATHS = (
    ("Generate_modern_stickman_character_20261027121745.jpg", "image/jpeg"),
    ("Remove_bag_and_animate_image_20261027125028.jpg", "image/jpeg"),
)

# Cloudflare error code 3036 / HTTP 429 means the account has exhausted the
# 10,000-neuron daily free allocation. This is a quota condition, not a bad token.
# Klein 4B is substantially cheaper and supports the same multi-reference workflow,
# so it is a safe no-paid-plan fallback for the remainder of the current run.
PREFERRED_MODEL = "@cf/black-forest-labs/flux-2-dev"
FALLBACK_MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
_quota_fallback_activated = False
_original_cloudflare = base._cloudflare


def _cloudflare_with_quota_fallback(prompt, refs=None):
    global _quota_fallback_activated

    # Once the daily quota has been detected, do not keep hitting the exhausted
    # preferred model on every subsequent image. Use the cheaper fallback directly.
    if _quota_fallback_activated:
        base.MODEL = FALLBACK_MODEL
        return _original_cloudflare(prompt, refs)

    try:
        return _original_cloudflare(prompt, refs)
    except RuntimeError as exc:
        message = str(exc)
        quota_exhausted = (
            "3036" in message
            or "used up your daily free allocation of 10,000 neurons" in message.lower()
            or "daily free allocation" in message.lower()
        )
        if not quota_exhausted:
            raise

        _quota_fallback_activated = True
        base.MODEL = FALLBACK_MODEL
        os.environ["CLOUDFLARE_IMAGE_MODEL"] = FALLBACK_MODEL
        print(
            "[CLOUDFLARE] Daily free neuron allocation exhausted on FLUX.2 dev; "
            f"switching to cheaper fallback model {FALLBACK_MODEL} for the remainder of this run."
        )
        return _original_cloudflare(prompt, refs)


base._cloudflare = _cloudflare_with_quota_fallback

base.STYLE_LOCK = """
CHISPA-PIFIA MASTER VISUAL STYLE — APPROVED CHARACTER BIBLE:
Use the supplied canonical reference images as the highest-priority visual authority.
The protagonist must look like the ORIGINAL character established by those references:
a very simple, expressive 2D storytime stickman/cartoon figure with a large smooth pale
round head, simple minimal body construction, thin clean limbs, large expressive eyes,
clear eyebrows and exaggerated mouth shapes. Preserve the exact recognizable character
proportions, face construction, silhouette, line quality and simplified anatomy shown in
the references. Clothing is NOT a defining requirement. Do not invent a hoodie, cargo
trousers, sneakers, backpack or other outfit merely because an older pipeline used them.
If the approved reference shows a minimally dressed or unclothed stickman, preserve that
minimal design.

STYLE TARGET:
Clean hand-drawn/digital 2D animation with bold readable outlines, simple shapes, strong
facial acting, exaggerated poses and expressive body language. The character must remain
flat/2D and cartoon-like rather than becoming a realistic person, 3D mascot, glossy CGI
figure, anime character or detailed human cartoon. Keep the character visually simple enough
that the same design can be reproduced across an entire Shorts channel.

SCENE ART:
Use story-appropriate illustrated environments that support the action without changing the
character's identity. Background detail may vary with the story, but must remain compatible
with the approved reference aesthetic. Use clear silhouettes, readable composition and
cinematic framing where useful. Do not turn every scene into a character model sheet.

CHARACTER CONTINUITY:
Reference 1 is the primary character identity anchor. Reference 2 is a secondary approved
style/action/character reference. Match the same head, eyes, brows, mouth, body proportions,
limb construction, outline language and overall 2D rendering in every shot. Expressions,
poses, camera angle and environment may change because the story changes. The character
identity must not.

ONE-SCENE RULE:
Generate exactly ONE coherent vertical story frame per request. Never reproduce the reference
sheet layout. Never create a collage, turnaround sheet, contact sheet or multiple panels.
The current narration beat is the only event to illustrate.

ABSOLUTE EXCLUSIONS:
no photorealistic humans, no live-action look, no 3D CGI, no glossy mascot, no anime/manga,
no realistic human anatomy, no old hoodie/cargo/sneaker identity lock, no random clothing,
no character redesign, no extra limbs/fingers, no duplicated protagonist unless the story
explicitly requires another person, no copied reference-sheet panels, no text, no captions,
no speech bubbles, no labels, no logos and no watermarks.
""".strip()

base.NO_TEXT = """
STRICT ZERO-TEXT IMAGE RULE:
Do not render captions, subtitles, title cards, speech bubbles, labels, logos, watermarks,
UI, readable signs, letters, numbers, symbols, menus, posters, packaging text, screen text
or typography. Keep environmental surfaces visually clean and non-readable. Never use text
as a storytelling shortcut. The story must be communicated through the character's acting,
objects, composition, expressions and visible cause-and-effect.
""".strip()


def _reference_blurb() -> str:
    return (
        "CANONICAL REFERENCE PRIORITY: Reference 1 is the primary approved character model. "
        "Reference 2 is the secondary approved character/style/action guide. Match their original "
        "simple 2D stickman construction, head shape, facial design, eye treatment, limb style, "
        "proportions, outline language and expressive acting. Do NOT import clothing or visual "
        "features from previous pipeline versions. Do not copy the reference-sheet composition. "
        "Render one single vertical story scene. The approved references define WHAT THE CHARACTER "
        "LOOKS LIKE; the narration defines WHAT THE CHARACTER IS DOING."
    )

base._reference_blurb = _reference_blurb

# The story generator still contains legacy wardrobe wording for its own character metadata.
# Strip that visual baggage before the image prompt is built so the approved reference images,
# rather than old prose, remain the authority for character appearance.
_original_scene_prompt = base._scene_prompt


def _scene_prompt(scene, characters, locations, previous_scene=None, attempt=1):
    sanitized = []
    for character in characters or []:
        item = dict(character) if isinstance(character, dict) else {}
        item["appearance"] = (
            "the approved canonical Chispa-Pifia 2D stickman character from reference image 1; "
            "preserve the reference character exactly; no invented clothing or redesign"
        )
        sanitized.append(item)
    return _original_scene_prompt(scene, sanitized, locations, previous_scene, attempt)


base._scene_prompt = _scene_prompt


if __name__ == "__main__":
    base.main()
