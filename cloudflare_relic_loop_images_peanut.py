"""Chispa-Pifia multi-provider image router.

Priority: Raphael web UI (quality-model preference, reference-guided where supported)
-> Cloudflare Workers AI (FLUX.2 Klein 4B, reference-guided fallback).
The existing renderer retains storyboard, continuity, OCR, and manifest handling.
"""
from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import time
from pathlib import Path

import requests
from PIL import Image, ImageOps, ImageFilter, ImageStat

import cloudflare_relic_loop_images as base
import raphael_browser

# The two checked-in Chispa-Pifia images are the canonical visual bible.
base.CANONICAL_REFERENCE_PATHS = (
    ("Generate_modern_stickman_character_20261027121745.jpg", "image/jpeg"),
    ("Remove_bag_and_animate_image_20261027125028.jpg", "image/jpeg"),
)

base.STYLE_LOCK = """
CHISPA-PIFIA APPROVED 2D STICKMAN STYLE — HARD LOCK:
Use the supplied canonical reference images as the highest-priority visual authority.
The protagonist must remain the same simple, expressive 2D storytime stickman shown in
those references. Preserve the same large smooth pale round head, face construction,
eye shape and spacing, eyebrows, mouth style, head-to-body ratio, silhouette, limb
construction, outline thickness and overall proportions. Do not redesign the character.
Do not invent clothing or accessories that are not established by the references or story.

UNIFIED 2D ANIMATION WORLD:
The character, props, objects and every background element must all belong to one
coherent illustrated 2D cartoon world. Use clean hand-drawn/digital outlines, simplified
shapes, expressive poses, readable silhouettes, and restrained 2D cel-shading. Rooms,
furniture, streets, vehicles, trees, phones and all other objects must also look illustrated.
Never place the flat 2D stickman into a photograph, photorealistic environment or 3D scene.
Keep the established reference palette and rendering language consistent between shots.

CHARACTER AND SCENE CONTINUITY:
Reference 1 is the primary character identity anchor. Reference 2 is the secondary
approved style/action guide. Preserve the same character identity across every scene.
Expressions, poses, camera angles and story locations may change, but character design
must not. Use the previously accepted frame as an additional continuity aid without
overriding the canonical references or story facts.

ONE-SCENE RULE:
Generate exactly one full-frame vertical story shot per request. Never reproduce a
reference-sheet layout, collage, turnaround sheet or multiple panels.

ABSOLUTE EXCLUSIONS:
no photorealism, no live action, no 3D CGI, no anime/manga, no glossy mascot redesign,
no realistic human anatomy, no character redesign, no random outfit changes, no extra
limbs or fingers, no duplicated protagonist unless the story explicitly requires it,
no copied reference-sheet panels, no text, captions, speech bubbles, labels, logos
or watermarks.
""".strip()

base.NO_TEXT = """
STRICT ZERO-TEXT IMAGE RULE:
Do not render captions, subtitles, title cards, speech bubbles, labels, logos, watermarks,
UI, readable signs, letters, numbers, symbols, menus, posters, packaging text, screen text
or typography. Keep environmental surfaces blank or non-readable. Communicate the story
only through character acting, objects, composition, expressions and visible cause/effect.
""".strip()

base._reference_blurb = lambda: (
    "CANONICAL REFERENCE PRIORITY: Reference 1 defines the exact protagonist identity. "
    "Reference 2 defines the approved 2D visual language and action/style guidance. "
    "Match the original simple stickman construction, facial design, proportions, outlines "
    "and flat illustrated rendering. Do not copy the reference-sheet layout. The references "
    "define character appearance; the narration defines the action."
)

_original_scene_prompt = base._scene_prompt
_SHOT_VARIATIONS = [
    "SHOT 1: dramatic wide establishing view from slightly above, show the ride and protagonist clearly; strong diagonal composition.",
    "SHOT 2: tight face-and-phone close-up, illustrated depth, carousel only as contextual shapes behind.",
    "SHOT 3: dynamic low-angle side view, protagonist gripping the moving ride; strong foreshortening and diagonal motion.",
    "SHOT 4: high overhead view looking down onto the carousel, clear circular geometry and protagonist position.",
    "SHOT 5: extreme close-up of the protagonist's hands and phone with expressive face partly visible; distinct foreground framing.",
    "SHOT 6: side-on medium tracking composition, protagonist off-centre, ride structure sweeping across the frame.",
    "SHOT 7: long view through foreground ride elements, protagonist's expression remains clearly readable.",
    "SHOT 8: rear three-quarter view over the protagonist's shoulder toward the story-critical object; change screen direction.",
    "SHOT 9: very wide view with strong foreground, middle-ground and background layers; new perspective.",
    "SHOT 10: low close-medium shot from platform level, emphasize physical action and reaction.",
    "SHOT 11: decisive payoff composition, protagonist large in frame, clear pose and prop interaction; distinct from earlier shots.",
]
def _peanut_scene_prompt(scene, characters, locations, previous_scene=None, attempt=1):
    cleaned = []
    for character in characters or []:
        item = dict(character) if isinstance(character, dict) else {}
        item["appearance"] = (
            "the approved canonical Chispa-Pifia 2D stickman from reference image 1; "
            "match its appearance exactly; do not redesign or invent wardrobe"
        )
        cleaned.append(item)
    prompt = _original_scene_prompt(scene, cleaned, locations, previous_scene, attempt)
    number = int(scene.get("number", 1)) if isinstance(scene, dict) else 1
    variation = _SHOT_VARIATIONS[(max(1, number) - 1) % len(_SHOT_VARIATIONS)]
    return (
        prompt + "\\n\\nMANDATORY SHOT-DIVERSITY DIRECTION — " + variation +
        "\\nDo not repeat the previous frame's camera distance, angle, framing, protagonist screen position, "
        "or dominant foreground shape. Preserve the same location and character identity, but visibly change "
        "the composition. The current beat's exact action remains the focal point. Never add unrelated props, "
        "characters, events, or visual gags. Use a distinct camera setup, not merely a different facial expression."
    )

base._scene_prompt = _peanut_scene_prompt

PROVIDER_EVENTS: list[dict] = []
PROVIDER_COUNTS: dict[str, int] = {"raphael": 0, "cloudflare": 0}

REFERENCE_FILES = base.CANONICAL_REFERENCE_PATHS
_CLOUDFLARE_ORIGINAL = base._cloudflare


def _record(provider: str, status: str, detail: str = "") -> None:
    PROVIDER_EVENTS.append({
        "provider": provider,
        "status": status,
        "detail": detail[:700],
    })
    print(f"[PROVIDER:{provider.upper()}] {status}" + (f": {detail[:500]}" if detail else ""))


def _normalize_image(blob: bytes) -> bytes:
    """Validate a real portrait source, then pad to exact 9:16 without blur/crop."""
    try:
        with Image.open(BytesIO(blob)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            ratio = height / max(1, width)
            if width < 512 or height < 768 or ratio < 1.45:
                raise RuntimeError(
                    f"Source is not a usable portrait image ({width}x{height}); "
                    "rejecting instead of manufacturing a blurred vertical canvas."
                )
            # Reject severely soft/placeholder captures. Real illustrated frames
            # should contain a meaningful amount of edge detail.
            edge = image.convert("L").filter(ImageFilter.FIND_EDGES)
            if float(ImageStat.Stat(edge).stddev[0]) < 8.0:
                raise RuntimeError("Source image is too blurred or visually empty.")
            target = (base.WIDTH, base.HEIGHT)
            fitted = ImageOps.contain(image, target, method=Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", target, (24, 27, 34))
            x = (target[0] - fitted.width) // 2
            y = (target[1] - fitted.height) // 2
            canvas.paste(fitted, (x, y))
            output = BytesIO()
            canvas.save(output, format="PNG", optimize=True)
            return output.getvalue()
    except Exception as exc:
        raise RuntimeError(f"Provider image failed validation: {type(exc).__name__}: {exc}") from exc


def _extract_image(response: requests.Response) -> bytes:
    content_type = response.headers.get("content-type", "").lower()
    if "image/" in content_type:
        return response.content
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Image API returned non-image, non-JSON response: {response.text[:500]}") from exc
    data = payload.get("data") or []
    if not data or not isinstance(data[0], dict):
        raise RuntimeError(f"Image API response contains no data[0]: {json.dumps(payload)[:700]}")
    item = data[0]
    encoded = item.get("b64_json")
    if encoded:
        try:
            return base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise RuntimeError("Image API returned invalid b64_json") from exc
    url = item.get("url")
    if url:
        downloaded = requests.get(url, timeout=90)
        downloaded.raise_for_status()
        return downloaded.content
    raise RuntimeError(f"Image API response has neither b64_json nor url: {json.dumps(item)[:500]}")


def _route_image(prompt: str, refs: list[tuple[str, bytes, str]] | None = None) -> bytes:
    providers = (
        ("raphael", lambda: _normalize_image(raphael_browser.generate(prompt, refs))),
        ("cloudflare", lambda: _normalize_image(_CLOUDFLARE_ORIGINAL(prompt, refs))),
    )
    failures = []
    for provider, generate in providers:
        try:
            image = generate()
            if not image:
                raise RuntimeError("provider returned an empty image")
            PROVIDER_COUNTS[provider] += 1
            _record(provider, "success")
            return image
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            failures.append(f"{provider}: {detail}")
            _record(provider, "fallback", detail)
    raise RuntimeError("All configured image providers failed. " + " || ".join(failures))


# The base renderer calls this function for each storyboard beat. Keep its
# story prompts, reference loading, previous-frame continuity and OCR pipeline.
base._cloudflare = _route_image
_original_main = base.main


def main() -> None:
    PROVIDER_EVENTS.clear()
    for key in PROVIDER_COUNTS:
        PROVIDER_COUNTS[key] = 0
    _record("router", "started", "Raphael browser UI (quality-model preference) -> Cloudflare FLUX.2 Klein 4B fallback")
    _original_main()

    manifest_path = base.RUN / "cloudflare_image_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["media_generation"] = "multi_provider_image_router"
        manifest["provider_priority"] = ["raphael", "cloudflare"]
        manifest["provider_counts"] = PROVIDER_COUNTS
        manifest["provider_events"] = PROVIDER_EVENTS
        manifest["reference_policy"] = {
            "raphael": "official web UI automation; selected high-quality model where available; canonical references uploaded when supported; no CAPTCHA/Turnstile bypass",
            "cloudflare": "FLUX.2 Klein 4B with canonical references plus previous accepted frame",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[PROVIDER SUMMARY] " + json.dumps(PROVIDER_COUNTS, sort_keys=True))


if __name__ == "__main__":
    main()
