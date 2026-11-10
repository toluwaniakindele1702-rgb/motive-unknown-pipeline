#!/usr/bin/env python3
"""Creative AI visual director and image-render quality guard for Relic Loop.

Creative AI authors a scene-specific visual brief; the configured provider router
(Raphael -> Cloudflare fallback) renders it. This module validates real image bytes,
normalizes portrait output, preserves accepted frames, and retries a rejected frame
once with a more explicit prompt. It never publishes or changes provider credentials.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable, Sequence
import json
import math

from PIL import Image, ImageFilter, ImageOps, ImageStat


@dataclass(frozen=True)
class ImageGenerationBrief:
    scene_number: int
    narration_beat: str
    protagonist_identity: str
    setting: str
    visual_style: str = (
        "polished expressive 2D stickman cartoon; pale round head; stable face, "
        "proportions and outfit; clean bold outlines; illustrated props and background; "
        "subtle 2D cel shading; cinematic but clearly hand-drawn"
    )
    previous_scene_state: str = ""
    shot_direction: str = "medium vertical 9:16 action shot, strong readable silhouette"
    important_props: tuple[str, ...] = ()
    avoid: tuple[str, ...] = (
        "photorealism", "3D CGI", "live action", "anime", "character redesign",
        "extra limbs or fingers", "duplicate characters", "collage", "split panels",
        "captions", "letters", "numbers", "logos", "watermarks", "unrelated props",
    )


@dataclass
class ImageInspection:
    valid: bool
    width: int = 0
    height: int = 0
    format: str = ""
    bytes_size: int = 0
    mean_luminance: float = 0.0
    luminance_stddev: float = 0.0
    edge_density: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid, "width": self.width, "height": self.height,
            "format": self.format, "bytes_size": self.bytes_size,
            "mean_luminance": round(self.mean_luminance, 2),
            "luminance_stddev": round(self.luminance_stddev, 2),
            "edge_density": round(self.edge_density, 4),
            "reasons": list(self.reasons),
        }


def build_image_prompt(brief: ImageGenerationBrief, refinement: int = 0) -> str:
    """Build a production-grade, beat-faithful image prompt with explicit locks."""
    if not brief.narration_beat.strip():
        raise ValueError("A narration beat is required for image generation")
    if not brief.protagonist_identity.strip():
        raise ValueError("A canonical protagonist identity is required")
    if not 1 <= int(brief.scene_number) <= 99:
        raise ValueError("Scene number must be between 1 and 99")
    props = ", ".join(x.strip() for x in brief.important_props if x.strip()) or "only props required by the beat"
    prior = brief.previous_scene_state.strip() or "Use the established story setting and canonical character design."
    refinement_note = (
        "QUALITY REFINEMENT: Make the protagonist and story-critical action larger and clearer; "
        "simplify competing background detail; sharpen the silhouette; preserve the exact beat."
        if refinement else ""
    )
    return f"""Create ONE finished story frame for Chispa-Pifia, scene {brief.scene_number}.
CANONICAL CHARACTER LOCK: {brief.protagonist_identity.strip()}. This is a fixed model sheet.
Never alter head shape, face, eye spacing, body proportions, outline weight, clothing or identity.
VISUAL STYLE LOCK: {brief.visual_style.strip()}. Character, props and environment must all be
illustrated in the same coherent 2D cartoon language. No realistic/photo background and no 3D.
EXACT NARRATION MOMENT: {brief.narration_beat.strip()}
VISIBLE ACTION: Show the protagonist physically performing or reacting to the exact action
above. Make the cause-and-effect readable in one glance. Do not add events or invent story facts.
SETTING: {brief.setting.strip()}
CONTINUITY STATE: {prior}
STORY-CRITICAL PROPS: {props}
COMPOSITION: {brief.shot_direction.strip()}. Vertical portrait 9:16. One dominant focal action,
clear foreground/midground/background separation, expressive face and body language, intentional
camera angle, readable lighting, clean edges, balanced negative space, phone-screen legibility.
Keep the protagonist fully visible unless a close-up is required by the beat. Maintain continuity
of object ownership, prop state, clothing, location, lighting direction and screen direction.
ZERO TEXT: no captions, typography, words, letters, numbers, symbols, speech bubbles, logos,
watermarks, labels, readable signs or interface elements anywhere.
STRICTLY AVOID: {", ".join(brief.avoid)}.
{refinement_note}
Return only the image, not a contact sheet, storyboard, collage, model sheet or multiple panels."""


def inspect_image(image_bytes: bytes, min_bytes: int = 2_048) -> ImageInspection:
    """Conservative technical/usability checks; not a substitute for human art review."""
    result = ImageInspection(valid=False, bytes_size=len(image_bytes or b""))
    if not image_bytes or len(image_bytes) < min_bytes:
        result.reasons.append(f"image payload is smaller than {min_bytes} bytes")
        return result
    try:
        with Image.open(BytesIO(image_bytes)) as probe:
            result.format = str(probe.format or "")
            result.width, result.height = probe.size
            probe.verify()
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            if image.width < 256 or image.height < 256:
                result.reasons.append("source resolution is below 256px on one edge")
            if image.width / image.height < 0.35 or image.width / image.height > 2.8:
                result.reasons.append("source aspect ratio is implausible")
            gray = image.convert("L")
            stats = ImageStat.Stat(gray)
            result.mean_luminance = stats.mean[0]
            result.luminance_stddev = stats.stddev[0]
            # Edge density is measured on a small normalized thumbnail. A very low
            # value can indicate a blank frame; flat cartoon art remains acceptable.
            thumb = gray.resize((128, 128), Image.Resampling.LANCZOS)
            edges = thumb.filter(ImageFilter.FIND_EDGES)
            edge_pixels = list(edges.getdata())
            result.edge_density = sum(1 for p in edge_pixels if p >= 24) / len(edge_pixels)
            if result.mean_luminance < 8:
                result.reasons.append("frame is almost completely black")
            if result.mean_luminance > 249 and result.luminance_stddev < 5:
                result.reasons.append("frame appears almost completely blank white")
            if result.luminance_stddev < 2.0:
                result.reasons.append("frame has almost no visible tonal variation")
            if result.edge_density < 0.001:
                result.reasons.append("frame appears blank or nearly featureless")
    except Exception as exc:
        result.reasons.append(f"image decode/verification failed: {type(exc).__name__}: {exc}")
    result.valid = not result.reasons
    return result


def normalize_portrait(image_bytes: bytes, width: int = 576, height: int = 1024) -> bytes:
    """Normalize any usable source ratio to a clean portrait PNG without distortion."""
    inspection = inspect_image(image_bytes)
    if not inspection.valid:
        raise ValueError("Rejected generated image: " + "; ".join(inspection.reasons))
    with Image.open(BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        # Fit rather than stretch. A modest vertical bias protects faces near the top.
        fitted = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.43))
        output = BytesIO()
        fitted.save(output, format="PNG", optimize=True)
        return output.getvalue()


def render_scene(
    brief: ImageGenerationBrief,
    provider_router: Callable[[str, list[tuple[str, bytes, str]] | None], bytes],
    references: list[tuple[str, bytes, str]] | None = None,
    output_path: str | Path | None = None,
    width: int = 576,
    height: int = 1024,
) -> tuple[bytes, dict]:
    """Render, inspect and retry once; a failed scene never overwrites a good frame."""
    attempts = []
    last_error = None
    for attempt in range(2):
        prompt = build_image_prompt(brief, refinement=attempt)
        try:
            raw = provider_router(prompt, references)
            inspection = inspect_image(raw)
            attempts.append({"attempt": attempt + 1, "inspection": inspection.to_dict()})
            if not inspection.valid:
                raise ValueError("; ".join(inspection.reasons))
            normalized = normalize_portrait(raw, width=width, height=height)
            final_check = inspect_image(normalized)
            if not final_check.valid:
                raise ValueError("normalized frame failed inspection: " + "; ".join(final_check.reasons))
            if output_path is not None:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(path.suffix + ".tmp")
                temp.write_bytes(normalized)
                temp.replace(path)
            return normalized, {
                "scene_number": brief.scene_number, "provider_router": "injected",
                "attempts": attempts, "final_inspection": final_check.to_dict(),
                "human_visual_approval_required": True,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt + 1, "error": last_error})
    raise RuntimeError(
        f"Scene {brief.scene_number} image generation failed after two quality attempts: {last_error}"
    )


def self_test() -> None:
    # Synthetic test fixtures validate the guard, not artistic quality.
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (576, 1024), "white")
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((170, 100, 410, 340), fill="#f0d9bd", outline="black", width=10)
    draw.line((290, 340, 290, 760), fill="black", width=18)
    draw.line((290, 450, 130, 600), fill="black", width=18)
    draw.line((290, 450, 450, 600), fill="black", width=18)
    draw.line((290, 760, 170, 950), fill="black", width=18)
    draw.line((290, 760, 410, 950), fill="black", width=18)
    buffer = BytesIO()
    canvas.save(buffer, format="PNG")
    good = buffer.getvalue()
    assert inspect_image(good).valid
    assert not inspect_image(b"not an image").valid
    brief = ImageGenerationBrief(
        scene_number=1, narration_beat="Milo grabs the runaway lunchbox.",
        protagonist_identity="Milo, canonical pale round-headed stickman in dark hoodie",
        setting="small illustrated kitchen",
        important_props=("red lunchbox",),
    )
    prompt = build_image_prompt(brief)
    assert "EXACT NARRATION MOMENT" in prompt and "ZERO TEXT" in prompt
    calls = []
    def fake_router(text, refs):
        calls.append(text)
        return good
    rendered, report = render_scene(brief, fake_router)
    assert inspect_image(rendered).valid and report["human_visual_approval_required"]
    assert len(calls) == 1
    print("Creative AI image generator self-test passed")


if __name__ == "__main__":
    self_test()
