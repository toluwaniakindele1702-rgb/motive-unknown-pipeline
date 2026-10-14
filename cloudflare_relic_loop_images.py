"""Generate Relic Loop fast-cut Shorts images with a hard visual style lock.

This module is intentionally separate from the legacy production pipeline. It reads
state/current_run/relic_loop_short.json, generates one Cloudflare image per beat,
rejects likely story text with OCR, and writes a beat manifest for assembly.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "state" / "current_run"
IMAGE_DIR = RUN / "images"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
MODEL = os.environ.get("CLOUDFLARE_IMAGE_MODEL", "@cf/black-forest-labs/flux-2-klein-4b").strip()
WIDTH = int(os.environ.get("RELIC_LOOP_IMAGE_WIDTH", "576"))
HEIGHT = int(os.environ.get("RELIC_LOOP_IMAGE_HEIGHT", "1024"))
MAX_RETRIES = int(os.environ.get("RELIC_LOOP_IMAGE_RETRIES", "3"))
OCR_STRICT = os.environ.get("RELIC_LOOP_OCR_STRICT", "1") != "0"

STYLE_LOCK = """
RELlC LOOP MASTER VISUAL STYLE — DO NOT REDESIGN:
modern hand-drawn 2D stickman animation illustration, based on the supplied Relic Loop
stickman reference: simple round head, simple tapered torso, thin expressive limbs,
clean slightly imperfect black ink linework, minimal oval eyes, highly expressive
mouth/eyebrow shapes, readable hands and exaggerated body language. Keep the character
simple; spend visual complexity on the action, environment, perspective and composition.
Clean graphic cartoon rendering with subtle flat-to-soft 2D shading, restrained color,
clear silhouettes, believable but simplified environments, strong foreground/background
separation, energetic comedy timing, dynamic perspective and polished Shorts storyboard
quality. Vertical 9:16 composition. The character identity must remain stable between
beats; do not redesign the head, body proportions or line language.

SHOT LANGUAGE:
Use purposeful shot variation: wide establishing shot, medium action shot, close-up
reaction, extreme close-up detail, over-the-shoulder reveal, low/high angle, side view,
front view, and dramatic perspective when the beat benefits from it. Never use camera
variation just for decoration: framing must make the current joke/action clearer.

ABSOLUTE STYLE EXCLUSIONS:
no photorealism, no live action, no stock photography, no anime, no manga, no Pixar/3D,
no glossy 3D mascot, no realistic human, no generic corporate illustration, no old
history-documentary art, no generic smiling portrait, no static character pose.
""".strip()

NO_TEXT = """
NO STORY TEXT: do not render captions, subtitles, title cards, speech bubbles, labels,
logos, watermarks, UI, signs with readable words, written explanations, or typography.
The image must communicate the beat through visible action, facial expression, objects,
composition and cause/effect. If incidental environmental writing appears naturally,
it must be tiny, non-readable and never plot-bearing.
""".strip()

TEXT_RE = re.compile(r"\b(?:caption|subtitle|title card|speech bubble|watermark|logo|typography|written explanation|text overlay|words on screen)\b", re.I)


def _cloudflare(prompt: str, refs: list[tuple[str, bytes]] | None = None) -> bytes:
    if not ACCOUNT_ID or not TOKEN:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required")
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{MODEL}"
    files = {}
    if refs:
        for idx, (name, blob) in enumerate(refs[:4]):
            files[f"input_image_{idx}"] = (name, blob, "image/jpeg")
    data = {"prompt": prompt, "width": str(WIDTH), "height": str(HEIGHT)}
    headers = {"Authorization": f"Bearer {TOKEN}"}
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, headers=headers, data=data, files=files or None, timeout=180)
            if r.status_code == 429:
                time.sleep(8 * (attempt + 1))
                continue
            r.raise_for_status()
            payload = r.json()
            image = ((payload.get("result") or {}).get("image"))
            if not image:
                raise RuntimeError(f"Cloudflare response had no result.image: {payload}")
            return base64.b64decode(image)
        except Exception as exc:
            last = exc
            if attempt + 1 < MAX_RETRIES:
                time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"Cloudflare image generation failed: {last}")


def _ocr_has_text(path: Path) -> bool:
    if not OCR_STRICT:
        return False
    try:
        proc = subprocess.run(
            ["tesseract", str(path), "stdout", "--psm", "11", "tsv"],
            text=True, capture_output=True, timeout=30, check=False,
        )
        for line in proc.stdout.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            word = parts[11].strip()
            try:
                conf = float(parts[10])
            except ValueError:
                conf = -1
            if word and conf >= 55 and len(re.sub(r"[^A-Za-z0-9]", "", word)) >= 2:
                return True
    except Exception:
        # If OCR itself fails, don't silently claim the image passed the gate.
        return True
    return False


def _reference_blurb() -> str:
    return (
        "The supplied Relic Loop reference establishes the character construction: "
        "round head, simple tapered torso, thin limbs, simple oval eyes, black expressive "
        "ink linework, hand-drawn asymmetry, minimal facial marks, readable hands and "
        "exaggerated poses. Preserve that identity in every beat."
    )


def _scene_prompt(scene: dict, characters: list[dict], locations: list[str]) -> str:
    cast = "; ".join(
        f"{c.get('name','character')}: {c.get('appearance','simple Relic Loop stickman')}"
        for c in characters
    )
    return f"""Create ONE vertical 9:16 image for a fast-cut Relic Loop absurd-comedy Short.

{STYLE_LOCK}

{_reference_blurb()}

CAST CONTINUITY:
{cast}

LOCATIONS AVAILABLE:
{', '.join(str(x) for x in locations)}

CURRENT BEAT — THIS IS THE ONLY STORY MOMENT TO ILLUSTRATE:
Beat text: {scene.get('beat_text','')}
Narration: {scene.get('narration','')}
Visible action: {scene.get('action','')}
Camera/composition: {scene.get('camera','')}

Make the exact action unmistakable. If the beat says a character turns, show the turn.
If something moves, show the movement. If the character reacts, show the reaction.
If an object changes state, show the changed state. Do not merely show the character
standing in the location. Use a strong, readable composition and expressive pose.

{NO_TEXT}

Do not add any extra characters, props, locations, story events or visual jokes that
are not useful to this beat. Do not turn this into a poster, thumbnail or character sheet.
This is a single frame from an animated story, so it must look like a real shot from
one continuous 2D cartoon Short.""".strip()


def main() -> None:
    package_path = RUN / "relic_loop_short.json"
    if not package_path.exists():
        raise SystemExit("Missing relic_loop_short.json; run the story generator first")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    scenes = package.get("scenes") or []
    if len(scenes) < 20:
        raise SystemExit(f"Refusing to render only {len(scenes)} beats; minimum is 20")

    # The exact user-supplied reference images are represented in the repository by a
    # text style lock for now; adding binary refs is optional because the prompt itself
    # is deliberately complete and the generator must also work on fresh GitHub runners.
    refs = []
    manifest = {
        "type": "relic_loop_cloudflare_image_render",
        "version": 1,
        "title": package.get("title"),
        "width": WIDTH,
        "height": HEIGHT,
        "model": MODEL,
        "reference_mode": "canonical_style_lock",
        "text_gate": "strict_ocr",
        "beats": [],
    }

    for index, scene in enumerate(scenes, 1):
        prompt = _scene_prompt(scene, package.get("characters") or [], package.get("locations") or [])
        if TEXT_RE.search(prompt):
            raise SystemExit(f"Prompt safety gate failed for beat {index}: story-text dependency detected")
        output = IMAGE_DIR / f"beat_{index:03d}.png"
        accepted = False
        for attempt in range(1, MAX_RETRIES + 1):
            blob = _cloudflare(prompt, refs)
            output.write_bytes(blob)
            if not _ocr_has_text(output):
                accepted = True
                break
            output.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                time.sleep(2 * attempt)
        if not accepted:
            raise SystemExit(f"Beat {index} failed the strict no-story-text OCR gate")
        manifest["beats"].append({
            "number": index,
            "duration_seconds": float(scene.get("duration_seconds", 1.0)),
            "image": str(output),
            "prompt": prompt,
        })
        print(f"[IMAGE {index}/{len(scenes)}] accepted")

    (RUN / "cloudflare_image_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[IMAGE] Rendered {len(scenes)} images to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
