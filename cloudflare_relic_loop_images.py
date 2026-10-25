"""Generate Relic Loop story images; image count follows meaningful story beats."""
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
OCR_RETRIES = max(1, int(os.environ.get("RELIC_LOOP_OCR_RETRIES", "3")))
OCR_STRICT = os.environ.get("RELIC_LOOP_OCR_STRICT", "1") != "0"
REFERENCE_B64 = ROOT / "assets" / "relic_loop_character_reference.jpg.b64"

STYLE_LOCK = """
CHISPA-PIFIA MASTER VISUAL STYLE — HARD LOCK, DO NOT REDESIGN:
Follow the supplied canonical standing-pose reference as the exact character model sheet,
not merely as loose inspiration. The main character is a stylized stickman with a large
smooth pale round head, large expressive oval black eyes, bold black eyebrows, small simple
mouth, tiny neck, compact developed dark-clothed torso, articulated arms, hands, legs and
feet. The body is more developed than line-only stick figures, but it must remain unmistakably
a round-headed stickman—not a realistic human, not a chibi child, and not an ordinary human
cartoon. Keep the head-to-body ratio, facial spacing, eye shape, limb thickness, silhouette,
hoodie, black cargo trousers, black-and-white high-top sneakers, and dark backpack when present.
Default outfit: black/dark charcoal hoodie with drawstrings, dark cargo trousers and black/white
sneakers; backpack appears only when the story establishes that the character carries it. Do not
add facial hair, nose detail, skin texture, logos, accessories or clothing changes without story need.

RENDERING AND ENVIRONMENTS:
Use cinematic stylized animation with polished soft dimensional shading on the character,
clean dark outlines, readable hands, strong facial acting, subtle material detail, and rich,
detailed environments. Environments may be more realistic and intricate than the character:
believable bedrooms, apartments, corridors, streets, classrooms and outdoor spaces, with depth,
practical objects, reflections, shadows and atmospheric perspective. Do not flatten the background
into a blank cartoon setting. Keep the character stylized against the detailed environment; never
turn the character photorealistic. Maintain a dark cinematic base palette (charcoal, black, cool
blue/steel grey) with controlled warm amber/orange practical-light accents. Lighting must match
time of day and story mood while keeping the face and action readable.

IDENTITY AND CONTINUITY RULES:
Use the standing-pose sheet for character shape, outfit, proportions and pose vocabulary. Use
the environment reference sheet for location detail, cinematic lighting, palette and camera
language. Treat both as canonical constraints. Do not copy the reference-sheet layout, captions,
labels or example story panels into a generated scene. Render only one full-frame story shot.
The same protagonist must retain the same head, eyes, brows, outfit, body proportions and footwear
in every scene. Preserve room layout, door/window positions, time of night, lighting direction,
prop placement and object state across adjacent shots unless narration explicitly changes them.
The prior accepted frame is a continuity aid, but never overrides the canonical character model
or story facts.

SHOT LANGUAGE:
Create vertical 9:16 cinematic story frames with deliberate composition. Vary wide establishing
shots, medium action shots, face close-ups, over-the-shoulder reveals, point-of-view inserts,
low/high angles and detail shots according to the exact beat. Expressions and pose must show
the narrated emotion and action. Keep the protagonist clearly readable and leave safe margins
around important subjects. Every frame must show a specific story moment, not a model-sheet pose.

ABSOLUTE EXCLUSIONS:
no photorealistic character, no live-action look, no anime/manga, no generic stick figure made
only of lines, no realistic human anatomy, no Pixar-like character redesign, no glossy mascot,
no style changes between shots, no changing eye shape or head size, no random outfit changes,
no extra fingers/limbs, no duplicate protagonist unless the story establishes one, no copying
reference-sheet panels, no text or typography, no watermark, no logo.
""".strip()

NO_TEXT = """
STRICT ZERO-TEXT IMAGE RULE: do not render captions, subtitles, title cards, speech bubbles,
labels, logos, watermarks, UI, signs with readable words, written explanations, letters,
numbers, symbols, menus, posters, packaging text, screen text or typography. Keep surfaces
blank or use tiny abstract non-readable environmental marks only when unavoidable. The image
must communicate the beat entirely through action, facial expression, objects, composition
and cause/effect.
""".strip()

STORY_TEXT_REQUEST_RE = re.compile(
    r"\b(?:add|show|display|write|render|include|put|read)\b[^.\n]{0,100}\b(?:caption|subtitle|title card|speech bubble|logo|watermark|readable words|text overlay|written words|sign|label|letters|numbers|text)\b",
    re.I,
)


VISUAL_REFERENCE_DIR = ROOT / "assets" / "visual_references"
CANONICAL_REFERENCE_PATHS = (
    ("chispa_character_standing_poses.png", "image/png"),
    ("chispa_environment_style_reference.png", "image/png"),
)


def _load_reference() -> list[tuple[str, bytes, str]]:
    """Load canonical visual guides first; keep legacy reference as fallback."""
    loaded: list[tuple[str, bytes, str]] = []
    for filename, mime in CANONICAL_REFERENCE_PATHS:
        path = VISUAL_REFERENCE_DIR / filename
        if path.is_file() and path.stat().st_size > 0:
            loaded.append((filename, path.read_bytes(), mime))
    if loaded:
        print("[REFERENCE] Loaded canonical visual references: " + ", ".join(x[0] for x in loaded))
        return loaded
    if REFERENCE_B64.exists():
        blob = base64.b64decode(REFERENCE_B64.read_text(encoding="utf-8").strip())
        if blob:
            print("[REFERENCE] WARNING: canonical Chispa-Pifia reference images are missing; using legacy reference only.")
            return [("relic_loop_character_reference.jpg", blob, "image/jpeg")]
    print("[REFERENCE] WARNING: no character reference image found; prompt-only mode.")
    return []


def _response_diagnostic(response: requests.Response) -> str:
    body = response.text.strip() or "<empty response body>"
    try:
        payload = response.json()
        errors = payload.get("errors") if isinstance(payload, dict) else None
        messages = payload.get("messages") if isinstance(payload, dict) else None
        result_keys = list((payload.get("result") or {}).keys()) if isinstance(payload, dict) and isinstance(payload.get("result"), dict) else []
        parts = [f"status={response.status_code}"]
        if errors:
            parts.append(f"errors={json.dumps(errors, ensure_ascii=False)}")
        if messages:
            parts.append(f"messages={json.dumps(messages, ensure_ascii=False)}")
        if result_keys:
            parts.append(f"result_keys={result_keys}")
        if len(parts) == 1:
            parts.append(f"json={json.dumps(payload, ensure_ascii=False)[:1200]}")
        return " | ".join(parts)
    except ValueError:
        return f"status={response.status_code} | body={body[:1200]}"


def _cloudflare(prompt: str, refs: list[tuple[str, bytes, str]] | None = None) -> bytes:
    if not ACCOUNT_ID or not TOKEN:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required")
    url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/{MODEL}"
    files: dict[str, tuple[None, str] | tuple[str, bytes, str]] = {
        "prompt": (None, prompt), "width": (None, str(WIDTH)), "height": (None, str(HEIGHT))
    }
    if refs:
        for idx, (name, blob, mime) in enumerate(refs[:4]):
            files[f"input_image_{idx}"] = (name, blob, mime)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[CLOUDFLARE] request attempt {attempt}/{MAX_RETRIES} model={MODEL} size={WIDTH}x{HEIGHT} refs={len(refs or [])}")
            r = requests.post(url, headers=headers, files=files, timeout=180)
            diagnostic = _response_diagnostic(r)
            print(f"[CLOUDFLARE] HTTP response: {diagnostic}")
            if r.status_code == 429:
                last = RuntimeError(f"Cloudflare rate limit: {diagnostic}")
                if attempt < MAX_RETRIES:
                    wait = 8 * attempt
                    print(f"[CLOUDFLARE] rate limited; retrying in {wait}s")
                    time.sleep(wait)
                    continue
                break
            if r.status_code >= 500:
                last = RuntimeError(f"Cloudflare server error: {diagnostic}")
                if attempt < MAX_RETRIES:
                    wait = 4 * attempt
                    print(f"[CLOUDFLARE] server error; retrying in {wait}s")
                    time.sleep(wait)
                    continue
                break
            if not r.ok:
                raise RuntimeError(f"Cloudflare API rejected request: {diagnostic}")
            try:
                payload = r.json()
            except ValueError as exc:
                raise RuntimeError(f"Cloudflare returned non-JSON success response: {diagnostic}") from exc
            image = ((payload.get("result") or {}).get("image")) if isinstance(payload, dict) else None
            if not image:
                raise RuntimeError(f"Cloudflare response had no result.image: {diagnostic}")
            try:
                decoded = base64.b64decode(image, validate=True)
            except Exception as exc:
                raise RuntimeError("Cloudflare returned an invalid base64 image payload") from exc
            if not decoded:
                raise RuntimeError("Cloudflare returned an empty image payload")
            return decoded
        except requests.RequestException as exc:
            last = RuntimeError(f"Cloudflare HTTP transport error: {type(exc).__name__}: {exc}")
            print(f"[CLOUDFLARE] transport error: {last}")
            if attempt < MAX_RETRIES:
                time.sleep(4 * attempt)
        except Exception as exc:
            last = exc
            print(f"[CLOUDFLARE] attempt failed: {type(exc).__name__}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(4 * attempt)
    raise RuntimeError(f"Cloudflare image generation failed after {MAX_RETRIES} attempts: {last}")


def _ocr_has_text(path: Path) -> bool:
    """Return True only when OCR finds convincing text-like content."""
    if not OCR_STRICT:
        return False
    try:
        proc = subprocess.run(
            ["tesseract", str(path), "stdout", "--psm", "11", "tsv"],
            text=True, capture_output=True, timeout=30, check=False,
        )
        candidates = []
        for line in proc.stdout.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            word = re.sub(r"[^A-Za-z0-9]", "", parts[11].strip())
            try:
                conf = float(parts[10])
            except ValueError:
                conf = -1
            if word and conf >= 60 and len(word) >= 3:
                candidates.append((word, conf))
        strong = [w for w, c in candidates if c >= 78 and len(w) >= 4]
        medium = [w for w, c in candidates if c >= 68 and len(w) >= 4]
        suspicious = len(strong) >= 2 or len(medium) >= 3
        if suspicious:
            print(f"[OCR] suspicious text candidates: strong={strong[:5]} medium={medium[:8]}")
        return suspicious
    except FileNotFoundError:
        if OCR_STRICT:
            raise SystemExit("Strict zero-text gate requires tesseract, but tesseract is not installed")
        return False
    except Exception as exc:
        if OCR_STRICT:
            raise SystemExit(f"Strict zero-text OCR check failed: {type(exc).__name__}: {exc}")
        print(f"[OCR] check unavailable: {type(exc).__name__}: {exc}")
        return False


def _reference_blurb() -> str:
    return (
        "REFERENCE IMAGE PRIORITY: Image reference 1 is the canonical main-character model sheet. "
        "Match its round pale head, oval eyes, eyebrow design, head/body ratio, dark hoodie, cargo "
        "trousers, sneakers, backpack design and silhouette closely. Image reference 2, when present, "
        "is the environment/style guide: match its detailed cinematic locations, dark cool palette, "
        "warm amber accents, lighting and camera language. Do not recreate either sheet as a collage "
        "and do not copy words or panels from them. Render one single edge-to-edge vertical scene. "
        "The reference images take priority over vague stylistic interpretation; story facts take "
        "priority over unrelated example scenes shown inside a reference sheet."
    )


def _scene_prompt(
    scene: dict,
    characters: list[dict],
    locations: list[str],
    previous_scene: dict | None = None,
    attempt: int = 1,
) -> str:
    cast = "; ".join(
        f"{c.get('name','protagonist')}: {c.get('appearance','simple human Relic Loop stickman')}"
        for c in characters
    )
    continuity = ""
    if previous_scene:
        continuity = (
            f"\nPREVIOUS STORY BEAT:\n{previous_scene.get('beat_text','')}\n"
            "The current frame is the direct next moment or consequence. Preserve the same protagonist,"
            " clothing, important props and environment unless the narration explicitly changes them.\n"
        )
    retry_note = ""
    if attempt > 1:
        retry_note = (
            "\nSTRICT TEXT-CLEAN RETRY: blank surfaces only. No signs, labels, menus, posters, packaging, "
            "screens, letters, numbers, symbols or readable marks anywhere in frame.\n"
        )
    return f"""Create ONE vertical 9:16 image for a Relic Loop / Chispa-Pifia absurd-comedy Short.

{STYLE_LOCK}

{_reference_blurb()}

CAST CONTINUITY:
{cast}

LOCATIONS AVAILABLE:
{', '.join(str(x) for x in locations)}
{continuity}
CURRENT BEAT — THIS IS THE ONLY MOMENT TO ILLUSTRATE:
Beat text: {scene.get('beat_text','')}
Narration for this shot: {scene.get('narration','')}
Visible action: {scene.get('action','')}
Camera/composition: {scene.get('camera','')}

NARRATION IS THE SOURCE OF TRUTH. Reenact the current beat literally. Do not summarize it,
and do not invent an unrelated visual joke. Clearly show WHO is present, WHERE they are,
WHAT exactly they are doing, and WHAT caused this moment. Do not add unrelated characters,
props, locations, plot events or visual gags.

{NO_TEXT}
{retry_note}
This is one literal shot from one continuous human-centered cartoon story.""".strip()


def main() -> None:
    package_path = RUN / "relic_loop_short.json"
    if not package_path.exists():
        raise SystemExit("Missing relic_loop_short.json; run the story generator first")
    package = json.loads(package_path.read_text(encoding="utf-8"))
    scenes = package.get("scenes") or []
    if not 8 <= len(scenes) <= 10:
        raise SystemExit(f"Storyboard beat count {len(scenes)} is outside the required 8-10 range")

    refs = _load_reference()
    manifest = {
        "type": "relic_loop_cloudflare_image_render", "version": 11,
        "title": package.get("title"), "width": WIDTH, "height": HEIGHT, "model": MODEL,
        "reference_mode": "canonical_character_and_environment_references_plus_previous_beat" if len(refs) >= 2 else ("legacy_reference_plus_previous_beat" if refs else "prompt_only"),
        "character_references": [name for name, _, _ in refs],
        "text_gate": "strict_ocr_reject_after_retry",
        "ocr_retries": OCR_RETRIES,
        "ocr_policy": "regenerate_then_fail_if_text_suspected",
        "cloudflare_diagnostics": "http_status_and_response_body_logged_on_every_attempt",
        "story_truth": "literal_scene_reenactment",
        "timing_truth": "assembly recalculates beat durations from actual TTS duration",
        "image_count_policy": "8-10 meaningful phrase-aligned story beats",
        "continuity_policy": "canonical_character_reference_plus_previous_accepted_frame",
        "beats": [],
    }

    previous_scene = None
    previous_image: bytes | None = None
    for index, scene in enumerate(scenes, 1):
        scene_fields = " ".join(
            str(scene.get(k, "")) for k in ("beat_text", "narration", "visual_prompt", "action", "camera")
        )
        if STORY_TEXT_REQUEST_RE.search(scene_fields):
            raise SystemExit(f"Prompt safety gate failed for beat {index}: deliberate story text requested")

        output = IMAGE_DIR / f"beat_{index:03d}.png"
        accepted = False
        ocr_suspected = False
        prompt = ""
        for attempt in range(1, OCR_RETRIES + 1):
            prompt = _scene_prompt(
                scene,
                package.get("characters") or [],
                package.get("locations") or [],
                previous_scene,
                attempt,
            )
            render_refs = list(refs)
            if previous_image:
                render_refs.append(("previous_accepted_beat.jpg", previous_image, "image/png"))
            blob = _cloudflare(prompt, render_refs)
            output.write_bytes(blob)
            if not _ocr_has_text(output):
                accepted = True
                break
            ocr_suspected = True
            print(f"[IMAGE {index}/{len(scenes)}] OCR suspected text; regenerating ({attempt}/{OCR_RETRIES})")
            output.unlink(missing_ok=True)
            if attempt < OCR_RETRIES:
                time.sleep(2 * attempt)

        if not accepted:
            raise SystemExit(
                f"Beat {index} rejected: generated image still contains OCR-detectable text after {OCR_RETRIES} attempts"
            )

        previous_image = output.read_bytes()
        manifest["beats"].append({
            "number": index,
            "duration_seconds": float(scene.get("duration_seconds", 1.0)),
            "image": str(output),
            "prompt": prompt,
            "beat_text": scene.get("beat_text"),
            "narration": scene.get("narration"),
            "ocr_suspected": ocr_suspected,
            "used_previous_frame_reference": previous_image is not None and index > 1,
        })
        previous_scene = scene
        print(f"[IMAGE {index}/{len(scenes)}] accepted")

    (RUN / "cloudflare_image_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[IMAGE] Rendered {len(scenes)} images to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
