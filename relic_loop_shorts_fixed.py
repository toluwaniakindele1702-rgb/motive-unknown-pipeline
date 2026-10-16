"""Relic Loop Shorts generator optimized for Groq free-tier token limits."""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from groq import Groq

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
RUN = STATE / "current_run"
HISTORY = STATE / "relic_loop_topic_history.json"
RUN.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("GROQ_WRITER_MODEL", "openai/gpt-oss-120b")
KEY = os.environ.get("GROQ_API_KEY", "").strip()

STYLE = ("Modern expressive 2D hand-drawn stickman cartoon: round head, simple tapered torso, "
         "thin expressive limbs, clean black ink linework, minimal eyes, expressive mouth and "
         "eyebrows, readable hands, exaggerated poses, flat-to-soft 2D shading, restrained color, "
         "clear silhouettes, believable simple environments, dynamic perspective, vertical 9:16. "
         "Consistent character identity. No photorealism, live action, stock photo, anime, manga, "
         "3D/Pixar, glossy mascot or documentary art.")

SYSTEM = f"""You are Relic Loop's creative director. Make one original fictional absurd-comedy YouTube Short.
Formula: instant weird situation -> escalation -> bigger problem -> absurd twist -> punchline.
Safe, non-graphic, visual and fast. Do NOT make an explainer, documentary, listicle or educational story.

STYLE: {STYLE}

Return ONLY JSON. Keep every field extremely compact so the response stays below the free-tier token limit.
Create EXACTLY 20 visual beats for a 20-45 second Short. Each beat is 0.8-2.2 seconds.
Change the image whenever the narration changes action, reaction, object state, reveal or important detail.
No captions, subtitles, title cards, speech bubbles, logos, watermarks, UI or plot text in images.

JSON shape:
{{"title":"...","hook":"...","narration":"55-85 words","characters":[{{"name":"A","appearance":"..."}}],"locations":["..."],"scenes":[{{"number":1,"duration_seconds":1.0,"beat_text":"...","narration":"...","visual_prompt":"...","camera":"...","action":"..."}}],"flow_prompt":"...","youtube_description":"...","tags":["...","..."]}}

Scene visual_prompt must say "stickman" and describe only what is visible. Keep each visual_prompt, camera and action short."""


def history() -> list[str]:
    try:
        value = json.loads(HISTORY.read_text(encoding="utf-8"))
        return [str(x).strip() for x in value if str(x).strip()][-60:] if isinstance(value, list) else []
    except Exception:
        return []


def save_history(items: list[str]) -> None:
    HISTORY.write_text(json.dumps(items[-100:], ensure_ascii=False, indent=2), encoding="utf-8")


def parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start:end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("Groq did not return valid JSON")


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text or ""))


def normalize_scenes(obj: dict[str, Any]) -> None:
    """Apply cheap local normalization instead of wasting another Groq request.

    The previous validator rejected prompts that were valid but omitted the literal word
    'stickman', and it also rejected negative instructions such as 'no logos' because
    the banned-word regex could not distinguish exclusions from requested text.
    """
    scenes = obj.get("scenes")
    if not isinstance(scenes, list):
        return
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        prompt = str(scene.get("visual_prompt", "")).strip()
        if prompt and "stickman" not in prompt.lower():
            scene["visual_prompt"] = "stickman " + prompt


def validate(obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("title", "hook", "narration", "characters", "locations", "scenes", "flow_prompt"):
        if not obj.get(key):
            errors.append(f"missing:{key}")
    wc = words(str(obj.get("narration", "")))
    if not 55 <= wc <= 85:
        errors.append(f"narration_words={wc}")
    scenes = obj.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 20:
        errors.append("scene_count")
        return errors
    runtime = 0.0
    for i, s in enumerate(scenes, 1):
        if not isinstance(s, dict):
            errors.append(f"scene_{i}_object")
            continue
        try:
            d = float(s.get("duration_seconds", 0))
        except (TypeError, ValueError):
            d = 0
        runtime += d
        if not 0.8 <= d <= 2.2:
            errors.append(f"scene_{i}_duration")
        for field in ("beat_text", "visual_prompt", "camera", "action", "narration"):
            if not str(s.get(field, "")).strip():
                errors.append(f"scene_{i}_{field}")
        p = str(s.get("visual_prompt", ""))
        if "stickman" not in p.lower():
            errors.append(f"scene_{i}_stickman")
    if not 20 <= runtime <= 45:
        errors.append(f"runtime={runtime:.1f}")
    return errors


def request(client: Groq, user_prompt: str, temperature: float = 0.8) -> str:
    result = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_completion_tokens=4200,
        response_format={"type": "json_object"},
    )
    return result.choices[0].message.content or ""


def make_prompt(topic: str | None) -> str:
    old = history()
    used = "" if not old else "Avoid these previous titles: " + "; ".join(old[-25:])
    seed = f"Story seed: {topic}." if topic else "Invent a highly shareable absurd premise with an immediate visual hook."
    return f"{seed}\n{used}\nWrite the complete package now. EXACTLY 20 compact scenes."


def generate(topic: str | None) -> dict[str, Any]:
    if not KEY:
        raise RuntimeError("GROQ_API_KEY is required")
    client = Groq(api_key=KEY)
    last: Exception | None = None
    for attempt in range(3):
        try:
            obj = parse_json(request(client, make_prompt(topic)))
            normalize_scenes(obj)
            errors = validate(obj)
            if not errors:
                obj["generated_at"] = datetime.now(timezone.utc).isoformat()
                obj["visual_policy"] = "No deliberate story text in images."
                obj["style_policy"] = STYLE
                return obj
            last = ValueError("validation failed: " + ", ".join(errors[:12]))
        except Exception as exc:
            last = exc
            msg = str(exc).lower()
            wait = 65 if any(x in msg for x in ("429", "rate limit", "rate_limit")) else 5 + attempt * 5
            if attempt < 2:
                print(f"[SHORTS] attempt {attempt + 1} failed: {exc}; waiting {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Could not generate a valid Relic Loop Short package: {last}")


def write(obj: dict[str, Any]) -> None:
    json_path = RUN / "relic_loop_short.json"
    flow_path = RUN / "flow_storyboard_prompt.txt"
    manifest_path = RUN / "flow_manifest.json"
    json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    scenes = obj["scenes"]
    lines = ["RELIC LOOP — FAST-CUT CLOUDFLARE IMAGE JOB", "", obj["flow_prompt"], "", "CANONICAL STYLE:", STYLE, "", "CAST:"]
    for c in obj["characters"]:
        lines.append(f"- {c['name']}: {c['appearance']}")
    lines += ["", "LOCATIONS:"] + [f"- {x}" for x in obj["locations"]] + ["", "IMAGE BEATS:"]
    for s in scenes:
        lines += [f"Beat {s['number']} — {s['duration_seconds']}s", f"Beat text: {s['beat_text']}", f"Narration: {s['narration']}", f"Visual: {s['visual_prompt']}", f"Camera: {s['camera']}", f"Action: {s['action']}", ""]
    lines += ["VOICEOVER:", obj["narration"], ""]
    flow_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "type": "relic_loop_fast_cut_image_short",
        "version": 5,
        "title": obj["title"],
        "runtime_seconds": round(sum(float(s["duration_seconds"]) for s in scenes), 2),
        "image_beat_count": len(scenes),
        "minimum_image_beats": 20,
        "target_image_beats": "20",
        "flow_template": "Storyboard Studio",
        "zero_text_gate": False,
        "text_policy": "no_deliberate_story_text",
        "media_generation": "cloudflare_images",
        "beat_policy": "new_image_at_phrase_comma_full_stop_tone_change_action_reaction_detail",
        "files": [str(json_path), str(flow_path)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    used = history()
    title = str(obj.get("title", "")).strip()
    if title and title.lower() not in {x.lower() for x in used}:
        used.append(title)
        save_history(used)
    print(f"[SHORTS] {obj['title']}")
    print(f"[SHORTS] {words(obj['narration'])} narration words")
    print(f"[SHORTS] {len(scenes)} image beats / {sum(float(s['duration_seconds']) for s in scenes):.1f}s")


if __name__ == "__main__":
    topic = os.environ.get("RELIC_LOOP_TOPIC", "").strip() or None
    write(generate(topic))
