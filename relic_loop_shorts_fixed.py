"""Robust Relic Loop Shorts story generator.

This is a replacement for the older planner when Groq returns malformed/truncated JSON.
"""
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

STYLE = """Modern expressive 2D hand-drawn stickman cartoon for Relic Loop. Round head,
simple tapered torso, thin expressive limbs, clean slightly imperfect black ink linework,
minimal oval eyes, expressive mouths and eyebrows, readable hands and exaggerated poses.
Clean graphic cartoon rendering, subtle flat-to-soft 2D shading, restrained color, clear
silhouettes, believable simplified environments, dynamic perspective and energetic comedy
timing. Vertical 9:16. Keep character identity, proportions, clothing and important props
consistent across beats. No photorealism, live action, stock photography, anime/manga,
Pixar/3D, glossy mascot art, realistic human photography or old documentary art.""".strip()

SYSTEM = f"""You are the creative director for Relic Loop, a Shorts-only absurd-comedy animation channel.
Create an original fictional micro-story: instant weird situation, fast escalation, visual
cause/effect, unexpected turn, punchline. Do not make a documentary, explainer, listicle,
motivational story or curiosity explainer. Keep it safe and non-graphic.

CANONICAL STYLE:
{STYLE}

IMAGE BEAT RULES:
- Minimum 20 scenes; target 25-40. More than 30 is allowed.
- New image at every meaningful phrase, clause, comma, full stop, tone change, action,
  reaction, reveal or important detail.
- Each beat must show the exact action, expression, object state or consequence being narrated.
- Adjacent beats must visibly differ in action, expression, framing, camera, object state or information.
- Duration must be 0.5-2.5 seconds; total runtime 20-45 seconds.
- No deliberate captions, subtitles, title cards, speech bubbles, logos, watermarks, UI or plot text.

Return ONLY valid JSON. Keep scene descriptions compact. Exact schema:
{{
  "title":"short title",
  "hook":"opening sentence",
  "narration":"55-105 words",
  "characters":[{{"name":"A","role":"...","appearance":"...","personality":"..."}}],
  "locations":["..."],
  "scenes":[{{"number":1,"duration_seconds":1.0,"beat_text":"...","narration":"...","visual_prompt":"stickman shot prompt","camera":"...","action":"...","transition":"..."}}],
  "flow_prompt":"concise full-story style prompt",
  "youtube_description":"...",
  "tags":["...","..."]
}}
"""


def history() -> list[str]:
    try:
        value = json.loads(HISTORY.read_text(encoding="utf-8"))
        return [str(x).strip() for x in value if str(x).strip()][-100:] if isinstance(value, list) else []
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
    depth = 0
    start = None
    quoted = False
    escaped = False
    for i, ch in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
            continue
        if ch == '"':
            quoted = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                start = None
    raise ValueError("Groq did not return valid JSON")


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text or ""))


def validate(obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("title", "hook", "narration", "characters", "locations", "scenes", "flow_prompt"):
        if not obj.get(key):
            errors.append(f"missing:{key}")
    count = words(str(obj.get("narration", "")))
    if not 55 <= count <= 105:
        errors.append(f"narration_words={count}")
    scenes = obj.get("scenes")
    if not isinstance(scenes, list) or not 20 <= len(scenes) <= 45:
        errors.append("scene_count")
        return errors
    runtime = 0.0
    forbidden = re.compile(r"\b(?:caption|subtitle|title card|speech bubble|watermark|logo|typography|written explanation)\b", re.I)
    wrong = re.compile(r"\b(?:photorealistic|live[- ]action|stock photo|anime|manga|pixar|3d render)\b", re.I)
    for i, s in enumerate(scenes, 1):
        if not isinstance(s, dict):
            errors.append(f"scene_{i}_object")
            continue
        try:
            d = float(s.get("duration_seconds", 0))
        except (TypeError, ValueError):
            d = 0
        runtime += d
        if not 0.5 <= d <= 2.5:
            errors.append(f"scene_{i}_duration")
        for field in ("beat_text", "visual_prompt", "camera", "action", "narration"):
            if not str(s.get(field, "")).strip():
                errors.append(f"scene_{i}_{field}")
        prompt = str(s.get("visual_prompt", ""))
        if "stickman" not in prompt.lower():
            errors.append(f"scene_{i}_stickman")
        if forbidden.search(prompt):
            errors.append(f"scene_{i}_text")
        if wrong.search(prompt):
            errors.append(f"scene_{i}_style")
    if not 20 <= runtime <= 45:
        errors.append(f"runtime={runtime:.1f}")
    return errors


def request(client: Groq, messages: list[dict[str, str]], temperature: float) -> str:
    result = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=8000,
        response_format={"type": "json_object"},
    )
    return result.choices[0].message.content or ""


def make_prompt(topic: str | None) -> str:
    old = history()
    used = "No previous topics recorded." if not old else "Previously used titles; do not repeat or make near-duplicates:\n" + "\n".join(f"- {x}" for x in old[-40:])
    seed = f"Use this seed: {topic}." if topic else "Invent a highly shareable absurd premise with an immediate visual hook."
    return f"{seed}\n\n{used}\n\nGenerate the complete Short package. Keep all scene fields compact. Use at least 20 visual beats, preferably 25-40."


def generate(topic: str | None) -> dict[str, Any]:
    if not KEY:
        raise RuntimeError("GROQ_API_KEY is required")
    client = Groq(api_key=KEY)
    last: Exception | None = None
    for attempt in range(3):
        try:
            obj = parse_json(request(client, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": make_prompt(topic)}], 0.9))
            errors = validate(obj)
            if errors:
                repair = "Repair this JSON and return ONLY JSON. Fix every error while preserving the story where possible. Keep at least 20 scenes and keep every field compact. Errors: " + ", ".join(errors) + "\nJSON:\n" + json.dumps(obj, ensure_ascii=False)
                obj = parse_json(request(client, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": repair}], 0.2))
                errors = validate(obj)
            if not errors:
                obj["generated_at"] = datetime.now(timezone.utc).isoformat()
                obj["visual_policy"] = "No deliberate story text in images."
                obj["style_policy"] = STYLE
                return obj
            last = ValueError("validation failed: " + ", ".join(errors))
        except Exception as exc:
            last = exc
            msg = str(exc).lower()
            time.sleep(8 + attempt * 8 if any(x in msg for x in ("429", "rate", "timeout")) else 3 + attempt * 2)
    raise RuntimeError(f"Could not generate a valid Relic Loop Short package: {last}")


def write(obj: dict[str, Any]) -> None:
    json_path = RUN / "relic_loop_short.json"
    flow_path = RUN / "flow_storyboard_prompt.txt"
    manifest_path = RUN / "flow_manifest.json"
    json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    scenes = obj["scenes"]
    lines = ["RELIC LOOP — FAST-CUT CLOUDFLARE IMAGE JOB", "", obj["flow_prompt"], "", "CANONICAL STYLE:", STYLE, "", "CAST:"]
    for c in obj["characters"]:
        lines.append(f"- {c['name']}: {c['appearance']}. Personality: {c['personality']}")
    lines += ["", "LOCATIONS:"] + [f"- {x}" for x in obj["locations"]] + ["", "IMAGE BEATS:"]
    for s in scenes:
        lines += [f"Beat {s['number']} — {s['duration_seconds']}s", f"Beat text: {s['beat_text']}", f"Narration: {s['narration']}", f"Visual: {s['visual_prompt']}", f"Camera: {s['camera']}", f"Action: {s['action']}", f"Transition: {s.get('transition', '')}", ""]
    lines += ["VOICEOVER:", obj["narration"], ""]
    flow_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "type": "relic_loop_fast_cut_image_short",
        "version": 4,
        "title": obj["title"],
        "runtime_seconds": round(sum(float(s["duration_seconds"]) for s in scenes), 2),
        "image_beat_count": len(scenes),
        "minimum_image_beats": 20,
        "target_image_beats": "25-40",
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
