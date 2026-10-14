"""Relic Loop Shorts-first story/image engine."""
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
STATE_DIR = ROOT / "state"
CURRENT_RUN = STATE_DIR / "current_run"
TOPIC_HISTORY = STATE_DIR / "relic_loop_topic_history.json"
WORK_DIR = ROOT / "run_work"
CURRENT_RUN.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("GROQ_WRITER_MODEL", "openai/gpt-oss-120b")
API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

STYLE = """
RELIC LOOP — CANONICAL CLOUDFLARE IMAGE STYLE:
- expressive 2D stickman cartoon illustration for YouTube Shorts
- hand-drawn feel with bold, clean black linework and simple graphic shapes
- stickman characters have simple heads/bodies/limbs but highly expressive faces,
  eyes, mouths, eyebrows, hands and poses; body language carries the comedy
- flat-to-soft 2D shading, crisp readable silhouettes, strong foreground/background separation
- colorful but controlled cartoon palette; simple believable environments with enough detail
  to explain the action without becoming visually cluttered
- dynamic compositions, exaggerated perspective, dramatic close-ups, wide shots,
  reaction shots, action poses and visual cause/effect
- every image must SHOW the exact beat being narrated; never use a generic portrait
  or a character simply looking at camera
- preserve character identity, clothing, proportions and recurring props across beats
- vertical 9:16 composition, designed for fast-cut Shorts editing
- playful absurd comedy, energetic and visually punchy, non-graphic
- NOT photorealistic, NOT live action, NOT stock photography, NOT anime,
  NOT Pixar/3D, NOT realistic human photography, NOT old documentary art
- NO captions, subtitles, title cards, speech bubbles, logos, watermarks,
  UI overlays or deliberate typography
- incidental environmental writing may exist only if naturally part of the setting
  and never carries the joke or plot
""".strip()

VISUAL_TEXT_POLICY = """
NO STORY TEXT IN IMAGES: Cloudflare image prompts must not request captions,
subtitles, title cards, speech bubbles, watermarks, logos, labels, UI text or
written explanations. The visual must communicate through action, expression,
objects, composition and cause/effect. Incidental real-world environmental writing
is allowed only when it naturally belongs in the scene and is not plot-bearing.
""".strip()

SCHEMA_PROMPT = """
Return ONLY valid JSON. No markdown and no commentary.
Schema:
{
  "title": "short original absurd title",
  "hook": "one punchy opening narration sentence",
  "narration": "complete voiceover for a 20-45 second Short",
  "characters": [
    {"name":"...", "role":"...", "appearance":"...", "personality":"..."}
  ],
  "locations": ["..."],
  "scenes": [
    {
      "number": 1,
      "duration_seconds": 1.2,
      "beat_text": "the exact tiny narration beat, usually one phrase/clause",
      "narration": "exact narration for this beat",
      "visual_prompt": "detailed Cloudflare-ready 2D stickman image prompt",
      "camera": "camera/composition for this single image",
      "action": "the exact visible action or reaction",
      "transition": "immediate visual change into the next beat"
    }
  ],
  "flow_prompt": "one complete prompt describing the same cast and visual style",
  "youtube_description": "short description",
  "tags": ["...", "..."]
}
"""

SYSTEM = f"""You are the creative director for Relic Loop, a Shorts-only YouTube animation channel.
Create original absurd/funny micro-stories designed for retention. Start immediately,
escalate quickly, contain a clear visual joke or unexpected turn, and end with a punchline.
Do not make documentaries, explainers, listicles, curiosity explainers, or motivational stories.
Keep it safe, non-graphic, and visually understandable without subtitles or captions.

{STYLE}

{VISUAL_TEXT_POLICY}

FAST-CUT IMAGE REQUIREMENTS:
- This is an IMAGE-BASED Short, not a small number of long shots.
- REQUIRED: minimum 20 visual beats per Short.
- TARGET: 25-40 visual beats per Short whenever the narration supports it.
- ALLOWED: more than 30 beats; do NOT cap a Short at 30 or 24 beats.
- A new image should be created at EVERY meaningful phrase, clause, comma, full stop,
  sudden tone change, reaction, action change, reveal, or important detail.
- If a sentence contains several distinct actions or details, split them into separate beats.
- Do NOT hold one image for several unrelated phrases.
- Prefer many short beats over 6-10 long scenes.
- Typical image duration: 0.6-1.8 seconds; use up to 2.5 seconds only when a beat genuinely needs it.
- Two adjacent beats must differ visibly in action, expression, framing, object state,
  camera angle, or story information. If the narration changes, the visual should change too.
- The first 1-2 seconds must contain a strong visual hook.
- Tiny details matter: if narration mentions an object, expression, movement, location,
  reaction, or consequence, show it in the corresponding image.
- Use extreme close-ups and reaction shots when a phrase calls for them.
- Keep recurring characters visually consistent across all images.
- Every image prompt must describe the exact action for that beat, not a generic scene.

{SCHEMA_PROMPT}
"""


def _load_topic_history() -> list[str]:
    try:
        data = json.loads(TOPIC_HISTORY.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()][-100:]
    except Exception:
        pass
    return []


def _save_topic_history(history: list[str]) -> None:
    TOPIC_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    TOPIC_HISTORY.write_text(json.dumps(history[-100:], ensure_ascii=False, indent=2), encoding="utf-8")


def _topic_context() -> str:
    history = _load_topic_history()
    if not history:
        return "No previous Relic Loop topics are recorded yet."
    recent = "\n".join(f"- {x}" for x in history[-40:])
    return (
        "These topics/titles have already been used by Relic Loop. Do NOT repeat them, "
        "and do not make a near-duplicate that merely changes a few words:\n" + recent
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    candidates = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:i + 1])
                start = None
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    raise ValueError("Groq did not return valid JSON")


def _words(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text or ""))


def _validate(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ("title", "hook", "narration", "characters", "locations", "scenes", "flow_prompt")
    for key in required:
        if not data.get(key):
            errors.append(f"missing:{key}")
    narration = str(data.get("narration", ""))
    if not 55 <= _words(narration) <= 115:
        errors.append(f"narration_words={_words(narration)}")
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not 20 <= len(scenes) <= 45:
        errors.append("scene_count_fast_cut")
        return errors
    total = 0.0
    forbidden = re.compile(r"\b(?:caption|subtitle|title card|watermark|speech bubble|typography|logo|written explanation)\b", re.I)
    for i, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            errors.append(f"scene_{i}_not_object")
            continue
        d = float(scene.get("duration_seconds", 0) or 0)
        total += d
        if not 0.5 <= d <= 2.5:
            errors.append(f"scene_{i}_duration")
        for field in ("beat_text", "visual_prompt", "camera", "action", "narration"):
            if not scene.get(field):
                errors.append(f"scene_{i}_missing_{field}")
        vp = str(scene.get("visual_prompt", ""))
        if forbidden.search(vp):
            errors.append(f"scene_{i}_text_dependency")
        if re.search(r"\b(?:photorealistic|live[- ]action|stock photo|portrait of a smiling man|anime|pixar|3d render)\b", vp, re.I):
            errors.append(f"scene_{i}_wrong_visual_style")
        if "stickman" not in vp.lower():
            errors.append(f"scene_{i}_missing_stickman")
    if not 20 <= total <= 45:
        errors.append(f"runtime={total:.1f}")
    return errors


def _prompt_for(topic: str | None = None) -> str:
    history = _topic_context()
    if topic:
        return f"Create an original absurd Relic Loop Short around this seed: {topic}.\n\n{history}\n\nBreak the narration into at least 20 immediate visual beats, preferably 25-40, at phrases, commas, full stops, tone changes and reactions. Do not cap the beat count at 24 or 30.\n\n{SYSTEM}"
    return f"Invent a highly shareable absurd Relic Loop Short premise with an immediate hook and strong visual punchline. Avoid overused meme formats and ALL previously used topics.\n\n{history}\n\nBreak the narration into at least 20 immediate visual beats, preferably 25-40, at phrases, commas, full stops, tone changes and reactions. Do not cap the beat count at 24 or 30.\n\n{SYSTEM}"


def generate(topic: str | None = None) -> dict[str, Any]:
    if not API_KEY:
        raise RuntimeError("GROQ_API_KEY is required")
    client = Groq(api_key=API_KEY)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": _prompt_for(topic)},
                ],
                temperature=0.95,
                max_completion_tokens=6000,
            )
            data = _extract_json(response.choices[0].message.content or "")
            errors = _validate(data)
            if errors:
                repair = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": "Repair this JSON so it passes every requirement. Return ONLY JSON.\nErrors: " + ", ".join(errors) + "\nJSON:\n" + json.dumps(data, ensure_ascii=False)},
                    ],
                    temperature=0.25,
                    max_completion_tokens=6000,
                )
                data = _extract_json(repair.choices[0].message.content or "")
                errors = _validate(data)
            if not errors:
                data["generated_at"] = datetime.now(timezone.utc).isoformat()
                data["visual_policy"] = VISUAL_TEXT_POLICY
                data["style_policy"] = STYLE
                return data
            last_error = ValueError("Short package validation failed: " + ", ".join(errors))
        except Exception as exc:
            last_error = exc
            if "429" in str(exc) or "rate" in str(exc).lower():
                time.sleep(8 + attempt * 8)
            else:
                time.sleep(2 + attempt * 2)
    raise RuntimeError(f"Could not generate a valid Relic Loop Short package: {last_error}")


def write_package(data: dict[str, Any]) -> None:
    json_path = CURRENT_RUN / "relic_loop_short.json"
    flow_path = CURRENT_RUN / "flow_storyboard_prompt.txt"
    manifest_path = CURRENT_RUN / "flow_manifest.json"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    scenes = data["scenes"]
    lines = [
        "RELIC LOOP — FAST-CUT CLOUDFLARE IMAGE JOB",
        "", data["flow_prompt"], "",
        "CANONICAL STYLE:", STYLE, "",
        "VISUAL TEXT POLICY:", VISUAL_TEXT_POLICY, "",
        "CAST:",
    ]
    for c in data["characters"]:
        lines.append(f"- {c['name']}: {c['appearance']}. Personality: {c['personality']}")
    lines += ["", "LOCATIONS:"] + [f"- {x}" for x in data["locations"]]
    lines += ["", "IMAGE BEATS:"]
    for s in scenes:
        lines += [
            f"Beat {s['number']} — {s['duration_seconds']}s",
            f"Beat text: {s['beat_text']}",
            f"Narration: {s['narration']}",
            f"Visual: {s['visual_prompt']}",
            f"Camera: {s['camera']}",
            f"Action: {s['action']}",
            f"Transition: {s['transition']}",
            "",
        ]
    lines += ["VOICEOVER:", data["narration"], ""]
    flow_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "type": "relic_loop_fast_cut_image_short",
        "version": 3,
        "title": data["title"],
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

    history = _load_topic_history()
    title = str(data.get("title", "")).strip()
    if title and title.lower() not in {x.lower() for x in history}:
        history.append(title)
        _save_topic_history(history)

    print(f"[IMAGE] Ready: {flow_path}")
    print(f"[IMAGE] JSON:  {json_path}")
    print(f"[IMAGE] Beats: {len(scenes)}")


def main() -> None:
    topic = os.environ.get("RELIC_LOOP_TOPIC", "").strip() or None
    data = generate(topic)
    write_package(data)
    print(f"[SHORTS] {data['title']}")
    print(f"[SHORTS] {_words(data['narration'])} narration words")
    print(f"[SHORTS] {len(data['scenes'])} image beats / {sum(float(s['duration_seconds']) for s in data['scenes']):.1f}s")


if __name__ == "__main__":
    main()
