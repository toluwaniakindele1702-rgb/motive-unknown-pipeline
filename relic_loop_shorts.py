"""Relic Loop Shorts-first story/Flow handoff engine."""
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
Relic Loop Shorts visual identity:
- modern expressive 2D stickman animation, hand-drawn/cartoon illustration feel
- bold clean linework, flat-to-soft 2D shading, graphic shapes, readable silhouettes
- simple but believable environments designed for animation
- highly expressive faces, poses, body language and physical comedy
- characters are clearly stickman/cartoon figures, NOT photographs or photoreal humans
- energetic camera movement: push-ins, whip pans, quick reveals, reaction shots
- strong visual cause/effect; every shot should visibly explain or advance the joke
- consistent character design across every scene
- cinematic vertical 9:16 composition
- playful absurd comedy, not horror and not graphic violence
- no photorealism, live-action photography, stock-photo appearance, anime, Pixar/3D look,
  old documentary illustration, generic smiling portraits, or static headshots
""".strip()

# We no longer impose the old OCR-style absolute zero-text wall on Flow-ready prompts.
# Flow is a stronger visual generator. We still prohibit captions/subtitles/typography as
# storytelling crutches, while allowing incidental/environmental writing when naturally
# present in a scene. The downstream Flow prompt should prioritize visual storytelling.
VISUAL_TEXT_POLICY = """
VISUAL TEXT POLICY: Do not use captions, subtitles, title cards, speech bubbles,
watermarks, logos, or typography as part of the storytelling. Do not ask for written
explanations inside the image. Natural incidental environmental writing is allowed
when it genuinely belongs in the setting, but it must never carry the joke or plot.
Communicate the story primarily through characters, actions, objects, expressions,
composition, camera movement and visual cause/effect.
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
      "duration_seconds": 3.5,
      "narration": "exact narration for this beat",
      "visual_prompt": "detailed 2D stickman visual prompt",
      "camera": "camera movement/composition",
      "action": "what visibly happens",
      "transition": "how this shot moves into the next"
    }
  ],
  "flow_prompt": "one complete prompt for Flow Storyboard Studio that can create the cast and storyboard from this package",
  "youtube_description": "short description",
  "tags": ["...", "..."]
}
"""

SYSTEM = f"""You are the creative director for Relic Loop, a Shorts-only YouTube animation channel.
Create original absurd/funny micro-stories designed for retention. The story should start immediately,
escalate quickly, contain a clear visual joke or unexpected turn, and end with a satisfying punchline/payoff.
Do not make it a documentary, explainer, listicle, curiosity explainer, or generic motivational story.
Keep it safe, non-graphic, and visually understandable without subtitles or captions.

{STYLE}

{VISUAL_TEXT_POLICY}

Short requirements:
- total narration target: 55-105 words
- total runtime target: 20-45 seconds
- 6-10 scenes, usually 2-5 seconds each
- each scene must materially change the action, framing, reaction, or visual information
- recurring characters must keep the same appearance in every scene
- narration must match what is visibly happening
- every scene must SHOW the action rather than merely showing a person smiling at camera
- avoid brand names and copyrighted characters

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
    if not isinstance(scenes, list) or not 6 <= len(scenes) <= 10:
        errors.append("scene_count")
        return errors
    total = 0.0
    forbidden = re.compile(r"\b(?:caption|subtitle|title card|watermark|speech bubble|typography)\b", re.I)
    for i, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            errors.append(f"scene_{i}_not_object")
            continue
        d = float(scene.get("duration_seconds", 0) or 0)
        total += d
        if not 2.0 <= d <= 6.0:
            errors.append(f"scene_{i}_duration")
        for field in ("visual_prompt", "camera", "action", "narration"):
            if not scene.get(field):
                errors.append(f"scene_{i}_missing_{field}")
        vp = str(scene.get("visual_prompt", ""))
        if forbidden.search(vp):
            errors.append(f"scene_{i}_text_dependency")
        if re.search(r"\b(?:photorealistic|live[- ]action|stock photo|portrait of a smiling man)\b", vp, re.I):
            errors.append(f"scene_{i}_wrong_visual_style")
    if not 20 <= total <= 45:
        errors.append(f"runtime={total:.1f}")
    return errors


def _prompt_for(topic: str | None = None) -> str:
    history = _topic_context()
    if topic:
        return f"Create an original absurd Relic Loop Short around this seed: {topic}.\n\n{history}\n\nMake the situation escalate into an unexpected visual punchline.\n\n{SYSTEM}"
    return f"Invent a highly shareable absurd Relic Loop Short premise with an immediate hook and a strong visual punchline. Avoid overused meme formats and ALL previously used topics.\n\n{history}\n\n{SYSTEM}"


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
                max_completion_tokens=2400,
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
                    max_completion_tokens=2400,
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
        "RELIC LOOP — FLOW STORYBOARD STUDIO JOB",
        "", data["flow_prompt"], "",
        "CANONICAL STYLE:", STYLE, "",
        "VISUAL TEXT POLICY:", VISUAL_TEXT_POLICY, "",
        "CAST:",
    ]
    for c in data["characters"]:
        lines.append(f"- {c['name']}: {c['appearance']}. Personality: {c['personality']}")
    lines += ["", "LOCATIONS:"] + [f"- {x}" for x in data["locations"]]
    lines += ["", "SCENES:"]
    for s in scenes:
        lines += [
            f"Scene {s['number']} — {s['duration_seconds']}s",
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
        "type": "relic_loop_flow_storyboard",
        "version": 2,
        "title": data["title"],
        "runtime_seconds": round(sum(float(s["duration_seconds"]) for s in scenes), 2),
        "scene_count": len(scenes),
        "flow_template": "Storyboard Studio",
        "zero_text_gate": False,
        "text_policy": "no_deliberate_story_text",
        "media_generation": "manual_in_flow",
        "files": [str(json_path), str(flow_path)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Record the title/topic only after the package has passed validation.
    history = _load_topic_history()
    title = str(data.get("title", "")).strip()
    if title and title.lower() not in {x.lower() for x in history}:
        history.append(title)
        _save_topic_history(history)

    print(f"[FLOW] Ready: {flow_path}")
    print(f"[FLOW] JSON:  {json_path}")


def main() -> None:
    topic = os.environ.get("RELIC_LOOP_TOPIC", "").strip() or None
    data = generate(topic)
    write_package(data)
    print(f"[SHORTS] {data['title']}")
    print(f"[SHORTS] {_words(data['narration'])} narration words")
    print(f"[SHORTS] {len(data['scenes'])} scenes / {sum(float(s['duration_seconds']) for s in data['scenes']):.1f}s")


if __name__ == "__main__":
    main()
