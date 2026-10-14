"""Relic Loop Shorts-first story/Flow handoff engine.

This module replaces the old long-form-first planning layer with direct,
20-45 second absurd-story Shorts. It does not automate or bypass Google Flow;
it produces a structured Flow Storyboard Studio package that can be pasted into
an eligible Flow project.
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
STATE_DIR = ROOT / "state"
CURRENT_RUN = STATE_DIR / "current_run"
WORK_DIR = ROOT / "run_work"
CURRENT_RUN.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("GROQ_WRITER_MODEL", "openai/gpt-oss-120b")
API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

STYLE = """
Relic Loop Shorts visual identity:
- modern expressive stickman animation, clean polished 2D illustration
- simple but appealing environments, strong silhouettes and readable poses
- exaggerated facial expressions, physical comedy and clear visual cause/effect
- energetic camera movement: push-ins, whip pans, quick reveals, reaction shots
- consistent character design across every scene
- cinematic composition adapted to vertical 9:16 Shorts
- playful absurd comedy, not horror and not graphic violence
- no photorealism, no anime, no 3D/Pixar look, no old documentary look
- visuals must communicate the joke/action without written language
""".strip()

NO_TEXT = """
ABSOLUTE ZERO-TEXT VISUAL RULE: generated visuals must contain NO letters, words,
numbers, captions, subtitles, titles, logos, watermarks, labels, signs, posters,
phone/computer UI text, speech bubbles, handwriting, book/page writing, fake text,
or pseudo-glyphs. If an object normally contains writing, make its surface blank,
crop the writing away, turn it away from camera, or replace it with a purely visual
symbol. The visual validator will reject text-bearing frames.
""".strip()

SCHEMA_PROMPT = """
Return ONLY valid JSON. No markdown and no commentary.
Schema:
{
  "title": "short curiosity/absurd title",
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
      "visual_prompt": "detailed visual-only prompt",
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
Do not make it a documentary, explainer, listicle, or generic motivational story.
Keep it safe, non-graphic, and visually understandable without text.

{STYLE}

{NO_TEXT}

Short requirements:
- total narration target: 55-105 words
- total runtime target: 20-45 seconds
- 6-10 scenes, usually 2-5 seconds each
- each scene must materially change the action, framing, reaction, or visual information
- recurring characters must keep the same appearance in every scene
- narration must match what is visibly happening
- do not rely on subtitles or on-screen writing
- avoid brand names and copyrighted characters

{SCHEMA_PROMPT}
"""


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
    # Salvage the last balanced object.
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
    forbidden = re.compile(r"\b(?:caption|subtitle|title card|logo|watermark|text|letter|word|signage|sign with|written|headline|label|UI|interface|screen text)\b", re.I)
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
        if forbidden.search(str(scene.get("visual_prompt", ""))):
            errors.append(f"scene_{i}_text_dependency")
    if not 20 <= total <= 45:
        errors.append(f"runtime={total:.1f}")
    return errors


def _prompt_for(topic: str | None = None) -> str:
    if topic:
        return f"Create an original absurd Relic Loop Short around this seed: {topic}. Make the situation escalate into an unexpected visual punchline.\n\n{SYSTEM}"
    return f"Invent a highly shareable absurd Relic Loop Short premise with an immediate hook and a strong visual punchline. Avoid overused meme formats.\n\n{SYSTEM}"


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
            if not errors:
                data["generated_at"] = datetime.now(timezone.utc).isoformat()
                data["visual_policy"] = NO_TEXT
                data["style_policy"] = STYLE
                return data
            # Ask for a compact deterministic repair rather than failing the run.
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
                data["visual_policy"] = NO_TEXT
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
        "RELlC LOOP — FLOW STORYBOARD STUDIO JOB",
        "", data["flow_prompt"], "",
        "CANONICAL STYLE:", STYLE, "",
        "ZERO-TEXT POLICY:", NO_TEXT, "",
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
        "version": 1,
        "title": data["title"],
        "runtime_seconds": round(sum(float(s["duration_seconds"]) for s in scenes), 2),
        "scene_count": len(scenes),
        "flow_template": "Storyboard Studio",
        "zero_text_gate": True,
        "media_generation": "manual_in_flow",
        "files": [str(json_path), str(flow_path)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
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
