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
RELIC LOOP / CHISPA-PIFIA CANONICAL VISUAL STYLE:
- expressive 2D stickman cartoon illustration for vertical comedy Shorts
- one canonical human protagonist: round white head, thin black limbs, simple tapered torso,
  minimal black facial marks, expressive eyebrows/mouth, readable hands and exaggerated poses
- hand-drawn feel with bold clean black linework and simple graphic shapes
- flat-to-soft 2D shading, restrained vivid palette, crisp silhouettes, strong foreground/background separation
- simple believable environments with enough detail to explain the action without visual clutter
- dynamic wide shots, medium action shots, close reactions, extreme details and dramatic perspective
- every frame shows the exact current story beat in progress; never a generic portrait
- preserve protagonist identity, clothing silhouette, proportions, line weight and recurring props
- vertical 9:16 composition with safe crop margins
- playful absurd comedy, energetic, non-graphic
- NOT photorealistic, NOT live action, NOT stock photography, NOT anime, NOT manga, NOT Pixar/3D
- NO captions, subtitles, title cards, speech bubbles, logos, watermarks, UI overlays or deliberate typography
""".strip()

VISUAL_TEXT_POLICY = """
ZERO STORY TEXT POLICY: generated images must not contain deliberate captions, subtitles,
title cards, speech bubbles, logos, watermarks, labels, UI text, readable signs, written
explanations, letters or numbers added for the story. The joke must work entirely through
visible action, expression, objects and composition. Natural environmental marks may exist
only when tiny, non-readable and non-plot-bearing; the renderer's OCR gate remains authoritative.
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
      "duration_seconds": 1.0,
      "beat_text": "one tiny narration beat",
      "narration": "exact narration for this beat",
      "visual_prompt": "detailed 2D stickman image prompt showing this exact action",
      "camera": "specific framing/composition",
      "action": "exact visible action or reaction",
      "transition": "what changes into the next beat"
    }
  ],
  "flow_prompt": "complete prompt describing the same cast and visual style",
  "youtube_description": "short description",
  "tags": ["...", "..."]
}
"""

SYSTEM = f"""You are the creative director for Relic Loop, the production system behind Chispa-Pifia's absurd stickman comedy Shorts.
Create original, highly visual micro-stories for 20-45 second vertical videos.

CORE FORMAT:
1. HOOK: the first 1-2 seconds must show a strange visual problem immediately.
2. GOAL: establish what the protagonist wants.
3. BAD DECISION: give them a simple, understandable but foolish solution.
4. CONSEQUENCE: show the direct result of that decision.
5. ESCALATION: each new problem must be caused by the previous one.
6. TWIST: introduce a reveal that is surprising but supported by an earlier detail.
7. PUNCHLINE: land the joke visually and end immediately; no outro or moral.

RETENTION RULES:
- Do not begin with greetings, explanations, setup narration or empty establishing shots.
- Introduce a new action, reaction, consequence, object state, reveal or useful story detail every 1-2 seconds.
- Prefer 18-28 meaningful visual beats. Never pad the story with redundant images.
- Most beats should be 0.7-1.8 seconds. A deliberate reaction/reveal may reach 2.5 seconds.
- Every adjacent beat must visibly change action, expression, framing, object state or story information.
- Use close reaction shots after major failures and before the punchline.
- The final beat must visually contain the payoff.
- The story must be understandable with audio muted; images cannot depend on captions.

WRITING RULES:
- Original absurd causal-chain comedy, not generic motivational content, explainers, listicles or documentaries.
- Avoid repetitive POV/mom/falling-asleep/phone-at-1%-style premises unless transformed into a genuinely original mechanism.
- Prefer concrete physical problems that create escalating visual consequences.
- Keep stories safe and non-graphic.

{STYLE}

{VISUAL_TEXT_POLICY}

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
        "These titles/topics have already been used. Do NOT repeat them or create near-duplicates:\n" + recent
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
    word_count = _words(narration)
    if not 55 <= word_count <= 105:
        errors.append(f"narration_words={word_count}")

    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not 18 <= len(scenes) <= 28:
        errors.append(f"scene_count={len(scenes) if isinstance(scenes, list) else 'invalid'}")
        return errors

    total = 0.0
    forbidden = re.compile(
        r"\b(?:caption|subtitle|title card|watermark|speech bubble|typography|logo|written explanation|text overlay|readable text|letters|numbers)\b",
        re.I,
    )
    previous_action = ""
    narration_parts: list[str] = []

    for i, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            errors.append(f"scene_{i}_not_object")
            continue
        try:
            d = float(scene.get("duration_seconds", 0) or 0)
        except (TypeError, ValueError):
            d = 0
        total += d
        if not 0.5 <= d <= 2.5:
            errors.append(f"scene_{i}_duration")
        for field in ("beat_text", "visual_prompt", "camera", "action", "transition", "narration"):
            if not str(scene.get(field, "")).strip():
                errors.append(f"scene_{i}_missing_{field}")
        vp = str(scene.get("visual_prompt", ""))
        action = str(scene.get("action", ""))
        beat_text = str(scene.get("beat_text", ""))
        narration_parts.append(str(scene.get("narration", "")))
        if forbidden.search(vp):
            errors.append(f"scene_{i}_text_dependency")
        if re.search(r"\b(?:photorealistic|live[- ]action|stock photo|anime|pixar|3d render|realistic human)\b", vp, re.I):
            errors.append(f"scene_{i}_wrong_visual_style")
        if "stickman" not in vp.lower():
            errors.append(f"scene_{i}_missing_stickman")
        if i == 1 and not re.search(r"\b(?:immediately|suddenly|already|stuck|falling|running|exploding|flying|crashing|chasing|hanging|covered|trapped|racing|escaping)\b", action, re.I):
            errors.append("weak_first_visual_hook")
        if i > 1 and action.strip().lower() == previous_action.strip().lower():
            errors.append(f"scene_{i}_duplicate_action")
        previous_action = action

    if not 20 <= total <= 45:
        errors.append(f"runtime={total:.1f}")

    scene_text = " ".join(narration_parts).strip()
    if scene_text != narration.strip():
        errors.append("beat_narration_mismatch")
    if scenes and scenes[-1].get("number") != len(scenes):
        errors.append("scene_numbering")
    return errors


def _prompt_for(topic: str | None = None) -> str:
    history = _topic_context()
    beat_instruction = (
        "Build exactly 18-28 meaningful beats. Give the first 2-3 beats an immediate visual hook, "
        "then use short action/reaction/consequence beats. Do not split every comma. Every beat must "
        "change what is visibly happening. Make the final 1-2 beats the payoff and stop immediately."
    )
    if topic:
        return f"Create an original absurd Relic Loop Short around this seed: {topic}.\n\n{history}\n\n{beat_instruction}\n\n{SYSTEM}"
    return f"Invent ONE original absurd Relic Loop Short premise that is visually ridiculous and easy to understand in a scrolling feed. Avoid all used topics and near-duplicates.\n\n{history}\n\n{beat_instruction}\n\n{SYSTEM}"


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
                temperature=0.92,
                max_completion_tokens=7000,
            )
            data = _extract_json(response.choices[0].message.content or "")
            errors = _validate(data)
            if errors:
                repair = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {
                            "role": "user",
                            "content": "Repair this JSON so it passes EVERY validation requirement. Do not merely remove errors; improve weak hooks, duplicate beats and causal continuity. Return ONLY JSON.\nErrors: " + ", ".join(errors) + "\nJSON:\n" + json.dumps(data, ensure_ascii=False),
                        },
                    ],
                    temperature=0.18,
                    max_completion_tokens=7000,
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
        "version": 4,
        "title": data["title"],
        "runtime_seconds": round(sum(float(s["duration_seconds"]) for s in scenes), 2),
        "image_beat_count": len(scenes),
        "minimum_image_beats": 18,
        "target_image_beats": "18-28 meaningful beats",
        "flow_template": "Storyboard Studio",
        "zero_text_gate": True,
        "text_policy": "reject_story_text_and_ocr_suspects",
        "media_generation": "cloudflare_images",
        "beat_policy": "causal_action_reaction_consequence_reveal_punchline",
        "quality_gate": "strict_storyboard_validation_before_image_quota_spend",
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
