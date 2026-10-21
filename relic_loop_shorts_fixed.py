"""Chispa-Pifia story generator: meaningful 45-60 second absurd comedy."""
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

STYLE = (
    "Modern expressive 2D hand-drawn stickman cartoon: round head, simple tapered torso, "
    "thin expressive limbs, clean black ink linework, minimal eyes, expressive mouth and eyebrows, "
    "readable hands, exaggerated poses, flat-to-soft 2D shading, restrained color, clear silhouettes, "
    "believable simple environments, dynamic perspective, vertical 9:16. Consistent human protagonist identity. "
    "No photorealism, live action, stock photo, anime, manga, 3D/Pixar, glossy mascot or documentary art."
)

SYSTEM = f"""You are the head writer and visual storyboard director for Chispa-Pifia, an absurd-comedy stickman Shorts channel.

Create ONE ORIGINAL, MEANINGFUL fictional story centered on a normal HUMAN PROTAGONIST.
The comedy must come from a believable human goal becoming increasingly ridiculous through the protagonist's own decisions.
The story must feel like a real mini-story, not a list of random jokes.

CORE STORY ENGINE — NON-NEGOTIABLE:
1. A relatable human problem or goal.
2. The protagonist makes a clear decision.
3. That decision causes a concrete consequence.
4. The protagonist reacts and makes another decision.
5. The new decision makes the problem worse.
6. A second escalation follows logically.
7. A surprising but foreshadowed twist changes how the earlier events are understood.
8. A short, funny final payoff ends the story.

MEANING TEST:
Before writing, silently explain the story in ONE sentence using cause and effect.
If you cannot explain why each major event happens, reject the idea and create a better one.
The viewer should be able to summarize the story afterward and understand what the protagonist was trying to accomplish.
Absurd does NOT mean meaningless. Every major event must have a reason.

LENGTH — HARD LIMIT:
The complete spoken narration MUST be 105-145 words, targeting roughly 120-135 words.
This should normally produce about 45-60 seconds of natural narration.
Do not pad the story. Do not rush it to hit a word count. If the idea cannot support a meaningful 45-60 second story, discard it and create a stronger premise.

STORY QUALITY:
- Start with an immediate concrete hook.
- Give the protagonist a clear goal that matters in the situation.
- Establish enough context that the viewer understands the stakes.
- Use 3-5 meaningful cause-and-effect escalations.
- Give the protagonist agency: their decisions should drive the story.
- Keep the world mostly normal until the absurdity grows naturally.
- Make the twist connected to something established earlier.
- End immediately after the strongest payoff.
- Natural conversational spoken English.
- No moral, lesson, explanation, generic outro or "subscribe" line.
- The story must still make sense if all visuals are removed.

GOOD PREMISE PATTERN:
A normal person wants something simple, makes a shortcut or bad decision, creates a practical problem, tries to repair it, creates a bigger practical problem, keeps improvising, and finally discovers an absurd consequence that was logically hiding in the original situation.

GOOD EXAMPLES OF PREMISE TYPES:
- returning something before a deadline
- trying to fix a small mistake before someone notices
- trying to save money and accidentally creating a much bigger expense
- trying to arrive somewhere on time after making one bad shortcut
- trying to impress someone by pretending to know what he is doing
- trying to recover a lost item without admitting he lost it
- trying to finish a simple job before a deadline
- trying to hide an embarrassing mistake while fixing it

ABSURDITY RULE:
The audience should think: "That is ridiculous, but I can see how one stupid decision caused the next thing."
Never make the audience think: "The writer just put a random thing here because it is weird."

ABSOLUTE NO-RANDOMNESS RULES:
- No random animals appearing as gags.
- No random dinosaurs, llamas, cats, dogs, monkeys or birds unless an animal is genuinely required by the established story.
- No objects suddenly becoming alive, talking, running, attacking or forming armies.
- No random livestreams, memes, websites, videos or digital jokes appearing without causal setup.
- No unrelated explosions, city-wide chaos or disasters just for spectacle.
- No "and then suddenly" events unless the sudden event is a direct consequence of the previous action.
- No meaningless surrealism.
- No object protagonist.
- No story whose only joke is "something impossible happened."
- No disconnected sequence of funny images.
- No generic "have you ever wondered" intro.

TWIST RULE:
The final twist must be seeded by an earlier detail, action, rule, person, object, deadline or misunderstanding.
Do not invent a completely new fact in the final sentence just to create a twist.
The viewer should be able to look back and say, "Oh. That's why that happened."

VISUAL TRUTH:
Narration is the source of truth. Every image is a literal reenactment of one meaningful moment in the narration.
Each visual beat must answer: WHO is there, WHERE are they, WHAT are they doing, and WHY is it happening now?
Never invent visual jokes that are absent from the story.

STORY-FIRST TIMING:
There is NO fixed image count. Create one beat whenever the story reaches a meaningful visual change: new action, reaction, consequence, location, object state, reveal or punchline detail.
Do not split every comma into an image. Do not combine distinct actions merely to reduce image count.
The narration is the master timeline. Scene narration fields must be contiguous exact excerpts from the master narration, in order, with no missing or duplicated words.
Final assembly uses the actual neural narration duration; scene duration values are only estimates.

CHARACTERS:
- The protagonist is a normal human represented as a stickman.
- The protagonist must actively make decisions.
- Other humans may appear only when the story requires them.
- Keep character roles and appearances consistent across scenes.

VISUAL STYLE: {STYLE}

SCENE RULES:
- Every scene describes exactly one meaningful moment.
- Every scene contains the protagonist or a directly relevant visible consequence.
- beat_text, narration, visual_prompt and action describe the same event.
- visual_prompt explicitly includes "human stickman" or "stickman protagonist".
- No captions, subtitles, title cards, speech bubbles, logos, watermarks, UI or plot text in images.

Return ONLY valid JSON with this shape:
{{
  "title":"...",
  "hook":"...",
  "narration":"complete spoken narration",
  "characters":[{{"name":"A","role":"human protagonist","appearance":"..."}}],
  "locations":["..."],
  "story_structure":{{
    "problem":"...",
    "goal":"...",
    "causal_chain":["problem -> decision -> consequence -> decision -> escalation -> twist -> payoff"],
    "escalations":["...","...","..."],
    "twist":"...",
    "punchline":"..."
  }},
  "scenes":[{{
    "number":1,
    "duration_seconds":3.0,
    "beat_text":"literal event happening now",
    "narration":"CONTIGUOUS EXACT EXCERPT FROM FULL NARRATION",
    "visual_prompt":"literal visible shot of the human stickman protagonist, no interpretation",
    "camera":"...",
    "action":"exact visible action"
  }}],
  "flow_prompt":"...",
  "youtube_description":"...",
  "tags":["...","..."]
}}
"""


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
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start:end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("Groq did not return valid JSON")


def words(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text or ""))


def normalize_scenes(obj: dict[str, Any]) -> None:
    scenes = obj.get("scenes")
    if not isinstance(scenes, list):
        return
    for i, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            continue
        scene["number"] = i
        prompt = str(scene.get("visual_prompt", "")).strip()
        if prompt and "stickman" not in prompt.lower():
            scene["visual_prompt"] = "human stickman protagonist " + prompt


def _norm_text(text: str) -> str:
    return " ".join(str(text or "").split())


def validate(obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ("title", "hook", "narration", "characters", "locations", "story_structure", "scenes", "flow_prompt")
    for key in required:
        if not obj.get(key):
            errors.append(f"missing:{key}")

    narration = _norm_text(str(obj.get("narration", "")))
    wc = words(narration)
    if not 105 <= wc <= 145:
        errors.append(f"narration_words={wc}_hard_limit_105_145")

    chars = obj.get("characters")
    if not isinstance(chars, list) or not chars:
        errors.append("no_characters")
    else:
        first = chars[0] if isinstance(chars[0], dict) else {}
        role = str(first.get("role", "")).lower()
        if "human" not in role and "person" not in role and "protagonist" not in role:
            errors.append("protagonist_not_human")

    structure = obj.get("story_structure")
    if isinstance(structure, dict):
        for key in ("problem", "goal", "causal_chain", "escalations", "twist", "punchline"):
            if not structure.get(key):
                errors.append(f"missing_structure:{key}")
        chain = structure.get("causal_chain")
        if isinstance(chain, list) and len(chain) < 4:
            errors.append("causal_chain_too_short")
        escalations = structure.get("escalations")
        if isinstance(escalations, list) and len(escalations) < 3:
            errors.append("too_few_meaningful_escalations")
    else:
        errors.append("story_structure")

    bad_random_patterns = [
        r"random (?:cat|dog|monkey|bird|llama|dinosaur)\b",
        r"random .*\b(?:livestream|yoga|meme|website)\b",
        r"(?:phone|chair|fridge|refrigerator|toilet|pizza|sandwich|coffee) .*\b(?:run|runs|walk|walks|talk|talks|attack|attacks|dance|dances|fight|fights)\b",
        r"(?:objects?|mugs?|chairs?|phones?) .*\bform (?:an )?(?:army|band|crowd)\b",
    ]
    combined_story = narration + " " + json.dumps(structure, ensure_ascii=False) if isinstance(structure, dict) else narration
    for pattern in bad_random_patterns:
        if re.search(pattern, combined_story.lower()):
            errors.append("random_object_or_creature_or_digital_nonsense")
            break

    scenes = obj.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        errors.append("no_visual_beats")
        return errors

    reconstructed = " ".join(_norm_text(str(s.get("narration", ""))) for s in scenes if isinstance(s, dict))
    if reconstructed != narration:
        errors.append("scene_narration_does_not_exactly_partition_master_narration")

    runtime = 0.0
    for i, scene in enumerate(scenes, 1):
        if not isinstance(scene, dict):
            errors.append(f"scene_{i}_object")
            continue
        try:
            duration = float(scene.get("duration_seconds", 0))
        except (TypeError, ValueError):
            duration = 0
        runtime += duration
        if not 0.25 <= duration <= 12.0:
            errors.append(f"scene_{i}_duration_estimate")
        for field in ("beat_text", "visual_prompt", "camera", "action", "narration"):
            if not str(scene.get(field, "")).strip():
                errors.append(f"scene_{i}_{field}")
        if "stickman" not in str(scene.get("visual_prompt", "")).lower():
            errors.append(f"scene_{i}_stickman")
        if len(str(scene.get("beat_text", ""))) < 12 or len(str(scene.get("action", ""))) < 8:
            errors.append(f"scene_{i}_too_vague")
        combined = " ".join(str(scene.get(k, "")) for k in ("beat_text", "narration", "visual_prompt", "action")).lower()
        if re.search(r"\b(?:random|for no reason|unrelated|suddenly appears)\b", combined):
            errors.append(f"scene_{i}_random_event")

    if runtime <= 0:
        errors.append("storyboard_runtime_invalid")
    return errors


def request(client: Groq, user_prompt: str, temperature: float = 0.78) -> str:
    result = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_completion_tokens=7000,
        response_format={"type": "json_object"},
    )
    return result.choices[0].message.content or ""


def make_prompt(topic: str | None, retry: bool = False) -> str:
    old = history()
    used = "" if not old else "Avoid repeating these previous titles: " + "; ".join(old[-25:])
    seed = f"Story seed: {topic}." if topic else "Invent an original, highly relatable everyday human problem that can naturally spiral into absurd comedy."
    retry_note = (
        "The previous package failed validation. Rebuild it from scratch. Do not patch the failed story. "
        "Make the causal chain clearer, make the protagonist's goal matter, and remove every event that exists only because it is weird. "
        "The final narration must be 105-145 words and every scene narration must exactly partition it."
        if retry else
        "Build the causal chain silently before writing. The story must be meaningful first and absurd second. "
        "Do not force extra events to reach length; if the premise is weak, replace it with a stronger premise."
    )
    return f"{seed}\n{used}\n{retry_note}\nWrite the complete JSON package now."


def generate(topic: str | None) -> dict[str, Any]:
    if not KEY:
        raise RuntimeError("GROQ_API_KEY is required")
    client = Groq(api_key=KEY)
    last: Exception | None = None
    for attempt in range(3):
        try:
            obj = parse_json(request(client, make_prompt(topic, retry=attempt > 0), temperature=0.78 if attempt == 0 else 0.58))
            normalize_scenes(obj)
            errors = validate(obj)
            if not errors:
                obj["generated_at"] = datetime.now(timezone.utc).isoformat()
                obj["visual_policy"] = "Narration is source of truth; every image literally reenacts its corresponding contiguous narration excerpt."
                obj["timing_policy"] = "Narration is the master timeline; final video duration follows neural narration; meaningful story beats determine image count."
                obj["story_policy"] = "meaningful_human_goal_causal_chain_absurd_comedy_45_60_seconds"
                obj["style_policy"] = STYLE
                return obj
            last = ValueError("validation failed: " + ", ".join(errors[:20]))
            print(f"[SHORTS] validation failed ({len(errors)} issue(s)): {errors[:10]}")
            if attempt < 2:
                time.sleep(3)
        except Exception as exc:
            last = exc
            msg = str(exc).lower()
            wait = 65 if any(x in msg for x in ("429", "rate limit", "rate_limit")) else 5 + attempt * 5
            if attempt < 2:
                print(f"[SHORTS] attempt {attempt + 1} failed: {exc}; waiting {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Could not generate a valid Chispa-Pifia story package: {last}")


def write(obj: dict[str, Any]) -> None:
    json_path = RUN / "relic_loop_short.json"
    flow_path = RUN / "flow_storyboard_prompt.txt"
    manifest_path = RUN / "flow_manifest.json"
    scenes = obj["scenes"]
    runtime_estimate = round(sum(float(s["duration_seconds"]) for s in scenes), 2)
    json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "CHISPA-PIFIA — MEANINGFUL STORY SHORT",
        "",
        "STORY POLICY: HUMAN PROTAGONIST + CLEAR GOAL + CAUSAL ABSURD COMEDY",
        "TIMING POLICY: NARRATION IS MASTER TIMELINE; MEANINGFUL STORY BEATS DETERMINE IMAGE COUNT",
        "",
        obj["flow_prompt"],
        "",
        "CANONICAL STYLE:", STYLE,
        "", "CAST:",
    ]
    for character in obj["characters"]:
        lines.append(f"- {character['name']}: {character['appearance']}")
    lines += ["", "STORY STRUCTURE:", json.dumps(obj["story_structure"], ensure_ascii=False), "", "LOCATIONS:"]
    lines += [f"- {location}" for location in obj["locations"]]
    lines += ["", "IMAGE BEATS — NARRATION IS SOURCE OF TRUTH:"]
    for scene in scenes:
        lines += [
            f"Beat {scene['number']} — estimated {scene['duration_seconds']}s",
            f"Beat text: {scene['beat_text']}",
            f"Narration: {scene['narration']}",
            f"Visual: {scene['visual_prompt']}",
            f"Camera: {scene['camera']}",
            f"Action: {scene['action']}",
            "",
        ]
    lines += ["VOICEOVER — MASTER TIMELINE:", obj["narration"], ""]
    flow_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "type": "chispa_pifia_story_image_short",
        "version": 14,
        "title": obj["title"],
        "story_word_count": words(obj["narration"]),
        "estimated_storyboard_seconds": runtime_estimate,
        "image_beat_count": len(scenes),
        "image_count_policy": "determined_by_meaningful_story_beats; no fixed count",
        "zero_text_gate": False,
        "text_policy": "no_deliberate_story_text",
        "story_policy": "meaningful_human_goal_causal_chain_absurd_comedy_45_60_seconds",
        "timing_policy": "narration_is_master_timeline",
        "media_generation": "cloudflare_images",
        "beat_policy": "meaningful_action_reaction_consequence_decision_reveal_payoff",
        "voice_policy": "natural_neural_narrator",
        "files": [str(json_path), str(flow_path)],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    used = history()
    title = str(obj.get("title", "")).strip()
    if title and title.lower() not in {x.lower() for x in used}:
        save_history(used + [title])


def main() -> None:
    topic = os.environ.get("RELIC_LOOP_TOPIC", "").strip() or None
    obj = generate(topic)
    write(obj)
    print(f"[SHORTS] generated: {obj['title']} | {words(obj['narration'])} narration words | {len(obj['scenes'])} visual beats")


if __name__ == "__main__":
    main()
