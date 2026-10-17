"""Relic Loop Shorts generator: story-first timing with literal visual beats."""
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
         "Consistent human protagonist identity. No photorealism, live action, stock photo, anime, "
         "manga, 3D/Pixar, glossy mascot or documentary art.")

SYSTEM = f"""You are Relic Loop's story director and visual storyboard director. Create ONE original fictional absurd-comedy YouTube Short about a HUMAN PROTAGONIST.

RELlC LOOP STORY BIBLE — NON-NEGOTIABLE:
- The protagonist must be a human person represented as a stickman. The human is the center of the story.
- Give the protagonist a clear immediate problem or goal.
- The protagonist must make decisions. Those decisions cause consequences.
- Every major event must logically cause, worsen, reveal or resolve the next event.
- Escalate through consequences 2-3 times, then deliver a specific twist/punchline.
- Keep the world mostly normal and believable; the absurdity should come from the human situation spiraling out of control.
- The narration must sound like someone telling a funny story to another person, not describing random images.
- The story must remain understandable with the visuals removed.
- The final sentence must land a payoff that connects to the setup.

ABSOLUTE STORY EXCLUSIONS:
- No inanimate-object protagonists or objects acting like humans.
- No mugs, phones, chairs, refrigerators, food, toilets, vehicles or other objects suddenly running, talking, attacking, dancing, forming armies or becoming characters.
- No random llamas, dinosaurs, animals or fantasy creatures unless a human encounters one as a clearly motivated part of an otherwise coherent story; never use one as a random gag.
- No disconnected chain of random events.
- No "and then suddenly" randomness.
- No meaningless city-wide chaos.
- No visual gag that has no cause in the story.
- No story whose only joke is that something impossible happened.

CAUSE-AND-EFFECT TEST:
Before returning JSON, mentally trace the story from the first sentence to the punchline. For every scene, answer: "Why is this happening now?" If the answer is not caused by the previous event or the protagonist's decision, rewrite it.

VISUAL TRUTH RULE — NON-NEGOTIABLE:
The narration is the source of truth. Images are a literal visual reenactment of the story, not independent jokes.
For every scene, show exactly what is happening in that beat at that moment. Never generate a cool unrelated pose just because it looks funny.

STORY-FIRST TIMING RULE — NON-NEGOTIABLE:
The narration is the master timeline. The finished video must last exactly long enough for the complete narration to be spoken naturally.
Do NOT target a fixed 25-40 second runtime. Do NOT make image count or image duration determine the story length.
First write the complete 70-110 word narration. Then divide that narration into 10-18 meaningful visual beats based on what is actually happening, not equal time slices.
Each beat's narration must be a contiguous excerpt of the full narration, in the same order, with no missing words and no duplicated words.
The sum of beat narration word counts must equal the full narration word count.
Beat duration_seconds is only an ESTIMATE used for storyboard planning; final assembly will measure the real neural narration and allocate time from the beat narration itself.
A single image may remain on screen longer when the narration needs it. A fast action may have a shorter beat. There must be no fixed timer cadence.
The story must be completely finished by the final beat. The final beat must contain the punchline/payoff.

STORY QUALITY:
- Immediate hook in the first sentence.
- Clear protagonist + problem/goal.
- 2-3 escalating consequences.
- Specific twist and punchline.
- Natural spoken English, contractions, punchy sentence lengths and conversational rhythm.
- No generic intro, no "have you ever wondered", no filler, no moral, no lesson.

STYLE: {STYLE}

Return ONLY JSON.
Create 10-18 visual beats for the complete story. TARGET 14, but choose the number based on story actions. NEVER split every phrase or comma into a new image.

JSON shape:
{{"title":"...","hook":"...","narration":"70-110 words","characters":[{{"name":"A","role":"human protagonist","appearance":"..."}}],"locations":["..."],"story_structure":{{"problem":"...","goal":"...","escalations":["...","..."],"twist":"...","punchline":"..."}},"scenes":[{{"number":1,"duration_seconds":2.0,"beat_text":"literal event happening now","narration":"CONTIGUOUS EXACT EXCERPT FROM THE FULL NARRATION","visual_prompt":"literal visible shot, no interpretation","camera":"...","action":"exact visible action"}}],"flow_prompt":"...","youtube_description":"...","tags":["...","..."]}}

SCENE RULES:
- Scene narration must be copied from the full narration, in order, with no rewriting.
- Concatenating scene narration fields with single spaces must reproduce the full narration, apart from harmless whitespace differences.
- Each scene must describe one literal moment from the continuous story.
- beat_text, narration, visual_prompt and action must describe the SAME event.
- visual_prompt must explicitly say "human stickman" or "stickman protagonist" and describe only visible things.
- Every scene must contain the protagonist or a directly relevant visible consequence of his previous action.
- Do not introduce a new prop, character, location or event unless the narration/story requires it.
- Do not invent visual jokes that are absent from the narration.
- No captions, subtitles, title cards, speech bubbles, logos, watermarks, UI or plot text in images.
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
    for key in ("title", "hook", "narration", "characters", "locations", "story_structure", "scenes", "flow_prompt"):
        if not obj.get(key):
            errors.append(f"missing:{key}")
    narration = _norm_text(str(obj.get("narration", "")))
    wc = words(narration)
    if not 70 <= wc <= 110:
        errors.append(f"narration_words={wc}")

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
        for key in ("problem", "goal", "escalations", "twist", "punchline"):
            if not structure.get(key):
                errors.append(f"missing_structure:{key}")
    else:
        errors.append("story_structure")

    bad_random_patterns = [
        r"mug .*\b(run|runs|sprint|sprints|attack|attacks|dance|dances|talk|talks)\b",
        r"(?:phone|chair|fridge|refrigerator|toilet|pizza|sandwich|coffee) .*\b(run|runs|walk|walks|talk|talks|attack|attacks|dance|dances|fight|fights)\b",
        r"(?:llama|dinosaur) .*\b(run|runs|chase|chases|attack|attacks|dance|dances)\b",
        r"(?:objects?|mugs?|chairs?|phones?) .*\bform (?:an )?(?:army|band|crowd)\b",
    ]
    for pattern in bad_random_patterns:
        if re.search(pattern, narration.lower()):
            errors.append("random_object_or_creature_nonsense")
            break

    scenes = obj.get("scenes")
    if not isinstance(scenes, list) or not 10 <= len(scenes) <= 18:
        errors.append("scene_count")
        return errors

    # Story-first timing: the scene narration must partition the master narration exactly.
    reconstructed = " ".join(_norm_text(str(s.get("narration", ""))) for s in scenes if isinstance(s, dict))
    if reconstructed != narration:
        errors.append("scene_narration_does_not_exactly_partition_master_narration")

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
        if not 0.5 <= d <= 8.0:
            errors.append(f"scene_{i}_duration_estimate")
        for field in ("beat_text", "visual_prompt", "camera", "action", "narration"):
            if not str(s.get(field, "")).strip():
                errors.append(f"scene_{i}_{field}")
        if "stickman" not in str(s.get("visual_prompt", "")).lower():
            errors.append(f"scene_{i}_stickman")
        if len(str(s.get("beat_text", ""))) < 12 or len(str(s.get("action", ""))) < 8:
            errors.append(f"scene_{i}_too_vague")
        combined = " ".join(str(s.get(k, "")) for k in ("beat_text", "narration", "visual_prompt", "action")).lower()
        if re.search(r"\b(random|suddenly appears|for no reason|somehow,? a (?:mug|llama|dinosaur))\b", combined):
            errors.append(f"scene_{i}_random_event")
    if not 20 <= runtime <= 70:
        errors.append(f"storyboard_estimated_runtime={runtime:.1f}")
    return errors


def request(client: Groq, user_prompt: str, temperature: float = 0.8) -> str:
    result = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_completion_tokens=5000,
        response_format={"type": "json_object"},
    )
    return result.choices[0].message.content or ""


def make_prompt(topic: str | None, retry: bool = False) -> str:
    old = history()
    used = "" if not old else "Avoid these previous titles: " + "; ".join(old[-25:])
    seed = f"Story seed: {topic}." if topic else "Invent a highly shareable absurd premise involving an ordinary human person facing a clear everyday problem."
    requirement = (
        "RETRY HARDER: The previous package failed validation. Return a complete coherent human story. "
        "Most importantly, the scene narration fields must be an exact ordered partition of the master narration, with no missing or duplicated words. "
        "Do not force a fixed runtime or fixed beat duration. Let the story's narration determine the final video length."
        if retry else
        "Return a coherent human-centered story. Write the complete narration first, then partition that exact narration into meaningful visual moments. The narration determines the final video length; images never determine it."
    )
    return f"{seed}\n{used}\n{requirement}\nWrite the complete package now."


def generate(topic: str | None) -> dict[str, Any]:
    if not KEY:
        raise RuntimeError("GROQ_API_KEY is required")
    client = Groq(api_key=KEY)
    last: Exception | None = None
    for attempt in range(3):
        try:
            obj = parse_json(request(client, make_prompt(topic, retry=attempt > 0)))
            normalize_scenes(obj)
            errors = validate(obj)
            if not errors:
                obj["generated_at"] = datetime.now(timezone.utc).isoformat()
                obj["visual_policy"] = "Narration is source of truth; every image literally reenacts its corresponding contiguous narration excerpt."
                obj["timing_policy"] = "Narration is master timeline; final video duration is measured from complete neural narration; beat durations are derived from narration segments, never the other way around."
                obj["story_policy"] = "human_protagonist_cause_effect_absurd_comedy"
                obj["style_policy"] = STYLE
                return obj
            last = ValueError("validation failed: " + ", ".join(errors[:16]))
            if attempt < 2:
                print(f"[SHORTS] validation failed ({len(errors)} issue(s)); regenerating package {attempt + 2}/3")
                time.sleep(3)
                continue
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
    runtime_estimate = round(sum(float(s["duration_seconds"]) for s in obj["scenes"]), 2)
    json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    scenes = obj["scenes"]
    lines = ["RELIC LOOP — STORY SHORT", "", "STORY POLICY: HUMAN PROTAGONIST + CAUSE/EFFECT ABSURD COMEDY", "TIMING POLICY: NARRATION IS MASTER TIMELINE; IMAGE BEATS FOLLOW NARRATION", "", obj["flow_prompt"], "", "CANONICAL STYLE:", STYLE, "", "CAST:"]
    for c in obj["characters"]:
        lines.append(f"- {c['name']}: {c['appearance']}")
    lines += ["", "STORY STRUCTURE:", json.dumps(obj["story_structure"], ensure_ascii=False), "", "LOCATIONS:"] + [f"- {x}" for x in obj["locations"]] + ["", "IMAGE BEATS — NARRATION IS SOURCE OF TRUTH:"]
    for s in scenes:
        lines += [f"Beat {s['number']} — estimated {s['duration_seconds']}s", f"Beat text: {s['beat_text']}", f"Narration: {s['narration']}", f"Visual: {s['visual_prompt']}", f"Camera: {s['camera']}", f"Action: {s['action']}", ""]
    lines += ["VOICEOVER — MASTER TIMELINE:", obj["narration"], ""]
    flow_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "type": "relic_loop_story_image_short",
        "version": 11,
        "title": obj["title"],
        "story_word_count": words(obj["narration"]),
        "estimated_storyboard_seconds": runtime_estimate,
        "image_beat_count": len(scenes),
        "minimum_image_beats": 10,
        "target_image_beats": 14,
        "maximum_image_beats": 18,
        "flow_template": "Storyboard Studio",
        "zero_text_gate": False,
        "text_policy": "no_deliberate_story_text",
        "story_policy": "human_protagonist_cause_effect_absurd_comedy",
        "visual_policy": "narration_is_source_of_truth_literal_scene_reenactment",
        "timing_policy": "narration_is_master_timeline",
        "media_generation": "cloudflare_images",
        "beat_policy": "meaningful_story_action_reaction_state_change_reveal_detail; no fixed cadence",
        "voice_policy": "natural_neural_narrator",
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
    print(f"[SHORTS] {len(scenes)} image beats / {runtime_estimate:.1f}s estimated storyboard")


if __name__ == "__main__":
    topic = os.environ.get("RELIC_LOOP_TOPIC", "").strip() or None
    write(generate(topic))
