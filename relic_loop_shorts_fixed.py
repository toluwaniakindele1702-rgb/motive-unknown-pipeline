"""Relic Loop Shorts generator: coherent human absurd-comedy stories with literal visual beats."""
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
For every scene, show exactly what is happening in that beat at that moment. If the protagonist is running, show him running. If he drops a phone, show the phone dropped. If he gets on a bus, show him getting on the bus. If he realizes a mistake, show the exact reaction and relevant situation.
Never generate a cool unrelated pose just because it looks funny.

STORY QUALITY:
- Immediate hook in the first sentence.
- Clear protagonist + problem/goal.
- 2-3 escalating consequences.
- Specific twist and punchline.
- Natural spoken English, contractions, punchy sentence lengths and conversational rhythm.
- No generic intro, no "have you ever wondered", no filler, no moral, no lesson.

STYLE: {STYLE}

Return ONLY JSON.
Create 12-16 visual beats for a 25-40 second Short. TARGET 14. NEVER return fewer than 12 or more than 16.
Use 1.4-3.2 seconds per beat. New image only for a meaningful story action, reaction, state change, reveal, location change or important detail. Do NOT split every phrase or comma into a new image.

JSON shape:
{{"title":"...","hook":"...","narration":"70-100 words","characters":[{{"name":"A","role":"human protagonist","appearance":"..."}}],"locations":["..."],"story_structure":{{"problem":"...","goal":"...","escalations":["...","..."],"twist":"...","punchline":"..."}},"scenes":[{{"number":1,"duration_seconds":2.0,"beat_text":"literal event happening now","narration":"exact narration covered by this shot","visual_prompt":"literal visible shot, no interpretation","camera":"...","action":"exact visible action"}}],"flow_prompt":"...","youtube_description":"...","tags":["...","..."]}}

SCENE RULES:
- Each scene must describe one literal moment from the continuous story.
- beat_text, narration, visual_prompt and action must describe the SAME event.
- visual_prompt must explicitly say "human stickman" or "stickman protagonist" and describe only visible things.
- Every scene must contain the protagonist or a directly relevant visible consequence of his previous action.
- Refer to the protagonist naturally by name or with a human reference such as he, she, him, her, the man, the guy, the person, or the protagonist. Do NOT force the name into every scene; natural pronouns are preferred for spoken storytelling.
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


def validate(obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("title", "hook", "narration", "characters", "locations", "story_structure", "scenes", "flow_prompt"):
        if not obj.get(key):
            errors.append(f"missing:{key}")
    wc = words(str(obj.get("narration", "")))
    if not 70 <= wc <= 100:
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

    narration = str(obj.get("narration", "")).lower()
    bad_random_patterns = [
        r"mug .*\b(run|runs|sprint|sprints|attack|attacks|dance|dances|talk|talks)\b",
        r"(?:phone|chair|fridge|refrigerator|toilet|pizza|sandwich|coffee) .*\b(run|runs|walk|walks|talk|talks|attack|attacks|dance|dances|fight|fights)\b",
        r"(?:llama|dinosaur) .*\b(run|runs|chase|chases|attack|attacks|dance|dances)\b",
        r"(?:objects?|mugs?|chairs?|phones?) .*\bform (?:an )?(?:army|band|crowd)\b",
    ]
    for pattern in bad_random_patterns:
        if re.search(pattern, narration):
            errors.append("random_object_or_creature_nonsense")
            break

    scenes = obj.get("scenes")
    if not isinstance(scenes, list) or not 12 <= len(scenes) <= 16:
        errors.append("scene_count")
        return errors
    runtime = 0.0
    human_reference_pattern = re.compile(r"\b(?:he|him|his|she|her|hers|man|woman|boy|girl|guy|person|human|protagonist)\b")
    for i, s in enumerate(scenes, 1):
        if not isinstance(s, dict):
            errors.append(f"scene_{i}_object")
            continue
        try:
            d = float(s.get("duration_seconds", 0))
        except (TypeError, ValueError):
            d = 0
        runtime += d
        if not 1.4 <= d <= 3.2:
            errors.append(f"scene_{i}_duration")
        for field in ("beat_text", "visual_prompt", "camera", "action", "narration"):
            if not str(s.get(field, "")).strip():
                errors.append(f"scene_{i}_{field}")
        combined = " ".join(str(s.get(k, "")) for k in ("beat_text", "narration", "visual_prompt", "action")).lower()
        if "stickman" not in str(s.get("visual_prompt", "")).lower():
            errors.append(f"scene_{i}_stickman")
        # The old validator required the protagonist's exact name in every scene.
        # That rejected natural storytelling where later scenes correctly use he/she/him/her.
        # Require a human reference instead; the visual prompt still requires a stickman.
        if not human_reference_pattern.search(combined):
            errors.append(f"scene_{i}_protagonist_reference_missing")
        # The same event must be represented across all scene fields; reject obvious contradictions.
        if len(str(s.get("beat_text", ""))) < 12 or len(str(s.get("action", ""))) < 8:
            errors.append(f"scene_{i}_too_vague")
        if re.search(r"\b(random|suddenly appears|for no reason|somehow,? a (?:mug|llama|dinosaur))\b", combined):
            errors.append(f"scene_{i}_random_event")
    if not 25 <= runtime <= 40:
        errors.append(f"runtime={runtime:.1f}")
    return errors


def request(client: Groq, user_prompt: str, temperature: float = 0.85) -> str:
    result = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_completion_tokens=4200,
        response_format={"type": "json_object"},
    )
    return result.choices[0].message.content or ""


def make_prompt(topic: str | None, retry: bool = False) -> str:
    old = history()
    used = "" if not old else "Avoid these previous titles: " + "; ".join(old[-25:])
    seed = f"Story seed: {topic}." if topic else "Invent a highly shareable absurd premise involving an ordinary human person facing a clear everyday problem."
    requirement = (
        "RETRY HARDER: The previous package failed validation. Return a complete coherent human story. "
        "The protagonist must have a problem/goal, make decisions, suffer logical consequences, escalate 2-3 times, then reach a specific twist and punchline. "
        "Every scene must literally depict the narration at that moment. Do not use random objects, animals or disconnected chaos."
        if retry else
        "Return a coherent human-centered story with a clear setup, cause-and-effect escalation, twist and punchline. Every image beat must literally depict what the narration says."
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
                obj["visual_policy"] = "Narration is source of truth; every image must literally reenact its corresponding story beat. No deliberate story text in images."
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
    runtime = round(sum(float(s["duration_seconds"]) for s in obj["scenes"]), 2)
    json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    scenes = obj["scenes"]
    lines = ["RELIC LOOP — STORY SHORT", "", "STORY POLICY: HUMAN PROTAGONIST + CAUSE/EFFECT ABSURD COMEDY", "", obj["flow_prompt"], "", "CANONICAL STYLE:", STYLE, "", "CAST:"]
    for c in obj["characters"]:
        lines.append(f"- {c['name']}: {c['appearance']}")
    lines += ["", "STORY STRUCTURE:", json.dumps(obj["story_structure"], ensure_ascii=False), "", "LOCATIONS:"] + [f"- {x}" for x in obj["locations"]] + ["", "IMAGE BEATS — NARRATION IS SOURCE OF TRUTH:"]
    for s in scenes:
        lines += [f"Beat {s['number']} — {s['duration_seconds']}s", f"Beat text: {s['beat_text']}", f"Narration: {s['narration']}", f"Visual: {s['visual_prompt']}", f"Camera: {s['camera']}", f"Action: {s['action']}", ""]
    lines += ["VOICEOVER:", obj["narration"], ""]
    flow_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "type": "relic_loop_story_image_short",
        "version": 11,
        "title": obj["title"],
        "runtime_seconds": runtime,
        "image_beat_count": len(scenes),
        "minimum_image_beats": 12,
        "target_image_beats": 14,
        "maximum_image_beats": 16,
        "flow_template": "Storyboard Studio",
        "zero_text_gate": False,
        "text_policy": "no_deliberate_story_text",
        "story_policy": "human_protagonist_cause_effect_absurd_comedy",
        "visual_policy": "narration_is_source_of_truth_literal_scene_reenactment",
        "story_validator": "natural_human_reference_not_exact_name_every_scene",
        "media_generation": "cloudflare_images",
        "beat_policy": "new_image_on_meaningful_story_action_reaction_state_change_reveal_detail",
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
    print(f"[SHORTS] {len(scenes)} image beats / {runtime:.1f}s")


if __name__ == "__main__":
    topic = os.environ.get("RELIC_LOOP_TOPIC", "").strip() or None
    write(generate(topic))
