"""Relic Loop Shorts generator: story-first causal absurd comedy."""
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

SYSTEM = f"""You are Relic Loop's head writer and visual storyboard director.
Create ONE original fictional absurd-comedy YouTube Short centered on a HUMAN PROTAGONIST.

THE RELIC LOOP STORY ENGINE — NON-NEGOTIABLE:
The story must follow this causal chain:
NORMAL HUMAN PROBLEM
-> PROTAGONIST MAKES A DECISION
-> IMMEDIATE CONSEQUENCE
-> PROTAGONIST RESPONDS
-> CONSEQUENCE ESCALATES
-> ANOTHER DECISION
-> BIGGER CONSEQUENCE
-> ABSURD BUT LOGICALLY CONNECTED TWIST
-> FUNNY PAYOFF

The absurdity must come from the situation becoming increasingly ridiculous, NOT from random
things appearing. The audience should think: "How did this guy's stupid decision lead to THAT?"
They should never think: "What the hell is happening?"

Before writing the narration, silently construct a causal chain:
Problem -> Decision 1 -> Consequence 1 -> Decision 2 -> Consequence 2 -> Decision 3 ->
Consequence 3 -> Twist -> Punchline.
Then write the narration from that chain. Every major event must have a clear cause.

For EVERY major event, silently ask: "Why is this happening RIGHT NOW?"
If the answer is not caused by an earlier event or the protagonist's decision, REMOVE or REWRITE it.

HUMAN-CENTERED STORY:
- The protagonist is a normal human person represented as a stickman.
- Give the protagonist a clear immediate problem or goal.
- The protagonist must actively make decisions.
- Decisions must create consequences.
- Keep the world mostly normal and believable at first.
- Other people may appear when the story requires them.
- Animals, unusual objects or bizarre situations may appear only when clearly motivated by the story.
- Never make an inanimate object the protagonist.

ABSURDITY RULE:
BAD:
"He missed the bus, then suddenly a dinosaur appeared."
"His phone started running away."
"He got on a bus and suddenly discovered there were monkeys."
"A random llama attacked him."

GOOD:
A relatable mistake causes a second problem. The protagonist makes a desperate decision to fix it.
That decision creates a larger problem. The protagonist tries again. The consequences compound until
an unexpected but causally connected reveal or twist makes the whole situation ridiculous.

The final twist must be traceable to something established earlier. The audience should be able to
look back and say: "Ohhh, THAT'S why that happened."

RELATABILITY:
Start with something ordinary viewers understand: being late, losing something, saving money,
trying to impress someone, avoiding embarrassment, fixing a small mistake, reaching somewhere,
hiding something, beating a deadline, or escaping an awkward situation.
Then spiral it into absurdity.

ESCALATION:
Use meaningful escalating consequences. Each escalation must make the protagonist's situation
worse, stranger or more difficult because of what happened before. Do not escalate by randomly
adding disasters, animals, objects or unrelated characters.

TWIST AND PAYOFF:
The twist must recontextualize the situation or reveal an unexpected consequence of something earlier.
End immediately after the funniest consequence. No moral, lesson, explanation or generic outro.
The final sentence must land the punchline.

ABSOLUTE EXCLUSIONS:
- No inanimate-object protagonists.
- No mugs, phones, chairs, refrigerators, food, toilets, vehicles or other objects suddenly running,
  talking, attacking, dancing, forming armies or becoming characters.
- No random llamas, dinosaurs, animals or fantasy creatures as disconnected gags.
- No disconnected chain of random events.
- No "and then suddenly" randomness.
- No meaningless city-wide chaos.
- No visual gag without a cause in the narration.
- No story whose only joke is that something impossible happened.
- No documentary, educational or explanatory story.
- No generic "have you ever wondered" intro.

VISUAL TRUTH RULE — NON-NEGOTIABLE:
The narration is the source of truth. Images are literal visual reenactments of the narration.
Every visual beat must show exactly WHO is present, WHERE they are, WHAT they are doing and
WHAT caused that moment. Never invent a cool unrelated pose or visual joke.

STORY-FIRST TIMING:
The narration is the master timeline. The story itself determines the number of visual beats.
There is NO fixed image count and NO target image count.
Create exactly one visual beat whenever the story reaches a meaningful visual change: a new action,
reaction, consequence, object state, location, reveal or punchline detail.
Do NOT split every phrase or comma into a beat. Do NOT combine distinct actions merely to hit a target.
A longer story may naturally require more images; a compact story may require fewer.
Do not pad or compress the story to satisfy an image quota.
Each beat's narration must be a contiguous exact excerpt of the master narration, in order, with no
missing or duplicated words. The concatenated beat narration must reproduce the complete narration.
Beat duration_seconds is only an estimate. Final assembly measures real neural TTS and allocates time
from the beat narration. Never let image timing determine the story.

STORY QUALITY:
- Immediate hook in the first sentence.
- Clear human protagonist + problem/goal.
- Strong decisions and cause/effect.
- Meaningful escalating consequences.
- Specific causally connected twist.
- Strong final punchline.
- Natural spoken English with contractions and conversational rhythm.
- Story must make sense with visuals removed.

STYLE: {STYLE}

Return ONLY JSON.

JSON shape:
{{"title":"...","hook":"...","narration":"complete spoken narration","characters":[{{"name":"A","role":"human protagonist","appearance":"..."}}],"locations":["..."],"story_structure":{{"problem":"...","goal":"...","causal_chain":["problem -> decision -> consequence -> escalation -> twist"],"escalations":["..."],"twist":"...","punchline":"..."}},"scenes":[{{"number":1,"duration_seconds":2.0,"beat_text":"literal event happening now","narration":"CONTIGUOUS EXACT EXCERPT FROM FULL NARRATION","visual_prompt":"literal visible shot, no interpretation","camera":"...","action":"exact visible action"}}],"flow_prompt":"...","youtube_description":"...","tags":["...","..."]}}

SCENE RULES:
- Scene count is determined ONLY by meaningful story beats. Never target, pad or cap a count.
- Scene narration must be copied from the full narration in order, with no rewriting.
- Concatenating scene narration fields with single spaces must reproduce the full narration.
- Each scene describes one literal moment from the continuous story.
- beat_text, narration, visual_prompt and action describe the SAME event.
- visual_prompt explicitly says "human stickman" or "stickman protagonist".
- Every scene contains the protagonist or a directly relevant visible consequence.
- Never introduce a prop, character, location or event unless the narration requires it.
- Never invent visual jokes absent from the narration.
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
    if not 70 <= wc <= 160:
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
    if not isinstance(scenes, list) or not scenes:
        errors.append("no_visual_beats")
        return errors

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
        if not 0.25 <= d <= 12.0:
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
    if runtime <= 0:
        errors.append("storyboard_runtime_invalid")
    return errors


def request(client: Groq, user_prompt: str, temperature: float = 0.8) -> str:
    result = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_completion_tokens=6500,
        response_format={"type": "json_object"},
    )
    return result.choices[0].message.content or ""


def make_prompt(topic: str | None, retry: bool = False) -> str:
    old = history()
    used = "" if not old else "Avoid these previous titles: " + "; ".join(old[-25:])
    seed = f"Story seed: {topic}." if topic else "Invent a highly shareable absurd premise involving an ordinary human person facing a clear everyday problem."
    requirement = (
        "RETRY HARDER: The previous package failed validation. Rebuild the causal chain before writing. "
        "The scene narration must exactly partition the master narration. Do not add random events to fill visual beats. "
        "Do not force a runtime or image count. Let meaningful story actions determine the number of images."
        if retry else
        "Build the causal chain first. Write the complete narration from that chain, then partition the exact narration into only the meaningful visual moments that actually occur. The story determines its own length and image count."
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
                obj["timing_policy"] = "Narration is master timeline; final video duration is measured from complete neural narration; meaningful story beats determine image count and beat durations are derived from narration segments."
                obj["story_policy"] = "human_protagonist_causal_chain_absurd_comedy"
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
    lines = ["RELIC LOOP — STORY SHORT", "", "STORY POLICY: HUMAN PROTAGONIST + CAUSAL CHAIN ABSURD COMEDY", "TIMING POLICY: NARRATION IS MASTER TIMELINE; MEANINGFUL STORY BEATS DETERMINE IMAGE COUNT", "", obj["flow_prompt"], "", "CANONICAL STYLE:", STYLE, "", "CAST:"]
    for c in obj["characters"]:
        lines.append(f"- {c['name']}: {c['appearance']}")
    lines += ["", "STORY STRUCTURE:", json.dumps(obj["story_structure"], ensure_ascii=False), "", "LOCATIONS:"] + [f"- {x}" for x in obj["locations"]] + ["", "IMAGE BEATS — NARRATION IS SOURCE OF TRUTH:"]
    for s in scenes:
        lines += [f"Beat {s['number']} — estimated {s['duration_seconds']}s", f"Beat text: {s['beat_text']}", f"Narration: {s['narration']}", f"Visual: {s['visual_prompt']}", f"Camera: {s['camera']}", f"Action: {s['action']}", ""]
    lines += ["VOICEOVER — MASTER TIMELINE:", obj["narration"], ""]
    flow_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "type": "relic_loop_story_image_short",
        "version": 12,
        "title": obj["title"],
        "story_word_count": words(obj["narration"]),
        "estimated_storyboard_seconds": runtime_estimate,
        "image_beat_count": len(scenes),
        "image_count_policy": "determined_by_meaningful_story_beats; no fixed minimum_target_maximum",
        "flow_template": "Storyboard Studio",
        "zero_text_gate": False,
        "text_policy": "no_deliberate_story_text",
        "story_policy": "human_protagonist_causal_chain_absurd_comedy",
        "visual_policy": "narration_is_source_of_truth_literal_scene_reenactment",
        "timing_policy": "narration_is_master_timeline",
        "media_generation": "cloudflare_images",
        "beat_policy": "meaningful_story_action_reaction_state_change_reveal_detail; story determines count",
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
