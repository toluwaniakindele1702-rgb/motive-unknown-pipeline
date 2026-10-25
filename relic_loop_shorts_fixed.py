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
    "CHISPA-PIFIA canonical style: a recognizable round-headed stickman character, not a line-only "
    "stick figure and not a realistic human. Large smooth pale round head, large oval black eyes, bold "
    "expressive eyebrows, small simple mouth, tiny neck, developed compact torso, articulated arms/hands/"
    "legs/feet. Fixed default outfit: dark charcoal hoodie with drawstrings, dark cargo trousers, black-and-white "
    "high-top sneakers; dark backpack only when story continuity says he has it. Keep identical head/body ratio, "
    "eye spacing, limb thickness, clothing silhouette and footwear across every scene. Cinematic stylized animation "
    "with soft dimensional shading and clean dark outlines on the character, paired with richly detailed, believable "
    "environments. Dark charcoal/cool blue/steel-grey base palette with restrained warm amber practical-light accents. "
    "Vertical 9:16. Use model-sheet and environment references as canonical constraints; show one specific story moment "
    "per frame. No photorealistic character, live action, anime/manga, Pixar redesign, generic line-stickman, style "
    "switching, random outfit changes, extra limbs/fingers, text, captions, logos or watermarks."
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

MEANING + CAUSAL-INTEGRITY TEST — NON-NEGOTIABLE:
Before writing, silently summarize the whole story in ONE sentence using cause and effect. If that summary sounds like a list of unrelated events, discard the premise.
For EVERY major beat, silently answer:
- What exactly caused this event?
- How does it follow from the previous action or established situation?
- What does the protagonist do in response, and how does that response create the next problem?
- Would the event make sense under the story's own rules and ordinary real-world logic?
If any answer is weak, rewrite the beat or replace the premise. Do not rely on coincidence to move the plot forward.
The viewer must understand the protagonist's goal, why each escalation happens, and how the ending connects to earlier details. Absurd does NOT mean meaningless.
CONTINUITY / PLAUSIBILITY AUDIT:
- Never create an obstacle that contradicts the setting's established rules without explaining the cause.
- Do not use implausible safety or building behavior as a cheap gag (for example, an exit inexplicably locked during a fire alarm/drill).
- Track where the protagonist, important objects, and other characters are; do not teleport or forget them between beats.
- If a character loses an object, becomes trapped, misses a deadline, or triggers an alarm, show the action or established rule that caused it.
- The twist must be surprising but fair: seed the key detail earlier, then reveal its meaning. Do not introduce a brand-new coincidence in the last sentence.
- Prefer one tightly connected chain of 3-5 escalations over adding extra events just to make the story feel bigger.

LENGTH — HARD LIMIT:
The complete spoken narration MUST be 105-145 words, targeting roughly 120-135 words.
This should normally produce about 45-60 seconds of natural narration.
Do not pad the story. Do not rush it to hit a word count. If the idea cannot support a meaningful 45-60 second story, discard it and create a stronger premise.

STORY QUALITY:
- Open in the first sentence with a concrete, instantly understandable problem AND a specific constraint, deadline, risk, or goal where useful. Avoid slow introductions.
- Make the viewer understand what the protagonist wants and why it matters within the first 1-2 sentences.
- Start the chain quickly: the protagonist makes a clear choice early, and that choice has a visible consequence.
- Use 3-5 meaningful escalations; each must be caused by the protagonist's prior action, the direct consequence of that action, or a clearly established rule.
- Escalate the COST or difficulty of the original goal; do not merely increase spectacle.
- Give the protagonist agency: their decisions should drive the story, including failed attempts to recover the situation.
- Keep the world mostly normal until the absurdity grows naturally from the chain.
- Keep continuity of people, locations, objects, time, and cause-and-effect.
- Make the twist connected to an earlier detail and make it feel inevitable in hindsight.
- Check for plot holes, contradictory rules, implausible coincidences, forgotten objects, unexplained escapes, and obstacles that exist only to force the next scene. Repair every one before returning JSON.
- End immediately after the strongest payoff; do not explain the joke afterward.
- Use natural conversational spoken English and short, clear sentences that are easy to narrate.
- No moral, lesson, generic outro or "subscribe" line.
- The story must still make sense if all visuals are removed.

GOOD PREMISE PATTERN:
A normal person wants something simple under a clear constraint (like a deadline or limited time), makes a shortcut or bad decision, causes a specific practical problem, tries to repair it, causes a worse practical problem, improvises again, and finally discovers an absurd consequence that was seeded in the original situation.
Every link must be explicit enough for a viewer to follow on first watch. The ending should pay off the original goal, not abandon it for an unrelated gag.

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

STORYBOARD BEAT COUNT — VISUAL PACING:
Create 8-10 distinct visual beats/scenes for a normal 45-60 second Short. This is a practical range, not a fixed duration per image. Seven or fewer is too sparse; more than 10 is not allowed.
Split the narration at natural sentence endings first, then at meaningful phrase/clause boundaries when a sentence contains multiple visible actions. Each image should cover one short spoken phrase or one compact sentence, not a whole paragraph. Aim for a new image as the spoken narration moves to the next meaningful phrase/action, roughly 10-16 words per beat on average for a 105-145 word narration, but use shorter phrases when a meaningful visual change happens and longer excerpts when one sentence describes a single continuous moment.
Do not assign every image the same number of seconds. Each scene's narration must be a contiguous exact excerpt from the master narration, in order, with no missing, duplicated, paraphrased, or reordered words. The editor must time each image to its own spoken excerpt, so shorter phrases display for less time and longer phrases display for longer time.
Do not pad with meaningless filler, duplicate poses, or split one unchanged moment arbitrarily. Preserve the coherent causal story while showing its meaningful visual changes.
Final assembly uses the actual neural narration as the master timeline; scene duration values are estimates only and must not control final image timing.

CHARACTERS:
- The protagonist is a normal human represented as a stickman.
- The protagonist must actively make decisions.
- Other humans may appear only when the story requires them.
- Keep character roles and appearances consistent across scenes.

VISUAL STYLE: {STYLE}

SCENE RULES:
- The complete storyboard MUST have 8-10 scenes; fewer than 8 or more than 10 is invalid.
- Divide scenes at natural sentence/phrase boundaries so each image changes with the spoken text; never use a fixed seconds-per-image rule.
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
    "causal_chain":[
      "1. Initial problem",
      "2. Protagonist decision",
      "3. Immediate consequence",
      "4. Attempt to fix it",
      "5. Further escalation",
      "6. Foreshadowed twist",
      "7. Punchline payoff"
    ],
    "escalations":["first cause-and-effect escalation","second escalation caused by the protagonist's response","third escalation that sets up the twist"],
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
        if isinstance(chain, list):
            # Accept a compact arrow-separated chain only if it clearly names multiple stages.
            chain_text = " ".join(str(item) for item in chain)
            stages = [part.strip() for part in re.split(r"\s*(?:->|→|\n|;|\|)\s*", chain_text) if part.strip()]
            if len(chain) < 4 and len(stages) < 5:
                errors.append("causal_chain_too_short")
        elif isinstance(chain, str):
            stages = [part.strip() for part in re.split(r"\s*(?:->|→|\n|;|\|)\s*", chain) if part.strip()]
            if len(stages) < 5:
                errors.append("causal_chain_too_short")
        else:
            errors.append("causal_chain_invalid")
        escalations = structure.get("escalations")
        if isinstance(escalations, list) and len(escalations) < 3:
            errors.append("too_few_meaningful_escalations")
    else:
        errors.append("story_structure")

    bad_random_patterns = [
        r"\brandom (?:cat|dog|monkey|bird|llama|dinosaur)\b",
        r"\b(?:a|an|the) random (?:livestream|yoga session|meme|website)\b",
        r"\b(?:phone|chair|fridge|refrigerator|toilet|pizza|sandwich|coffee) (?:suddenly )?(?:runs away|walks away|starts talking|attacks him|attacks her|starts dancing|starts fighting)\b",
        r"\b(?:objects?|mugs?|chairs?|phones?) (?:suddenly )?form (?:an )?(?:army|band|crowd)\b",
    ]
    # Check the story itself, not the entire JSON blob or production metadata.
    structure_text = json.dumps(structure, ensure_ascii=False) if isinstance(structure, dict) else ""
    combined_story = narration + " " + structure_text
    for pattern in bad_random_patterns:
        if re.search(pattern, combined_story.lower()):
            errors.append("random_object_or_creature_or_digital_nonsense")
            break

    scenes = obj.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        errors.append("no_visual_beats")
        return errors
    if not 8 <= len(scenes) <= 10:
        errors.append(f"visual_beat_count={len(scenes)}_hard_limit_8_10")

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
        "Audit cause and effect beat by beat: every event needs an established cause, every attempted fix must change the situation, and the ending must pay off the original goal. Repair plot holes, contradictory rules, implausible coincidences, forgotten objects/characters, and unexplained location changes. Do not use implausible safety/building behavior as a cheap gag (for example, an exit inexplicably locked during a fire alarm/drill). "
        "Write causal_chain as a JSON ARRAY of at least 5 separate short strings, one causal stage per item; never put the entire chain in one array item. "
        "Write at least 3 separate escalations, each caused by the protagonist's response to the previous problem. "
        "Make the first sentence an immediate concrete hook, make the protagonist's goal matter, remove every event that exists only because it is weird, and ensure the twist is seeded earlier. "
        "The final narration must be 105-145 words and every scene narration must exactly partition it. "
        "The storyboard MUST contain 8-10 separate scenes. Seven or fewer is too sparse; more than ten is invalid. Divide narration at natural sentence boundaries and then meaningful phrase/clause boundaries, with each image showing the phrase currently being spoken. Keep exact contiguous narration excerpts; do not use filler or duplicate moments. Do not set a fixed seconds-per-image duration."
        if retry else
        "Build and audit the full causal chain silently before writing. For every beat, verify what caused it, how it follows from the prior beat, and how it changes the protagonist's next decision. Repair plot holes, contradictory rules, implausible coincidences, forgotten objects/characters, and unexplained location changes. "
        "Use a concrete first-sentence hook with a clear goal and constraint; start the protagonist's first consequential decision early. Each escalation must worsen the original problem because of an established cause, not merely add spectacle. The twist must be seeded earlier and pay off the original goal. "
        "Then output causal_chain as at least 5 separate JSON array items, not one combined sentence. Output at least 3 separate escalation items. The story must be meaningful first and absurd second. "
        "The storyboard MUST contain 8-10 separate scenes for the 45-60 second narration. Make each image correspond to a natural sentence or short spoken phrase; split long sentences at meaningful action/reaction/consequence/reveal boundaries. Each scene must contain a short exact contiguous narration excerpt. Do not add filler, duplicate moments, or assign a fixed duration to every image. "
        "Do not force extra plot events to reach length; show the existing causal story in more visual detail. If the premise is weak, replace it with a stronger premise."
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
                obj["timing_policy"] = "Narration is the master timeline; 8-10 image beats follow exact contiguous spoken phrase excerpts; image durations scale to each excerpt's share of the actual voiceover duration, with no fixed seconds-per-image."
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
        "TIMING POLICY: 8-10 IMAGES; EACH IMAGE FOLLOWS ITS SPOKEN PHRASE; NO FIXED IMAGE DURATION",
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
        "image_count_policy": "8_to_10_phrase_aligned_visual_beats",
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
