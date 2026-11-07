"""Chispa-Pifia story generator: story-first 60-90 second mystery, suspense, dilemma and twist Shorts."""
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
REVIEW_MODEL = os.environ.get("GROQ_REVIEWER_MODEL", "openai/gpt-oss-20b")
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

SYSTEM = f"""You write one-minute scripts for Chispa-Pifia, a Shorts channel about absurd but coherent stickman comedy.

Turn an ordinary situation into a funny escalating incident through a character choice, misunderstanding, overconfidence, failed plan or clever solution. Absurd is welcome; random disconnected events are not. Use natural spoken English, a strong immediate hook, clear cause and effect, visual actions, and a satisfying comedic payoff. No filler, moral, lesson, subscribe line or forced twist.

The spoken narration must be 125-145 words. Return one complete JSON package with title, hook, narration, characters, locations, story_structure, scenes, flow_prompt, youtube_description and tags. Include a human stickman protagonist, 10-12 ordered scenes whose narration excerpts exactly reconstruct the master narration, and concrete 45-80 word visual prompts for each scene. Scene 1 must show the opening problem clearly. No text in images. story_structure must include problem, goal, causal_chain, escalations, twist and punchline. If there is no twist, say so plainly. Return JSON only.

VISUAL STYLE: {STYLE}
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
    if not 125 <= wc <= 145:
        errors.append(f"narration_words={wc}_hard_limit_125_145_one_minute")

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
    if not 10 <= len(scenes) <= 14:
        errors.append(f"visual_beat_count={len(scenes)}_hard_limit_10_14")

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
        visual_description = _norm_text(str(scene.get("visual_prompt", "")))
        if "stickman" not in visual_description.lower():
            errors.append(f"scene_{i}_stickman")
        visual_words = words(visual_description)
        if not 45 <= visual_words <= 115:
            errors.append(f"scene_{i}_visual_prompt_words={visual_words}_expected_45_115")
        if i == 1 and visual_words < 55:
            errors.append("opening_visual_prompt_too_short")
        if i == 1 and re.search(r"\b(?:empty room|standing neutrally|standing still|generic room|far in the distance)\b", visual_description, re.I):
            errors.append("opening_visual_prompt_generic_or_low_impact")
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
        # Keep prompt + reserved completion budget below Groq's 8k TPM request ceiling.
        # The production wrapper uses compact 150-210 word stories and 10-14 beats.
        max_completion_tokens=4500,
        response_format={"type": "json_object"},
    )
    return result.choices[0].message.content or ""




def review_story(client: Groq, obj: dict[str, Any]) -> dict[str, Any]:
    """Independent semantic quality gate; structural validity alone is not enough."""
    review_system = """You are a demanding senior story editor for short-form animated storytelling.
You are NOT the writer and must not praise weak work to be polite. Evaluate whether a real viewer would want to keep watching.
Judge only the actual story and narration, not how many required fields it fills.

Score each category from 1 to 10:
1. HOOK: immediate, specific curiosity; no generic setup.
2. CAUSALITY: each event follows from a previous choice, consequence, clue, or established rule.
3. CHARACTER AGENCY: the protagonist makes understandable choices that drive the story.
4. ESCALATION: stakes or difficulty meaningfully rise; beats are not repetitive.
5. ORIGINALITY: premise and execution feel specific, not like a stock AI story or a familiar trope with names swapped.
6. PAYOFF: ending is surprising but fair, emotionally or comically satisfying, and resolves the central goal.
7. NATURALNESS: sounds like a human storyteller, not a checklist, synopsis, or awkward AI-generated prose.
8. VISUAL STORYTELLING: actions and reveals can be understood clearly in images without relying on text.
9. COHERENCE: no plot holes, arbitrary coincidences, unexplained behavior, missing objects, or contradictions.
10. SHORTS RETENTION: each beat creates a reason to keep watching; no padding.
11. OPENING IMAGE: scene 1 is a specific, visually arresting, truthful hook that can be understood at phone size immediately; it shows a strong action or visible consequence, expressive acting, and a story-critical prop, rather than a generic pose or empty establishing shot.

Be strict. A story that merely follows the requested structure is NOT automatically good.
Reject vague stakes, predictable or unearned twists, random events, fake complexity, repetitive failed attempts, characters acting irrationally for convenience, and endings that explain rather than deliver a payoff.
Do not reward a story for including a twist if the twist is obvious, unsupported, or irrelevant to the original problem.
Return ONLY JSON:
{"approved":true|false,"overall_score":1,"scores":{"hook":1,"causality":1,"agency":1,"escalation":1,"originality":1,"payoff":1,"naturalness":1,"visual_storytelling":1,"coherence":1,"retention":1,"opening_image":1},"fatal_flaws":["specific flaw"],"rewrite_instructions":["specific actionable fix"]}
Approval gate: overall_score must be at least 8, opening_image must be at least 8, and no major flaw may remain. If the opening image is generic, visually weak, too distant, static, cluttered, or not instantly understandable at phone size, reject the candidate even when the narration is strong. Give specific rewrite instructions for scene 1. Be honest; do not inflate scores."""
    review_input = {
        "title": obj.get("title", ""),
        "hook": obj.get("hook", ""),
        "narration": obj.get("narration", ""),
        "story_structure": obj.get("story_structure", {}),
        "scenes": [
            {"beat_text": s.get("beat_text", ""), "narration": s.get("narration", ""), "visual_prompt": s.get("visual_prompt", ""), "camera": s.get("camera", ""), "action": s.get("action", "")}
            for s in obj.get("scenes", []) if isinstance(s, dict)
        ],
    }
    result = client.chat.completions.create(
        model=REVIEW_MODEL,
        messages=[
            {"role": "system", "content": review_system},
            {"role": "user", "content": json.dumps(review_input, ensure_ascii=False)},
        ],
        temperature=0.15,
        max_completion_tokens=1400,
        response_format={"type": "json_object"},
    )
    return parse_json(result.choices[0].message.content or "")


def make_prompt(topic: str | None, retry: bool = False, feedback: str = "") -> str:
    history_text = "; ".join(history()[-20:])
    seed = f"Start from this idea: {topic}" if topic else "Invent a fresh absurd-but-coherent stickman comedy situation."
    return "\\n".join(part for part in (
        seed,
        "Avoid repeating these previous titles or premises: " + history_text if history_text else "",
        "Write a complete 125-145 word spoken script and its 10-12 visual scenes.",
        "Make the situation funny through understandable cause and effect, character choices and a satisfying payoff.",
        "Return the complete required JSON package only.",
        "Recovery instruction: " + feedback if retry and feedback else "",
    ) if part)

def generate(topic: str | None) -> dict[str, Any]:
    if not KEY:
        raise RuntimeError("GROQ_API_KEY is required")
    client = Groq(api_key=KEY)
    last: Exception | None = None
    feedback = ""
    for attempt in range(3):
        try:
            obj = parse_json(request(
                client,
                make_prompt(topic, retry=attempt > 0, feedback=feedback),
                temperature=0.82 if attempt == 0 else 0.62,
            ))
            normalize_scenes(obj)
            errors = validate(obj)
            if errors:
                last = ValueError("structural validation failed: " + ", ".join(errors[:20]))
                print(f"[SHORTS] structural validation failed ({len(errors)} issue(s)): {errors[:10]}")
                feedback = "The previous draft failed machine validation: " + ", ".join(errors[:20])
                if attempt < 2:
                    time.sleep(2)
                    continue
                break

            # A valid JSON structure is not proof of a good story. Have a separate editor
            # score the actual narrative and require a rewrite when quality is below threshold.
            review = review_story(client, obj)
            score = int(review.get("overall_score", 0) or 0)
            approved = review.get("approved") is True and score >= 8
            print(f"[SHORTS] independent story review: approved={approved}, score={score}/10")
            print("[SHORTS] review scores: " + json.dumps(review.get("scores", {}), ensure_ascii=False))
            if not approved:
                flaws = review.get("fatal_flaws", [])
                fixes = review.get("rewrite_instructions", [])
                feedback = json.dumps({
                    "overall_score": score,
                    "scores": review.get("scores", {}),
                    "fatal_flaws": flaws,
                    "rewrite_instructions": fixes,
                }, ensure_ascii=False)
                last = ValueError(f"story quality review rejected draft ({score}/10)")
                print(f"[SHORTS] quality gate rejected draft: {feedback[:1800]}")
                if attempt < 2:
                    time.sleep(2)
                    continue
                break

            obj["story_quality_review"] = {
                "reviewer_model": REVIEW_MODEL,
                "approved": True,
                "overall_score": score,
                "scores": review.get("scores", {}),
                "fatal_flaws": review.get("fatal_flaws", []),
            }
            obj["generated_at"] = datetime.now(timezone.utc).isoformat()
            obj["visual_policy"] = "Narration is source of truth; every image literally reenacts its corresponding contiguous narration excerpt."
            obj["timing_policy"] = "Narration is the master timeline; 10-14 image beats follow exact contiguous spoken phrase excerpts; image durations scale to each excerpt's share of the actual voiceover duration, with no fixed seconds-per-image."
            obj["story_policy"] = "story_first_mystery_suspense_dilemma_twist_60_90_seconds"
            obj["style_policy"] = STYLE
            return obj
        except Exception as exc:
            last = exc
            msg = str(exc).lower()
            wait = 65 if any(x in msg for x in ("429", "rate limit", "rate_limit")) else 5 + attempt * 5
            if attempt < 2:
                print(f"[SHORTS] attempt {attempt + 1} failed: {exc}; waiting {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Could not generate a valid Chispa-Pifia story package after quality checks: {last}")


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
        "STORY POLICY: MINI-MYSTERY / SUSPENSE / DILEMMA / PROBLEM-SOLVING / FAIR TWIST; COMEDY OPTIONAL",
        "TIMING POLICY: 10-14 IMAGES; EACH IMAGE FOLLOWS ITS SPOKEN PHRASE; NO FIXED IMAGE DURATION",
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
        "image_count_policy": "10_to_12_phrase_aligned_visual_beats",
        "zero_text_gate": "strict_ocr_reject_after_retry_in_image_renderer",
        "text_policy": "no_deliberate_story_text",
        "story_policy": "absurd_but_coherent_stickman_comedy_about_60_seconds",
        "story_quality_gate": "independent_editor_score_minimum_8_of_10",
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
