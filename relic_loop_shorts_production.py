"""Script-first generator for Chispa-Pifia one-minute stickman comedy Shorts."""
from __future__ import annotations

import json
import os
import re
import time

import relic_loop_shorts_fixed as base

MODEL = os.environ.get("GROQ_WRITER_MODEL", "openai/gpt-oss-120b")
base.MODEL = MODEL
base.REVIEW_MODEL = os.environ.get("GROQ_REVIEWER_MODEL", "openai/gpt-oss-20b")

base.STYLE = (
    "Chispa-Pifia: expressive 2D stickman characters with pale round heads, clear eyes and "
    "eyebrows, simple mouths, readable poses, bold clean outlines and illustrated environments. "
    "Vertical 9:16. No text, captions, speech bubbles, logos or watermarks."
)

SCRIPT_SYSTEM = """You write the actual one-minute spoken scripts for Chispa-Pifia.

CHANNEL CONCEPT:
Chispa-Pifia is absurd, funny stickman storytelling. Start with a relatable ordinary situation, then let a character's bad decision, overconfidence, misunderstanding, strange-but-clear rule, or failed plan make it hilariously worse. The events must connect through understandable cause and effect. The humour should come from what the characters do and what happens next, not random disconnected nonsense.

WRITE ONLY THE SCRIPT:
- Create a complete spoken narration of 125-145 words, suitable for about 60 seconds.
- Open immediately on the funniest, strangest, or most urgent moment. No greeting, channel introduction, "one day", or "have you ever".
- Give the protagonist a clear goal. Let their choices drive the escalation.
- End with a strong, satisfying comedic payoff. A twist is optional and must never be forced.
- Use natural conversational English that sounds good read aloud.
- Keep every important event visual and easy to understand in a simple 2D stickman image.
- No moral, lesson, subscribe request, filler, or generic outro.
- Do not write scene breakdowns, story structure, camera directions, visual prompts, or storyboard instructions.

Return a compact JSON object with only:
{"title":"short title","narration":"the complete spoken script","characters":[{"name":"character name","role":"protagonist or supporting character"}],"locations":["setting name"]}

Keep the JSON small. Characters and locations are brief continuity notes only; do not add descriptions or scene plans. The narration must be 125-145 words."""
 
VISUAL_SYSTEM = """You are the visual storyboard artist for Chispa-Pifia, an absurd-but-coherent 2D stickman comedy channel.

For each supplied narration excerpt, describe ONE specific frozen story moment that literally illustrates that excerpt. Preserve cause and effect, character identity, props and location continuity. Use expressive stickman acting, a clear readable action, a story-critical object when relevant, and a strong composition. No random visual gag and no event that is not in the supplied excerpt.

Each visual_prompt must be 50-75 words; scene 1 must be 55-75 words. Include the stickman character in every prompt. Images are vertical 9:16, illustrated 2D cartoons, not photorealistic or 3D. Absolutely no text, captions, speech bubbles, logos, watermarks, letters, numbers, or readable signs.

Return JSON only:
{"scenes":[{"number":1,"visual_prompt":"...","camera":"...","action":"..."}]}

Return exactly one result for every supplied scene number. Do not rewrite or alter the narration excerpts."""

def _is_rate_limited(message: str) -> bool:
    text = message.lower()
    return any(x in text for x in ("429", "rate limit", "rate_limit", "tokens per minute", "tpm"))

def _call_json(client, system: str, user: str, max_tokens: int, temperature: float) -> dict:
    result = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return base.parse_json(result.choices[0].message.content or "")

def _script_prompt(topic: str | None, attempt: int, feedback: str, previous: list[str]) -> str:
    parts = []
    if topic:
        parts.append("Use this idea as the starting point: " + topic)
    else:
        parts.append("Invent a fresh Chispa-Pifia stickman comedy premise yourself.")
    if previous:
        parts.append("Avoid repeating these recent titles: " + "; ".join(previous[-20:]))
    if attempt > 1 and feedback:
        parts.append("The previous script did not meet the simple script requirements. Correct this: " + feedback)
    parts.append("Return only the compact script JSON requested. Do not create a storyboard.")
    return "\n\n".join(parts)

def _split_narration(text: str, scene_count: int = 11) -> list[str]:
    """Split normalized narration into balanced exact contiguous excerpts."""
    narration = " ".join(str(text or "").split())
    words = list(re.finditer(r"\b[\w'’-]+\b", narration))
    count = len(words)
    if count < scene_count * 8:
        raise ValueError("Narration is too short to create meaningful visual beats")
    scene_count = min(scene_count, count // 8)
    boundaries = []
    previous = 0
    for scene_index in range(1, scene_count):
        ideal = round(scene_index * count / scene_count)
        minimum = previous + 8
        maximum = count - (scene_count - scene_index) * 8
        candidates = range(minimum, maximum + 1)

        def boundary_cost(position: int) -> tuple[float, int]:
            gap = narration[words[position - 1].end():words[position].start()]
            if re.search(r"[.!?][\"')\]]?$", gap):
                punctuation_penalty = 0.0
            elif re.search(r"[;:][\"')\]]?$", gap):
                punctuation_penalty = 0.7
            elif "," in gap:
                punctuation_penalty = 1.2
            else:
                punctuation_penalty = 2.0
            return (abs(position - ideal) + punctuation_penalty, abs(position - ideal))

        chosen = min(candidates, key=boundary_cost)
        boundaries.append(words[chosen].start())
        previous = chosen

    chunks = []
    start = 0
    for boundary in boundaries:
        chunk = narration[start:boundary].strip()
        if chunk:
            chunks.append(chunk)
        start = boundary
    final_chunk = narration[start:].strip()
    if final_chunk:
        chunks.append(final_chunk)
    if len(chunks) != scene_count or " ".join(chunks) != narration:
        raise ValueError("Deterministic scene splitting did not preserve the master narration")
    return chunks

def _normalise_characters(raw) -> list[dict]:
    characters = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            characters.append({
                "name": name,
                "role": "human stickman protagonist" if not characters else "human stickman supporting character",
                "appearance": (
                    "the canonical Chispa-Pifia 2D stickman design from the approved visual references; "
                    "preserve the same round head, facial design, proportions and illustrated style"
                ),
            })
    if not characters:
        characters = [{
            "name": "Protagonist",
            "role": "human stickman protagonist",
            "appearance": (
                "the canonical Chispa-Pifia 2D stickman design from the approved visual references; "
                "preserve the same round head, facial design, proportions and illustrated style"
            ),
        }]
    return characters

def _normalise_locations(raw) -> list[str]:
    if isinstance(raw, list):
        values = [str(x).strip() for x in raw if isinstance(x, str) and x.strip()]
        if values:
            return values[:8]
    return ["The setting established by the narration and each scene description"]

def _fallback_visual(scene: dict, first: bool = False) -> dict:
    excerpt = scene["narration"]
    opening = "This is the opening hook frame. " if first else ""
    prompt = (
        "A single vertical 9:16 illustrated 2D cartoon frame in the approved Chispa-Pifia stickman style. "
        "Show the same expressive round-headed stickman protagonist with a clear readable face and pose. "
        + opening
        + "Visually reenact this exact spoken moment: " + excerpt + ". "
        "Show the action and its immediate consequence through body language and the relevant story prop. "
        "Use a coherent illustrated setting, bold clean outlines, one strong focal action and clear foreground "
        "separation. Keep the character design consistent. No text, captions, symbols or watermarks."
    )
    return {
        "visual_prompt": prompt,
        "camera": "Close medium shot, eye-level vertical 9:16 composition with the key action clearly readable." if first else "Medium shot, eye-level vertical 9:16 composition with the key action clearly readable.",
        "action": "Show the protagonist physically performing the exact action described in this narration excerpt.",
    }

def _visual_batch(client, title: str, narration: str, characters: list[dict], locations: list[str], batch: list[dict]) -> list[dict]:
    payload = {
        "title": title,
        "master_narration": narration,
        "characters": [{"name": c["name"], "role": c["role"]} for c in characters],
        "locations": locations,
        "scenes": [{"number": s["number"], "exact_narration_excerpt": s["narration"]} for s in batch],
    }
    feedback = ""
    for attempt in range(1, 3):
        try:
            user = json.dumps(payload, ensure_ascii=False)
            if feedback:
                user += "\n\nFix these exact output problems: " + feedback
            obj = _call_json(client, VISUAL_SYSTEM, user, 1600, 0.68)
            returned = obj.get("scenes")
            if not isinstance(returned, list) or len(returned) != len(batch):
                raise ValueError("visual response did not return the exact requested scene count")
            by_number = {}
            for item in returned:
                if isinstance(item, dict):
                    try:
                        by_number[int(item.get("number"))] = item
                    except (TypeError, ValueError):
                        continue
            output = []
            errors = []
            for scene in batch:
                item = by_number.get(scene["number"], {})
                prompt = str(item.get("visual_prompt", "")).strip()
                camera = str(item.get("camera", "")).strip()
                action = str(item.get("action", "")).strip()
                wc = base.words(prompt)
                minimum = 55 if scene["number"] == 1 else 45
                if "stickman" not in prompt.lower():
                    errors.append(f"scene {scene['number']} visual prompt must explicitly include stickman")
                if not minimum <= wc <= 115:
                    errors.append(f"scene {scene['number']} visual prompt has {wc} words; minimum is {minimum}")
                if len(camera) < 5 or len(action) < 8:
                    errors.append(f"scene {scene['number']} camera/action is incomplete")
                output.append({
                    "number": scene["number"],
                    "visual_prompt": prompt,
                    "camera": camera,
                    "action": action,
                })
            if errors:
                raise ValueError("; ".join(errors))
            return output
        except Exception as exc:
            feedback = str(exc)
            print(f"[CHISPA] visual batch attempt {attempt}/2 failed: {exc}")
            if attempt == 1:
                time.sleep(65 if _is_rate_limited(feedback) else 3)
    print("[CHISPA] using deterministic visual descriptions for this batch after model formatting failure")
    return [
        {"number": scene["number"], **_fallback_visual(scene, scene["number"] == 1)}
        for scene in batch
    ]

def _build_scenes(client, title: str, narration: str, characters: list[dict], locations: list[str]) -> list[dict]:
    excerpts = _split_narration(narration, 11)
    total_words = base.words(narration)
    scenes = []
    for index, excerpt in enumerate(excerpts, 1):
        scenes.append({
            "number": index,
            "duration_seconds": round(65.0 * base.words(excerpt) / max(total_words, 1), 2),
            "beat_text": excerpt,
            "narration": excerpt,
            "visual_prompt": "",
            "camera": "",
            "action": "",
        })
    for start in range(0, len(scenes), 4):
        batch = scenes[start:start + 4]
        visual_results = _visual_batch(client, title, narration, characters, locations, batch)
        for scene, visual in zip(batch, visual_results):
            scene.update(visual)
    return scenes

def _make_story_structure(scenes: list[dict]) -> dict:
    excerpts = [s["narration"] for s in scenes]
    return {
        "problem": excerpts[0],
        "goal": "The protagonist tries to deal with the situation introduced in the opening.",
        "causal_chain": [excerpts[i] for i in sorted(set([0, 2, 4, 6, 8, len(excerpts) - 1])) if i < len(excerpts)],
        "escalations": [excerpts[i] for i in (3, 6, 8) if i < len(excerpts)],
        "twist": "No forced twist; the comedy comes from the situation escalating naturally.",
        "punchline": excerpts[-1],
    }

def generate(topic: str | None) -> dict:
    if not base.KEY:
        raise RuntimeError("GROQ_API_KEY is required")
    client = base.Groq(api_key=base.KEY)
    previous_titles = base.history()
    last_error = ""
    candidates = []
    selected = None

    for attempt in (1, 2):
        try:
            raw = _call_json(
                client,
                SCRIPT_SYSTEM,
                _script_prompt(topic, attempt, last_error, previous_titles),
                1300,
                0.9 if attempt == 1 else 0.82,
            )
            title = str(raw.get("title", "")).strip() or "A Very Bad Idea"
            narration = " ".join(str(raw.get("narration", "")).split())
            word_count = base.words(narration)
            if not 125 <= word_count <= 145:
                raise ValueError(f"script narration has {word_count} words; required 125-145")
            characters = _normalise_characters(raw.get("characters"))
            locations = _normalise_locations(raw.get("locations"))
            scenes = _build_scenes(client, title, narration, characters, locations)
            obj = {
                "title": title,
                "hook": scenes[0]["narration"],
                "narration": narration,
                "characters": characters,
                "locations": locations,
                "story_structure": _make_story_structure(scenes),
                "scenes": scenes,
                "flow_prompt": (
                    "Create a continuous one-minute absurd-but-coherent Chispa-Pifia stickman comedy. "
                    "Use the approved 2D stickman references for character identity and style. "
                    "Illustrate each supplied narration excerpt literally, in order, with consistent characters, "
                    "props and settings. Vertical 9:16. No text in any image."
                ),
                "youtube_description": (
                    "An absurd stickman comedy story from Chispa-Pifia. #shorts #stickman #comedy"
                ),
                "tags": ["shorts", "stickman", "comedy", "absurd comedy", "storytime"],
            }
            errors = base.validate(obj)
            if errors:
                raise ValueError("package validation: " + ", ".join(errors[:18]))
            review = base.review_story(client, obj)
            score = int(review.get("overall_score", 0) or 0)
            approved = review.get("approved") is True and score >= 8
            print(f"[CHISPA] script review: approved={approved}, score={score}/10")
            print("[CHISPA] review scores: " + json.dumps(review.get("scores", {}), ensure_ascii=False))
            candidates.append((score, obj, review, attempt))
            if approved:
                selected = (score, obj, review, attempt, False)
                break
            last_error = "Reviewer feedback: " + json.dumps({
                "fatal_flaws": review.get("fatal_flaws", []),
                "rewrite_instructions": review.get("rewrite_instructions", []),
            }, ensure_ascii=False)
            print("[CHISPA] reviewer requested a stronger one-minute script")
        except Exception as exc:
            last_error = str(exc)
            print(f"[CHISPA] script attempt {attempt}/2 failed: {exc}")
            if attempt < 2:
                wait = 65 if _is_rate_limited(last_error) else 4
                print(f"[CHISPA] retrying script generation after {wait}s")
                time.sleep(wait)

    if selected is None:
        if not candidates:
            raise RuntimeError("Could not produce a valid one-minute Chispa-Pifia package: " + last_error)
        score, obj, review, attempt = max(candidates, key=lambda item: item[0])
        selected = (score, obj, review, attempt, True)

    score, obj, review, attempt, fallback = selected
    obj["story_quality_review"] = {
        "reviewer_model": base.REVIEW_MODEL,
        "approved": bool(review.get("approved") is True and score >= 8),
        "overall_score": score,
        "scores": review.get("scores", {}),
        "fatal_flaws": review.get("fatal_flaws", []),
        "fallback_used": fallback,
        "generation_mode": "script_only_then_python_scene_split_and_visual_batches",
    }
    obj["generated_at"] = base.datetime.now(base.timezone.utc).isoformat()
    obj["visual_policy"] = "Python owns scene count and exact narration excerpts; visuals illustrate each excerpt."
    obj["timing_policy"] = "Approximately 60-second spoken script; visual beat durations follow narration word share."
    obj["story_policy"] = "absurd_but_coherent_stickman_comedy_about_60_seconds"
    obj["style_policy"] = base.STYLE
    print(
        f"[CHISPA] selected attempt {attempt}; {base.words(obj['narration'])} words; "
        f"{len(obj['scenes'])} scenes; review={score}/10; fallback={fallback}"
    )
    return obj

def main() -> None:
    topic = os.environ.get("RELIC_LOOP_TOPIC", "").strip() or None
    base.write(generate(topic))

if __name__ == "__main__":
    main()
