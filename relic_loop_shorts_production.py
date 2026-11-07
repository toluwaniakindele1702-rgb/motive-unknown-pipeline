"""Compact one-pass script and storyboard generator for Chispa-Pifia."""
from __future__ import annotations

import json
import os
import time

import relic_loop_shorts_fixed as base

MODEL = os.environ.get("GROQ_WRITER_MODEL", "openai/gpt-oss-120b")
base.MODEL = MODEL
base.REVIEW_MODEL = os.environ.get("GROQ_REVIEWER_MODEL", "openai/gpt-oss-20b")

base.STYLE = (
    "Chispa-Pifia visual identity: expressive simple 2D stickman characters, pale round heads, "
    "clear eyes and eyebrows, simple mouths, thin clean limbs, bold readable outlines, strong "
    "poses and facial acting. Vertical 9:16. No text, captions, speech bubbles, logos or watermarks."
)

SYSTEM = """You write one-minute story scripts for Chispa-Pifia, a Shorts channel built around absurd, funny stickman situations.

CHANNEL IDEA:
Make an ordinary situation become hilariously absurd through a character's bad decision, overconfidence, misunderstanding, strange but understandable rule, escalating mishap, or clever solution. The story should feel like a short entertaining incident acted out by stickman characters. Be imaginative and funny, but keep cause and effect understandable. Absurd does not mean random: every event must connect to the situation or a character's choice.

SCRIPT RULES:
- Write a complete spoken script of 125-145 words, suitable for about 60 seconds.
- Start with the most interesting moment or problem. No introductions like "Have you ever", "One day", or "This is a story about".
- Use conversational, natural spoken English.
- Give the main character a clear goal and let their choices drive the funny escalation.
- Build toward a strong, satisfying comedic payoff. A twist is optional, never forced.
- Keep it visual: actions, reactions, objects and locations should be easy to show in simple stickman images.
- No moral, lesson, subscribe request, filler, or generic outro.
- Do not copy a previous title or premise supplied in the user message.

Return one complete JSON object only, with these keys:
title, hook, narration, characters, locations, story_structure, scenes, flow_prompt, youtube_description, tags.

The narration is the actual script and must contain 125-145 words.
characters is a non-empty array of character objects, each with non-empty name, role and appearance. The first character is the human stickman protagonist.
locations is a non-empty array of setting names.
story_structure contains problem, goal, causal_chain (at least 4 short items), escalations (at least 3 short items), twist, and punchline. If there is no twist, say "No twist; the comedy comes from the escalating situation."
Create 10-12 scenes that cover the narration in order. Every scene narration must be a contiguous excerpt from the master narration; all excerpts together, in order, must reproduce the narration exactly when joined with spaces. Do not add events.
Each scene has number, duration_seconds, beat_text, narration, visual_prompt, camera, action. Set duration_seconds to a plausible estimate between 3 and 8 seconds. Give each scene a 45-80 word visual_prompt (55-80 words for scene 1) describing one specific frozen moment: stickman pose, facial expression, key prop, setting, composition and camera angle. Scene 1 must show the opening problem/action clearly in a close or medium shot. No text in images.
flow_prompt briefly describes the complete visual direction and must include the visual style.
youtube_description is one short sentence. tags is an array of relevant short tags.
Do not include analysis or any fields beyond the requested JSON."""


def make_prompt(topic: str | None, attempt: int = 1, feedback: str = "") -> str:
    old = base.history()
    previous = ""
    if old:
        previous = "Avoid repeating these recent titles or premises: " + "; ".join(old[-20:])
    seed = (
        "Use this supplied idea as the starting point: " + topic
        if topic else
        "Invent a fresh, funny Chispa-Pifia stickman situation. Choose a strong premise yourself."
    )
    retry = (
        "This is a replacement attempt. Fix the specific problem below and use a different premise "
        "if the previous premise was weak. " + feedback
        if attempt > 1 and feedback else ""
    )
    return "\n".join(x for x in (seed, previous, retry, "Write the complete one-minute script package now.") if x)


def generate(topic: str | None) -> dict:
    if not base.KEY:
        raise RuntimeError("GROQ_API_KEY is required")
    client = base.Groq(api_key=base.KEY)
    last_error = None
    candidates = []

    for attempt in (1, 2):
        try:
            result = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": make_prompt(topic, attempt, str(last_error or ""))},
                ],
                temperature=0.92 if attempt == 1 else 0.82,
                max_completion_tokens=4200,
                response_format={"type": "json_object"},
            )
            obj = base.parse_json(result.choices[0].message.content or "")
            base.normalize_scenes(obj)
            errors = base.validate(obj)
            if errors:
                raise ValueError("package validation: " + ", ".join(errors[:16]))

            review = base.review_story(client, obj)
            score = int(review.get("overall_score", 0) or 0)
            approved = review.get("approved") is True and score >= 7
            print(f"[CHISPA] one-minute script review: approved={approved}, score={score}/10")
            print("[CHISPA] review scores: " + json.dumps(review.get("scores", {}), ensure_ascii=False))
            candidates.append((score, obj, review, attempt))
            if approved:
                selected = (score, obj, review, attempt, False)
                break
            last_error = "Reviewer feedback: " + json.dumps({
                "fatal_flaws": review.get("fatal_flaws", []),
                "rewrite_instructions": review.get("rewrite_instructions", []),
            }, ensure_ascii=False)
            print("[CHISPA] reviewer requested a stronger script.")
        except Exception as exc:
            last_error = str(exc)
            print(f"[CHISPA] generation attempt {attempt}/2 failed: {exc}")
            if attempt < 2:
                message = str(exc).lower()
                wait_seconds = 65 if any(
                    marker in message
                    for marker in ("429", "rate limit", "rate_limit", "tokens per minute", "tpm")
                ) else 4
                print(f"[CHISPA] retrying after {wait_seconds}s")
                time.sleep(wait_seconds)

    else:
        selected = None

    if "selected" not in locals() or selected is None:
        if not candidates:
            raise RuntimeError("Could not generate a valid one-minute Chispa-Pifia script: " + str(last_error))
        score, obj, review, attempt = max(candidates, key=lambda item: item[0])
        selected = (score, obj, review, attempt, True)

    score, obj, review, attempt, fallback = selected
    obj["story_quality_review"] = {
        "reviewer_model": base.REVIEW_MODEL,
        "approved": bool(review.get("approved") is True and score >= 7),
        "overall_score": score,
        "scores": review.get("scores", {}),
        "fatal_flaws": review.get("fatal_flaws", []),
        "fallback_used": fallback,
        "generation_mode": "single_compact_one_minute_script",
    }
    obj["generated_at"] = base.datetime.now(base.timezone.utc).isoformat()
    obj["visual_policy"] = "Every scene illustrates its corresponding spoken script excerpt."
    obj["timing_policy"] = "One-minute spoken script; visual beats follow the narration."
    obj["story_policy"] = "absurd_but_coherent_stickman_comedy_about_60_seconds"
    obj["style_policy"] = base.STYLE
    print(
        f"[CHISPA] selected attempt {attempt}; {base.words(obj['narration'])} words; "
        f"{len(obj['scenes'])} scenes; review={score}/10; fallback={fallback}"
    )
    return obj


def main() -> None:
    topic = os.environ.get("RELIC_LOOP_TOPIC", "").strip() or None
    obj = generate(topic)
    base.write(obj)


if __name__ == "__main__":
    main()
