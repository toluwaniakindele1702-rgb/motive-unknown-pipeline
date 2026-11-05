"""Production story generator for Chispa-Pifia.

Uses GPT-OSS 120B for both writing and review, with multiple independent
quality attempts and a best-draft fallback so a strict final review cannot
kill an otherwise usable production run.
"""
from __future__ import annotations

import json
import os
import time

import relic_loop_shorts_fixed as base

MODEL = "openai/gpt-oss-120b"
base.MODEL = MODEL
base.REVIEW_MODEL = os.environ.get("GROQ_REVIEWER_MODEL", "openai/gpt-oss-20b")

base.STYLE = (
    "CHISPA-PIFIA approved visual identity: simple expressive 2D storytime stickman based on the "
    "canonical visual references. Large smooth pale round head, expressive eyes and eyebrows, "
    "simple mouth, minimal body construction, thin clean limbs, bold readable outlines and strong "
    "facial/body acting. Clothing is optional and must not define the character. Never redesign the "
    "character as realistic, 3D, anime or a detailed human. Vertical 9:16; one literal story moment "
    "per frame; no text, captions, speech bubbles, logos or watermarks."
)

base.SYSTEM = f"""You are the lead story writer and storyboard director for Chispa-Pifia.

WRITE A GOOD STORY FIRST. Do not write a checklist disguised as a story. Create one original, entertaining mini-story that a person would naturally tell a friend. Vary the genre: everyday mishap, social misunderstanding, clever problem-solving, dilemma, suspense, discovery, irony, comedy, emotional payoff, or mystery. Do not default to mysteries or force a twist.

STORY QUALITY:
- Start inside a specific situation. The first sentence must make the viewer curious immediately.
- Establish what the protagonist wants and what is in the way.
- Let the protagonist make understandable choices. Each important event must follow from a previous action, consequence, or established fact.
- Build 2-4 meaningful developments. Each must change the situation, not repeat the same failed attempt.
- End with a satisfying payoff that resolves or reframes the original problem. A twist is optional; never add one just to surprise.
- Use natural spoken English, concrete details, varied sentence rhythm, and believable reactions. Avoid generic AI phrasing, exposition dumps, filler, moral lessons, and generic outros.
- The story must make sense without the images. No random spectacle, unexplained coincidences, or characters behaving irrationally just to move the plot.

LENGTH: 150-210 spoken words. Aim for about 165-190. Do not pad.

STORYBOARD:
- Create 10-12 meaningful scenes, not filler.
- Scene narration excerpts must reproduce the complete master narration exactly, in order, without omissions, duplication, or paraphrase.
- Each scene depicts one precise moment from its narration excerpt.
- visual_prompt must be concrete and drawable: protagonist pose/expression, key prop/action, story-supported location, composition and shot angle. Keep it concise (about 45-75 words).
- Scene 1 is the opening hook image: show the protagonist mid-action or the visible consequence/discovery; use a close or medium view, expressive acting and one clear focal point. Do not use an empty establishing shot or neutral standing pose.
- Maintain the approved simple expressive 2D stickman character. No text, captions, speech bubbles, logos or watermarks. {base.STYLE}

Return ONLY valid JSON with these keys:
{{
"title":"...",
"hook":"...",
"narration":"complete spoken narration",
"characters":[{{"name":"...","role":"human protagonist","appearance":"approved simple 2D stickman"}}],
"locations":["..."],
"story_structure":{{"problem":"...","goal":"...","causal_chain":["stage 1","stage 2","stage 3","stage 4","stage 5"],"escalations":["development 1","development 2","development 3"],"payoff":"...","twist":"optional; may be empty"}},
"scenes":[{{"number":1,"duration_seconds":4.0,"beat_text":"...","narration":"exact contiguous excerpt","visual_prompt":"45-75 word drawable frame description","camera":"shot and angle","action":"visible action"}}],
"flow_prompt":"...",
"youtube_description":"...",
"tags":["..."]
}}
Do not include markdown fences or commentary. Check that the JSON is complete and all scene excerpts together exactly reproduce the narration.
"""


def make_prompt(topic: str | None, retry: bool = False, feedback: str = "", attempt: int = 1) -> str:
    old = base.history()
    used = "" if not old else "Avoid repeating these recent titles: " + "; ".join(old[-8:])
    if topic:
        seed = f"Story seed: {topic}. Keep its central idea, but make the story original and coherent."
    else:
        seed = (
            "Invent an original Chispa-Pifia story. Choose whichever genre gives the strongest "
            "human story; do not default to mystery, absurdity, or a twist."
        )
    retry_note = ""
    if retry:
        retry_note = (
            f"This is attempt {attempt}. The previous draft failed. Start with a different premise "
            "unless a user-provided seed must be retained. Correct the specific validation/review "
            "feedback below. Prioritize a natural, entertaining story over decorative JSON detail.\n"
        )
    feedback_block = ("Feedback to fix:\n" + feedback[:900] + "\n") if feedback else ""
    return (
        f"{seed}\n{used}\n{retry_note}{feedback_block}"
        "Write 150-210 words of natural spoken narration and 10-12 meaningful scenes. "
        "Use exact contiguous narration excerpts that reconstruct the full narration in order. "
        "Keep visual prompts concise and drawable (45-75 words). Include every required JSON key. "
        "A twist is optional; deliver a satisfying payoff either way. Return complete valid JSON only."
    )


base.make_prompt = make_prompt


def production_generate(topic: str | None) -> dict:
    """Generate up to three candidates, reviewing only structurally valid drafts.

    Stop immediately when a candidate passes the strict 8/10 quality gate.
    Invalid drafts receive concrete validator feedback before the next attempt.
    If neither passes, choose the highest-scoring structurally valid candidate.
    
    """
    if not base.KEY:
        raise RuntimeError("GROQ_API_KEY is required")

    client = base.Groq(api_key=base.KEY)
    candidates: list[tuple[int, dict, dict, int]] = []
    feedback = ""
    last_error = None
    selected = None

    # A third bounded writer attempt gives structural/API failures one recovery
    # opportunity without adding reviewer calls for invalid drafts.
    for attempt in (1, 2, 3):
        try:
            prompt = make_prompt(
                topic,
                retry=(attempt > 1),
                feedback=feedback,
                attempt=attempt,
            )
            obj = base.parse_json(base.request(
                client,
                prompt,
                temperature=0.82 if attempt == 1 else (0.72 if attempt == 2 else 0.62),
            ))
            base.normalize_scenes(obj)
            errors = base.validate(obj)
            if errors:
                last_error = "structural validation failed: " + ", ".join(errors[:20])
                print(f"[SHORTS] candidate {attempt} rejected by local validation: {last_error}")
                feedback = "The previous candidate failed deterministic validation: " + last_error
                continue

            review = base.review_story(client, obj)
            score = int(review.get("overall_score", 0) or 0)
            approved = review.get("approved") is True and score >= 8
            print(
                f"[SHORTS] independent review: candidate={attempt}/3, "
                f"approved={approved}, score={score}/10"
            )
            print("[SHORTS] review scores: " + json.dumps(review.get("scores", {}), ensure_ascii=False))

            candidates.append((score, obj, review, attempt))
            feedback = json.dumps({
                "overall_score": score,
                "scores": review.get("scores", {}),
                "fatal_flaws": review.get("fatal_flaws", []),
                "rewrite_instructions": review.get("rewrite_instructions", []),
            }, ensure_ascii=False)

            if approved:
                selected = (score, obj, review, attempt, False)
                break
            last_error = f"candidate {attempt} did not pass the 8/10 independent review gate"
            if attempt < 3:
                print(f"[SHORTS] candidate {attempt} failed review; generating a distinct replacement.")
        except Exception as exc:
            last_error = str(exc)
            print(f"[SHORTS] candidate {attempt} failed: {exc}")
            if attempt < 3:
                feedback = (
                    "The previous response failed at the API or JSON stage. For this retry, follow the "
                    "current schema exactly: title, hook, narration, characters, locations, story_structure "
                    "with problem, goal, causal_chain (at least five stages), escalations (at least three), "
                    "payoff, optional twist, 10-12 scenes, flow_prompt, youtube_description, and tags. "
                    "Use 150-210 narration words and exact contiguous scene excerpts. Return compact, complete "
                    "valid JSON only. Do not add fields or requirements from an older schema."
                )

    else:
        selected = None

    if 'selected' not in locals() or selected is None:
        if not candidates:
            raise RuntimeError(
                "No structurally valid, reviewed story candidate was produced within the three-candidate limit: "
                + str(last_error)
            )
        score, obj, review, chosen_attempt = max(
            candidates,
            key=lambda item: (
                item[0],
                int((item[2].get("scores", {}) or {}).get("hook", 0) or 0),
                int((item[2].get("scores", {}) or {}).get("payoff", 0) or 0),
                int((item[2].get("scores", {}) or {}).get("coherence", 0) or 0),
            ),
        )
        selected = (score, obj, review, chosen_attempt, True)

    score, obj, review, attempt_used, fallback = selected
    obj["story_quality_review"] = {
        "reviewer_model": base.REVIEW_MODEL,
        "approved": bool(review.get("approved") is True and score >= 8),
        "overall_score": score,
        "scores": review.get("scores", {}),
        "fatal_flaws": review.get("fatal_flaws", []),
        "rewrite_instructions": review.get("rewrite_instructions", []),
        "fallback_used": fallback,
        "fallback_reason": (
            "No candidate passed the strict 8/10 review gate; "
            "the highest-scoring structurally valid candidate was selected."
            if fallback else None
        ),
        "candidate_count": len(candidates),
        "selected_candidate": attempt_used,
        "max_candidates": 3,
    }
    obj["generated_at"] = base.datetime.now(base.timezone.utc).isoformat()
    obj["visual_policy"] = "Narration is source of truth; every image literally reenacts its corresponding contiguous narration excerpt."
    obj["timing_policy"] = "Narration is the master timeline; 10-14 image beats follow exact contiguous spoken phrase excerpts; image durations scale to each excerpt's share of the actual voiceover duration, with no fixed seconds-per-image."
    obj["story_policy"] = "varied_storytime_60_90_seconds_three_attempt_reviewed"
    obj["style_policy"] = base.STYLE
    print(
        f"[SHORTS] selected candidate {attempt_used}; score={score}/10; "
        f"reviewed_candidates={len(candidates)}; fallback={fallback}; "
        "hard maximum=3 writer calls and 3 reviewer calls."
    )
    return obj

if __name__ == "__main__":
    topic = os.environ.get("RELIC_LOOP_TOPIC", "").strip() or None
    obj = production_generate(topic)
    base.write(obj)
    print(
        f"[SHORTS] generated: {obj['title']} | "
        f"{base.words(obj['narration'])} narration words | "
        f"{len(obj['scenes'])} visual beats | model={MODEL} | "
        f"review_score={obj.get('story_quality_review', {}).get('overall_score')} | "
        f"fallback={obj.get('story_quality_review', {}).get('fallback_used', False)}"
    )
