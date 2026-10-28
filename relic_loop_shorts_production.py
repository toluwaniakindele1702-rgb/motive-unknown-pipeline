"""Production story generator for Chispa-Pifia.

Uses GPT-OSS 120B for both writing and review, while adding a resilient production
retry loop so one mediocre draft cannot kill an entire scheduled video run.
"""
from __future__ import annotations

import json
import os
import time

import relic_loop_shorts_fixed as base

MODEL = "openai/gpt-oss-120b"
base.MODEL = MODEL
base.REVIEW_MODEL = MODEL

base.STYLE = (
    "CHISPA-PIFIA approved visual identity: simple expressive 2D storytime stickman based on the "
    "canonical visual references. Large smooth pale round head, expressive eyes and eyebrows, "
    "simple mouth, minimal body construction, thin clean limbs, bold readable outlines and strong "
    "facial/body acting. Clothing is optional and must not define the character. Never redesign the "
    "character as realistic, 3D, anime or a detailed human. Vertical 9:16; one literal story moment "
    "per frame; no text, captions, speech bubbles, logos or watermarks."
)

base.SYSTEM = f"""You are the lead writer and storyboard director for Chispa-Pifia, a storytime animated Shorts channel.

Your job is NOT to make every video a mystery. The channel needs a varied stream of genuinely entertaining, easy-to-follow short stories with strong hooks and satisfying endings.

STORY MIX — ROTATE NATURALLY:
Choose the strongest category for the premise rather than forcing a mystery every time. Possible categories include:
- relatable everyday problems or embarrassing situations
- clever problem-solving and escalating mistakes
- social misunderstandings or awkward situations
- secrets, discoveries and mini-mysteries
- suspenseful situations where the audience needs to know what happens next
- dilemmas and difficult choices
- ironic situations where a character's plan backfires
- clever reversals and fair twists
- light emotional or satisfying stories
- absurd/comedic situations grounded in a coherent setup

Do NOT label the category in the narration. Do NOT cycle through categories mechanically. Choose whatever produces the strongest story.

CORE QUALITY RULE:
Every story must have a clear protagonist, a concrete desire/problem, understandable cause-and-effect, meaningful escalation or development, and a satisfying payoff. A twist is welcome when it genuinely improves the story, but a twist is NOT mandatory if a strong payoff works better.

The viewer should understand the situation almost immediately and want to know what happens next. The ending should feel earned, surprising when appropriate, and satisfying in hindsight.

CAUSALITY AUDIT — NON-NEGOTIABLE:
Before returning JSON, silently summarize the story as one causal chain. For every major beat ask:
1. What caused this?
2. What does the protagonist do because of it?
3. How does that decision change the situation?
4. Why does the next event logically follow?
5. Does the ending pay off something established earlier?
If the premise depends on random coincidence, unexplained behavior, teleporting objects/people, or an event that exists only to create spectacle, replace or rewrite it.

CHARACTER AGENCY:
The protagonist should make decisions and react intelligently within the situation. Do not make characters irrational simply to force the plot forward.

RETENTION:
Open immediately with the concrete situation. Create a new meaningful question, complication, decision, reveal or consequence whenever the story naturally demands one. Do not pad the narration with generic introductions or filler.

NATURAL STORYTELLING:
Use conversational spoken English. Write like a human telling a great short story aloud, not like an AI-generated synopsis or checklist. No moral, lesson, subscribe line, generic outro or unnecessary explanation.

LENGTH:
150-210 spoken words. Target roughly 165-195 words for a 60-90 second narration. Do not pad to hit the number.

VISUAL TRUTH:
Every scene must literally show one meaningful moment from the narration. The narration is the source of truth. No visual gag may contradict or invent story events.

STORYBOARD:
Create 10-14 meaningful visual beats. Each scene's narration must be a contiguous exact excerpt from the master narration, in order, with no missing, duplicated, paraphrased or reordered words. Split at natural sentence/action/reaction boundaries rather than fixed time intervals. Do not create filler scenes.

CHARACTER:
The protagonist is a human represented by the approved simple 2D stickman visual identity. Keep roles and locations consistent.

VISUAL STYLE: {base.STYLE}

ABSOLUTE NO-RANDOMNESS RULES:
- No random animals or creatures appearing solely as jokes.
- No random explosions, disasters or city-wide chaos solely for spectacle.
- No objects randomly becoming alive or attacking people without a real story reason.
- No disconnected sequence of funny images.
- No meaningless surrealism.
- No generic "have you ever wondered" opening.
- No ending that introduces a completely new fact just to manufacture a twist.

Return ONLY valid JSON with this shape:
{{
  "title":"...",
  "hook":"...",
  "narration":"complete spoken narration",
  "characters":[{{"name":"A","role":"human protagonist","appearance":"approved canonical 2D stickman character"}}],
  "locations":["..."],
  "story_structure":{{
    "problem":"...",
    "goal":"...",
    "causal_chain":["initial situation","decision","consequence","response","escalation or development","reveal if used","payoff"],
    "escalations":["meaningful development 1","meaningful development 2","meaningful development 3"],
    "twist":"...",
    "punchline":"..."
  }},
  "scenes":[{{
    "number":1,
    "duration_seconds":3.0,
    "beat_text":"literal event happening now",
    "narration":"CONTIGUOUS EXACT EXCERPT FROM FULL NARRATION",
    "visual_prompt":"literal visible shot of the human stickman protagonist",
    "camera":"...",
    "action":"exact visible action"
  }}],
  "flow_prompt":"...",
  "youtube_description":"...",
  "tags":["...","..."]
}}
"""


def make_prompt(topic: str | None, retry: bool = False, feedback: str = "", attempt: int = 1) -> str:
    old = base.history()
    used = "" if not old else "Avoid repeating these previous titles or near-identical premises: " + "; ".join(old[-30:])

    if topic:
        seed = f"Story seed: {topic}. Build the strongest story possible from this seed."
    else:
        seed = (
            "Invent an original Chispa-Pifia story. Do NOT default to a mystery. Choose naturally among "
            "relatable mishap, social misunderstanding, clever problem-solving, dilemma, suspense, "
            "discovery, ironic situation, comedy, emotional payoff or a fair twist."
        )

    if retry:
        retry_note = (
            f"This is recovery attempt {attempt}. The previous draft was rejected. DO NOT preserve its core "
            "premise, setting, conflict, gimmick, or ending. Build a genuinely different story premise "
            "unless the user explicitly supplied a story seed. Change the setting and central problem when "
            "possible. Do not keep rewriting the same failed idea with different wording. "
            "Fix every reported flaw, especially weak hooks, repetitive beats, fake escalation, coincidence-driven "
            "plotting, predictable/unearned endings, irrational character choices, synopsis-like narration, "
            "and visual ambiguity."
        )
    else:
        retry_note = (
            "Before writing, silently test the premise for originality, causality, character agency, retention "
            "and payoff. If the premise is weak or generic, discard it and invent a stronger one."
        )

    feedback_block = (
        "\nINDEPENDENT REVIEW FEEDBACK — fix every point in the new story:\n" + feedback + "\n"
        if feedback else ""
    )
    return (
        f"{seed}\n{used}\n{retry_note}\n{feedback_block}\n"
        "Write the complete JSON package now. Make the first sentence concrete and immediately interesting. "
        "Use 10-14 meaningful scenes, exact contiguous narration excerpts, and a coherent causal chain."
    )


base.make_prompt = make_prompt


def production_generate(topic: str | None) -> dict:
    """Generate a story with up to five independent quality attempts.

    The original base generator only allowed three attempts. In production that made a
    single strict reviewer decision capable of killing the whole scheduled run. This wrapper
    keeps the structural validator and quality gate but gives the writer more chances and,
    on retry, explicitly forces a new premise instead of repeatedly patching the same draft.
    """
    if not base.KEY:
        raise RuntimeError("GROQ_API_KEY is required")

    client = base.Groq(api_key=base.KEY)
    last = None
    feedback = ""

    for attempt in range(1, 6):
        try:
            prompt = make_prompt(topic, retry=attempt > 1, feedback=feedback, attempt=attempt)
            temperature = 0.82 if attempt == 1 else 0.72
            obj = base.parse_json(base.request(client, prompt, temperature=temperature))
            base.normalize_scenes(obj)

            errors = base.validate(obj)
            if errors:
                last = ValueError("structural validation failed: " + ", ".join(errors[:20]))
                print(f"[SHORTS] structural validation failed on attempt {attempt}: {errors[:10]}")
                feedback = "Machine validation failed: " + ", ".join(errors[:20])
                if attempt < 5:
                    time.sleep(2)
                    continue
                break

            review = base.review_story(client, obj)
            score = int(review.get("overall_score", 0) or 0)
            approved = review.get("approved") is True and score >= 8
            print(f"[SHORTS] independent story review: approved={approved}, score={score}/10, attempt={attempt}/5")
            print("[SHORTS] review scores: " + json.dumps(review.get("scores", {}), ensure_ascii=False))

            if not approved:
                feedback = json.dumps({
                    "overall_score": score,
                    "scores": review.get("scores", {}),
                    "fatal_flaws": review.get("fatal_flaws", []),
                    "rewrite_instructions": review.get("rewrite_instructions", []),
                }, ensure_ascii=False)
                last = ValueError(f"story quality review rejected draft ({score}/10)")
                print(f"[SHORTS] quality gate rejected draft: {feedback[:1800]}")
                if attempt < 5:
                    time.sleep(2)
                    continue
                break

            obj["story_quality_review"] = {
                "reviewer_model": MODEL,
                "approved": True,
                "overall_score": score,
                "scores": review.get("scores", {}),
                "fatal_flaws": review.get("fatal_flaws", []),
            }
            obj["generated_at"] = base.datetime.now(base.timezone.utc).isoformat()
            obj["visual_policy"] = "Narration is source of truth; every image literally reenacts its corresponding contiguous narration excerpt."
            obj["timing_policy"] = "Narration is the master timeline; 10-14 image beats follow exact contiguous spoken phrase excerpts; image durations scale to each excerpt's share of the actual voiceover duration, with no fixed seconds-per-image."
            obj["story_policy"] = "varied_storytime_60_90_seconds"
            obj["style_policy"] = base.STYLE
            return obj

        except Exception as exc:
            last = exc
            message = str(exc).lower()
            wait = 65 if any(x in message for x in ("429", "rate limit", "rate_limit")) else 5 + attempt * 3
            print(f"[SHORTS] attempt {attempt} failed: {exc}")
            if attempt < 5:
                time.sleep(wait)

    raise RuntimeError(f"Could not generate a valid Chispa-Pifia story package after production retries: {last}")


if __name__ == "__main__":
    topic = os.environ.get("RELIC_LOOP_TOPIC", "").strip() or None
    obj = production_generate(topic)
    base.write(obj)
    print(
        f"[SHORTS] generated: {obj['title']} | "
        f"{base.words(obj['narration'])} narration words | "
        f"{len(obj['scenes'])} visual beats | model={MODEL}"
    )
