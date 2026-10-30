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

HOOK QUALITY — NON-NEGOTIABLE:
The first sentence is the retention gate. It must immediately present a specific, concrete situation with an unanswered question, consequence, deadline, risk, embarrassment, unusual constraint, or other reason to keep listening. It must sound like the beginning of an actual story, not a topic introduction.
Before writing, silently generate three different first-sentence hooks and choose the strongest one. Reject generic openings such as "Have you ever...", "One day...", "This guy...", "Something strange happened...", "He never expected...", or any vague setup that delays the actual story.
Do not explain the premise before starting it. Start inside the problem.

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

FINAL SILENT QUALITY CHECK:
Before outputting JSON, score the draft mentally on hook, causality, agency, escalation, originality, payoff, naturalness, visual clarity, coherence and retention. If any major category is weak, discard the draft and write a stronger premise instead. Do not output a merely acceptable story when a stronger premise can be created.

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
            "and visual ambiguity. Generate three new hook candidates internally and select the strongest before "
            "writing the narration. The new first sentence must work as a standalone retention hook."
        )
    else:
        retry_note = (
            "Before writing, silently test the premise for originality, causality, character agency, retention "
            "and payoff. If the premise is weak or generic, discard it and invent a stronger one. Generate three "
            "candidate first sentences internally and choose the strongest. The first sentence must start inside "
            "a concrete problem, decision, risk, deadline, embarrassment or unanswered situation; never use a "
            "generic introduction."
        )

    feedback_block = (
        "\nINDEPENDENT REVIEW FEEDBACK — fix every point in the new story:\n" + feedback + "\n"
        if feedback else ""
    )
    return (
        f"{seed}\n{used}\n{retry_note}\n{feedback_block}\n"
        "Write the complete JSON package now. Make the first sentence concrete and immediately interesting. "
        "Use 10-14 meaningful scenes, exact contiguous narration excerpts, and a coherent causal chain. "
        "Do not sacrifice story quality to satisfy the JSON fields; the actual narration is the product."
    )


base.make_prompt = make_prompt


def production_generate(topic: str | None) -> dict:
    """Generate one story package per run; validate locally without a separate LLM reviewer.

    The writer prompt already contains the editorial requirements and silent quality audit.
    A second model call for review, followed by as many as seven rewrites, consumed the
    daily Groq allowance disproportionately. Production now makes exactly one Groq request.
    Invalid structure fails fast and is visible in diagnostics rather than burning more tokens.
    """
    if not base.KEY:
        raise RuntimeError("GROQ_API_KEY is required")

    client = base.Groq(api_key=base.KEY)
    prompt = make_prompt(topic, retry=False, feedback="", attempt=1)
    obj = base.parse_json(base.request(client, prompt, temperature=0.82))
    base.normalize_scenes(obj)

    errors = base.validate(obj)
    if errors:
        raise RuntimeError(
            "Story failed local structural validation; no additional Groq rewrite was attempted "
            "to protect the daily token budget: " + ", ".join(errors[:20])
        )

    # This records that the story passed deterministic checks, not an independent LLM score.
    obj["story_quality_review"] = {
        "reviewer_model": None,
        "approved": None,
        "review_method": "single_pass_writer_self_audit_plus_local_structural_validation",
        "independent_llm_review": False,
        "fallback_used": False,
        "attempt": 1,
    }
    obj["generated_at"] = base.datetime.now(base.timezone.utc).isoformat()
    obj["visual_policy"] = "Narration is source of truth; every image literally reenacts its corresponding contiguous narration excerpt."
    obj["timing_policy"] = "Narration is the master timeline; 10-14 image beats follow exact contiguous spoken phrase excerpts; image durations scale to each excerpt's share of the actual voiceover duration, with no fixed seconds-per-image."
    obj["story_policy"] = "varied_storytime_60_90_seconds_single_pass"
    obj["style_policy"] = base.STYLE
    print("[SHORTS] single-pass story generation; independent LLM review disabled to conserve Groq tokens.")
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
