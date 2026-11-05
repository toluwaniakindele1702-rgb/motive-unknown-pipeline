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

VISUAL DESCRIPTION DIRECTOR — HIGH PRIORITY:
The visual_prompt is not a vague caption or a paraphrase of narration. It is the production-ready image description that will be sent directly to the image generator. Write it like an experienced animation art director describing one specific finished frame to an illustrator.
For EVERY scene, specify:
- the exact visible instant: what is happening at this precise moment, not a general summary;
- protagonist placement, body pose, hand/arm action, head direction, facial expression and eye-line;
- the important story prop(s), their exact position/state, and how the character interacts with them;
- the location's distinctive visible features and spatial layout, with foreground, middle-ground and background details;
- a deliberate shot size and angle, plus a clear focal point and visual reading order;
- lighting direction, time-of-day cues, atmosphere and a restrained color/contrast treatment that supports the emotion;
- one or two small contextual details that make the frame feel specific and lived-in, only when supported by the story.
Make descriptions concrete and drawable: name visible objects, positions, gestures, expressions and spatial relationships. Avoid abstract phrases such as "dramatic scene", "he feels nervous", "cinematic atmosphere" or "a mysterious place" unless followed by observable visual evidence. Never rely on text, signs, UI or speech bubbles to explain the story. Do not add props, characters, actions or plot facts that the narration does not establish.
A strong frame must read instantly at mobile size: one dominant action, clear silhouette, expressive acting, simple visual hierarchy, and a background that supports rather than competes with the focal action. Vary shot composition across the storyboard when the story beat changes; do not repeat the same medium shot or standing pose.
OPENING IMAGE — SCENE 1 IS THE MOST IMPORTANT FRAME:
Scene 1 is the first visual impression and must be art-directed as the opening hook, not as a routine establishing shot. Translate the first narration beat into one striking, truthful image: show the protagonist in the middle of the unusual action, urgent problem, visible consequence or revealing discovery. Put the protagonist and the key object/consequence prominently in frame, with a strong readable expression, clear gesture, bold silhouette and immediate visual contrast. Prefer a close or medium shot unless the story absolutely requires a wide view. Make the unanswered visual question obvious at a glance, but do not reveal the ending or invent extra danger. Never use an empty location, neutral standing pose, tiny distant figure or generic room as the opening frame.
Keep each visual_prompt approximately 55-95 words. Write in concrete visual language, not abstract mood labels. For every prompt, describe a frozen, drawable instant and include: (1) protagonist position and unmistakable pose/gesture, (2) exact facial expression and eye-line, (3) the key prop or consequence and its position/state, (4) a distinctive story-supported location detail, (5) foreground/middle-ground/background separation, (6) shot size/angle and focal hierarchy, and (7) light direction and a small set of intentional colors. Use only details supported by the narration. Avoid generic phrases such as "standing in a room", "looking worried", "cinematic scene" or "dramatic lighting" unless made physically specific.

OPENING FRAME QUALITY BAR — SCENE 1 MUST WIN THE FIRST GLANCE:
Treat scene 1 as a thumbnail-like first impression inside the actual story, not a setup illustration. Choose the single most visually arresting instant contained in the opening narration excerpt: the protagonist caught mid-action, the visible consequence of a mistake, an unusual object relationship, a revealing discovery, or a clearly readable dilemma. The protagonist's face/body and the story-critical prop or consequence must occupy enough of the frame to read on a phone. Use a close or medium composition, strong silhouette, clear gesture, purposeful perspective and high local contrast. Make the viewer ask one immediate visual question ("what just happened?" or "what will he do?") without giving away the payoff. The image must still depict only the opening beat; do not borrow a later scene, invent danger, add spectacle, or use misleading clickbait. Avoid a plain full-body standing pose, empty establishing shot, tiny distant protagonist, generic centered composition, or a background that competes with the action. Scene 1 should be the strongest and most specific visual description in the entire storyboard.

STORYBOARD:
Create 10-14 meaningful visual beats. Each scene's narration must be a contiguous exact excerpt from the master narration, in order, with no missing, duplicated, paraphrased or reordered words. Split at natural sentence/action/reaction boundaries rather than fixed time intervals. Do not create filler scenes. Every visual_prompt must be a specific production-ready image description following the visual director requirements above.

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
    "visual_prompt":"55-95 word art-directed image description. Scene 1: visually arresting truthful opening hook, protagonist mid-action, key prop/consequence prominent, expressive face and gesture, clear focal hierarchy, layered setting, deliberate angle and lighting. Every later scene: one precise frozen instant with pose, expression, prop state, setting, depth and composition.",
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
        "OUTPUT BUDGET / COMPLETENESS: Return exactly 10 scenes (not 4, 8, or 9). Keep narration 150-175 words. "
        "Each scene visual_prompt MUST be 55-65 words, concise but drawable; camera and action each one short phrase. "
        "Keep beat_text short. Keep flow_prompt to 1-2 concise sentences, youtube_description to 1-2 sentences, "
        "and tags to 5-8 short strings. Do not repeat the narration inside other fields. "
        "Every required top-level key in the schema must be present, especially flow_prompt, youtube_description, and tags. "
        "Scene narration excerpts must reproduce the full narration exactly, in order, with no omissions or additions. "
        "Use natural sentence/clause boundaries to divide it into exactly 10 meaningful scenes. "
        "Before responding, verify the JSON closes correctly and contains all 10 scenes and all required top-level keys. "
        "Do not end the response early. Do not use markdown fences or commentary."
    )


base.make_prompt = make_prompt


def production_generate(topic: str | None) -> dict:
    """Generate at most two candidates, independently review each, and return the best.

    Candidate 1 is reviewed first. If it passes the strict 8/10 gate, stop immediately.
    Otherwise candidate 2 is generated with the first review's actionable feedback.
    If neither passes, choose the highest-scoring structurally valid candidate.
    Hard cap: two writer calls and two reviewer calls per run.
    """
    if not base.KEY:
        raise RuntimeError("GROQ_API_KEY is required")

    client = base.Groq(api_key=base.KEY)
    candidates: list[tuple[int, dict, dict]] = []
    feedback = ""
    last_error = None

    for attempt in (1, 2):
        try:
            prompt = make_prompt(
                topic,
                retry=(attempt == 2),
                feedback=feedback,
                attempt=attempt,
            )
            obj = base.parse_json(base.request(
                client,
                prompt,
                temperature=0.84 if attempt == 1 else 0.78,
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
                f"[SHORTS] independent review: candidate={attempt}/2, "
                f"approved={approved}, score={score}/10"
            )
            print("[SHORTS] review scores: " + json.dumps(review.get("scores", {}), ensure_ascii=False))

            candidates.append((score, obj, review))
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
            if attempt == 1:
                print("[SHORTS] first candidate failed; generating one distinct replacement only.")
        except Exception as exc:
            last_error = str(exc)
            print(f"[SHORTS] candidate {attempt} failed: {exc}")
            if attempt == 1:
                feedback = (
                    "The previous candidate could not be completed or reviewed. "
                    "Create a fresh, structurally complete story with especially careful JSON and scene validation."
                )

    else:
        selected = None

    if 'selected' not in locals() or selected is None:
        if not candidates:
            raise RuntimeError(
                "No structurally valid, reviewed story candidate was produced within the two-candidate limit: "
                + str(last_error)
            )
        score, obj, review = max(
            candidates,
            key=lambda item: (
                item[0],
                int((item[2].get("scores", {}) or {}).get("hook", 0) or 0),
                int((item[2].get("scores", {}) or {}).get("payoff", 0) or 0),
                int((item[2].get("scores", {}) or {}).get("coherence", 0) or 0),
            ),
        )
        selected = (score, obj, review, 2, True)

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
            "Neither of the maximum two candidates passed the strict 8/10 review gate; "
            "the highest-scoring structurally valid candidate was selected."
            if fallback else None
        ),
        "candidate_count": len(candidates),
        "selected_candidate": attempt_used,
        "max_candidates": 2,
    }
    obj["generated_at"] = base.datetime.now(base.timezone.utc).isoformat()
    obj["visual_policy"] = "Narration is source of truth; every image literally reenacts its corresponding contiguous narration excerpt."
    obj["timing_policy"] = "Narration is the master timeline; 10-14 image beats follow exact contiguous spoken phrase excerpts; image durations scale to each excerpt's share of the actual voiceover duration, with no fixed seconds-per-image."
    obj["story_policy"] = "varied_storytime_60_90_seconds_two_candidate_reviewed"
    obj["style_policy"] = base.STYLE
    print(
        f"[SHORTS] selected candidate {attempt_used}; score={score}/10; "
        f"reviewed_candidates={len(candidates)}; fallback={fallback}; "
        "hard maximum=2 writer calls and 2 reviewer calls."
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
