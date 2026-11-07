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
        "Use 10-14 meaningful scenes, exact contiguous narration excerpts, and a coherent causal chain. "
        "Do not sacrifice story quality to satisfy the JSON fields; the actual narration is the product."
    )


base.make_prompt = make_prompt


def _phase_json(client, system_prompt: str, user_prompt: str, temperature: float, token_budget: int) -> dict:
    """One bounded JSON generation phase with a single repair attempt."""
    last_error = None
    for attempt in range(2):
        try:
            result = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt if attempt == 0 else (
                        user_prompt + "\n\nREPAIR: The previous response was incomplete or invalid. "
                        "Return one complete valid JSON object now. Do not omit required fields. "
                        "Keep the response within the requested scope."
                    )},
                ],
                temperature=temperature if attempt == 0 else max(0.35, temperature - 0.12),
                max_completion_tokens=token_budget,
                response_format={"type": "json_object"},
            )
            content = result.choices[0].message.content or ""
            return base.parse_json(content)
        except Exception as exc:
            last_error = exc
            print(f"[SHORTS] bounded JSON phase attempt {attempt + 1}/2 failed: {exc}")
    raise RuntimeError(f"JSON phase failed after one repair attempt: {last_error}")


STORY_SYSTEM = """You are the lead writer for Chispa-Pifia, a storytime Shorts channel.
Write an original, genuinely entertaining spoken story. Do not default to mystery or twists.
Choose naturally among relatable mishaps, clever problem-solving, social misunderstandings,
dilemmas, suspense, discoveries, irony, comedy, and emotional or satisfying payoffs.

The story must have a concrete protagonist, desire/problem, clear cause and effect,
meaningful escalation, character agency, and an earned payoff. Start inside a specific
situation with a strong first-sentence hook. Use natural conversational spoken English,
not a synopsis, checklist, moral, or generic introduction. No filler, subscribe line,
or forced twist. Length: 150-210 spoken words.

Before answering, silently audit the causal chain and ensure every major event follows
from an established cause or a character decision. Keep the story visually depictable.
Return ONLY one JSON object with exactly these keys:
"title", "hook", "narration", "characters", "locations", "story_structure".
characters must be a non-empty array of objects with name, role, appearance.
The protagonist role must identify a human/person.
story_structure must contain problem, goal, causal_chain (at least 5 separate array items),
escalations (at least 3 separate items), twist, and punchline.
Keep twist and punchline non-empty: if there is no twist, state that plainly; describe the actual payoff in punchline. Never invent a twist just to fill a field.
Do not include scenes, visual prompts, flow_prompt, markdown, or commentary."""


def _story_prompt(topic: str | None, prior_feedback: str = "") -> str:
    old = base.history()
    used = "" if not old else "Avoid repeating these previous titles or near-identical premises: " + "; ".join(old[-30:])
    seed = (
        f"Story seed: {topic}. Build the strongest coherent story from it."
        if topic else
        "Invent an original Chispa-Pifia story. Choose the strongest premise; do not default to a mystery."
    )
    feedback = ("\nPrevious review/validation feedback to fix: " + prior_feedback) if prior_feedback else ""
    return seed + "\n" + used + feedback + "\nWrite the complete story core now."


STORYBOARD_SYSTEM = f"""You are an expert storyboard director for Chispa-Pifia.
The supplied narration is FINAL and is the source of truth. Do not rewrite, shorten,
expand, correct, paraphrase, reorder, or omit any narration words.
Create 10-14 meaningful visual beats, each containing a contiguous exact excerpt copied
from the narration, in the same order. When joined with single spaces and whitespace
normalized, the excerpts must reconstruct the entire narration exactly.
Split at natural sentence/action/reaction boundaries; do not invent events or filler.

Each scene must have: number, duration_seconds, beat_text, narration, visual_prompt,
camera, action. Each visual_prompt must be 55-95 words, describe one frozen drawable
moment, include the approved simple 2D human stickman identity, pose/gesture, expression
and eye-line, story-critical prop and its state, specific setting, depth, shot/angle,
focal hierarchy, and lighting/color. Scene 1 must be a truthful, arresting close/medium
opening image with the protagonist mid-action and the key prop/consequence prominent.
No text, captions, signs, speech bubbles, logos, watermarks, or unsupported story details.
Use only story facts in the supplied package. Vary composition when the beat changes.
Duration should be positive and proportionate to the spoken excerpt, not fixed equally.
Return ONLY JSON with keys flow_prompt and scenes. flow_prompt should briefly describe
the complete storyboard direction and preserve the canonical visual style:
{base.STYLE}"""


def _storyboard_prompt(story: dict) -> str:
    payload = {
        "title": story.get("title", ""),
        "narration": story.get("narration", ""),
        "characters": story.get("characters", []),
        "locations": story.get("locations", []),
        "story_structure": story.get("story_structure", {}),
    }
    return (
        "Build the storyboard for this completed story. Copy narration excerpts exactly; "
        "do not rewrite the narration. Ensure 10-14 meaningful scenes and return complete JSON.\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _attach_storyboard(story: dict, client) -> dict:
    board = _phase_json(
        client, STORYBOARD_SYSTEM, _storyboard_prompt(story),
        temperature=0.48, token_budget=3400,
    )
    if not isinstance(board.get("scenes"), list):
        raise ValueError("storyboard phase returned no scenes array")
    story["scenes"] = board["scenes"]
    story["flow_prompt"] = str(board.get("flow_prompt", "")).strip()
    base.normalize_scenes(story)
    errors = base.validate(story)
    if errors:
        raise ValueError("storyboard validation failed: " + ", ".join(errors[:20]))
    return story


def production_generate(topic: str | None) -> dict:
    """Generate the narration and storyboard in separate bounded calls, then review."""
    if not base.KEY:
        raise RuntimeError("GROQ_API_KEY is required")

    client = base.Groq(api_key=base.KEY)
    feedback = ""
    candidates: list[tuple[int, dict, dict, int]] = []
    last_error = None

    # Two complete story attempts. Each attempt has a compact story-core call,
    # a separate storyboard call (with one repair), and an independent review.
    for attempt in (1, 2):
        try:
            story = _phase_json(
                client, STORY_SYSTEM, _story_prompt(topic, feedback),
                temperature=0.84 if attempt == 1 else 0.76,
                token_budget=1800,
            )
            if not isinstance(story.get("narration"), str):
                raise ValueError("story phase omitted narration")
            narration_words = base.words(story["narration"])
            if not 150 <= narration_words <= 210:
                raise ValueError(f"story narration has {narration_words} words; expected 150-210")
            story = _attach_storyboard(story, client)

            review = base.review_story(client, story)
            score = int(review.get("overall_score", 0) or 0)
            approved = review.get("approved") is True and score >= 8
            print(
                f"[SHORTS] independent review: candidate={attempt}/2, "
                f"approved={approved}, score={score}/10"
            )
            print("[SHORTS] review scores: " + json.dumps(review.get("scores", {}), ensure_ascii=False))
            candidates.append((score, story, review, attempt))
            feedback = json.dumps({
                "overall_score": score,
                "scores": review.get("scores", {}),
                "fatal_flaws": review.get("fatal_flaws", []),
                "rewrite_instructions": review.get("rewrite_instructions", []),
            }, ensure_ascii=False)
            if approved:
                selected = (score, story, review, attempt, False)
                break
            last_error = f"candidate {attempt} did not pass the 8/10 independent review gate"
            print("[SHORTS] reviewer requested a stronger replacement story.")
        except Exception as exc:
            last_error = str(exc)
            print(f"[SHORTS] candidate {attempt} failed: {exc}")
            feedback = (
                "Previous attempt failed: " + str(exc) +
                ". Write a fresh coherent story. Keep narration 150-210 words. "
                "The storyboard will be generated separately, so return only the six story-core fields."
            )

    else:
        selected = None

    if 'selected' not in locals() or selected is None:
        if not candidates:
            raise RuntimeError(
                "No complete, structurally valid story package was produced after two staged attempts: "
                + str(last_error)
            )
        score, story, review, attempt_used = max(
            candidates,
            key=lambda item: (
                item[0],
                int((item[2].get("scores", {}) or {}).get("hook", 0) or 0),
                int((item[2].get("scores", {}) or {}).get("payoff", 0) or 0),
                int((item[2].get("scores", {}) or {}).get("coherence", 0) or 0),
            ),
        )
        selected = (score, story, review, attempt_used, True)

    score, story, review, attempt_used, fallback = selected
    story["story_quality_review"] = {
        "reviewer_model": base.REVIEW_MODEL,
        "approved": bool(review.get("approved") is True and score >= 8),
        "overall_score": score,
        "scores": review.get("scores", {}),
        "fatal_flaws": review.get("fatal_flaws", []),
        "rewrite_instructions": review.get("rewrite_instructions", []),
        "fallback_used": fallback,
        "fallback_reason": (
            "Neither staged candidate passed the strict 8/10 review gate; "
            "the highest-scoring structurally valid candidate was selected."
            if fallback else None
        ),
        "candidate_count": len(candidates),
        "selected_candidate": attempt_used,
        "max_candidates": 2,
        "generation_mode": "story_core_then_storyboard",
    }
    story["generated_at"] = base.datetime.now(base.timezone.utc).isoformat()
    story["visual_policy"] = "Narration is source of truth; every image reenacts its exact contiguous narration excerpt."
    story["timing_policy"] = "Narration is the master timeline; 10-14 phrase-aligned image beats scale to their spoken excerpts."
    story["story_policy"] = "varied_storytime_60_90_seconds_staged_generation_reviewed"
    story["style_policy"] = base.STYLE
    print(
        f"[SHORTS] selected candidate {attempt_used}; score={score}/10; "
        f"reviewed_candidates={len(candidates)}; fallback={fallback}; "
        "generation=separate_story_and_storyboard_phases."
    )
    return story

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
