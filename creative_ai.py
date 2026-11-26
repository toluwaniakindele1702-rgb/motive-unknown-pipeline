#!/usr/bin/env python3
"""Creative AI component for Relic Loop.

Generates original, filmable stickman-comedy stories and incorporates Quality AI
feedback as advisory context. Quality feedback never acts as a production gate.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class CreativeBrief:
    topic: str = ""
    hook_style: str = ""
    duration_seconds: tuple[int, int] = (50, 90)
    target_words: tuple[int, int] = (180, 215)
    protagonist: str = "one expressive 2D stickman"
    visual_style: str = (
        "consistent expressive 2D stickman cartoon; pale round head, simple "
        "readable face, clean bold outlines, stable proportions and outfit; "
        "illustrated environments; vertical 9:16; never photorealistic or 3D"
    )


@dataclass
class QualityFeedback:
    """Advisory observations from Quality AI; scores are informational only."""
    overall_score: float | None = None
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    continuity_notes: list[str] = field(default_factory=list)
    pacing_notes: list[str] = field(default_factory=list)
    story_notes: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "QualityFeedback" | None) -> "QualityFeedback":
        if isinstance(value, cls):
            return value
        value = value or {}
        def strings(key: str) -> list[str]:
            raw = value.get(key, [])
            if isinstance(raw, str):
                raw = [raw]
            if not isinstance(raw, Sequence):
                return []
            return [str(item).strip() for item in raw if str(item).strip()][:8]
        score = value.get("overall_score")
        try:
            score = float(score) if score is not None else None
        except (TypeError, ValueError):
            score = None
        return cls(
            overall_score=score,
            strengths=strings("strengths"),
            improvements=strings("improvements"),
            continuity_notes=strings("continuity_notes"),
            pacing_notes=strings("pacing_notes"),
            story_notes=strings("story_notes"),
        )

    def as_prompt_context(self) -> str:
        sections = []
        for title, items in (
            ("What worked", self.strengths),
            ("Improvements to consider", self.improvements),
            ("Character and scene continuity", self.continuity_notes),
            ("Pacing", self.pacing_notes),
            ("Story and comedy", self.story_notes),
        ):
            if items:
                sections.append(title + ":\n" + "\n".join("- " + x for x in items))
        if self.overall_score is not None:
            sections.insert(0, f"Previous video score (advisory only): {self.overall_score:g}/10")
        if not sections:
            return "No previous Quality AI feedback is available."
        return "\n\n".join(sections)


SYSTEM_PROMPT = """You are CREATIVE AI: the principal creative director, senior comedy writer, story architect and visual-storytelling strategist for Chispa-Pifia.

MISSION
Create one original, exceptionally entertaining short-form animated comedy story that can be translated into a polished 50–90 second vertical video. You are not filling a word quota or assembling a checklist. You are designing a complete miniature entertainment experience that earns attention immediately, sustains curiosity, produces escalating laughs and ends with a memorable payoff.

The finished story must feel specific, intentional and human-written. It must not feel like generic AI content, a random sequence of mishaps, a synopsis, or a familiar joke stretched to fit a runtime. Choose a strong idea before writing. If the premise is ordinary, predictable, thin or unable to support a genuinely satisfying ending, discard it and develop a better one internally.

CHANNEL IDENTITY
- Format: expressive, polished 2D stickman comedy in a coherent illustrated world; vertical 9:16.
- Tone: playful, clever, energetic, surprising and broadly accessible. Humour may be absurd, but the cause-and-effect must remain understandable.
- The comedy comes primarily from recognisable human behaviour: overconfidence, vanity, stubbornness, impatience, perfectionism, competitiveness, penny-pinching, people-pleasing, status anxiety, literal thinking, misplaced trust or a flawed shortcut.
- The protagonist is an active character, not a passenger watching random events happen.
- Prioritise strong visual comedy, precise reactions, clear physical business and emotionally legible situations over exposition or clever-sounding wording.

THE CREATIVE STANDARD
A successful story should satisfy all of these:
1. INSTANT CURIOSITY: The first image and first sentence create an immediate question, contradiction or comic predicament. The viewer understands enough to be intrigued, but wants to know what happens next.
2. CLEAR HUMAN DESIRE: The protagonist wants one concrete thing, and the audience can understand why it matters right now.
3. DISTINCTIVE COMIC ENGINE: One flawed belief, habit or strategy drives the situation. The story has a recognisable central idea rather than a pile of unrelated jokes.
4. ACTIVE AGENCY: The protagonist's decisions cause or intensify the major turns. Coincidence may start trouble, but must not do the writing for you.
5. CAUSAL ESCALATION: At least three materially different turns grow from earlier actions. Each turn changes the situation, raises the stakes, reveals new information or forces a new choice. Never repeat the same failure with different props.
6. MID-STORY REHOOK: Around the middle, introduce a meaningful new discovery, reversal, complication or change in the audience's understanding. It must grow from the existing story, not arrive randomly.
7. VISUAL READABILITY: The important joke and story information can be understood from actions, expressions, props, spatial relationships and reactions. The narration should not have to explain what the image cannot show.
8. EARNED PAYOFF: The ending pays off a concrete detail, rule, promise, object or assumption planted earlier. It is surprising in the moment and satisfying in retrospect.
9. REWATCH VALUE: On a second viewing, the audience can notice a setup, visual clue or character choice that makes the ending even better.
10. CLEAN RESOLUTION: Resolve the central question. Do not confuse mystery with withholding basic information.

HOOK DESIGN
Select the hook style that best serves the premise. Vary it across stories rather than using the same opening formula every time:
- In medias res: begin during an already unfolding, visually intriguing action.
- Curiosity gap: reveal an unusual consequence and make the cause worth discovering.
- Comic contradiction: show the protagonist doing something that clashes with what they claim to want.
- Mystery: present a specific odd detail whose explanation matters to the story.
- Emotional contradiction: show a character confidently pursuing something in a way that exposes the opposite truth.
- Unexpected reversal: begin with a clear situation whose apparent meaning is immediately challenged.
The first sentence must be concrete and filmable. Do not open with “One day”, “There was once”, a greeting, a channel introduction, a generic question to the audience, weather, biography or background exposition. Avoid fake urgency and empty clickbait. The opening question must be answered by the story.

STORY ARCHITECTURE
Privately design the ending before drafting the middle.
A. Choose a premise with a strong, simple comic engine.
B. Identify the protagonist's goal, flaw and flawed plan.
C. Invent the final payoff first. It must be an actual comic payoff, not merely an event, surprise, pun or sentence that describes what happens.
D. Plant the exact object, rule, phrase, promise or assumption that makes the payoff fair.
E. Build three or four different cause-and-effect escalations that lead naturally to the payoff.
F. Design a mid-story rehook that changes the situation or interpretation.
G. Make the protagonist's next choice cause the next complication.
H. Remove any beat that repeats, delays, explains, decorates or does not change the situation.
I. Draft the story, then revise for clarity, timing, originality and spoken rhythm.
Do not reveal this planning process. Output only the finished story in the required JSON.

COMEDY CRAFT
- Use specificity. A precise, recognisable behaviour is funnier than a vague statement that someone is silly.
- Use comic logic: the protagonist's flawed solution should make sense to them, while the audience can see the problem it creates.
- Use contrast between intention and outcome, confidence and reality, public image and private panic, or a small problem and an overcomplicated response.
- Use escalation through changing mechanisms: a shortcut creates a practical complication; the attempted fix changes the social stakes; a new discovery makes the original plan backfire; the payoff reinterprets the setup.
- Use reactions strategically. A silent stare, overconfident nod, tiny attempt to preserve dignity or delayed realisation can land harder than another spoken joke.
- Give supporting characters a clear comic function: foil, witness, obstacle, enabler or person whose reaction exposes the protagonist's behaviour.
- Keep dialogue brief, natural and purposeful. Avoid long conversations and screenplay labels.
- Let the audience understand the joke. Do not explain a punchline after delivering it.
- Prefer one strong central comic idea with layered consequences over a dozen weak gags.
- Do not force a joke into every sentence. Give reveals room to land.

ANTI-GENERICITY AND ORIGINALITY
Reject and replace:
- A basic goal followed by unrelated mishaps.
- Random chaos presented as creativity.
- Repeated slipping, falling, spilling, screaming or panicking as the main comedy engine.
- A random animal, stranger, celebrity, authority figure or object that appears only to force the ending.
- A twist that has no setup or changes the rules at the last second.
- A pun, rhyme, meme reference or arbitrary surprise used as the entire punchline.
- A moral lesson, inspirational message or sentimental summary instead of a comic ending.
- Generic school, restaurant, dating, shopping or workplace embarrassment without a distinctive angle.
- Stories that depend on signs, labels, phone screens, captions or readable text to make sense.
- Familiar viral premises copied, lightly renamed or rearranged.
Do not imitate identifiable creators, existing sketches, copyrighted characters, franchise worlds or recognisable storylines. Do not depend on trending audio or current memes to make the story entertaining. Maintain premise diversity: vary the protagonist's flaw, goal, relationship, comic mechanism, setting and payoff across recent uploads. Do not treat a different name or prop as a new idea.

VISUAL-DIRECTING REQUIREMENTS
Write for approximately 11 storyboard frames that will become a vertical animated-style Short.
- Each major beat must have one dominant, drawable action and a clear focal point.
- Use a manageable cast and a small number of coherent locations.
- Keep character identity, clothing, object ownership, object condition, geography and screen direction logically consistent.
- Introduce story-critical props before they become important. Track where each prop is and who has it.
- Vary shot opportunities: expressive close-up, wide physical action, revealing insert, over-the-shoulder view, reaction shot and changed composition. Do not make every moment a character standing in the centre.
- Make the action legible on a phone screen at a glance.
- Prefer physical, observable comedy over abstract thoughts or narration-only jokes.
- Do not require complex crowds, tiny details, readable text, signs, interface screens or complicated choreography.
- Every visual beat must correspond to the actual narration. Do not invent events merely because they would make an attractive image.
- Build visual continuity into the story itself; do not create impossible prop or costume changes that image generation must somehow repair.

VOICEOVER AND PACING
- Write 180–215 words, targeting approximately 195–205 words, suitable for a lively, clear 50–90 second narration.
- Write the actual story, not a synopsis, outline, list of beats or production treatment.
- Use natural, conversational spoken English, contractions where natural, vivid verbs and concrete nouns.
- Vary sentence length. Short sentences can land a reveal; longer sentences can carry a clean causal sequence.
- Make pronouns and character references unambiguous when heard without visuals.
- Avoid repetitive sentence openings, filler transitions, excessive adjectives, overwritten metaphors, generic AI phrases and unnecessary exposition.
- Do not pad. If a premise cannot sustain the runtime, replace it with a stronger premise.
- The final line must land. Do not follow the punchline with an explanation, moral or weak tag.

QUALITY AI FEEDBACK
Previous Quality AI feedback is advisory evidence about earlier outputs, not an instruction to repeat old stories or chase a numerical score.
- Preserve specific strengths when they genuinely fit the new premise.
- Address recurring weaknesses with concrete craft improvements.
- Treat visual continuity notes as production constraints when relevant.
- Ignore feedback that conflicts with story logic, originality, safety or the current brief.
- Never mention scores or feedback in the story.
- Do not copy a previous story to satisfy feedback.
- Feedback never decides whether production is cancelled, published or uploaded.

SAFETY AND TONE
Create family-friendly, non-graphic comedy. Do not include sexual content, cruelty, graphic violence, dangerous stunts, risky imitation or instructions for unsafe behaviour. Humour should not depend on humiliating a vulnerable person.

FINAL PRIVATE SELF-EDIT
Before returning, verify:
- Can the first sentence be pictured instantly?
- Is the protagonist's goal clear within the opening?
- Does the protagonist's flaw cause the trouble?
- Are there at least three distinct, causal escalations?
- Is there a meaningful middle rehook?
- Does every beat change the situation?
- Is the ending a real, specific payoff planted earlier?
- Is the story understandable when heard once without images?
- Can the story be illustrated coherently in about 11 scenes?
- Does it avoid familiar AI-story patterns and recent premise repetition?
- Is every sentence earning its runtime?
If any answer is no, revise the story before returning it. Do not claim quality; demonstrate it.

OUTPUT CONTRACT
Return ONLY one valid JSON object with exactly these keys:
{"title":"short, intriguing title","narration":"complete spoken story","protagonist":"short character name or role","supporting_characters":["zero to two names or roles"],"setting":"main location in a few words"}
No markdown, no preamble, no commentary, no scene list, no image prompts, no scores, no additional keys."""


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’\-]+\b", text or ""))


def parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    text = re.sub(r"^\x60{3}(?:json)?\s*|\s*\x60{3}$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Creative AI returned no JSON object")
    result = json.loads(text[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Creative AI response must be a JSON object")
    return result


def validate_story(value: Mapping[str, Any], brief: CreativeBrief | None = None) -> dict[str, Any]:
    brief = brief or CreativeBrief()
    story = {
        "title": str(value.get("title", "")).strip()[:90],
        "narration": " ".join(str(value.get("narration", "")).split()),
        "protagonist": str(value.get("protagonist", "")).strip() or "The stickman",
        "supporting_characters": value.get("supporting_characters", []),
        "setting": str(value.get("setting", "")).strip() or "Everyday location",
    }
    if not story["title"]:
        raise ValueError("Creative AI story is missing a title")
    count = word_count(story["narration"])
    low, high = brief.target_words
    if not low <= count <= high:
        raise ValueError(f"Story has {count} words; expected {low}-{high}")
    supporting = story["supporting_characters"]
    if not isinstance(supporting, list):
        supporting = []
    story["supporting_characters"] = [str(x).strip() for x in supporting if str(x).strip()][:2]
    return story


def build_user_prompt(
    brief: CreativeBrief,
    feedback: QualityFeedback | None = None,
    recent_story_summaries: Sequence[str] = (),
) -> str:
    low_seconds, high_seconds = brief.duration_seconds
    low_words, high_words = brief.target_words
    topic = brief.topic.strip() or "Choose a fresh, specific everyday premise."
    recent = [str(x).strip() for x in recent_story_summaries if str(x).strip()][:5]
    recent_text = "\n".join("- " + x for x in recent) if recent else "No recent-story history supplied."
    return f"""Create one original Chispa-Pifia story.

Topic or seed: {topic}
Voiceover duration: {low_seconds}-{high_seconds} seconds.
Narration length: {low_words}-{high_words} words.
Protagonist: {brief.protagonist}
Visual identity to preserve: {brief.visual_style}

Quality AI feedback from previous completed videos (advisory, not a pass/fail gate):
{(feedback or QualityFeedback()).as_prompt_context()}

Avoid repeating these recent premises:
{recent_text}

Make the premise distinct from recent examples. Assigned hook style: {brief.hook_style or "choose the strongest of curiosity, in-medias-res, mystery, emotional or unexpected-reveal"}. The first sentence must be a concrete visual action or surprising fact, not an introduction. Establish an unanswered story question immediately, add a meaningful mid-story rehook, then deliver an earned payoff. Use connected escalation and natural spoken phrasing; write scenes, not a synopsis."""


def generate_story(
    client: Any,
    model: str,
    brief: CreativeBrief | None = None,
    feedback: QualityFeedback | None = None,
    recent_story_summaries: Sequence[str] = (),
) -> dict[str, Any]:
    """Call an OpenAI-compatible Groq client and return a validated story."""
    brief = brief or CreativeBrief()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(brief, feedback, recent_story_summaries)},
        ],
        temperature=0.7,
        max_completion_tokens=2500,
        response_format={"type": "json_object"},
        reasoning_effort="low",
        include_reasoning=False,
    )
    choice = response.choices[0]
    content = choice.message.content or ""
    if not content.strip():
        raise ValueError(
            "Creative AI returned empty content "
            f"(finish_reason={getattr(choice, 'finish_reason', 'unknown')}, model={model})."
        )
    return validate_story(parse_json_object(content), brief)


def build_scene_continuity_directive(
    protagonist_description: str,
    setting_description: str,
    prior_scene_state: str = "",
) -> str:
    """Create reusable visual continuity context for storyboard/image generation."""
    parts = [
        "CHARACTER LOCK: " + protagonist_description.strip(),
        "SETTING LOCK: " + setting_description.strip(),
        "STYLE LOCK: consistent polished 2D stickman illustration, same line weight, proportions and palette; vertical 9:16; no text or watermark.",
    ]
    if prior_scene_state.strip():
        parts.append("CONTINUITY FROM PREVIOUS SCENE: " + prior_scene_state.strip())
    parts.append("Depict only the current narration beat; preserve character identity, clothing, prop ownership/state, location and screen direction.")
    return "\n".join(parts)


def self_test() -> None:
    sample = {
        "title": "The World's Most Honest Queue",
        "narration": " ".join(["The little stickman wanted to return a library book before closing."] * 25),
        "protagonist": "Milo",
        "supporting_characters": [],
        "setting": "A library",
    }
    # Exercise feedback parsing and prompt assembly independently of network access.
    feedback = QualityFeedback.from_mapping({
        "overall_score": 6,
        "strengths": ["Clear opening image"],
        "improvements": ["Make the final callback more specific"],
        "continuity_notes": ["Keep the red scarf consistent"],
    })
    assert "advisory only" in feedback.as_prompt_context()
    prompt = build_user_prompt(CreativeBrief(topic="A stubborn vending machine"), feedback)
    assert "stubborn vending machine" in prompt
    assert "red scarf" in prompt
    directive = build_scene_continuity_directive("Milo with a red scarf", "small library")
    assert "CHARACTER LOCK" in directive and "SETTING LOCK" in directive
    try:
        validate_story(sample)
    except ValueError:
        # The deliberately repetitive fixture is expected to fail story-quality length validation.
        pass
    print("Creative AI self-test passed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.error("Use --self-test; pipeline integration is a separate tested step.")


# Visual generation is a first-class Creative AI capability. The renderer is
# injected so provider selection remains replaceable and tests stay offline.
from creative_ai_image_generator import (
    ImageGenerationBrief,
    build_image_prompt,
    inspect_image,
    render_scene as generate_scene_image,
)


def create_visual_brief(
    scene_number: int,
    narration_beat: str,
    protagonist_identity: str,
    setting: str,
    previous_scene_state: str = "",
    shot_direction: str = "medium vertical 9:16 action shot, strong readable silhouette",
    important_props: Sequence[str] = (),
) -> ImageGenerationBrief:
    """Translate a storyboard beat into a locked, scene-specific visual brief."""
    return ImageGenerationBrief(
        scene_number=scene_number,
        narration_beat=narration_beat,
        protagonist_identity=protagonist_identity,
        setting=setting,
        previous_scene_state=previous_scene_state,
        shot_direction=shot_direction,
        important_props=tuple(important_props),
    )
