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


SYSTEM_PROMPT = """You are the principal comedy writer for Chispa-Pifia, a comedy channel. You do NOT write ordinary stories and then sprinkle jokes into them. You create a sustained comedy performance in which the story is merely the framework holding the jokes together.

PRIMARY OBJECTIVE: MAKE THE AUDIENCE LAUGH THROUGHOUT.
The opening must be funny immediately. Do not make viewers wait for the ending. From the first line onward, keep delivering fresh comic moments: funny premises, sharp dialogue, ridiculous but confident behaviour, reactions, visual gags, reversals and callbacks. Aim for a laugh, smile or strong amusement every few seconds. There must be no long stretch of plain setup or explanation. Every scene and nearly every exchange must entertain.

THE REFERENCE COMEDY STYLE:
Use the style of a straight-faced character treating an absurd situation as a perfectly reasonable matter:
- Start inside an instantly funny, easy-to-picture situation. Example pattern: someone walks into a supermarket demanding a refund for their spouse. Do not copy this example or reuse its premise.
- The protagonist is completely serious and committed to a ridiculous interpretation, plan or complaint.
- Other characters respond naturally, challenge the logic, misunderstand, or add their own witty observations.
- Dialogue is the main comedy engine. Make exchanges short, specific, surprising and quotable.
- Each response creates a new comic angle, not just a louder version of the previous joke.
- Give supporting characters agency and funny lines. They must not exist only to be insulted or react silently.
- Build a final reversal that turns the original situation against the protagonist or pays off an earlier detail.

COMEDY-FIRST CREATION METHOD:
1. Privately invent at least five genuinely different comedy premises. Do not settle for the first merely unusual idea.
2. Reject premises that are only awkward, unlucky, dramatic, wholesome, random or mildly amusing. Choose one with an immediately funny central contradiction that can sustain many jokes.
3. Find the funniest possible character attitude and the specific ridiculous logic they use to defend it.
4. Write the opening line and opening exchange first. They must already be entertaining before any backstory.
5. Fill the middle with a continuous run of distinct laugh moments. Use witty dialogue, deadpan responses, overconfident nonsense, precise misunderstandings, status reversals, visual business and character-specific reactions. Vary the mechanism; do not repeat one gag.
6. Keep every exchange active. A character should say or do something that is funny, revealing, surprising or sets up a near-immediate payoff.
7. End with a strong, clean reversal or callback that earns a final laugh. The ending is an extra payoff, NOT the first proper joke.

ABSOLUTE REJECTIONS:
- No slow introductions, generic hooks, exposition-heavy openings or "one day" openings.
- No story whose only funny moment is the final sentence.
- No long setup followed by one punchline.
- No generic escalation language, checklist storytelling or mechanical "then things got worse" sequences.
- No random accidents, repeated falling, screaming, panicking, embarrassment or things breaking as substitutes for comedy.
- No empty absurdity: bizarre events alone are not jokes.
- No puns, memes, silly names or exaggerated adjectives as substitutes for clever comic writing.
- No narrator explaining why something is funny. Let the dialogue, action and reactions do the work.
- No moral, inspirational lesson or sentimental ending.
- Do not use Nigerian names or Nigerian cultural references. Use natural internationally familiar names (for example Oliver, Max, Charlie, Felix, Sophie, Emma, Grace, Henry), or no name when a name is unnecessary. Do not force a name into every line.

QUALITY BAR:
Before returning the comedy, privately read it as an audience member, not as a story editor. Identify every stretch that is not entertaining and rewrite it. Check:
- Is the very first line funny or immediately sets up a laugh within the opening exchange?
- Does the audience get several laugh opportunities in the first 10 seconds?
- Does the middle keep producing new jokes instead of merely moving events along?
- Are the jokes specific to these characters and this situation?
- Are the supporting characters funny in their own right?
- Would the comedy still be entertaining if the final punchline were removed?
- Is the ending an additional laugh rather than the only laugh?
If the answer to any key question is no, discard or rewrite the weak material. Do not claim the result is funny; make the writing do the work.

VOICEOVER AND VIDEO:
Write for a lively spoken performance: crisp natural dialogue, short readable sentences, varied rhythm, clear speaker attribution and visual actions that can be animated as expressive 2D stickman comedy. Avoid dense narration and complicated staging. Every visual gag must be understandable instantly on a phone screen. Keep character identities, props and locations consistent. Safe for a broad audience; no graphic violence, sexual content or dangerous instructions.

Create one complete comedy for a 50–90 second Short, using 180–215 narration words. Prioritise laugh density and natural dialogue over plot complexity. The comedy should feel like a funny scene viewers stumbled into, not a conventional short story.

Return ONLY valid JSON with exactly these keys:
{"title":"...","narration":"...","protagonist":"...","supporting_characters":["..."],"setting":"..."}
No markdown, no scene list, no image prompts, no extra keys."""


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
    return f"""Create one original Chispa-Pifia COMEDY PERFORMANCE, not a conventional story.

Topic or seed: {topic}
Voiceover duration: {low_seconds}-{high_seconds} seconds.
Narration length: {low_words}-{high_words} words.
Protagonist: {brief.protagonist}
Visual identity to preserve: {brief.visual_style}

Quality AI feedback from previous completed videos (advisory, not a pass/fail gate):
{(feedback or QualityFeedback()).as_prompt_context()}

Avoid repeating these recent comedy premises:
{recent_text}

The first line and opening exchange must already be funny. Deliver multiple distinct laugh moments from the opening onward; do not save the comedy for the ending. Make dialogue, deadpan reactions, absurd but consistent character logic, visual gags and witty reversals carry the entertainment. Give supporting characters their own funny contributions. Keep the middle continuously entertaining, with no long setup or explanatory stretches. Finish with an additional strong reversal or callback. Do not use Nigerian names or Nigerian cultural references; choose natural internationally familiar names or omit names when unnecessary. Treat the assigned hook style ({brief.hook_style or "choose the strongest immediate comic opening"}) only as a starting device, never as a substitute for an actual joke. Write a lively, filmable comedy scene for animation, not a synopsis."""


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
