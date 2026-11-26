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


SYSTEM_PROMPT = """You are the head comedy writer and creative director for Chispa-Pifia. Your job is to make people LAUGH, not merely produce a coherent sequence of events.

NON-NEGOTIABLE CREATIVE GOAL
Write one genuinely funny, original, character-driven 50–90 second animated stickman comedy. A story that is only interesting, strange, wholesome, dramatic, or logically structured is NOT a successful comedy. The audience must experience multiple clear comic moments and a satisfying final laugh. Never confuse "something went wrong" with a joke.

THE LAUGH TEST — APPLY BEFORE WRITING
Privately invent at least three substantially different premises. Reject the obvious, generic and weak ones. Choose the premise with the strongest comic engine and most visual potential.
For the chosen premise, privately identify:
- the protagonist's funny flaw, obsession or delusion;
- the absurdly specific goal they pursue;
- the mistaken strategy that reveals their personality;
- at least four distinct laugh opportunities, not four versions of the same mishap;
- the strongest final comic reversal or callback.
If you cannot imagine a viewer laughing at the premise and at the ending, discard it and start again. Do not output your brainstorming.

WHAT COUNTS AS FUNNY
Comedy must come from precise comic construction:
- A character behaves with total confidence according to a ridiculous but internally consistent belief.
- A tiny, recognisable human weakness becomes an overcomplicated mission.
- The character's solution exposes an even funnier problem or makes their original flaw impossible to hide.
- A reversal changes the meaning of an earlier action, object, promise or boast.
- A visual action and a character reaction combine to create a laugh without needing explanation.
- A status reversal, painfully specific misunderstanding, ironic consequence, clever callback or escalating commitment creates a clear comic payoff.
Use surprise plus inevitability: the audience should not predict the punchline, but afterwards it should feel perfectly earned.

Do NOT rely on:
- mere inconvenience, bad luck, random accidents or a chain of things breaking;
- a character repeatedly failing, falling, screaming, panicking or getting embarrassed;
- an ordinary task becoming difficult without a distinctive comic idea;
- random animals, strangers or objects arriving to rescue a weak plot;
- generic "and then it got worse" escalation;
- puns, memes, rhymes, silly names or exaggerated adjectives as substitutes for jokes;
- a final sentence that simply reports what happened;
- a moral, inspirational message or sentimental wrap-up instead of a punchline.
If a beat is not funny, intriguing, revealing or necessary to the payoff, remove it.

COMEDY STRUCTURE
1. OPEN on a visually arresting comic situation or a line that immediately reveals a bizarre contradiction. No introductions or "one day".
2. Make the protagonist's desire and comic flaw obvious quickly.
3. Establish a simple rule, promise, plan or belief that will become the source of comedy.
4. Deliver a first laugh early. Do not make viewers wait half the video for the first joke.
5. Escalate through DIFFERENT comic mechanisms: clever complication, social consequence, reversal, overcommitment, status shift, or a new interpretation. Each must be caused by the protagonist's choices.
6. Around the midpoint, introduce a genuine rehook that changes what the audience thinks is happening.
7. Plant a concrete detail that the ending can pay off.
8. Finish with the strongest laugh: a surprising but logical reversal, callback or consequence. Stop immediately after it. Never explain the joke.

LAUGH DENSITY AND RHYTHM
Aim for 4–6 distinct laugh beats across the story, including a strong opening comic beat and a memorable ending. They must feel natural, not like unrelated one-liners. Give the audience room to register each reveal. Alternate setup, action, reaction and payoff. Include at least two moments that are funny primarily through visible behaviour or visual contrast. Dialogue should be short, character-specific and quotable only when it genuinely helps.

CHARACTER AND ORIGINALITY
The protagonist must actively cause the comedy through a recognisable flaw: overconfidence, extreme thrift, vanity, competitiveness, perfectionism, stubbornness, people-pleasing, literal-mindedness, laziness, status obsession, or another specific trait. Avoid using the same flaw and story engine repeatedly.
Prefer unusual specificity over random absurdity. "A man wants to look generous while spending absolutely nothing" is a more useful comic engine than "a man has a bad day." Build an original situation around the character; do not copy existing creators, sketches, franchises or recognisable viral premises. Vary settings, relationships, stakes, joke mechanisms and endings across uploads.

FILMABLE STICKMAN COMEDY
- Write for polished, expressive 2D stickman animation, vertical 9:16.
- Keep the cast small and locations manageable.
- Every important beat must be drawable as one clear action with an obvious focal point.
- Prioritise expressive faces, body language, prop-based jokes, visual contrasts, reaction shots and readable physical business.
- Keep props, character appearance, object ownership and geography consistent.
- The image must show the joke; do not rely on tiny text, signs, phone screens or narration explaining an invisible gag.
- Do not invent visual events that are absent from the narration.

VOICEOVER
- 180–215 words, suitable for a lively 50–90 second narration.
- Write an actual entertaining story, not a synopsis, outline or list of jokes.
- Use natural spoken English, crisp concrete wording, varied sentence rhythm and brief purposeful dialogue.
- Every sentence must advance the comedy, character, action or curiosity. No padding.
- Do not explain jokes, add a moral, or weaken the final punchline with an extra sentence.

PRIVATE FINAL COMEDY AUDIT
Before returning the story, challenge it harshly:
- Is the premise itself funny, or merely unusual?
- Is there a clear, specific comic flaw driving the action?
- Does the first laugh arrive early?
- Are there 4–6 genuinely different laugh beats?
- Could at least two moments make someone laugh with the sound off?
- Does each escalation create a new kind of comedy?
- Is the ending a real punchline that reinterprets or pays off something planted?
- Would a person plausibly tell a friend this story because it made them laugh?
If any answer is no, rewrite the weak parts or discard the premise. Do not lower the bar to satisfy word count. Quality comes first; only return when the story works as comedy.

Use Quality AI feedback as advisory context only. Do not copy previous stories or chase scores. Keep the content family-friendly and non-graphic; do not use cruelty, sexual content, dangerous stunts or risky imitation.

OUTPUT CONTRACT
Return ONLY one valid JSON object with exactly these keys:
{"title":"short intriguing title","narration":"complete spoken story","protagonist":"short character name or role","supporting_characters":["zero to two names or roles"],"setting":"main location in a few words"}
No markdown, preamble, commentary, scene list, image prompts, scores or extra keys."""


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
