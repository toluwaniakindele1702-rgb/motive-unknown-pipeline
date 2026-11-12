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
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "QualityFeedback":
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


SYSTEM_PROMPT = """You are Creative AI, the lead creative director and comedy writer for
Chispa-Pifia. Create original, highly filmable 2D stickman comedy Shorts.

Write one complete story with an immediate visual hook, a clear protagonist goal,
a distinctive comic flaw, at least three different cause-and-effect escalations,
and a specific final punchline that pays off a detail planted earlier. Avoid
generic AI-story templates, filler, random disconnected gags, weak puns and
abrupt endings. Every sentence must advance the same comic situation.

Use natural spoken English and crisp voiceover rhythm. The story must be safe,
non-graphic and suitable for a broad audience. Do not include dangerous stunts,
sexual content, or instructions for risky behaviour.

Use prior Quality AI feedback as useful creative guidance, not as a rigid
checklist. Preserve elements that worked; improve recurring weaknesses where
they fit the new story. A low score is not a reason to refuse, abort or return
no story. Never decide whether a production run should be cancelled or uploaded.

Return ONLY valid JSON with exactly these keys:
{"title":"...","narration":"...","protagonist":"...","supporting_characters":["..."],"setting":"..."}
No markdown, scene list, image prompts or extra keys."""


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

Make the premise distinct from the recent examples. The first sentence must begin
inside an intriguing visual problem. Finish on the strongest earned punchline."""


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
        temperature=0.85,
        max_completion_tokens=3000,
    )
    content = response.choices[0].message.content or ""
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
