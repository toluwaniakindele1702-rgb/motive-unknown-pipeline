#!/usr/bin/env python3
"""Standalone Quality AI for Relic Loop.

Reviews completed story/video evidence and emits advisory feedback. This module
never decides whether production, rendering, or publishing may continue.
It has no YouTube, GitHub-write, pipeline-trigger, or deployment integration.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence


DIMENSIONS = (
    "hook", "story_coherence", "pacing", "comedic_payoff",
    "originality", "visual_clarity", "visual_continuity",
    "audio_clarity", "technical_polish",
)

SYSTEM_PROMPT = """You are Quality AI, an evidence-led reviewer for short-form,
family-friendly 2D stickman comedy videos.

Review only the evidence supplied. Never invent details about unseen images,
unheard audio, or unavailable video. Distinguish observed facts from uncertain
inferences. Give specific, actionable feedback, not generic praise.

Assess: opening hook, story coherence, pacing, setup/payoff, originality,
visual clarity, visual continuity, audio clarity, and technical polish.
A missing evidence source must be marked not_assessed, not guessed.
Scores are advisory diagnostic measurements, not pass/fail decisions.

IMPORTANT GOVERNANCE:
- You do not approve, reject, cancel, pause, block, or authorize production.
- You do not decide whether a video is uploaded or published.
- Low scores must never stop a run.
- Return feedback for Creative AI to consider, not commands to production.

Return only JSON with this shape:
{
 "overall_score": number|null,
 "dimensions": {
   "<dimension>": {"score": number|null, "status":"assessed|not_assessed",
                   "evidence":[string], "recommendations":[string]}
 },
 "strengths":[string], "improvements":[string],
 "continuity_notes":[string], "pacing_notes":[string], "story_notes":[string],
 "uncertainties":[string]
}
Scores must be from 1 to 10. Use null and not_assessed when evidence is absent.
Limit each list to 6 concise items. Do not include extra keys."""


@dataclass
class ReviewConfig:
    min_story_words: int = 180
    max_story_words: int = 215
    min_duration_seconds: float = 50.0
    max_duration_seconds: float = 90.0


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'’\-]+\b", text or ""))


def _safe_list(value: Any, limit: int = 6) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return []
    return [str(item).strip()[:500] for item in value if str(item).strip()][:limit]


def _score(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not 1 <= number <= 10:
        return None
    return round(number, 1)


def _json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    text = re.sub(r"^\x60{3}(?:json)?\s*|\s*\x60{3}$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Quality evaluator returned no JSON object")
    result = json.loads(text[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("Quality evaluator response must be a JSON object")
    return result


def _empty_dimensions() -> dict[str, dict[str, Any]]:
    return {
        key: {"score": None, "status": "not_assessed", "evidence": [],
              "recommendations": []}
        for key in DIMENSIONS
    }


def _deterministic_checks(payload: Mapping[str, Any], config: ReviewConfig) -> tuple[dict, list[str], list[str]]:
    """Report verifiable metadata checks without pretending they are artistic review."""
    dimensions = _empty_dimensions()
    observations: list[str] = []
    recommendations: list[str] = []

    narration = str(payload.get("narration") or payload.get("transcript") or "").strip()
    if narration:
        count = _word_count(narration)
        dimensions["story_coherence"]["evidence"].append(
            f"Narration contains {count} words; configured target is "
            f"{config.min_story_words}-{config.max_story_words}."
        )
        dimensions["story_coherence"]["status"] = "assessed"
        if config.min_story_words <= count <= config.max_story_words:
            observations.append("Narration length is within the configured target.")
            dimensions["story_coherence"]["score"] = 7.0
        else:
            recommendations.append(
                f"Review narration length ({count} words) against the "
                f"{config.min_story_words}-{config.max_story_words}-word target."
            )
            dimensions["story_coherence"]["score"] = 5.0
    else:
        observations.append("Narration was not supplied; story content was not assessed.")

    duration = payload.get("duration_seconds")
    if duration is not None:
        try:
            seconds = float(duration)
            in_range = config.min_duration_seconds <= seconds <= config.max_duration_seconds
            dimensions["technical_polish"]["status"] = "assessed"
            dimensions["technical_polish"]["evidence"].append(
                f"Reported duration is {seconds:.2f}s; target is "
                f"{config.min_duration_seconds:g}-{config.max_duration_seconds:g}s."
            )
            dimensions["technical_polish"]["score"] = 7.0 if in_range else 5.0
            if not in_range:
                recommendations.append("Review final runtime against the target duration.")
        except (TypeError, ValueError):
            observations.append("Duration metadata was invalid and was not assessed.")

    scenes = payload.get("scenes")
    if isinstance(scenes, list) and scenes:
        dimensions["visual_continuity"]["evidence"].append(
            f"{len(scenes)} scene descriptor(s) supplied; semantic continuity still "
            "requires visual evidence or human review."
        )
        observations.append("Scene descriptors are available for evaluator review.")

    inspections = payload.get("image_inspections")
    if isinstance(inspections, list) and inspections:
        valid = sum(1 for item in inspections if isinstance(item, Mapping) and item.get("valid") is True)
        dimensions["visual_clarity"]["status"] = "assessed"
        dimensions["visual_clarity"]["evidence"].append(
            f"{valid}/{len(inspections)} supplied image(s) passed the upstream technical image checks. "
            "This does not establish artistic quality."
        )
        dimensions["visual_clarity"]["score"] = round(10 * valid / len(inspections), 1)
        if valid != len(inspections):
            recommendations.append("Inspect technically rejected frames and regenerate or correct them.")
    else:
        observations.append("No image inspection data was supplied; visual quality was not assessed.")

    return dimensions, observations, recommendations


def _normalise_evaluation(raw: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = _empty_dimensions()
    supplied = raw.get("dimensions", {})
    if isinstance(supplied, Mapping):
        for key in DIMENSIONS:
            item = supplied.get(key)
            if not isinstance(item, Mapping):
                continue
            score = _score(item.get("score"))
            status = "assessed" if score is not None and item.get("status") != "not_assessed" else "not_assessed"
            dimensions[key] = {
                "score": score if status == "assessed" else None,
                "status": status,
                "evidence": _safe_list(item.get("evidence")),
                "recommendations": _safe_list(item.get("recommendations")),
            }
    return {
        "overall_score": _score(raw.get("overall_score")),
        "dimensions": dimensions,
        "strengths": _safe_list(raw.get("strengths")),
        "improvements": _safe_list(raw.get("improvements")),
        "continuity_notes": _safe_list(raw.get("continuity_notes")),
        "pacing_notes": _safe_list(raw.get("pacing_notes")),
        "story_notes": _safe_list(raw.get("story_notes")),
        "uncertainties": _safe_list(raw.get("uncertainties")),
    }


def build_evaluator_prompt(payload: Mapping[str, Any], deterministic_observations: Sequence[str]) -> str:
    # Deliberately accept descriptions and metadata, not binary files or credentials.
    allowed = {
        key: payload.get(key)
        for key in (
            "title", "narration", "transcript", "duration_seconds", "scenes",
            "image_descriptions", "image_inspections", "audio_observations",
            "technical_observations", "human_review_notes",
        )
        if payload.get(key) is not None
    }
    return (
        "Review this completed-video evidence. Treat user-supplied descriptions as "
        "evidence, not guaranteed ground truth. Mark unavailable modalities not_assessed.\n\n"
        "VERIFIABLE PRE-CHECKS:\n" + json.dumps(list(deterministic_observations), ensure_ascii=False)
        + "\n\nSUBMISSION:\n" + json.dumps(allowed, ensure_ascii=False)
    )


def review_submission(
    payload: Mapping[str, Any],
    client: Any = None,
    model: str | None = None,
    config: ReviewConfig | None = None,
) -> dict[str, Any]:
    """Produce an advisory review; deterministic mode works without network access."""
    if not isinstance(payload, Mapping):
        raise TypeError("Review payload must be a mapping")
    config = config or ReviewConfig()
    dimensions, observations, recommendations = _deterministic_checks(payload, config)
    evaluation: dict[str, Any] = {
        "overall_score": None, "dimensions": dimensions,
        "strengths": [], "improvements": recommendations,
        "continuity_notes": [], "pacing_notes": [], "story_notes": [],
        "uncertainties": observations,
    }
    evaluator_error = None

    if client is not None and model:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_evaluator_prompt(payload, observations)},
                ],
                temperature=0.2,
                max_completion_tokens=2200,
            )
            model_result = _normalise_evaluation(_json_object(response.choices[0].message.content or ""))
            # Keep deterministic evidence and append model observations; model may not erase facts.
            for key in DIMENSIONS:
                base, candidate = evaluation["dimensions"][key], model_result["dimensions"][key]
                if candidate["status"] == "assessed":
                    base.update(candidate)
                elif base["status"] == "not_assessed" and candidate["evidence"]:
                    base["evidence"] = candidate["evidence"]
            for key in ("strengths", "improvements", "continuity_notes", "pacing_notes", "story_notes", "uncertainties"):
                evaluation[key] = _safe_list(evaluation[key] + model_result[key])
            evaluation["overall_score"] = model_result["overall_score"]
        except Exception as exc:
            evaluator_error = f"{type(exc).__name__}: {exc}"
            evaluation["uncertainties"].append(
                "Language-model evaluation was unavailable; deterministic checks are retained."
            )

    # Derive an aggregate only from dimensions with evidence. Do not imply certainty.
    assessed = [v["score"] for v in evaluation["dimensions"].values()
                if v["status"] == "assessed" and v["score"] is not None]
    if evaluation["overall_score"] is None and assessed:
        evaluation["overall_score"] = round(sum(assessed) / len(assessed), 1)

    return {
        "schema_version": "1.0",
        "review_type": "advisory_quality_review",
        **evaluation,
        "evidence_coverage": {
            "assessed_dimensions": [k for k, v in evaluation["dimensions"].items() if v["status"] == "assessed"],
            "not_assessed_dimensions": [k for k, v in evaluation["dimensions"].items() if v["status"] != "assessed"],
        },
        "confidence": "medium" if len(assessed) >= 4 else "low",
        "evaluator_error": evaluator_error,
        "creative_ai_feedback": {
            "overall_score": evaluation["overall_score"],
            "strengths": evaluation["strengths"],
            "improvements": evaluation["improvements"],
            "continuity_notes": evaluation["continuity_notes"],
            "pacing_notes": evaluation["pacing_notes"],
            "story_notes": evaluation["story_notes"],
        },
        "governance": {
            "advisory_only": True,
            "may_block_production": False,
            "may_cancel_or_pause_run": False,
            "may_authorize_or_trigger_upload": False,
            "production_decision": None,
        },
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run standalone advisory Quality AI review")
    parser.add_argument("input", help="Path to a JSON evidence file")
    parser.add_argument("--output", default="-", help="Output JSON path (default stdout)")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as handle:
        payload = json.load(handle)
    result = review_submission(payload)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
