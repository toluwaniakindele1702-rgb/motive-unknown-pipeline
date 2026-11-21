#!/usr/bin/env python3
"""Isolated orchestration for Creative AI, Quality AI and Maintenance AI.

This pipeline is deliberately separate from Chispa-Pifia V2 production. It
creates a story, optionally hands media to Quality AI, feeds advisory feedback
into the next Creative AI request, and asks Maintenance AI for a repair proposal
if a technical stage fails. It never renders through production adapters,
uploads, publishes, schedules production, or deploys repairs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from creative_ai import CreativeBrief, QualityFeedback, generate_story
from quality_ai import build_client_from_env as build_quality_client, review_submission
from quality_ai_media import cloudflare_config, inspect_media_files
from maintenance_ai import propose_repair


def _groq_client(environ: Mapping[str, str] | None = None):
    env = environ if environ is not None else os.environ
    key = (env.get("GROQ_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is required for Creative AI story generation")
    from groq import Groq
    return Groq(api_key=key)


def _safe_media_paths(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        raise TypeError("Media paths must be a list")
    return [str(Path(value).expanduser()) for value in values if str(value).strip()]


def _maintenance_report(error: Exception, client: Any, source_files: Mapping[str, str] | None):
    if client is None:
        return {"attempted": False, "reason": "No Maintenance AI model client is configured."}
    try:
        proposal = propose_repair(
            failure_context=f"{type(error).__name__}: {error}",
            source_files=source_files or {},
            client=client,
        )
        # Deliberately do not apply or deploy candidate patches here.
        proposal.pop("patch", None)
        return {"attempted": True, "proposal": proposal,
                "deployment": "not_performed", "production_touched": False}
    except Exception as diagnostic_error:
        return {"attempted": True,
                "diagnostic_error": f"{type(diagnostic_error).__name__}: {diagnostic_error}",
                "deployment": "not_performed", "production_touched": False}


def run_pipeline(
    request: Mapping[str, Any],
    *,
    creative_client: Any,
    quality_client: Any = None,
    maintenance_client: Any = None,
    image_renderer: Callable[[dict[str, Any]], Sequence[str]] | None = None,
    video_renderer: Callable[[dict[str, Any], Sequence[str]], str | None] | None = None,
    maintenance_source_files: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run one isolated story/review cycle; media renderers are injected adapters."""
    if not isinstance(request, Mapping):
        raise TypeError("Pipeline request must be a JSON object")

    result: dict[str, Any] = {
        "pipeline": "relic-loop-isolated-v1",
        "status": "running",
        "stages": [],
        "production_touched": False,
        "published": False,
        "deployment": "not_performed",
    }
    try:
        brief = CreativeBrief(
            topic=str(request.get("topic") or ""),
            duration_seconds=(50, 90),
            target_words=(180, 215),
        )
        feedback = QualityFeedback.from_mapping(request.get("previous_quality_feedback"))
        story = generate_story(
            creative_client,
            model=str(request.get("creative_model") or os.getenv("CREATIVE_AI_MODEL", "openai/gpt-oss-120b")),
            brief=brief,
            feedback=feedback,
            recent_story_summaries=request.get("recent_story_summaries") or (),
        )
        result["story"] = story
        result["stages"].append({"name": "creative_ai", "status": "completed"})

        image_paths = _safe_media_paths(request.get("image_paths"))
        if image_renderer is not None:
            image_paths = _safe_media_paths(image_renderer(story))
        video_path = request.get("video_path")
        if video_renderer is not None:
            video_path = video_renderer(story, image_paths)

        evidence: dict[str, Any] = {
            "title": story["title"],
            "narration": story["narration"],
            "scenes": request.get("scenes") or [],
        }
        if image_paths or video_path or request.get("audio_path"):
            account, token, model = cloudflare_config()
            media = inspect_media_files(
                image_paths=image_paths,
                audio_path=request.get("audio_path"),
                video_path=str(video_path) if video_path else None,
                account_id=account,
                api_token=token,
                model=model,
                scene_descriptions=request.get("scene_descriptions") or (),
            )
            evidence.update(media)
        elif request.get("duration_seconds") is not None:
            evidence["duration_seconds"] = request.get("duration_seconds")

        review = review_submission(evidence, client=quality_client,
                                   model=os.getenv("QUALITY_AI_MODEL") if quality_client else None)
        result["quality_review"] = review
        result["next_creative_feedback"] = review["creative_ai_feedback"]
        result["stages"].append({
            "name": "quality_ai", "status": "completed",
            "media_reviewed": bool(image_paths or video_path or request.get("audio_path")),
            "advisory_only": True,
        })
        result["status"] = "completed"
        return result
    except Exception as error:
        result["status"] = "failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        result["stages"].append({"name": "pipeline", "status": "failed"})
        result["maintenance_ai"] = _maintenance_report(
            error, maintenance_client, maintenance_source_files
        )
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the isolated Relic Loop AI pipeline")
    parser.add_argument("--input", required=True, help="JSON request file")
    parser.add_argument("--output", default="relic_loop_result.json")
    args = parser.parse_args()
    request = json.loads(Path(args.input).read_text(encoding="utf-8"))
    client = _groq_client()
    quality_client, _ = build_quality_client()
    result = run_pipeline(request, creative_client=client, quality_client=quality_client,
                          maintenance_client=client)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "story"}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
