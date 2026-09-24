"""
Motive Unknown v2 — curiosity-first automated history video factory.

Design goals
------------
- Daily, unattended GitHub Actions execution.
- Curiosity-driven historical questions instead of generic topics.
- GPT-OSS 120B for research/storytelling; GPT-OSS 20B for lightweight
  structuring/SEO tasks.
- Local/open-weight Kokoro TTS (no paid voice API).
- No per-clip external image API calls. A deterministic cartoon/motion-comic renderer
  creates richer local illustrations quickly and avoids image-provider 429 loops.
- Each narration scene is split into 2-5 visual beats so the picture changes frequently.
- No Ken Burns zoom and no burned-in subtitles.
- Dedicated thumbnail generation separate from video scenes.
- Idempotent stage files so a rerun can skip already-completed stages.
- Safe YouTube default: private uploads until the owner changes the setting.

Required GitHub Secrets
-----------------------
GROQ_API_KEY
YOUTUBE_TOKEN_JSON
YOUTUBE_CLIENT_SECRET_JSON

Optional GitHub Variables / Secrets
-----------------------------------
YOUTUBE_PRIVACY_STATUS     default: private
KOKORO_VOICE               default: am_onyx
KOKORO_SPEED               default: 0.96
CHANNEL_NAME               optional, used in prompts/description

Optional repo asset
-------------------
assets/music/background.mp3  royalty-free / licensed music only

The workflow file supplied with this package runs daily and also supports a
manual "voice_test" mode before committing to a full production run.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
import soundfile as sf
from groq import Groq
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
CURRENT_RUN_DIR = STATE_DIR / "current_run"
WORK_DIR = ROOT / "run_work"
SCENE_DIR = WORK_DIR / "scenes"
AUDIO_DIR = WORK_DIR / "audio"
THUMB_DIR = WORK_DIR / "thumbnails"
OUTPUT_DIR = WORK_DIR / "output"

for d in (STATE_DIR, CURRENT_RUN_DIR, WORK_DIR, SCENE_DIR, AUDIO_DIR, THUMB_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_RESEARCH_MODEL = os.environ.get("GROQ_RESEARCH_MODEL", "openai/gpt-oss-120b")
GROQ_WRITER_MODEL = os.environ.get("GROQ_WRITER_MODEL", "openai/gpt-oss-120b")
GROQ_LIGHT_MODEL = os.environ.get("GROQ_LIGHT_MODEL", "openai/gpt-oss-20b")

KOKORO_VOICE = os.environ.get("KOKORO_VOICE", "am_onyx").strip()
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", "0.96"))

CHANNEL_NAME = os.environ.get("CHANNEL_NAME", "Motive Unknown").strip()
YOUTUBE_PRIVACY_STATUS = os.environ.get("YOUTUBE_PRIVACY_STATUS", "private").strip().lower()
if YOUTUBE_PRIVACY_STATUS not in {"private", "public", "unlisted"}:
    YOUTUBE_PRIVACY_STATUS = "private"

VIDEO_W, VIDEO_H = 1280, 720
VIDEO_FPS = 30
AUDIO_SR = 24000
MUSIC_PATH = ROOT / "assets" / "music" / "background.mp3"

SCENE_MIN = 18
SCENE_MAX = 38
SCRIPT_MIN_WORDS = 1700
SCRIPT_MAX_WORDS = 2600

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Small, reusable color palette for a child-friendly illustrated look.
PAPER = (248, 244, 232)
INK = (38, 37, 35)
WHITE = (255, 255, 255)
BLACK = (15, 15, 15)
SKY = (220, 236, 246)
GRASS = (175, 203, 146)
SAND = (228, 199, 143)
STONE = (166, 163, 150)
RED = (185, 75, 63)
BLUE = (75, 113, 158)
GOLD = (214, 167, 66)
BROWN = (128, 91, 57)
GREEN = (77, 126, 81)
PURPLE = (117, 88, 138)

# ---------------------------------------------------------------------------
# Files / persistence
# ---------------------------------------------------------------------------
HISTORY_PATH = STATE_DIR / "content_history.json"
RUN_META_PATH = STATE_DIR / "last_run.json"
RESEARCH_PATH = WORK_DIR / "research.txt"
TOPIC_PATH = WORK_DIR / "topic.json"
STORY_PATH = WORK_DIR / "story.json"
SEO_PATH = WORK_DIR / "seo.json"
SCRIPT_PATH = WORK_DIR / "script.json"
MANIFEST_PATH = WORK_DIR / "manifest.json"


# Small text-only artifacts survive a fresh GitHub Actions runner.
CURRENT_TOPIC_PATH = CURRENT_RUN_DIR / "topic.json"
CURRENT_RESEARCH_PATH = CURRENT_RUN_DIR / "research.txt"
CURRENT_STORY_PATH = CURRENT_RUN_DIR / "story.json"
CURRENT_SCRIPT_PATH = CURRENT_RUN_DIR / "script.json"
CURRENT_SEO_PATH = CURRENT_RUN_DIR / "seo.json"
CURRENT_UPLOAD_PATH = CURRENT_RUN_DIR / "upload.json"


def atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_history() -> dict[str, Any]:
    data = load_json(HISTORY_PATH, {"videos": []})
    if not isinstance(data, dict) or not isinstance(data.get("videos"), list):
        return {"videos": []}
    return data


def save_history(data: dict[str, Any]) -> None:
    atomic_write_json(HISTORY_PATH, data)


def checkpoint(stage: str, **extra: Any) -> None:
    data = {
        "stage": stage,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    atomic_write_json(RUN_META_PATH, data)
    print(f"[CHECKPOINT] {stage}")


def _save_current_json(path: Path, data: Any) -> None:
    atomic_write_json(path, data)


def _save_current_text(path: Path, content: str) -> None:
    atomic_write_text(path, content)


def _load_current_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return load_json(path, None)


def _load_current_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def clear_current_run() -> None:
    for path in CURRENT_RUN_DIR.glob("*"):
        if path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                print(f"[STATE] Could not remove {path}: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def require_secret(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment secret: {name}")
    return value


def clean_json_text(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def extract_last_json_object(raw: str) -> dict[str, Any]:
    """Find the last balanced JSON object that parses successfully."""
    candidates: list[str] = []
    stack = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}" and stack:
            stack -= 1
            if stack == 0 and start is not None:
                candidates.append(raw[start : i + 1])
                start = None
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    raise RuntimeError("No valid JSON object found in model response.")


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", text))


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_wrap_for_image(text: str, width_chars: int) -> list[str]:
    return textwrap.wrap(normalize_spaces(text), width=width_chars, break_long_words=False)


def safe_filename(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    return text[:max_len].strip("_") or "untitled"


def parse_wait_seconds(resp: requests.Response | Exception, default: float = 8.0) -> float:
    if isinstance(resp, requests.Response):
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return max(float(ra) + 1.0, default)
            except ValueError:
                pass
        match = re.search(r"try again in ([\d.]+)s", resp.text or "", flags=re.IGNORECASE)
        if match:
            return max(float(match.group(1)) + 1.0, default)
    return default


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------
if not GROQ_API_KEY:
    client: Groq | None = None
else:
    client = Groq(api_key=GROQ_API_KEY)


def _exception_status(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status from Groq SDK exceptions."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    text = str(exc)
    match = re.search(r"\b(400|401|403|404|409|429|500|502|503|504)\b", text)
    return int(match.group(1)) if match else None


def _is_non_retryable(exc: Exception) -> bool:
    status = _exception_status(exc)
    if status in {400, 401, 403, 404}:
        return True
    text = str(exc).lower()
    return any(token in text for token in (
        "does not support", "unsupported parameter", "invalid_request_error",
        "invalid api key", "authentication", "permission denied",
    ))


def _retry_wait(exc: Exception, attempt: int, base: float = 8.0) -> float:
    message = str(exc)
    match = re.search(r"try again in ([\d.]+)s", message, flags=re.IGNORECASE)
    if match:
        return max(float(match.group(1)) + 1.0, base * attempt)
    retry_after = getattr(exc, "headers", None)
    if retry_after and hasattr(retry_after, "get"):
        value = retry_after.get("retry-after")
        if value:
            try:
                return max(float(value) + 1.0, base * attempt)
            except (TypeError, ValueError):
                pass
    return base * attempt


def _tool_results_text(response: Any) -> str:
    """Extract server-side browser-search results when message.content is empty."""
    try:
        message = response.choices[0].message
    except Exception:
        return ""

    executed = getattr(message, "executed_tools", None)
    if not executed:
        return ""

    chunks: list[str] = []
    for tool in executed:
        try:
            if hasattr(tool, "model_dump"):
                data = tool.model_dump()
            elif isinstance(tool, dict):
                data = tool
            else:
                data = vars(tool)
        except Exception:
            data = {"tool": str(tool)}

        results = data.get("search_results") if isinstance(data, dict) else None
        if isinstance(results, dict):
            results = results.get("results", [])
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip()
                url = str(item.get("url", "")).strip()
                content = str(item.get("content", "")).strip()
                if title or content:
                    chunks.append(f"TITLE: {title}\nURL: {url}\nSNIPPET: {content}")

    return "\n\n".join(chunks).strip()


def groq_call(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_completion_tokens: int,
    temperature: float = 0.6,
    attempts: int = 5,
) -> str:
    """Normal Groq call with intelligent retry behavior."""
    if client is None:
        raise RuntimeError("GROQ_API_KEY is required for this step.")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort="low",
            )
            content = response.choices[0].message.content or ""
            if content.strip():
                return content.strip()
            raise RuntimeError(f"Groq returned an empty response for model {model}.")
        except Exception as exc:
            last_error = exc
            if _is_non_retryable(exc):
                raise RuntimeError(f"Groq rejected the request for {model}: {exc}") from exc
            if attempt >= attempts:
                break
            wait = _retry_wait(exc, attempt)
            print(f"[GROQ RETRY] {model} attempt {attempt}/{attempts}: {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Groq call failed after {attempts} attempts: {last_error}")


def groq_browser_search(
    model: str,
    prompt: str,
    *,
    max_completion_tokens: int = 2800,
    attempts: int = 3,
) -> str:
    """Run browser search WITHOUT structured output/citation_options.

    Groq documents that Browser Search is not compatible with structured outputs.
    We therefore keep browsing as a plain-text stage and structure the result in a
    separate Groq call. If the SDK returns empty message.content but exposes the
    executed search results, those results are used as the search payload.
    """
    if client is None:
        raise RuntimeError("GROQ_API_KEY is required for this step.")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_completion_tokens,
                temperature=0.4,
                reasoning_effort="low",
                tool_choice="required",
                tools=[{"type": "browser_search"}],
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content

            tool_text = _tool_results_text(response)
            if tool_text:
                print("[GROQ SEARCH] message.content was empty; using executed browser-search results.")
                return tool_text

            raise RuntimeError("Groq browser-search call returned neither final text nor search-result payload.")
        except Exception as exc:
            last_error = exc
            if _is_non_retryable(exc):
                raise RuntimeError(f"Groq browser-search request was rejected: {exc}") from exc
            if attempt >= attempts:
                break
            wait = _retry_wait(exc, attempt, base=10.0)
            print(f"[GROQ SEARCH RETRY] {model} attempt {attempt}/{attempts}: {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Groq browser-search failed after {attempts} attempts: {last_error}")


def groq_json(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_completion_tokens: int,
    temperature: float,
    attempts: int = 3,
) -> dict[str, Any]:
    """Structured JSON call. Never combine this with Browser Search."""
    if client is None:
        raise RuntimeError("GROQ_API_KEY is required for this step.")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise RuntimeError("Groq JSON call returned an empty response.")
            return extract_last_json_object(clean_json_text(raw))
        except Exception as exc:
            last_error = exc
            if _is_non_retryable(exc):
                raise RuntimeError(f"Groq JSON request was rejected for {model}: {exc}") from exc
            if attempt >= attempts:
                break
            wait = _retry_wait(exc, attempt)
            print(f"[GROQ JSON RETRY] {model} attempt {attempt}/{attempts}: {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"Groq JSON call failed after {attempts} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Topic scouting / research
# ---------------------------------------------------------------------------
TOPIC_PROMPT = """
You are a curiosity-first history channel producer.

Your job is NOT to pick a generic history topic. Generate one specific historical
QUESTION that a normal person might genuinely wonder about after hearing it.

Good patterns:
- Why did ...?
- How did ...?
- What happened when ...?
- Why were ... so important to ...?
- How did an ordinary decision cause ...?
- Why did people suddenly start/stop ...?
- What was really happening behind ...?

Prefer stories involving a famous civilization, ruler, event, object, belief, or
place when there is a genuinely strange or surprising question attached to it.
Avoid fake mysteries, conspiracy framing, paranormal claims presented as fact,
and "history of X" topics.

The eventual video should be understandable to an intelligent young teenager,
while still being interesting to adults.

Return exactly this JSON object:
{
  "question": "one irresistible historical question",
  "topic": "short internal topic label",
  "era": "short era/civilization label",
  "why_curious": "2-4 sentences explaining why the question creates curiosity",
  "search_angles": ["angle 1", "angle 2", "angle 3", "angle 4"]
}
""".strip()


def choose_topic(history: dict[str, Any]) -> dict[str, Any]:
    previous = []
    for item in history.get("videos", [])[-60:]:
        q = item.get("question") or item.get("topic") or item.get("title")
        if q:
            previous.append(q)
    history_text = "\n".join(f"- {x}" for x in previous) or "(no previous videos recorded)"

    search_prompt = f"""
Search the web for genuinely curiosity-driven historical story ideas suitable for a
YouTube channel. Focus on real history, museums, universities, archives, reputable
history/reference sources, and questions that make a normal viewer think: 'Why did
that happen?' or 'How was that possible?'

Look across different eras and regions. Favor specific questions rather than broad
subjects. Return useful findings with the names of the historical subject, the
curiosity question it suggests, and enough factual context to judge whether the idea
can support a 10-15 minute video.

Avoid conspiracy claims, paranormal claims presented as fact, and questions already
used in this channel's recent history list:
{history_text}
""".strip()

    search_findings = groq_browser_search(
        GROQ_LIGHT_MODEL,
        search_prompt,
        max_completion_tokens=2600,
        attempts=3,
    )

    selection_prompt = f"""
You are the final topic selector for a curiosity-first history YouTube channel.

Use the web findings below to select ONE specific historical question.
Do not invent a more exciting claim than the evidence supports.
Do not repeat or closely imitate the channel's previous questions.
The question should be naturally intriguing to a young teenager and an adult, while
having enough real evidence for a 10-15 minute story.

Previous channel questions:
{history_text}

WEB FINDINGS:
{search_findings}

Return exactly this JSON object:
{{
  "question": "one specific, irresistible historical question",
  "topic": "short internal topic label",
  "era": "short era/civilization label",
  "why_curious": "2-4 sentences explaining the curiosity",
  "search_angles": ["angle 1", "angle 2", "angle 3", "angle 4"]
}}
""".strip()

    data = groq_json(
        GROQ_LIGHT_MODEL,
        [{"role": "user", "content": selection_prompt}],
        max_completion_tokens=1100,
        temperature=0.7,
        attempts=3,
    )
    for key in ("question", "topic", "era", "why_curious", "search_angles"):
        if not data.get(key):
            raise RuntimeError(f"Topic selector missing field: {key}")
    if count_words(str(data["question"])) < 4:
        raise RuntimeError("Topic question is too short.")
    if not isinstance(data["search_angles"], list):
        raise RuntimeError("Topic search_angles must be a list.")
    return data


def research_topic(topic: dict[str, Any]) -> str:
    search_prompt = f"""
Research this historical question deeply using browser search:
{topic['question']}

Topic: {topic['topic']}
Era/civilization: {topic['era']}
Search angles: {json.dumps(topic.get('search_angles', []), ensure_ascii=False)}

Find reliable evidence from museums, universities, national archives, reputable
reference works, academic/history institutions, and strong primary-source or
reference pages where possible. Search for the central answer, important people,
places, objects, chronology, competing interpretations, and any claim that is
commonly exaggerated online.

Return detailed search findings. Include source titles and URLs when available.
Clearly distinguish established facts from disputed, legendary, or uncertain claims.
""".strip()

    search_findings = groq_browser_search(
        GROQ_RESEARCH_MODEL,
        search_prompt,
        max_completion_tokens=4200,
        attempts=4,
    )

    synthesis_prompt = f"""
You are the lead historical researcher for a documentary channel.

Turn the browser-search findings below into a compact, accurate research dossier for
another writer. Do not invent facts or sources. Preserve uncertainty and disagreement.

CENTRAL QUESTION:
{topic['question']}

BROWSER-SEARCH FINDINGS:
{search_findings}

Return plain text with exactly these headings:
1. CORE ANSWER
2. STORY BEATS (10-16 numbered beats)
3. IMPORTANT PEOPLE / PLACES / OBJECTS
4. DISPUTES OR UNCERTAINTY
5. SOURCES (at least 6 source titles + URLs when available)

The story writer will use this dossier as the factual backbone, so every surprising
claim should be traceable to the supplied search findings.
""".strip()

    return groq_call(
        GROQ_RESEARCH_MODEL,
        [{"role": "user", "content": synthesis_prompt}],
        max_completion_tokens=4200,
        temperature=0.35,
        attempts=4,
    )


# ---------------------------------------------------------------------------
# Story architecture + script
# ---------------------------------------------------------------------------
STORY_ARCHITECT_PROMPT = """
You are a story architect for a premium history YouTube channel.

Turn the research dossier into a suspenseful STORY PLAN, not a textbook outline.
The video must have one central curiosity question and a satisfying answer.

Structure:
1. Cold open: start with the strangest or most important moment, without a long intro.
2. Immediate question: make the viewer understand what they are trying to figure out.
3. Minimal context: only the background needed to understand the mystery.
4. Escalation: each section should reveal something that changes the viewer's picture.
5. Midpoint turn: a fact, decision, discovery, or contradiction that surprises us.
6. Deepening: show why the obvious explanation is not enough.
7. Payoff: answer the central question using the strongest evidence.
8. Ending: leave the viewer with one memorable implication or final twist grounded in fact.

Avoid chronological padding. Avoid atmosphere-for-atmosphere's-sake.

Return JSON:
{
  "central_question": "...",
  "hook": "1-3 sentence cold open concept",
  "story_arc": [
    {"section": 1, "purpose": "...", "reveal": "...", "required_facts": ["..."]}
  ],
  "ending_payoff": "..."
}
""".strip()


def build_story_plan(topic: dict[str, Any], research: str) -> dict[str, Any]:
    base_prompt = (
        STORY_ARCHITECT_PROMPT
        + "\n\nTOPIC:\n"
        + json.dumps(topic, ensure_ascii=False)
        + "\n\nRESEARCH DOSSIER:\n"
        + research
    )

    last_plan: dict[str, Any] | None = None
    for attempt in range(1, 4):
        prompt = base_prompt
        if attempt > 1:
            prompt += """
REPAIR PASS: The previous story plan was too thin.
Return at least 10 substantial story_arc sections. Each section must contain:
section (integer), purpose, reveal, and required_facts (2-4 evidence-backed facts).
Do not repeat sections or pad with generic suspense.
""".strip()
        try:
            plan = groq_json(
                GROQ_LIGHT_MODEL,
                [{"role": "user", "content": prompt}],
                max_completion_tokens=4200,
                temperature=0.45,
                attempts=2,
            )
        except Exception as exc:
            print(f"[STORY] architect attempt {attempt} failed: {exc}")
            continue

        last_plan = plan
        arc = plan.get("story_arc")
        if isinstance(arc, list) and len(arc) >= 8:
            valid = all(
                isinstance(item, dict)
                and item.get("purpose")
                and item.get("reveal")
                and isinstance(item.get("required_facts"), list)
                for item in arc
            )
            if valid:
                print(f"[STORY] validated: {len(arc)} story sections")
                return plan

        print(
            f"[STORY] architect attempt {attempt} returned insufficient structure "
            f"({len(arc) if isinstance(arc, list) else 0} sections)"
        )

    keys = sorted(last_plan.keys()) if isinstance(last_plan, dict) else []
    raise RuntimeError(
        "Story architect could not produce at least 8 complete story sections after 3 attempts. "
        f"Last response keys: {keys}"
    )




SCRIPTWRITER_PROMPT = """
You are the head writer of an excellent history storytelling channel.

Write a 10-15 minute narration that answers one irresistible historical question.
The target audience is a curious young teenager AND adults. The language is simple,
but the thinking is not childish.

VOICE OF THE WRITING
- Modern, conversational, confident, vivid.
- Sounds like a brilliant storyteller talking directly to the viewer.
- Short and medium sentences mixed for rhythm.
- Use concrete actions and decisions instead of decorative prose.
- Reveal information in the order that creates curiosity.
- Ask a question only when it genuinely advances the story.
- Use humor lightly when it fits the facts.
- Let characters make choices and let consequences matter.
- Make the viewer feel like they are discovering the answer with you.

DO NOT WRITE LIKE
- a school essay
- an old-fashioned novel
- an encyclopedia
- a travel brochure
- a movie trailer full of fake suspense

Avoid filler such as "the sun was shining," "the water was calm," "little did they know,"
"in the annals of history," and long scenery descriptions unless the detail changes the story.
Never add a fact simply to make the script longer.
Never invent dialogue or inner thoughts and present them as historical facts.
When evidence is uncertain or disputed, say so naturally.
Do not use graphic descriptions.

VERY IMPORTANT: the first 20-30 seconds must make a viewer who has never heard of this
story think: "Wait, why did that happen?"

Return JSON in exactly this shape:
{
  "title_question": "the central curiosity question",
  "era": "short era/civilization label",
  "scenes": [
    {
      "id": 1,
      "narration": "35-100 words of narration for this scene",
      "setting": "simple place description",
      "characters": ["role/person 1", "role/person 2"],
      "action": "what the characters are visibly doing",
      "props": ["prop 1", "prop 2"],
      "mood": "curious / tense / triumphant / confused / etc"
    }
  ],
  "thumbnail": {
    "headline": "2-5 words, not the full title",
    "subject": "main visual subject",
    "supporting_prop": "one strong prop or symbol",
    "emotion": "clear facial/body emotion",
    "composition": "left_subject_right_prop / right_subject_left_prop / central_subject",
    "preferred_variant": 1
  }
}

Scene count: 18-38.
Total narration: 1700-2600 words.
Each scene must describe a distinct, useful visual moment. Do not create a new scene just
because a sentence changed. Scenes can hold for several seconds.
""".strip()


def write_script(topic: dict[str, Any], research: str, plan: dict[str, Any]) -> dict[str, Any]:
    prompt = (
        SCRIPTWRITER_PROMPT
        + "\n\nTOPIC:\n"
        + json.dumps(topic, ensure_ascii=False)
        + "\n\nSTORY PLAN:\n"
        + json.dumps(plan, ensure_ascii=False)
        + "\n\nRESEARCH DOSSIER:\n"
        + research
    )
    script = groq_json(
        GROQ_WRITER_MODEL,
        [{"role": "user", "content": prompt}],
        max_completion_tokens=7200,
        temperature=0.78,
        attempts=3,
    )

    # Do not kill the run just because the model under-produced. First allow a
    # lenient structural validation, then repair the same story to target length.
    validate_script(script, allow_short=True)
    atomic_write_json(SCRIPT_PATH, script)
    _save_current_json(CURRENT_SCRIPT_PATH, script)
    total_words = _script_word_count(script)
    scene_count = len(script["scenes"])
    avg_target = max(50, min(105, round(2050 / scene_count)))
    min_scene_words = max(40, avg_target - 20)
    max_scene_words = min(120, avg_target + 15)
    needs_repair = total_words < SCRIPT_MIN_WORDS or total_words > SCRIPT_MAX_WORDS
    needs_repair = needs_repair or any(
        not (min_scene_words <= count_words(str(scene.get("narration", ""))) <= max_scene_words)
        for scene in script["scenes"]
    )
    if needs_repair:
        print(f"[SCRIPT] repair pass needed: ~{total_words} words")
        script = _repair_script_length(topic, research, plan, script)

    validate_script(script)
    issues = _script_style_issues(script)
    if issues:
        print(f"[SCRIPT] style polish pass needed: {issues}")
        script = _polish_script_style(topic, research, plan, script, issues)

    validate_script(script)
    return script

def validate_script(script: dict[str, Any], allow_short: bool = False) -> None:
    scenes = script.get("scenes")
    if not isinstance(scenes, list) or not (SCENE_MIN <= len(scenes) <= SCENE_MAX):
        raise RuntimeError(f"Script scene count must be {SCENE_MIN}-{SCENE_MAX}; got {len(scenes) if isinstance(scenes, list) else 0}.")
    total_words = 0
    for i, scene in enumerate(scenes, 1):
        if not scene.get("narration"):
            raise RuntimeError(f"Scene {i} is missing narration.")
        words = count_words(scene["narration"])
        if words < 25 or words > 120:
            raise RuntimeError(f"Scene {i} narration is {words} words; expected 25-120.")
        for key in ("setting", "characters", "action", "props", "mood"):
            if key not in scene:
                raise RuntimeError(f"Scene {i} missing {key}.")
        total_words += words
    if not allow_short and not (SCRIPT_MIN_WORDS <= total_words <= SCRIPT_MAX_WORDS):
        raise RuntimeError(f"Total script word count {total_words} outside {SCRIPT_MIN_WORDS}-{SCRIPT_MAX_WORDS}.")
    print(f"[SCRIPT] validated: {len(scenes)} scenes, ~{total_words} words")


SCRIPT_STYLE_RED_FLAGS = (
    "the sun was shining",
    "the water was calm",
    "little did they know",
    "in the annals of history",
    "through the mists of time",
    "from the dawn of time",
)


def _script_word_count(script: dict[str, Any]) -> int:
    return sum(count_words(str(scene.get("narration", ""))) for scene in script.get("scenes", []))


def _script_style_issues(script: dict[str, Any]) -> list[str]:
    full_text = " ".join(str(scene.get("narration", "")) for scene in script.get("scenes", []))
    lower = full_text.lower()
    issues = [f"Avoid stale phrase: {phrase}" for phrase in SCRIPT_STYLE_RED_FLAGS if phrase in lower]
    q_count = full_text.count("?")
    if q_count > max(8, len(script.get("scenes", [])) // 2):
        issues.append("Too many rhetorical questions; keep only questions that genuinely advance the story.")
    return issues


def _repair_script_length(topic: dict[str, Any], research: str, plan: dict[str, Any], script: dict[str, Any]) -> dict[str, Any]:
    """
    Expand narration in small batches. Partial progress is written to the
    persistent current-run state after every successful batch.
    """
    current_words = _script_word_count(script)
    scene_count = len(script["scenes"])
    avg_target = max(50, min(105, round(2050 / scene_count)))
    min_scene_words = max(45, avg_target - 20)
    max_scene_words = min(120, avg_target + 15)

    repaired = json.loads(json.dumps(script, ensure_ascii=False))
    atomic_write_json(SCRIPT_PATH, repaired)
    _save_current_json(CURRENT_SCRIPT_PATH, repaired)

    print(
        f"[SCRIPT] batch repair: {current_words} words -> target ~2050 "
        f"({min_scene_words}-{max_scene_words} words/scene)"
    )

    batch_size = 6
    for start in range(0, scene_count, batch_size):
        end = min(scene_count, start + batch_size)
        batch = repaired["scenes"][start:end]
        success = False

        for attempt in range(1, 4):
            prompt = f"""
You are repairing narration for scenes {start + 1}-{end} of a history video.

Expand ONLY the narration text. Do not change scene IDs, setting, characters, action, props, or mood.
Preserve factual meaning and supported claims. Do not invent facts, dialogue, motives, or events.

Each narration should be roughly {min_scene_words}-{max_scene_words} words.
Add useful historical explanation, consequences, decisions, evidence, and transitions.
Do NOT add scenery filler, repetition, generic suspense, fake dialogue, or decorative prose.
Keep the modern, conversational storyteller voice.

Return JSON only:
{{"narrations": ["one narration per scene, in order"]}}

TOPIC:
{json.dumps(topic, ensure_ascii=False)}

STORY PLAN:
{json.dumps(plan, ensure_ascii=False)}

SCENES:
{json.dumps(
    [{"id": scene["id"], "narration": scene["narration"], "setting": scene["setting"],
      "characters": scene["characters"], "action": scene["action"], "props": scene["props"],
      "mood": scene["mood"]} for scene in batch],
    ensure_ascii=False,
)}
""".strip()

            try:
                result = groq_json(
                    GROQ_WRITER_MODEL,
                    [{"role": "user", "content": prompt}],
                    max_completion_tokens=2400,
                    temperature=0.50,
                    attempts=3,
                )
                narrations = result.get("narrations")
                if not isinstance(narrations, list) or len(narrations) != len(batch):
                    raise RuntimeError(
                        f"expected {len(batch)} narrations, got "
                        f"{len(narrations) if isinstance(narrations, list) else 0}"
                    )

                for scene, narration in zip(batch, narrations):
                    words = count_words(str(narration))
                    # Do not fail an otherwise healthy run over a few words on one scene.
                    if not (45 <= words <= 120):
                        raise RuntimeError(
                            f"scene {scene['id']} returned {words} words; accepted range is 45-120"
                        )
                    scene["narration"] = str(narration).strip()

                success = True
                break
            except Exception as exc:
                print(f"[SCRIPT] repair batch {start + 1}-{end}, attempt {attempt} failed: {exc}")

        if not success:
            raise RuntimeError(f"Could not repair narration batch {start + 1}-{end} after 3 attempts.")

        # Persist progress so a fresh scheduled/manual run resumes here.
        atomic_write_json(SCRIPT_PATH, repaired)
        _save_current_json(CURRENT_SCRIPT_PATH, repaired)
        print(f"[SCRIPT] saved repair progress through scene {end} (~{_script_word_count(repaired)} words)")

    final_words = _script_word_count(repaired)

    # If the batches still landed slightly short overall, top up in the same
    # bounded batches rather than failing because of one underlong scene.
    if final_words < SCRIPT_MIN_WORDS:
        deficit = SCRIPT_MIN_WORDS - final_words
        print(f"[SCRIPT] top-up needed: {deficit} words")
        topup_per_scene = max(8, min(30, int(np.ceil(deficit / scene_count)) + 3))

        for start in range(0, scene_count, batch_size):
            end = min(scene_count, start + batch_size)
            batch = repaired["scenes"][start:end]
            prompt = f"""
Lightly expand the narration for these history-video scenes.

Preserve every fact and the existing wording as much as practical. Add only useful explanation,
context, consequences, or transitions. Do not add scenery, repetition, dialogue, or unsupported claims.

Add roughly {topup_per_scene} useful words to EACH narration. Never take any scene above 120 words.

Return JSON only:
{{"narrations": ["one updated narration per scene, in order"]}}

SCENES:
{json.dumps(
    [{"id": scene["id"], "narration": scene["narration"]} for scene in batch],
    ensure_ascii=False,
)}
""".strip()
            result = groq_json(
                GROQ_WRITER_MODEL,
                [{"role": "user", "content": prompt}],
                max_completion_tokens=1800,
                temperature=0.40,
                attempts=3,
            )
            narrations = result.get("narrations")
            if not isinstance(narrations, list) or len(narrations) != len(batch):
                raise RuntimeError("Top-up returned the wrong number of narrations.")
            for scene, narration in zip(batch, narrations):
                words = count_words(str(narration))
                if not (45 <= words <= 120):
                    raise RuntimeError(
                        f"Top-up produced {words} words for scene {scene['id']}; expected 45-120."
                    )
                scene["narration"] = str(narration).strip()
            atomic_write_json(SCRIPT_PATH, repaired)
            _save_current_json(CURRENT_SCRIPT_PATH, repaired)

        final_words = _script_word_count(repaired)

    if not (SCRIPT_MIN_WORDS <= final_words <= SCRIPT_MAX_WORDS):
        raise RuntimeError(
            f"Script repair finished at {final_words} words; "
            f"expected {SCRIPT_MIN_WORDS}-{SCRIPT_MAX_WORDS}."
        )

    validate_script(repaired)
    print(f"[SCRIPT] repair successful: ~{final_words} words")
    return repaired


def _polish_script_style(topic: dict[str, Any], research: str, plan: dict[str, Any], script: dict[str, Any], issues: list[str]) -> dict[str, Any]:
    prompt = f"""
Polish this finished history script because it triggered these style checks:
{json.dumps(issues, ensure_ascii=False)}

Keep the same factual meaning, central question, scene order, characters, settings, actions, props,
and approximate length. Remove old-fashioned novel phrasing, decorative scenery, filler, and excessive
rhetorical questions. Make it sound like a brilliant modern storyteller speaking directly to a curious
teenager and adults. Do not invent facts, dialogue, motives, or events.

Return JSON only in the same schema as the input.

TOPIC:
{json.dumps(topic, ensure_ascii=False)}

STORY PLAN:
{json.dumps(plan, ensure_ascii=False)}

RESEARCH:
{research}

DRAFT:
{json.dumps(script, ensure_ascii=False)}
""".strip()
    polished = groq_json(
        GROQ_WRITER_MODEL,
        [{"role": "user", "content": prompt}],
        max_completion_tokens=6500,
        temperature=0.55,
        attempts=2,
    )
    validate_script(polished)
    return polished


# ---------------------------------------------------------------------------
# Kokoro TTS
# ---------------------------------------------------------------------------
_kokoro_pipeline = None


def get_kokoro_pipeline():
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        try:
            from kokoro import KPipeline
        except Exception as exc:
            raise RuntimeError(
                "Kokoro is not installed. The GitHub workflow should install it "
                "and espeak-ng before running the pipeline."
            ) from exc
        print(f"[TTS] Loading Kokoro voice pipeline for {KOKORO_VOICE}...")
        _kokoro_pipeline = KPipeline(lang_code="a")
    return _kokoro_pipeline


def synthesize_kokoro(text: str) -> np.ndarray:
    pipeline = get_kokoro_pipeline()
    chunks: list[np.ndarray] = []
    generator = pipeline(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED)
    for _, _, audio in generator:
        if audio is None:
            continue
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size:
            chunks.append(arr)
    if not chunks:
        raise RuntimeError("Kokoro produced no audio.")
    return np.concatenate(chunks)


def generate_voiceovers(script: dict[str, Any]) -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(int(AUDIO_SR * 0.04), dtype=np.float32)
    for idx, scene in enumerate(script["scenes"], 1):
        out = AUDIO_DIR / f"scene_{idx:03d}.wav"
        if out.exists() and out.stat().st_size > 10000:
            print(f"[TTS] Reusing {out.name}")
            continue
        print(f"[TTS] Scene {idx}/{len(script['scenes'])}: generating")
        audio = synthesize_kokoro(scene["narration"])
        sf.write(out, audio, AUDIO_SR)
        if idx != len(script["scenes"]):
            # Add a tiny silence to keep cuts clean without sounding like a gap.
            sf.write(out, np.concatenate([audio, silence]), AUDIO_SR)
    checkpoint("voiceovers_complete")


def voice_test() -> Path:
    sample = (
        "For years, historians thought they knew the answer. Then one tiny detail "
        "started causing problems. The people involved had left clues behind. The clues "
        "looked ordinary. But put them together, and the story suddenly changes. Why? "
        "Because the obvious explanation leaves one very strange question unanswered."
    )
    out = ROOT / "voice_test_onyx.wav"
    audio = synthesize_kokoro(sample)
    sf.write(out, audio, AUDIO_SR)
    print(f"[VOICE TEST] wrote {out} ({len(audio) / AUDIO_SR:.1f}s) using {KOKORO_VOICE}")
    return out


# ---------------------------------------------------------------------------
# Cartoon / motion-comic renderer
# ---------------------------------------------------------------------------
def load_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


FONT_24 = load_font(FONT_REGULAR, 24)
FONT_28 = load_font(FONT_BOLD, 28)
FONT_42 = load_font(FONT_BOLD, 42)
FONT_64 = load_font(FONT_BOLD, 64)
FONT_88 = load_font(FONT_BOLD, 88)


def contains_any(text: str, words: Iterable[str]) -> bool:
    text = text.lower()
    return any(w.lower() in text for w in words)


def split_visual_beats(narration: str) -> list[str]:
    """Turn a scene into a small number of visual beats, preserving narration order."""
    text = normalize_spaces(narration)
    if not text:
        return [""]

    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    if not sentences:
        sentences = [text]

    # Split very long sentences at natural punctuation so one image does not sit
    # on screen for too long.
    expanded: list[str] = []
    for sentence in sentences:
        if count_words(sentence) > 22:
            parts = [p.strip() for p in re.split(r"(?<=[,;:])\s+", sentence) if p.strip()]
            if len(parts) > 1:
                bucket = ""
                for part in parts:
                    candidate = f"{bucket} {part}".strip()
                    if count_words(candidate) <= 18 or not bucket:
                        bucket = candidate
                    else:
                        expanded.append(bucket)
                        bucket = part
                if bucket:
                    expanded.append(bucket)
            else:
                expanded.append(sentence)
        else:
            expanded.append(sentence)

    word_total = count_words(text)
    target_beats = max(2, min(5, int(np.ceil(word_total / 18))))
    target_beats = min(target_beats, len(expanded))

    # Merge adjacent short pieces until the target count is reached.
    while len(expanded) > target_beats:
        best_idx = min(
            range(len(expanded) - 1),
            key=lambda i: count_words(expanded[i]) + count_words(expanded[i + 1]),
        )
        expanded[best_idx] = f"{expanded[best_idx]} {expanded[best_idx + 1]}".strip()
        del expanded[best_idx + 1]

    while len(expanded) < target_beats:
        idx = max(range(len(expanded)), key=lambda i: count_words(expanded[i]))
        words = expanded[idx].split()
        if len(words) < 10:
            break
        cut = len(words) // 2
        expanded[idx:idx + 1] = [
            " ".join(words[:cut]),
            " ".join(words[cut:]),
        ]

    return expanded[:5] or [text]


def era_theme(era: str, setting: str) -> dict[str, Any]:
    all_text = f"{era} {setting}".lower()
    if contains_any(all_text, ["egypt", "pharaoh", "nile", "pyramid"]):
        return {"sky_top": (241, 216, 172), "sky_bottom": (249, 237, 210), "ground": (218, 187, 120), "accent": GOLD, "building": (188, 154, 93), "water": (75, 150, 177), "kind": "egypt"}
    if contains_any(all_text, ["china", "chinese", "han", "qin", "ming", "imperial"]):
        return {"sky_top": (170, 211, 237), "sky_bottom": (235, 239, 239), "ground": (188, 176, 143), "accent": RED, "building": (170, 79, 68), "water": (72, 139, 176), "kind": "china"}
    if contains_any(all_text, ["rome", "roman", "latin"]):
        return {"sky_top": (179, 210, 231), "sky_bottom": (238, 235, 224), "ground": (170, 158, 139), "accent": RED, "building": (174, 157, 138), "water": (83, 143, 166), "kind": "rome"}
    if contains_any(all_text, ["viking", "norse", "scandinavia"]):
        return {"sky_top": (151, 193, 219), "sky_bottom": (226, 232, 234), "ground": (135, 157, 141), "accent": BLUE, "building": (121, 91, 67), "water": (65, 126, 163), "kind": "viking"}
    if contains_any(all_text, ["japan", "japanese", "shogun", "samurai"]):
        return {"sky_top": (203, 223, 239), "sky_bottom": (248, 235, 226), "ground": (149, 179, 130), "accent": RED, "building": (151, 89, 74), "water": (83, 140, 171), "kind": "japan"}
    if contains_any(all_text, ["aztec", "maya", "inca", "mesoamerica"]):
        return {"sky_top": (173, 218, 213), "sky_bottom": (232, 239, 224), "ground": (124, 167, 102), "accent": GOLD, "building": (148, 124, 84), "water": (70, 145, 163), "kind": "meso"}
    if contains_any(all_text, ["medieval", "castle", "europe", "kingdom"]):
        return {"sky_top": (166, 201, 226), "sky_bottom": (234, 235, 228), "ground": (151, 174, 137), "accent": PURPLE, "building": (121, 128, 142), "water": (74, 137, 165), "kind": "medieval"}
    return {"sky_top": (169, 216, 240), "sky_bottom": (240, 242, 234), "ground": (163, 193, 133), "accent": BLUE, "building": (143, 151, 160), "water": (80, 145, 172), "kind": "generic"}


def draw_gradient_sky(img: Image.Image, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    strip = Image.new("RGB", (1, VIDEO_H))
    px = strip.load()
    for y in range(VIDEO_H):
        t = y / max(1, VIDEO_H - 1)
        px[0, y] = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
    img.paste(strip.resize((VIDEO_W, VIDEO_H)), (0, 0))


def draw_cloud(draw: ImageDraw.ImageDraw, x: int, y: int, scale: float = 1.0) -> None:
    fill = (246, 248, 245)
    draw.ellipse([x, y + 18, x + int(95 * scale), y + int(55 * scale)], fill=fill)
    draw.ellipse([x + int(30 * scale), y, x + int(120 * scale), y + int(60 * scale)], fill=fill)
    draw.ellipse([x + int(65 * scale), y + int(12 * scale), x + int(155 * scale), y + int(60 * scale)], fill=fill)


def draw_tree(draw: ImageDraw.ImageDraw, x: int, ground_y: int, scale: float = 1.0) -> None:
    trunk = max(7, int(12 * scale))
    draw.line([x, ground_y - int(80 * scale), x, ground_y], fill=BROWN, width=trunk)
    canopy = GREEN
    for ox, oy, r in [(-35, -90, 45), (0, -115, 56), (38, -92, 44), (5, -65, 48)]:
        rr = int(r * scale)
        draw.ellipse([x + int(ox * scale) - rr, ground_y + int(oy * scale) - rr,
                      x + int(ox * scale) + rr, ground_y + int(oy * scale) + rr], fill=canopy)


def draw_background(draw: ImageDraw.ImageDraw, theme: dict[str, Any], setting: str, beat_text: str, seed: int) -> None:
    all_text = f"{setting} {beat_text}".lower()
    # Broad, softly layered cartoon landscape.
    for pts, fill in [
        ([(0, 415), (180, 305), (390, 400), (610, 280), (840, 400), (1060, 295), (1280, 405), (1280, 505), (0, 505)], (208, 211, 196)),
        ([(0, 455), (220, 350), (450, 440), (690, 335), (900, 455), (1120, 355), (1280, 445), (1280, 525), (0, 525)], (184, 190, 173)),
    ]:
        draw.polygon(pts, fill=fill)

    draw.rectangle([0, 500, VIDEO_W, VIDEO_H], fill=theme["ground"])
    draw.ellipse([1065, 62, 1165, 162], fill=(247, 205, 91), outline=(230, 184, 68), width=3)

    rng = random.Random(seed)
    if contains_any(all_text, ["river", "nile", "harbor", "sea", "ship", "lake"]):
        draw.rectangle([0, 400, VIDEO_W, 535], fill=theme["water"])
        for y in [425, 455, 485, 515]:
            for _ in range(8):
                x = rng.randint(30, 1220)
                draw.line([x, y, x + rng.randint(30, 90), y], fill=(154, 205, 214), width=2)
    elif contains_any(all_text, ["desert", "sand"]):
        draw.rectangle([0, 405, VIDEO_W, VIDEO_H], fill=(229, 199, 139))
        for _ in range(18):
            x = rng.randint(0, VIDEO_W)
            y = rng.randint(480, 695)
            draw.arc([x, y, x + 70, y + 25], 180, 360, fill=(204, 168, 111), width=2)
    elif contains_any(all_text, ["street", "market", "town", "city"]):
        for x, h in [(60, 180), (250, 130), (470, 210), (820, 145), (1030, 195)]:
            roof = theme["building"]
            draw.rectangle([x, 500 - h, x + 145, 500], fill=roof, outline=INK, width=4)
            draw.polygon([(x - 8, 500 - h), (x + 70, 455 - h), (x + 153, 500 - h)], fill=theme["accent"], outline=INK)
            for wx in [x + 30, x + 88]:
                draw.rectangle([wx, 390, wx + 28, 430], fill=(242, 218, 156), outline=INK, width=3)
    elif contains_any(all_text, ["palace", "temple", "court", "throne"]):
        draw.rectangle([165, 295, 1115, 500], fill=theme["building"], outline=INK, width=5)
        for x in range(230, 1080, 125):
            draw.rectangle([x, 315, x + 28, 500], fill=(214, 194, 158), outline=INK, width=3)
        draw.polygon([(460, 295), (640, 190), (820, 295)], fill=theme["accent"], outline=INK, width=5)
        draw.rectangle([575, 372, 705, 500], fill=(94, 70, 54), outline=INK, width=4)
    elif contains_any(all_text, ["battlefield", "battle", "war", "camp", "army"]):
        for x in [140, 330, 1030, 1170]:
            draw.line([x, 505, x + 38, 340], fill=BROWN, width=9)
            draw.polygon([(x + 38, 340), (x + 94, 363), (x + 38, 388)], fill=theme["accent"], outline=INK)
        draw.ellipse([720, 420, 790, 490], fill=STONE, outline=INK, width=4)
    elif theme["kind"] == "egypt":
        draw.polygon([(90, 500), (265, 255), (440, 500)], fill=(201, 172, 110), outline=INK)
        draw.polygon([(785, 500), (920, 305), (1055, 500)], fill=(209, 177, 113), outline=INK)
        draw_tree(draw, 1110, 505, 0.7)
    elif theme["kind"] == "china":
        draw.rectangle([130, 315, 1150, 500], fill=theme["building"], outline=INK, width=5)
        draw.polygon([(90, 315), (640, 190), (1190, 315)], fill=RED, outline=INK)
        for x in range(210, 1100, 130):
            draw.rectangle([x, 335, x + 25, 500], fill=(109, 74, 52))
    elif theme["kind"] == "rome":
        for x in [150, 330, 510, 770, 950, 1130]:
            draw.rectangle([x, 280, x + 58, 500], fill=theme["building"], outline=INK, width=3)
            draw.arc([x - 8, 225, x + 66, 315], 180, 360, fill=theme["building"], width=13)
    elif theme["kind"] == "japan":
        draw.polygon([(180, 305), (370, 210), (560, 305)], fill=theme["building"], outline=INK)
        draw.rectangle([240, 305, 500, 500], fill=theme["building"], outline=INK, width=4)
        draw.ellipse([850, 175, 1050, 375], outline=RED, width=22)
        draw_tree(draw, 760, 500, 0.8)
    elif theme["kind"] == "meso":
        draw.polygon([(160, 500), (310, 270), (460, 500)], fill=theme["building"], outline=INK)
        draw.polygon([(800, 500), (950, 260), (1100, 500)], fill=theme["building"], outline=INK)
        draw_tree(draw, 1080, 500, 0.85)
    elif theme["kind"] == "viking":
        draw.polygon([(250, 500), (475, 320), (700, 500)], fill=theme["building"], outline=INK)
        draw.polygon([(610, 500), (860, 300), (1110, 500)], fill=theme["building"], outline=INK)
    else:
        draw_cloud(draw, 120, 95, 0.8)
        draw_cloud(draw, 865, 125, 0.65)
        draw_tree(draw, 120, 505, 0.75)
        draw_tree(draw, 1180, 505, 0.75)

    # A few small background marks make the world feel hand-drawn instead of empty.
    for _ in range(14):
        x = rng.randint(40, 1240)
        y = rng.randint(180, 475)
        draw.arc([x, y, x + 26, y + 14], 200, 330, fill=(125, 135, 139), width=2)


def archetype_style(label: str, era: str) -> dict[str, Any]:
    t = f"{label} {era}".lower()
    if contains_any(t, ["pharaoh", "egyptian", "scribe"]):
        return {"skin": (225, 181, 137), "shirt": GOLD if "pharaoh" in t else (240, 232, 202), "pants": (119, 91, 62), "hair": (54, 44, 39), "kind": "egypt"}
    if contains_any(t, ["roman", "legionary", "centurion"]):
        return {"skin": (226, 184, 141), "shirt": RED, "pants": STONE, "hair": (62, 48, 39), "kind": "rome"}
    if contains_any(t, ["chinese", "emperor", "courtier", "mandarin"]):
        return {"skin": (229, 190, 150), "shirt": RED if "emperor" in t else BLUE, "pants": (69, 61, 57), "hair": (30, 28, 28), "kind": "china"}
    if contains_any(t, ["viking", "norse", "warrior"]):
        return {"skin": (224, 181, 142), "shirt": BROWN, "pants": (73, 68, 62), "hair": (75, 51, 38), "kind": "viking"}
    if contains_any(t, ["samurai", "japanese", "shogun"]):
        return {"skin": (231, 191, 151), "shirt": BLACK if "samurai" in t else RED, "pants": (57, 54, 50), "hair": (30, 28, 28), "kind": "japan"}
    if contains_any(t, ["king", "queen", "monarch", "noble"]):
        return {"skin": (229, 186, 148), "shirt": PURPLE, "pants": (71, 63, 76), "hair": (64, 43, 35), "kind": "royal"}
    return {"skin": (227, 187, 148), "shirt": BLUE, "pants": (68, 75, 84), "hair": (58, 46, 39), "kind": "generic"}


def draw_cartoon_person(
    draw: ImageDraw.ImageDraw,
    x: int,
    ground_y: int,
    scale: float,
    label: str,
    era: str,
    action: str,
    mood: str,
    flip: bool = False,
    beat_index: int = 0,
) -> None:
    style = archetype_style(label, era)
    s = scale
    direction = -1 if flip else 1
    head_r = int(38 * s)
    head_y = ground_y - int(180 * s)
    shoulder_y = head_y + int(58 * s)
    hip_y = ground_y - int(65 * s)
    hand_r = max(7, int(9 * s))
    line_w = max(4, int(5 * s))
    act = f"{action} {mood}".lower()

    # Ground shadow.
    draw.ellipse(
        [x - int(48 * s), ground_y - int(8 * s), x + int(48 * s), ground_y + int(10 * s)],
        fill=(109, 105, 94),
    )

    # Legs and shoes.
    draw.polygon(
        [(x - int(18 * s), hip_y), (x + int(8 * s), hip_y), (x + int(12 * s), ground_y - int(12 * s)),
         (x - int(8 * s), ground_y - int(10 * s))],
        fill=style["pants"], outline=INK,
    )
    draw.polygon(
        [(x + int(5 * s), hip_y), (x + int(25 * s), hip_y), (x + int(45 * s), ground_y - int(12 * s)),
         (x + int(30 * s), ground_y - int(7 * s))],
        fill=style["pants"], outline=INK,
    )
    draw.ellipse([x - int(28 * s), ground_y - int(10 * s), x + int(4 * s), ground_y + int(2 * s)], fill=INK)
    draw.ellipse([x + int(28 * s), ground_y - int(8 * s), x + int(57 * s), ground_y + int(4 * s)], fill=INK)

    # Torso and neck.
    draw.polygon(
        [(x - int(31 * s), shoulder_y), (x + int(31 * s), shoulder_y),
         (x + int(28 * s), hip_y), (x - int(28 * s), hip_y)],
        fill=style["shirt"], outline=INK,
    )
    draw.rectangle([x - int(11 * s), head_y + head_r - 2, x + int(11 * s), shoulder_y + 8], fill=style["skin"], outline=INK, width=line_w)

    # Arms: use the sentence/action to create different poses.
    if contains_any(act, ["point", "pointing", "indicate"]):
        far = (x + direction * int(92 * s), shoulder_y - int(52 * s))
        near = (x - direction * int(48 * s), shoulder_y + int(46 * s))
    elif contains_any(act, ["raise", "signal", "lift", "hold up"]):
        far = (x + direction * int(48 * s), shoulder_y - int(78 * s))
        near = (x - direction * int(50 * s), shoulder_y + int(42 * s))
    elif contains_any(act, ["hold", "carry", "read", "show", "present"]):
        far = (x + direction * int(58 * s), shoulder_y + int(8 * s))
        near = (x - direction * int(34 * s), shoulder_y + int(22 * s))
    elif contains_any(act, ["cross", "think", "consider"]):
        far = (x + direction * int(42 * s), shoulder_y + int(5 * s))
        near = (x + direction * int(5 * s), shoulder_y + int(35 * s))
    elif contains_any(act, ["wave", "greet"]):
        far = (x + direction * int(55 * s), shoulder_y - int(35 * s))
        near = (x + direction * int(74 * s), shoulder_y - int(92 * s))
    else:
        far = (x + direction * int(46 * s), shoulder_y + int(44 * s))
        near = (x - direction * int(46 * s), shoulder_y + int(44 * s))

    for end in (far, near):
        draw.line([x, shoulder_y, *end], fill=INK, width=max(5, int(7 * s)))
        draw.ellipse([end[0] - hand_r, end[1] - hand_r, end[0] + hand_r, end[1] + hand_r], fill=style["skin"], outline=INK, width=2)

    # Head and ears.
    draw.ellipse([x - head_r, head_y - head_r, x + head_r, head_y + head_r], fill=style["skin"], outline=INK, width=line_w)
    for side in (-1, 1):
        draw.ellipse(
            [x + side * (head_r - 2) - int(7 * s), head_y - int(13 * s),
             x + side * (head_r - 2) + int(7 * s), head_y + int(13 * s)],
            fill=style["skin"], outline=INK, width=2,
        )

    # Hair / hat.
    if style["kind"] in {"china", "viking", "japan"}:
        draw.arc([x - head_r + 2, head_y - head_r, x + head_r - 2, head_y + 14], 185, 355, fill=style["hair"], width=int(15 * s))
    elif style["kind"] == "royal":
        draw.polygon(
            [(x - head_r - 3, head_y - int(9 * s)),
             (x - int(20 * s), head_y - head_r - int(16 * s)),
             (x, head_y - int(5 * s)),
             (x + int(20 * s), head_y - head_r - int(16 * s)),
             (x + head_r + 3, head_y - int(9 * s))],
            fill=GOLD, outline=INK,
        )
    elif style["kind"] == "egypt":
        draw.polygon(
            [(x - head_r - 4, head_y - int(4 * s)),
             (x, head_y - head_r - int(23 * s)),
             (x + head_r + 4, head_y - int(4 * s))],
            fill=GOLD, outline=INK,
        )
    else:
        draw.arc([x - head_r + 2, head_y - head_r + 4, x + head_r - 2, head_y + 14], 188, 352, fill=style["hair"], width=int(11 * s))

    # Expressive face.
    face_mood = mood.lower()
    if contains_any(face_mood, ["confused", "curious", "worried", "uncertain"]):
        brow_y = head_y - int(12 * s)
        draw.line([x - int(22 * s), brow_y + int(5 * s), x - int(7 * s), brow_y - int(4 * s)], fill=INK, width=max(2, int(3 * s)))
        draw.line([x + int(7 * s), brow_y - int(4 * s), x + int(22 * s), brow_y + int(5 * s)], fill=INK, width=max(2, int(3 * s)))
        draw.arc([x - int(12 * s), head_y + int(10 * s), x + int(12 * s), head_y + int(25 * s)], 20, 160, fill=INK, width=max(2, int(3 * s)))
    elif contains_any(face_mood, ["triumphant", "happy", "relieved"]):
        draw.line([x - int(22 * s), head_y - int(6 * s), x - int(8 * s), head_y - int(10 * s)], fill=INK, width=max(2, int(3 * s)))
        draw.line([x + int(8 * s), head_y - int(10 * s), x + int(22 * s), head_y - int(6 * s)], fill=INK, width=max(2, int(3 * s)))
        draw.arc([x - int(14 * s), head_y + int(9 * s), x + int(14 * s), head_y + int(31 * s)], 200, 340, fill=INK, width=max(2, int(3 * s)))
    elif contains_any(face_mood, ["angry", "tense", "frustrated"]):
        draw.line([x - int(22 * s), head_y - int(10 * s), x - int(8 * s), head_y - int(3 * s)], fill=INK, width=max(2, int(3 * s)))
        draw.line([x + int(8 * s), head_y - int(3 * s), x + int(22 * s), head_y - int(10 * s)], fill=INK, width=max(2, int(3 * s)))
        draw.line([x - int(11 * s), head_y + int(22 * s), x + int(12 * s), head_y + int(22 * s)], fill=INK, width=max(2, int(3 * s)))
    else:
        draw.line([x - int(21 * s), head_y - int(5 * s), x - int(8 * s), head_y - int(7 * s)], fill=INK, width=max(2, int(3 * s)))
        draw.line([x + int(8 * s), head_y - int(7 * s), x + int(21 * s), head_y - int(5 * s)], fill=INK, width=max(2, int(3 * s)))
        draw.ellipse([x - int(13 * s), head_y + int(2 * s), x - int(5 * s), head_y + int(10 * s)], fill=INK)
        draw.ellipse([x + int(5 * s), head_y + int(2 * s), x + int(13 * s), head_y + int(10 * s)], fill=INK)
        draw.line([x - int(10 * s), head_y + int(24 * s), x + int(11 * s), head_y + int(24 * s)], fill=INK, width=max(2, int(3 * s)))

    # Small period-appropriate clothing cues.
    if style["kind"] == "rome":
        draw.rectangle([x - int(30 * s), shoulder_y + int(3 * s), x + int(30 * s), shoulder_y + int(11 * s)], fill=WHITE)
    if contains_any(label, ["soldier", "warrior", "guard", "samurai", "roman"]):
        draw.polygon(
            [(x + direction * int(50 * s), shoulder_y + int(6 * s)),
             (x + direction * int(77 * s), shoulder_y - int(55 * s)),
             (x + direction * int(86 * s), shoulder_y - int(50 * s)),
             (x + direction * int(60 * s), shoulder_y + int(17 * s))],
            fill=STONE, outline=INK,
        )


def draw_prop(draw: ImageDraw.ImageDraw, x: int, y: int, prop: str, scale: float = 1.0) -> None:
    p = prop.lower()
    s = scale
    if contains_any(p, ["scroll", "document", "letter", "papyrus", "map"]):
        draw.rectangle([x - int(55 * s), y - int(22 * s), x + int(55 * s), y + int(22 * s)], fill=PAPER, outline=INK, width=4)
        draw.line([x - int(35 * s), y - int(4 * s), x + int(32 * s), y - int(4 * s)], fill=(110, 104, 95), width=3)
        draw.line([x - int(28 * s), y + int(8 * s), x + int(22 * s), y + int(8 * s)], fill=(110, 104, 95), width=3)
    elif contains_any(p, ["sword", "blade", "weapon"]):
        draw.line([x - int(12 * s), y + int(42 * s), x + int(62 * s), y - int(42 * s)], fill=STONE, width=int(10 * s))
        draw.line([x - int(5 * s), y + int(25 * s), x + int(20 * s), y + int(48 * s)], fill=BROWN, width=int(9 * s))
        draw.line([x + int(10 * s), y + int(18 * s), x + int(32 * s), y - int(2 * s)], fill=GOLD, width=int(6 * s))
    elif contains_any(p, ["shield"]):
        draw.polygon([(x, y - int(55 * s)), (x + int(45 * s), y - int(30 * s)),
                      (x + int(37 * s), y + int(35 * s)), (x, y + int(58 * s)),
                      (x - int(37 * s), y + int(35 * s)), (x - int(45 * s), y - int(30 * s))],
                     fill=BLUE, outline=INK)
        draw.line([x, y - int(42 * s), x, y + int(43 * s)], fill=WHITE, width=int(5 * s))
    elif contains_any(p, ["crown"]):
        pts = [(x - 42, y + 28), (x - 30, y - 24), (x, y + 5), (x + 28, y - 25), (x + 42, y + 28)]
        draw.polygon(pts, fill=GOLD, outline=INK)
        draw.rectangle([x - 42, y + 20, x + 42, y + 34], fill=GOLD, outline=INK)
    elif contains_any(p, ["torch", "fire"]):
        draw.rectangle([x - 7, y, x + 7, y + 58], fill=BROWN)
        draw.polygon([(x, y - 26), (x - 17, y + 5), (x, y + 18), (x + 17, y + 5)],
                     fill=GOLD, outline=RED)
    elif contains_any(p, ["book", "tablet", "stone"]):
        draw.polygon([(x - 42, y - 32), (x + 33, y - 42), (x + 42, y + 34), (x - 33, y + 42)],
                     fill=STONE, outline=INK)
        for i in [-14, 2, 18]:
            draw.line([x - 22, y + i, x + 22, y + i - 2], fill=INK, width=2)
    elif contains_any(p, ["coin", "gold", "money", "treasure"]):
        for ox in (-24, 0, 24):
            draw.ellipse([x + ox - 16, y - 10, x + ox + 16, y + 22], fill=GOLD, outline=INK, width=3)
    elif contains_any(p, ["ship", "boat"]):
        draw.polygon([(x - 82, y), (x + 82, y), (x + 48, y + 38), (x - 58, y + 38)], fill=BROWN, outline=INK)
        draw.line([x, y, x, y - 100], fill=INK, width=6)
        draw.polygon([(x, y - 92), (x + 55, y - 40), (x, y - 40)], fill=PAPER, outline=INK)
    elif contains_any(p, ["temple", "gate", "building"]):
        draw.rectangle([x - 72, y - 48, x + 72, y + 48], fill=theme_color_from_prop(p), outline=INK, width=4)
        draw.polygon([(x - 88, y - 48), (x, y - 98), (x + 88, y - 48)], fill=GOLD, outline=INK)
    elif contains_any(p, ["desk", "table"]):
        draw.rectangle([x - 75, y - 32, x + 75, y + 2], fill=BROWN, outline=INK, width=4)
        draw.line([x - 55, y + 2, x - 65, y + 55], fill=INK, width=5)
        draw.line([x + 55, y + 2, x + 65, y + 55], fill=INK, width=5)
    else:
        draw.rectangle([x - 34, y - 30, x + 34, y + 30], fill=STONE, outline=INK, width=4)
        draw.ellipse([x - 12, y - 10, x + 12, y + 10], outline=INK, width=3)


def theme_color_from_prop(_: str) -> tuple[int, int, int]:
    return (171, 92, 65)


def symbolic_hint(beat_text: str, mood: str) -> str | None:
    t = f"{beat_text} {mood}".lower()
    if contains_any(t, ["why", "question", "mystery", "wonder"]):
        return "?"
    if contains_any(t, ["secret", "hidden", "unknown", "surprise"]):
        return "!"
    if contains_any(t, ["money", "coin", "tax", "trade", "price"]):
        return "money"
    if contains_any(t, ["law", "rule", "decree", "order", "document", "letter"]):
        return "document"
    if contains_any(t, ["battle", "war", "soldier", "army", "fight"]):
        return "shield"
    if contains_any(t, ["king", "queen", "emperor", "pharaoh", "throne"]):
        return "crown"
    if contains_any(t, ["ship", "sea", "voyage", "sail"]):
        return "ship"
    return None


def draw_symbolic_hint(draw: ImageDraw.ImageDraw, hint: str, x: int, y: int, mood: str) -> None:
    if hint == "?":
        draw.ellipse([x - 48, y - 48, x + 48, y + 48], fill=WHITE, outline=INK, width=4)
        draw.text((x - 17, y - 36), "?", font=FONT_64, fill=INK)
    elif hint == "!":
        draw.ellipse([x - 48, y - 48, x + 48, y + 48], fill=WHITE, outline=INK, width=4)
        draw.text((x - 12, y - 38), "!", font=FONT_64, fill=INK)
    else:
        draw_prop(draw, x, y, hint, 0.9)


def make_scene(
    scene: dict[str, Any],
    era: str,
    output_path: Path,
    beat_text: str = "",
    beat_index: int = 0,
    beat_count: int = 1,
) -> None:
    img = Image.new("RGB", (VIDEO_W, VIDEO_H), PAPER)
    draw_gradient_sky(img, (174, 214, 238), (244, 243, 232))
    draw = ImageDraw.Draw(img)
    theme = era_theme(era, scene.get("setting", ""))
    draw_background(
        draw,
        theme,
        scene.get("setting", ""),
        beat_text,
        hash((scene.get("id", 0), beat_index, beat_text)) & 0xFFFFFFFF,
    )

    chars = scene.get("characters") or ["historical figure"]
    rng = random.Random(hash((scene.get("id", 0), beat_index, "layout")) & 0xFFFFFFFF)
    if len(chars) == 1:
        base = [390, 650, 910][beat_index % 3]
        positions = [(base + rng.randint(-20, 20), 610)]
    else:
        spacing = 250 if len(chars) == 2 else 205
        center = 620 + ((beat_index % 2) * 70 - 35)
        start = center - spacing * (min(len(chars), 4) - 1) / 2
        positions = [(int(start + i * spacing), 610 - (i % 2) * 5) for i in range(min(len(chars), 4))]

    action = str(scene.get("action", ""))
    mood = str(scene.get("mood", "curious"))
    for idx, label in enumerate(chars[:4]):
        x, y = positions[idx]
        local_scale = 1.0 if len(chars) <= 2 else 0.85
        if beat_index % 2 == 1:
            local_scale *= 1.03
        draw_cartoon_person(draw, x, y, local_scale, str(label), era, action, mood, flip=(idx % 2 == 1), beat_index=beat_index)

    props = [str(x) for x in (scene.get("props") or [])[:3]]
    prop_positions = [(1080, 520), (155, 535), (640, 420)]
    for idx, prop in enumerate(props):
        draw_prop(draw, *prop_positions[idx], prop, 1.0 if idx == 0 else 0.88)

    extra = symbolic_hint(beat_text, mood)
    if extra and not any(contains_any(p, [extra]) for p in props):
        hint_pos = (1040, 245) if beat_index % 2 == 0 else (180, 240)
        draw_symbolic_hint(draw, extra, *hint_pos, mood)

    # Foreground framing shapes change across beats like an animated storyboard.
    if beat_index % 3 == 1:
        draw.rectangle([0, 570, 95, VIDEO_H], fill=(103, 84, 69))
        draw.line([70, 570, 120, 470], fill=INK, width=18)
    elif beat_index % 3 == 2:
        draw.rectangle([1185, 570, VIDEO_W, VIDEO_H], fill=(103, 84, 69))
        draw.line([1210, 570, 1165, 470], fill=INK, width=18)

    draw.rectangle([16, 16, VIDEO_W - 16, VIDEO_H - 16], outline=INK, width=6)
    img.save(output_path, quality=94)


def render_scenes(script: dict[str, Any]) -> None:
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    for idx, scene in enumerate(script["scenes"], 1):
        beats = split_visual_beats(str(scene.get("narration", "")))
        for beat_idx, beat_text in enumerate(beats, 1):
            out = SCENE_DIR / f"scene_{idx:03d}_beat_{beat_idx:02d}.jpg"
            if out.exists() and out.stat().st_size > 10000:
                print(f"[SCENE] Reusing {out.name}")
                continue
            print(f"[SCENE] Scene {idx}/{len(script['scenes'])} beat {beat_idx}/{len(beats)}")
            make_scene(
                scene,
                script["era"],
                out,
                beat_text=beat_text,
                beat_index=beat_idx - 1,
                beat_count=len(beats),
            )
    # Legacy one-image-per-scene files are no longer part of the video.
    checkpoint("scenes_complete", visual_beats=sum(len(split_visual_beats(str(s.get("narration", "")))) for s in script["scenes"]))


# ---------------------------------------------------------------------------
# FFmpeg / video assembly
# ---------------------------------------------------------------------------
def run_cmd(args: list[str], label: str) -> None:
    print(f"[FFMPEG] {label}")
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {result.returncode}).\n"
            f"STDERR:\n{result.stderr[-4000:]}"
        )


def audio_duration(path: Path) -> float:
    data, sr = sf.read(path, always_2d=False)
    length = len(data) / sr
    return float(length)


def build_concat_file(paths: list[Path], output: Path, durations: list[float] | None = None) -> None:
    lines: list[str] = []
    for idx, path in enumerate(paths):
        lines.append(f"file '{path.resolve().as_posix()}'")
        if durations is not None:
            lines.append(f"duration {max(0.25, durations[idx]):.4f}")
    if durations is not None and paths:
        lines.append(f"file '{paths[-1].resolve().as_posix()}'")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _visual_beat_durations(narration: str, total_audio_duration: float) -> list[float]:
    beats = split_visual_beats(narration)
    if len(beats) == 1:
        return [total_audio_duration]

    weights = [max(1, count_words(x)) for x in beats]
    total_weight = sum(weights)
    raw = [total_audio_duration * w / total_weight for w in weights]

    # Keep very short flashes readable, then renormalize to the exact audio duration.
    adjusted = [max(1.15, d) for d in raw]
    scale = total_audio_duration / sum(adjusted)
    return [d * scale for d in adjusted]


def build_video(script: dict[str, Any], out_path: Path) -> float:
    scenes = script["scenes"]
    audio_paths = [AUDIO_DIR / f"scene_{i:03d}.wav" for i in range(1, len(scenes) + 1)]
    for path in audio_paths:
        if not path.exists():
            raise RuntimeError(f"Missing narration asset: {path}")

    all_images: list[Path] = []
    all_durations: list[float] = []
    scene_audio_durations: list[float] = []

    for idx, scene in enumerate(scenes, 1):
        audio_path = AUDIO_DIR / f"scene_{idx:03d}.wav"
        duration = audio_duration(audio_path)
        scene_audio_durations.append(duration)
        beats = split_visual_beats(str(scene.get("narration", "")))
        beat_paths = [
            SCENE_DIR / f"scene_{idx:03d}_beat_{beat_idx:02d}.jpg"
            for beat_idx in range(1, len(beats) + 1)
        ]
        for path in beat_paths:
            if not path.exists():
                raise RuntimeError(f"Missing visual beat asset: {path}")
        beat_durations = _visual_beat_durations(str(scene.get("narration", "")), duration)
        all_images.extend(beat_paths)
        all_durations.extend(beat_durations)

    total = sum(scene_audio_durations)
    if not all_images:
        raise RuntimeError("No visual beat images were generated.")

    video_list = WORK_DIR / "video_concat.txt"
    build_concat_file(all_images, video_list, all_durations)
    video_silent = OUTPUT_DIR / "video_silent.mp4"
    run_cmd(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(video_list),
            "-vf", f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-r", str(VIDEO_FPS),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(video_silent),
        ],
        "render motion-comic visual sequence",
    )

    audio_list = WORK_DIR / "audio_concat.txt"
    build_concat_file(audio_paths, audio_list)
    full_audio = OUTPUT_DIR / "full_audio.wav"
    run_cmd(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(audio_list),
            "-ar", str(AUDIO_SR), "-ac", "1", "-c:a", "pcm_s16le",
            str(full_audio),
        ],
        "concatenate narration",
    )

    mixed_audio = OUTPUT_DIR / "full_audio_mixed.m4a"
    if MUSIC_PATH.exists():
        run_cmd(
            [
                "ffmpeg", "-y",
                "-i", str(full_audio),
                "-stream_loop", "-1", "-i", str(MUSIC_PATH),
                "-filter_complex",
                "[0:a]highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11[n];"
                "[1:a]volume=0.06[m];[n][m]amix=inputs=2:duration=first:dropout_transition=2[a]",
                "-map", "[a]", "-c:a", "aac", "-b:a", "192k",
                str(mixed_audio),
            ],
            "mix background music",
        )
    else:
        run_cmd(
            [
                "ffmpeg", "-y", "-i", str(full_audio),
                "-af", "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11",
                "-c:a", "aac", "-b:a", "192k",
                str(mixed_audio),
            ],
            "normalize narration",
        )

    run_cmd(
        [
            "ffmpeg", "-y",
            "-i", str(video_silent),
            "-i", str(mixed_audio),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path),
        ],
        "mux final video",
    )
    print(
        f"[VIDEO] ready: {out_path} (~{total / 60:.1f} min, "
        f"{len(all_images)} visual beats)"
    )
    checkpoint("video_complete", duration_seconds=round(total, 2), visual_beats=len(all_images))
    return total


# ---------------------------------------------------------------------------
# YouTube upload
# ---------------------------------------------------------------------------
def upload_video(video_path: Path, thumbnail_path: Path, seo: dict[str, Any]) -> str:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    token_text = require_secret("YOUTUBE_TOKEN_JSON")
    client_secret_text = require_secret("YOUTUBE_CLIENT_SECRET_JSON")
    token_path = ROOT / "youtube_token.json"
    client_secret_path = ROOT / "client_secret.json"
    token_path.write_text(token_text, encoding="utf-8")
    client_secret_path.write_text(client_secret_text, encoding="utf-8")

    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    credentials = Credentials.from_authorized_user_file(str(token_path), scopes)
    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            raise RuntimeError("YouTube token is invalid/expired and has no usable refresh token.")

    youtube = build("youtube", "v3", credentials=credentials)

    # Avoid duplicate uploads if a previous runner uploaded before another stage failed.
    existing_upload = _load_current_json(CURRENT_UPLOAD_PATH)
    if isinstance(existing_upload, dict) and existing_upload.get("video_id"):
        video_id = str(existing_upload["video_id"])
        print(f"[YOUTUBE] Reusing previously uploaded video {video_id}")
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path),
            ).execute()
            print("[YOUTUBE] thumbnail set on reused upload")
        except Exception as exc:
            print(f"[YOUTUBE] thumbnail upload warning: {exc}")
        return video_id

    body = {
        "snippet": {
            "title": seo["title"][:100],
            "description": seo["description"][:4900],
            "tags": seo["tags"][:15],
            "categoryId": "24",
        },
        "status": {
            "privacyStatus": YOUTUBE_PRIVACY_STATUS,
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", chunksize=-1, resumable=True)
    response = youtube.videos().insert(part="snippet,status", body=body, media_body=media).execute()
    video_id = response["id"]
    _save_current_json(CURRENT_UPLOAD_PATH, {
        "video_id": video_id,
        "privacy_status": YOUTUBE_PRIVACY_STATUS,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    })
    print(f"[YOUTUBE] uploaded {video_id} ({YOUTUBE_PRIVACY_STATUS})")

    try:
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg"),
        ).execute()
        print("[YOUTUBE] thumbnail set")
    except Exception as exc:
        print(f"[YOUTUBE] thumbnail upload warning: {exc}")

    return video_id


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------
def update_history(topic: dict[str, Any], seo: dict[str, Any], video_id: str | None) -> None:
    history = load_history()
    item = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "question": topic.get("question"),
        "topic": topic.get("topic"),
        "era": topic.get("era"),
        "title": seo.get("title"),
        "video_id": video_id,
    }
    history["videos"].append(item)
    # Keep the file useful but bounded.
    history["videos"] = history["videos"][-200:]
    save_history(history)


def save_manifests(topic: dict[str, Any], research: str, story: dict[str, Any], script: dict[str, Any], seo: dict[str, Any]) -> None:
    atomic_write_json(TOPIC_PATH, topic)
    RESEARCH_PATH.write_text(research, encoding="utf-8")
    atomic_write_json(STORY_PATH, story)
    atomic_write_json(SCRIPT_PATH, script)
    atomic_write_json(SEO_PATH, seo)
    atomic_write_json(
        MANIFEST_PATH,
        {
            "topic": topic,
            "script_file": str(SCRIPT_PATH.relative_to(ROOT)),
            "scene_count": len(script.get("scenes", [])),
            "title": seo.get("title"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def main(mode: str = "full") -> None:
    print(f"=== {CHANNEL_NAME} / Motive Unknown v3 ===")
    print(f"MODE={mode} | KOKORO_VOICE={KOKORO_VOICE} | SPEED={KOKORO_SPEED}")

    if mode == "voice_test":
        voice_test()
        return

    require_secret("GROQ_API_KEY")
    require_secret("YOUTUBE_TOKEN_JSON")
    require_secret("YOUTUBE_CLIENT_SECRET_JSON")

    checkpoint("starting")
    history = load_history()

    # Reuse small text artifacts saved by a previous failed Actions run.
    topic = _load_current_json(CURRENT_TOPIC_PATH)
    topic_resumed = isinstance(topic, dict) and bool(topic.get("question"))
    if topic_resumed:
        print("[RESUME] Reusing saved topic.")
    else:
        topic = choose_topic(history)
        _save_current_json(CURRENT_TOPIC_PATH, topic)
    checkpoint("topic_complete", question=topic["question"], resumed=topic_resumed)

    research = _load_current_text(CURRENT_RESEARCH_PATH)
    research_resumed = bool(research)
    if research_resumed:
        print("[RESUME] Reusing saved research.")
    else:
        research = research_topic(topic)
        _save_current_text(CURRENT_RESEARCH_PATH, research)
    RESEARCH_PATH.write_text(research, encoding="utf-8")
    checkpoint("research_complete", resumed=research_resumed)

    story = _load_current_json(CURRENT_STORY_PATH)
    story_resumed = isinstance(story, dict) and bool(story.get("story_arc"))
    if story_resumed:
        print("[RESUME] Reusing saved story plan.")
    else:
        story = build_story_plan(topic, research)
        _save_current_json(CURRENT_STORY_PATH, story)
    atomic_write_json(STORY_PATH, story)
    checkpoint("story_plan_complete", resumed=story_resumed)

    script = _load_current_json(CURRENT_SCRIPT_PATH)
    if isinstance(script, dict):
        try:
            validate_script(script, allow_short=True)
            saved_words = _script_word_count(script)
            if SCRIPT_MIN_WORDS <= saved_words <= SCRIPT_MAX_WORDS:
                print(f"[RESUME] Reusing saved validated script (~{saved_words} words).")
            else:
                print(f"[RESUME] Reusing saved partial script (~{saved_words} words) and continuing repair.")
                script = _repair_script_length(topic, research, story, script)
                _save_current_json(CURRENT_SCRIPT_PATH, script)
        except Exception as exc:
            print(f"[RESUME] Saved script invalid; regenerating: {exc}")
            script = None

    if script is None:
        script = write_script(topic, research, story)


    atomic_write_json(SCRIPT_PATH, script)
    checkpoint("script_complete", scene_count=len(script["scenes"]), words=_script_word_count(script))

    # Audio/scenes are regenerated on fresh runners from the saved script.
    generate_voiceovers(script)
    render_scenes(script)

    seo = _load_current_json(CURRENT_SEO_PATH)
    seo_resumed = isinstance(seo, dict) and bool(seo.get("title") and seo.get("description") and seo.get("tags"))
    if seo_resumed:
        print("[RESUME] Reusing saved SEO package.")
    else:
        seo = build_seo(topic, script, research)
        _save_current_json(CURRENT_SEO_PATH, seo)
    atomic_write_json(SEO_PATH, seo)
    checkpoint("seo_complete", title=seo["title"], resumed=seo_resumed)

    thumb = make_thumbnail(script, seo["title"])
    video_path = OUTPUT_DIR / "final_video.mp4"
    build_video(script, video_path)

    video_id = upload_video(video_path, thumb, seo)
    update_history(topic, seo, video_id)
    atomic_write_json(
        MANIFEST_PATH,
        {
            "topic": topic,
            "title": seo["title"],
            "video_id": video_id,
            "privacy_status": YOUTUBE_PRIVACY_STATUS,
            "scene_count": len(script["scenes"]),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    clear_current_run()
    checkpoint("complete", video_id=video_id, title=seo["title"])
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "voice_test"], default="full")
    args = parser.parse_args()
    main(args.mode)
