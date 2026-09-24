"""
Motive Unknown v2 — curiosity-first automated history video factory.

Design goals
------------
- Daily, unattended GitHub Actions execution.
- Curiosity-driven historical questions instead of generic topics.
- GPT-OSS 120B for research/storytelling; GPT-OSS 20B for lightweight
  structuring/SEO tasks.
- Local/open-weight Kokoro TTS (no paid voice API).
- Polished AI-generated historical illustrations using Cloudflare Workers AI.
- Each narration scene is split into 1-4 visual beats so the picture changes frequently.
- A persistent style-reference image plus prior-frame references improve visual continuity.
- No Ken Burns zoom and no burned-in subtitles.
- Dedicated thumbnail generation separate from video scenes.
- Idempotent stage files so a rerun can skip already-completed stages.
- Safe YouTube default: private uploads until the owner changes the setting.

Required GitHub Secrets
-----------------------
GROQ_API_KEY
YOUTUBE_TOKEN_JSON
YOUTUBE_CLIENT_SECRET_JSON
CLOUDFLARE_ACCOUNT_ID
CLOUDFLARE_API_TOKEN
HUGGINGFACE_TOKEN
REPLICATE_API_TOKEN

Optional GitHub Variables / Secrets
-----------------------------------
YOUTUBE_PRIVACY_STATUS     default: private
KOKORO_VOICE               default: am_onyx
KOKORO_SPEED               default: 0.96
CHANNEL_NAME               optional, used in prompts/description

Optional repo assets
--------------------
assets/music/background.mp3  royalty-free / licensed music only
assets/visual_style_reference.jpg.b64  embedded JPEG style reference used as an image-model style anchor

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

# Polished AI illustration generation.
IMAGE_PROVIDER_ORDER = [
    x.strip().lower()
    for x in os.environ.get("IMAGE_PROVIDERS", "cloudflare,huggingface,replicate").split(",")
    if x.strip()
]
if not IMAGE_PROVIDER_ORDER:
    IMAGE_PROVIDER_ORDER = ["cloudflare", "huggingface", "replicate"]
IMAGE_PROVIDER = IMAGE_PROVIDER_ORDER[0]
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
CLOUDFLARE_IMAGE_MODEL = os.environ.get(
    "CLOUDFLARE_IMAGE_MODEL",
    "@cf/black-forest-labs/flux-2-klein-4b",
).strip()
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
HUGGINGFACE_IMAGE_MODEL = os.environ.get("HUGGINGFACE_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell").strip()
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "").strip()
REPLICATE_IMAGE_MODEL = os.environ.get("REPLICATE_IMAGE_MODEL", "black-forest-labs/flux-1.1-pro").strip()
IMAGE_W = 1024
IMAGE_H = 576
MAX_VISUAL_BEATS_PER_VIDEO = 36
STYLE_REFERENCE_B64 = ROOT / "assets" / "visual_style_reference.jpg.b64"

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


def visual_test() -> None:
    """Generate a few polished sample frames without LLM, TTS, or YouTube."""
    if IMAGE_PROVIDER != "cloudflare":
        raise RuntimeError(
            f"Unsupported IMAGE_PROVIDER={IMAGE_PROVIDER!r}. "
            "The visual test currently requires cloudflare."
        )
    require_secret("CLOUDFLARE_ACCOUNT_ID")
    require_secret("CLOUDFLARE_API_TOKEN")

    test_dir = WORK_DIR / "visual_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    style_ref = _small_reference(_decode_style_reference(), "visual_test_style")

    samples = [
        (
            "A powerful ancient African ruler stands before a thriving riverside city at sunrise, "
            "wearing historically grounded royal garments and holding a ceremonial staff; traders, "
            "boats, stone buildings, palms, and a busy market fill the layered background."
        ),
        (
            "Inside an ancient royal workshop, skilled artisans examine a mysterious object on a table "
            "while the ruler and advisors watch closely; detailed period tools, fabrics, architecture, "
            "lamps, shelves, and expressive faces create a believable historical scene."
        ),
        (
            "At dusk, a small group of historical travelers crosses a monumental desert road toward a "
            "distant walled city and temple complex; banners move in the wind, pack animals carry supplies, "
            "and the composition feels like a polished illustrated documentary frame."
        ),
    ]

    previous: Path | None = None
    outputs: list[str] = []
    for idx, description in enumerate(samples, 1):
        out = test_dir / f"sample_{idx:02d}.jpg"
        refs = [style_ref]
        if previous is not None:
            refs.append(_small_reference(previous, f"visual_test_prev_{idx:02d}"))

        prompt = f"""
{POLISHED_VISUAL_STYLE}

Create a polished historical documentary illustration in 16:9.
The image should feel like a premium hand-illustrated storybook frame rather than clip-art.
Keep human anatomy convincing, faces expressive, environments richly layered, and historical
details coherent. Use the supplied style reference as the visual anchor.

SCENE:
{description}

Continuity:
Preserve the established illustration language and recurring character design from the
reference frames, while creating a distinct composition and setting.

No readable text, captions, subtitles, logos, watermarks, letters, numbers, pseudo-writing,
modern objects, modern roads, asphalt, lane markings, traffic signs, power lines, cars,
photorealism, 3D CGI, anime, stick figures, doodles, or flat geometric art. Use a genuinely
period-appropriate road or path rather than a modern paved roadway. Flags and banners must be
blank or non-readable unless the scene explicitly requires a documented inscription.
""".strip()

        print(f"[VISUAL TEST] Generating sample {idx}/{len(samples)}")
        provider_used = _generate_image_with_fallback(
            prompt,
            out,
            _image_seed(9000, idx - 1),
            refs,
        )
        previous = out
        outputs.append({"path": str(out.relative_to(ROOT)), "provider": provider_used})

    checkpoint(
        "visual_test_complete",
        image_providers=IMAGE_PROVIDER_ORDER,
        primary_image_provider=IMAGE_PROVIDER,
        image_model=CLOUDFLARE_IMAGE_MODEL,
        samples=outputs,
    )
    print("[VISUAL TEST] Complete. Inspect the generated sample images in the workflow artifact.")



# ---------------------------------------------------------------------------
# Polished AI illustration renderer
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
    """Turn narration into 1-4 natural visual beats, roughly one image every ~60 words, with at most two beats per scene."""
    text = normalize_spaces(narration)
    if not text:
        return [""]

    sentences = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    if not sentences:
        sentences = [text]

    expanded: list[str] = []
    for sentence in sentences:
        if count_words(sentence) > 24:
            parts = [p.strip() for p in re.split(r"(?<=[,;:])\s+", sentence) if p.strip()]
            if len(parts) > 1:
                bucket = ""
                for part in parts:
                    candidate = f"{bucket} {part}".strip()
                    if count_words(candidate) <= 22 or not bucket:
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
    target_beats = max(1, min(2, int(np.ceil(word_total / 60))))
    target_beats = min(target_beats, len(expanded))

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
        if len(words) < 14:
            break
        cut = len(words) // 2
        expanded[idx:idx + 1] = [
            " ".join(words[:cut]),
            " ".join(words[cut:]),
        ]

    return expanded[:4] or [text]


POLISHED_VISUAL_STYLE = """
Polished 2D historical cartoon illustration for a premium educational YouTube
documentary. Cinematic storybook composition, expressive human characters with
believable anatomy, expressive faces and gestures, period-appropriate clothing
and architecture, richly detailed environments, layered foreground/midground/
background depth, crisp hand-drawn ink contours, clean painterly/cel-shaded
color, warm natural lighting, subtle texture, strong focal subject, visual
storytelling in every frame, appealing to teenagers and adults.

Do NOT make stick figures, doodles, primitive geometric drawings, flat clip-art,
photorealism, 3D CGI, anime, modern objects, UI elements, captions, subtitles,
logos, watermarks, readable text, letters, numbers, pseudo-writing, or written words
inside the image. Avoid modern roads, asphalt, lane markings, traffic signs, power lines,
streetlights, cars, modern furniture, modern tools, and other anachronistic infrastructure
unless the narration explicitly requires a modern setting. Use historically plausible
roads, materials, tools, clothing, architecture, and transport for the stated era.
""".strip()


def _image_seed(scene_id: int, beat_index: int) -> int:
    return ((scene_id + 1) * 10007 + (beat_index + 1) * 7919) & 0x7FFFFFFF


def _decode_style_reference() -> Path:
    out = WORK_DIR / "visual_style_reference.jpg"
    if out.exists() and out.stat().st_size > 1000:
        return out
    if not STYLE_REFERENCE_B64.exists():
        raise RuntimeError(
            "Missing embedded visual style reference at assets/visual_style_reference.jpg.b64"
        )
    try:
        data = STYLE_REFERENCE_B64.read_bytes()
    except Exception as exc:
        raise RuntimeError("Embedded visual style reference could not be read.") from exc
    if len(data) < 1000 or data[:2] != b"\xff\xd8":
        raise RuntimeError("Embedded visual style reference is not a valid JPEG.")
    out.write_bytes(data)
    return out


def _small_reference(path: Path, label: str) -> Path:
    out = WORK_DIR / f"{label}_reference.jpg"
    if out.exists() and out.stat().st_size > 1000:
        return out
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((448, 448))
        image.save(out, format="JPEG", quality=82, optimize=True)
    return out


class ImageProviderError(RuntimeError):
    def __init__(self, provider: str, message: str, *, disable_for_run: bool = False):
        super().__init__(message)
        self.provider = provider
        self.disable_for_run = disable_for_run


def _save_provider_image_bytes(out_path: Path, raw: bytes, provider: str) -> None:
    if len(raw) < 1000:
        raise ImageProviderError(provider, "Provider returned an empty or tiny image payload.")
    try:
        out_path.write_bytes(raw)
        with Image.open(out_path) as image:
            image.convert("RGB").save(out_path, format="JPEG", quality=92, optimize=True)
    except Exception as exc:
        raise ImageProviderError(provider, f"Provider returned invalid image data: {exc}") from exc


def _huggingface_image(prompt: str, out_path: Path, seed: int) -> None:
    if not HUGGINGFACE_TOKEN:
        raise ImageProviderError("huggingface", "HUGGINGFACE_TOKEN is not configured.", disable_for_run=True)
    try:
        from huggingface_hub import InferenceClient
    except Exception as exc:
        raise ImageProviderError("huggingface", f"huggingface_hub is unavailable: {exc}", disable_for_run=True) from exc
    print(f"[IMAGE] Hugging Face {HUGGINGFACE_IMAGE_MODEL} seed={seed}")
    try:
        client = InferenceClient(api_key=HUGGINGFACE_TOKEN)
        image = client.text_to_image(prompt, model=HUGGINGFACE_IMAGE_MODEL, seed=seed)
        image.save(out_path, format="JPEG", quality=92, optimize=True)
    except Exception as exc:
        status_match = re.search(r"\b(400|401|402|403|404|408|409|429|500|502|503|504)\b", str(exc))
        status = int(status_match.group(1)) if status_match else None
        disable = status in {401, 402, 403, 404, 429}
        raise ImageProviderError("huggingface", f"Hugging Face image generation failed: {str(exc)[:1800]}", disable_for_run=disable) from exc


def _replicate_image(prompt: str, out_path: Path, seed: int) -> None:
    if not REPLICATE_API_TOKEN:
        raise ImageProviderError("replicate", "REPLICATE_API_TOKEN is not configured.", disable_for_run=True)
    url = f"https://api.replicate.com/v1/models/{REPLICATE_IMAGE_MODEL}/predictions"
    headers = {"Authorization": f"Bearer {REPLICATE_API_TOKEN}", "Content-Type": "application/json", "Prefer": "wait=60"}
    payload = {"input": {"prompt": prompt, "seed": seed, "aspect_ratio": "16:9", "output_format": "webp", "output_quality": 90, "safety_tolerance": 2, "prompt_upsampling": False}}
    print(f"[IMAGE] Replicate {REPLICATE_IMAGE_MODEL} seed={seed}")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=90)
    except requests.RequestException as exc:
        raise ImageProviderError("replicate", f"Replicate request failed: {exc}") from exc
    if response.status_code not in {200, 201, 202}:
        disable = response.status_code in {401, 402, 403, 404, 429}
        raise ImageProviderError("replicate", f"Replicate failed with HTTP {response.status_code}: {response.text[:1600]}", disable_for_run=disable)
    try:
        data = response.json()
    except Exception as exc:
        raise ImageProviderError("replicate", "Replicate returned invalid JSON.") from exc
    status = str(data.get("status", "")).lower()
    prediction_url = data.get("url")
    if prediction_url and status not in {"succeeded", "failed", "canceled"}:
        deadline = time.time() + 300
        while time.time() < deadline:
            time.sleep(5)
            poll = requests.get(prediction_url, headers={"Authorization": f"Bearer {REPLICATE_API_TOKEN}"}, timeout=60)
            if poll.status_code != 200:
                raise ImageProviderError("replicate", f"Replicate polling returned HTTP {poll.status_code}: {poll.text[:1000]}")
            data = poll.json()
            status = str(data.get("status", "")).lower()
            if status in {"succeeded", "failed", "canceled"}:
                break
    if status != "succeeded":
        error = data.get("error") or "prediction did not succeed"
        disable = "credit" in str(error).lower() or "billing" in str(error).lower() or "payment" in str(error).lower()
        raise ImageProviderError("replicate", f"Replicate prediction ended with status={status}: {str(error)[:1600]}", disable_for_run=disable)
    output = data.get("output")
    output_url = output if isinstance(output, str) else (output[0] if isinstance(output, list) and output and isinstance(output[0], str) else None)
    if not output_url:
        raise ImageProviderError("replicate", f"Replicate returned no output URL: {json.dumps(data)[:1600]}")
    try:
        image_response = requests.get(output_url, timeout=120)
    except requests.RequestException as exc:
        raise ImageProviderError("replicate", f"Replicate image download failed: {exc}") from exc
    if image_response.status_code != 200:
        raise ImageProviderError("replicate", f"Replicate image download failed with HTTP {image_response.status_code}.")
    _save_provider_image_bytes(out_path, image_response.content, "replicate")


def _generate_image_with_fallback(prompt: str, out_path: Path, seed: int, references: list[Path]) -> str:
    disabled = getattr(_generate_image_with_fallback, "_disabled", set())
    attempted = []
    for provider in IMAGE_PROVIDER_ORDER:
        if provider in disabled:
            continue
        attempted.append(provider)
        try:
            if provider == "cloudflare":
                _cloudflare_image(prompt, out_path, seed, references)
            elif provider == "huggingface":
                _huggingface_image(prompt, out_path, seed)
            elif provider == "replicate":
                _replicate_image(prompt, out_path, seed)
            else:
                print(f"[IMAGE] Unknown provider {provider!r}, skipping.")
                continue
            print(f"[IMAGE] provider used: {provider}")
            return provider
        except ImageProviderError as exc:
            print(f"[IMAGE FALLBACK] {provider} failed: {exc}")
            if exc.disable_for_run:
                disabled.add(provider)
                setattr(_generate_image_with_fallback, "_disabled", disabled)
                print(f"[IMAGE FALLBACK] disabling {provider} for the rest of this run.")
        except Exception as exc:
            print(f"[IMAGE FALLBACK] {provider} unexpected failure: {exc}")
    raise RuntimeError("All configured image providers failed. Attempted: " + (", ".join(attempted) or "(none)"))

def _cloudflare_image(prompt: str, out_path: Path, seed: int, references: list[Path]) -> None:
    if "cloudflare" not in IMAGE_PROVIDER_ORDER:
        raise ImageProviderError("cloudflare", "Cloudflare is not enabled in IMAGE_PROVIDERS.", disable_for_run=True)
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_TOKEN:
        raise ImageProviderError(
            "cloudflare",
            "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required for polished image generation.",
            disable_for_run=True,
        )

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{CLOUDFLARE_ACCOUNT_ID}/ai/run/{CLOUDFLARE_IMAGE_MODEL}"
    )
    data = {
        "prompt": prompt,
        "width": str(IMAGE_W),
        "height": str(IMAGE_H),
        "seed": str(seed),
    }
    files: dict[str, tuple[str, bytes, str]] = {}
    for idx, ref in enumerate(references[:4]):
        files[f"input_image_{idx}"] = (
            ref.name,
            ref.read_bytes(),
            "image/jpeg",
        )

    print(f"[IMAGE] Cloudflare {CLOUDFLARE_IMAGE_MODEL} seed={seed}")
    try:
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
            data=data,
            files=files or None,
            timeout=180,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Cloudflare image request failed: {exc}") from exc

    if response.status_code != 200:
        message = response.text[:2500]
        disable = response.status_code in {401, 403, 429}
        raise ImageProviderError(
            "cloudflare",
            f"Cloudflare image generation failed with HTTP {response.status_code}: {message}",
            disable_for_run=disable,
        )

    try:
        payload = response.json()
        result = payload.get("result")
        if isinstance(result, dict):
            image_b64 = result.get("image")
        else:
            image_b64 = None
    except Exception as exc:
        raise ImageProviderError("cloudflare", "Cloudflare image response was not valid JSON.") from exc

    if not isinstance(image_b64, str) or not image_b64.strip():
        raise ImageProviderError("cloudflare", "Cloudflare image response did not contain result.image.")

    try:
        out_path.write_bytes(base64.b64decode(image_b64))
    except Exception as exc:
        raise ImageProviderError("cloudflare", "Cloudflare returned invalid base64 image data.") from exc

    with Image.open(out_path) as image:
        image.convert("RGB").save(out_path, format="JPEG", quality=92, optimize=True)


def make_visual_prompt(
    scene: dict[str, Any],
    era: str,
    beat_text: str,
    beat_index: int,
    beat_count: int,
    has_previous_reference: bool,
) -> str:
    shot_types = [
        "wide cinematic establishing shot",
        "medium character interaction shot",
        "dynamic over-the-shoulder storytelling shot",
        "closer emotional or important-object shot",
    ]
    shot = shot_types[beat_index % len(shot_types)]
    chars = ", ".join(str(x) for x in (scene.get("characters") or [])[:4]) or "historical people"
    props = ", ".join(str(x) for x in (scene.get("props") or [])[:4]) or "period-appropriate objects"

    continuity = (
        "A previous generated frame is supplied as a reference. Preserve recurring "
        "character design, clothing colors, facial proportions, and overall illustration "
        "style from that reference, but create a new shot and do not copy its background."
        if has_previous_reference
        else
        "Establish the visual character designs now so later shots can remain consistent."
    )

    return f"""
{POLISHED_VISUAL_STYLE}

ERA / HISTORICAL CONTEXT:
{era}

SETTING:
{scene.get("setting", "historical location")}

CHARACTERS:
{chars}

VISIBLE ACTION:
{scene.get("action", "characters interacting naturally")}

PROPS / SYMBOLS:
{props}

MOOD:
{scene.get("mood", "curious")}

NARRATION BEAT:
{beat_text}

SHOT DIRECTION:
{shot}. Use clear staging and strong depth. Let the main action be easy to understand
at a glance. Vary camera distance and composition from the previous beat while keeping
the same visual world.

CONTINUITY:
{continuity}

Create a finished, polished illustration. No readable text, lettering, numbers,
inscriptions, pseudo-text, logos, or symbols resembling modern writing anywhere in the image.
Flags, banners, walls, tablets, scrolls, signs, and books must have blank or non-readable
surfaces unless the narration explicitly requires a specific historical inscription.
""".strip()


def render_scenes(script: dict[str, Any]) -> None:
    SCENE_DIR.mkdir(parents=True, exist_ok=True)
    style_ref = _small_reference(_decode_style_reference(), "style")
    total_beats = sum(
        len(split_visual_beats(str(scene.get("narration", ""))))
        for scene in script["scenes"]
    )
    if total_beats > MAX_VISUAL_BEATS_PER_VIDEO:
        raise RuntimeError(
            f"Planned {total_beats} visual beats, above the configured Cloudflare budget cap of "
            f"{MAX_VISUAL_BEATS_PER_VIDEO}. The pipeline intentionally limits daily image generation "
            "to stay below the free allocation more reliably."
        )

    previous_image: Path | None = None
    previous_characters: set[str] = set()

    for idx, scene in enumerate(script["scenes"], 1):
        beats = split_visual_beats(str(scene.get("narration", "")))
        current_characters = {
            normalize_spaces(str(x)).lower()
            for x in (scene.get("characters") or [])
            if normalize_spaces(str(x))
        }
        for beat_idx, beat_text in enumerate(beats, 1):
            out = SCENE_DIR / f"scene_{idx:03d}_beat_{beat_idx:02d}.jpg"
            if out.exists() and out.stat().st_size > 10000:
                print(f"[IMAGE] Reusing {out.name}")
                previous_image = out
                continue

            refs = [style_ref]
            use_previous = previous_image is not None and (
                beat_idx > 1 or bool(current_characters & previous_characters)
            )
            if use_previous:
                refs.append(_small_reference(previous_image, f"prev_{idx:03d}_{beat_idx:02d}"))

            prompt = make_visual_prompt(
                scene,
                script.get("era", "History"),
                beat_text,
                beat_idx - 1,
                len(beats),
                has_previous_reference=use_previous,
            )
            print(f"[IMAGE] Scene {idx}/{len(script['scenes'])} beat {beat_idx}/{len(beats)}")
            provider_used = _generate_image_with_fallback(
                prompt,
                out,
                _image_seed(idx, beat_idx - 1),
                refs,
            )
            previous_image = out

        previous_characters = current_characters

    checkpoint(
        "scenes_complete",
        visual_beats=total_beats,
        image_provider=IMAGE_PROVIDER,
        image_model=CLOUDFLARE_IMAGE_MODEL,
    )



# ---------------------------------------------------------------------------
# SEO
# ---------------------------------------------------------------------------
SEO_PROMPT = """
You are the YouTube packaging editor for a history storytelling channel.

The video answers a specific historical curiosity question.
Create metadata that is discoverable without sounding like spam.

Rules:
- One primary title under 70 characters. Natural curiosity, no fake claims.
- Two alternate titles.
- Description: the first two lines should clearly explain the question and why the story matters.
  Then a concise spoiler-light summary, followed by a Sources section using the supplied sources.
- Tags: 12-15 relevant terms. Tags are secondary; do not stuff unrelated keywords.
- Thumbnail headline: 2-5 words that complement the title instead of repeating it.

Return JSON only:
{
  "title": "...",
  "alternate_titles": ["...", "..."],
  "description": "...",
  "tags": ["..."],
  "primary_keywords": ["..."],
  "thumbnail_headline": "..."
}
""".strip()


SEO_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "maxLength": 70},
        "alternate_titles": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        },
        "description": {"type": "string", "minLength": 80},
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 12,
            "maxItems": 15,
        },
        "primary_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 8,
        },
        "thumbnail_headline": {"type": "string", "minLength": 2, "maxLength": 40},
    },
    "required": [
        "title",
        "alternate_titles",
        "description",
        "tags",
        "primary_keywords",
        "thumbnail_headline",
    ],
    "additionalProperties": False,
}


def build_seo(topic: dict[str, Any], script: dict[str, Any], research: str) -> dict[str, Any]:
    prompt = (
        SEO_PROMPT
        + "\n\nCENTRAL QUESTION:\n"
        + topic["question"]
        + "\n\nSCRIPT:\n"
        + "\n\n".join(scene["narration"] for scene in script["scenes"])
        + "\n\nRESEARCH SOURCES / NOTES:\n"
        + research[-12000:]
        + """

IMPORTANT JSON RULES:
- "description" must be one plain string value. Do not put Tags, Sources, or any other JSON key inside it.
- "tags" must be a JSON array of strings, separate from description.
- "alternate_titles" must contain exactly two strings.
- Return only the JSON object. No markdown fences.
"""
    )

    if client is None:
        raise RuntimeError("GROQ_API_KEY is required for SEO generation.")

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.chat.completions.create(
                model=GROQ_LIGHT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=3200,
                temperature=0.45,
                reasoning_effort="low",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "motive_unknown_seo",
                        "strict": True,
                        "schema": SEO_JSON_SCHEMA,
                    },
                },
            )
            raw = response.choices[0].message.content or ""
            if not raw.strip():
                raise RuntimeError("Strict SEO JSON call returned an empty response.")
            seo = json.loads(raw)
            if not seo.get("title") or not seo.get("description") or not seo.get("tags"):
                raise RuntimeError("SEO output is incomplete.")
            break
        except Exception as exc:
            last_error = exc
            if _is_non_retryable(exc):
                raise RuntimeError(f"SEO request was rejected: {exc}") from exc
            if attempt >= 3:
                raise RuntimeError(f"SEO generation failed after 3 attempts: {exc}") from exc
            wait = _retry_wait(exc, attempt, base=3.0)
            print(f"[SEO RETRY] attempt {attempt}/3: {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)

    if not isinstance(seo, dict):
        raise RuntimeError("SEO generation returned an invalid object.")

    seo["title"] = normalize_spaces(str(seo["title"]))[:70]
    seo["alternate_titles"] = [
        normalize_spaces(str(x))[:100]
        for x in seo.get("alternate_titles", [])[:2]
    ]
    seo["description"] = str(seo["description"]).strip()
    seo["tags"] = [
        normalize_spaces(str(x))
        for x in seo.get("tags", [])
        if normalize_spaces(str(x))
    ][:15]
    seo["primary_keywords"] = [
        normalize_spaces(str(x))
        for x in seo.get("primary_keywords", [])
        if normalize_spaces(str(x))
    ]
    seo["thumbnail_headline"] = normalize_spaces(
        str(seo.get("thumbnail_headline", "History Mystery"))
    )[:40]

    if len(seo["alternate_titles"]) != 2 or not (12 <= len(seo["tags"]) <= 15):
        raise RuntimeError("SEO output did not satisfy title/tag requirements.")

    return seo



# ---------------------------------------------------------------------------
# AI thumbnail
# ---------------------------------------------------------------------------
def make_thumbnail(script: dict[str, Any], title: str) -> Path:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    final = OUTPUT_DIR / "thumbnail.jpg"
    cached_ai = THUMB_DIR / "thumbnail_ai.jpg"

    thumb = script.get("thumbnail", {}) or {}
    subject = str(thumb.get("subject", "historical figure")).strip()
    prop = str(thumb.get("supporting_prop", "important historical object")).strip()
    emotion = str(thumb.get("emotion", "surprised and curious")).strip()
    composition = str(thumb.get("composition", "left_subject_right_prop")).strip()
    era = str(script.get("era", "History")).strip()

    side_note = {
        "left_subject_right_prop": "Place the main subject prominently on the left and the important object or symbol on the right.",
        "right_subject_left_prop": "Place the main subject prominently on the right and the important object or symbol on the left.",
        "central_subject": "Place the main subject prominently near the center with the important object or symbol clearly visible beside them.",
    }.get(composition, "Use a strong asymmetrical YouTube thumbnail composition with a clear focal subject.")

    style_ref = _small_reference(_decode_style_reference(), "thumbnail_style")

    prompt = f"""
{POLISHED_VISUAL_STYLE}

Create a polished 16:9 YouTube thumbnail illustration for a history mystery documentary.

ERA:
{era}

MAIN SUBJECT:
{subject}

IMPORTANT OBJECT / SYMBOL:
{prop}

EXPRESSION / BODY LANGUAGE:
{emotion}

COMPOSITION:
{side_note}

Make the main subject large enough to read clearly at thumbnail size. Use a dramatic but
fact-grounded moment, strong silhouette separation, rich historical detail, expressive faces,
and a clean focal hierarchy. Keep the image visually bold without becoming cluttered.

No readable text, letters, numbers, pseudo-writing, captions, subtitles, logos, watermarks,
modern infrastructure, modern clothing, cars, asphalt lane markings, or other anachronisms.
""".strip()

    if not cached_ai.exists() or cached_ai.stat().st_size < 10000:
        _generate_image_with_fallback(prompt, cached_ai, 71003, [style_ref])

    with Image.open(cached_ai) as base:
        image = base.convert("RGB").resize((1280, 720))

    draw = ImageDraw.Draw(image)
    headline = normalize_spaces(
        str(thumb.get("headline") or title or "HISTORY MYSTERY")
    ).upper()
    headline = headline[:36]

    font = FONT_88
    max_width = 720
    words = headline.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font, stroke_width=2)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    lines = lines[:3]

    x = 45 if "right" not in composition else 720
    y = 45
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=3)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        draw.rounded_rectangle(
            [x - 12, y - 12, min(1260, x + w + 24), y + h + 24],
            radius=18,
            fill=BLACK,
        )
        draw.text(
            (x, y),
            line,
            font=font,
            fill=WHITE,
            stroke_width=2,
            stroke_fill=BLACK,
        )
        y += h + 24

    draw.rounded_rectangle(
        [35, 650, 370, 705],
        radius=14,
        fill=BLACK,
    )
    draw.text(
        (52, 660),
        era[:28],
        font=FONT_28,
        fill=WHITE,
    )

    image.save(final, format="JPEG", quality=94, optimize=True)
    print(f"[THUMBNAIL] AI thumbnail ready: {final}")
    return final


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


def validate_youtube_credentials() -> None:
    """Preflight the stored YouTube OAuth credentials before expensive work."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request

    token_text = require_secret("YOUTUBE_TOKEN_JSON")
    client_secret_text = require_secret("YOUTUBE_CLIENT_SECRET_JSON")

    try:
        token_info = json.loads(token_text)
        client_info = json.loads(client_secret_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("YouTube credential secrets must contain valid JSON.") from exc

    if not isinstance(token_info, dict):
        raise RuntimeError("YOUTUBE_TOKEN_JSON must contain a JSON object.")
    if not isinstance(client_info, dict):
        raise RuntimeError("YOUTUBE_CLIENT_SECRET_JSON must contain a JSON object.")

    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    credentials = Credentials.from_authorized_user_info(token_info, scopes)

    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except Exception as exc:
                raise RuntimeError(
                    "YouTube OAuth refresh failed. The saved refresh token may be expired or revoked."
                ) from exc
        else:
            raise RuntimeError(
                "YouTube OAuth credentials are invalid and do not have a usable refresh token."
            )

    print("[YOUTUBE PREFLIGHT] OAuth credentials are usable.")



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

    if mode == "visual_test":
        visual_test()
        return

    require_secret("GROQ_API_KEY")
    require_secret("YOUTUBE_TOKEN_JSON")
    require_secret("YOUTUBE_CLIENT_SECRET_JSON")
    supported = {"cloudflare", "huggingface", "replicate"}
    unknown = [x for x in IMAGE_PROVIDER_ORDER if x not in supported]
    if unknown:
        raise RuntimeError(f"Unsupported image providers: {unknown}")
    if "cloudflare" in IMAGE_PROVIDER_ORDER:
        require_secret("CLOUDFLARE_ACCOUNT_ID")
        require_secret("CLOUDFLARE_API_TOKEN")

    validate_youtube_credentials()

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
    parser.add_argument("--mode", choices=["full", "voice_test", "visual_test"], default="full")
    args = parser.parse_args()
    main(args.mode)
