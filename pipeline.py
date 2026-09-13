"""
Motive Unknown — automated video pipeline (Ken Burns / cartoon-illustration
format). Runs unattended on GitHub Actions. No interactive input anywhere.

Required GitHub Secrets (Settings -> Secrets and variables -> Actions):
  GROQ_API_KEY          - your Groq API key
  YOUTUBE_TOKEN_JSON    - the full contents of your youtube_token.json file
  YOUTUBE_CLIENT_SECRET_JSON - the full contents of your client_secret_....json file

No local character assets required anymore — every scene's image is
generated on the fly (Pollinations.ai, free, no key) from that beat's gist.
"""

import os
import re
import time
import json
import asyncio
import subprocess

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------
# 0. Load secrets from environment (GitHub injects these at runtime)
# ---------------------------------------------------------------------
GROQ_KEY = os.environ["GROQ_API_KEY"]

with open("youtube_token.json", "w") as f:
    f.write(os.environ["YOUTUBE_TOKEN_JSON"])
with open("client_secret.json", "w") as f:
    f.write(os.environ["YOUTUBE_CLIENT_SECRET_JSON"])

print("Secrets loaded.")

# ---------------------------------------------------------------------
# 1. LLM connection (Groq)
# ---------------------------------------------------------------------
import requests
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import tool

# WORKAROUND for a known CrewAI bug (crewAIInc/crewAI#5886): CrewAI's own
# code injects an Anthropic-style 'cache_breakpoint' property into every
# system message, but the function that's supposed to strip it back out for
# non-Anthropic providers never actually gets called. Groq's API has no
# concept of that field and rejects the whole request outright with
# "property 'cache_breakpoint' is unsupported" — this has nothing to do
# with which Groq model is selected, it happens for any Groq/OpenAI-
# compatible provider. No-op'ing the injection function fixes it.
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

llm = LLM(
    model="groq/openai/gpt-oss-120b",
    api_key=GROQ_KEY,
    timeout=300,
    max_retries=5,
    max_tokens=2048,  # Groq's free tier caps openai/gpt-oss-120b at 8000
                       # TOKENS PER MINUTE total. Topic Scout/Researcher/SEO
                       # only need short outputs so this still leaves headroom.
)


def call_with_retry(llm_obj, prompt, attempts=5, base_delay=10):
    last_error = None
    for i in range(attempts):
        try:
            return llm_obj.call(prompt)
        except Exception as e:
            last_error = e
            wait = base_delay * (i + 1)
            print(f"LLM call failed (attempt {i+1}/{attempts}): {e}\nRetrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {attempts} attempts: {last_error}")


GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_ID = "openai/gpt-oss-120b"


def _call_groq_direct(messages, max_tokens=8192, temperature=0.4, attempts=6, base_delay=15):
    """Calls Groq's chat/completions endpoint directly with raw `requests`,
    bypassing CrewAI/litellm entirely — used for the Scriptwriter's outline
    and per-beat generation calls: full control over exactly what's sent,
    independent of CrewAI's system-prompt templating. openai/gpt-oss-120b IS
    a reasoning model, but Groq puts its reasoning trace in a separate
    "reasoning" field on the response rather than mixing it into "content" —
    include_reasoning: false below tells Groq to drop that field entirely.

    On a 429 (rate limit — Groq's free tier is only 8000 tokens/minute for
    this model), this parses the actual suggested wait time out of Groq's
    error response ("Please try again in 12.915s") rather than guessing."""
    last_error = None
    for i in range(attempts):
        try:
            resp = requests.post(
                GROQ_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL_ID,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "include_reasoning": False,
                },
                timeout=120,
            )
            if resp.status_code == 429:
                wait = base_delay * (i + 1)
                retry_after = resp.headers.get("retry-after")
                if retry_after:
                    try:
                        wait = max(wait, float(retry_after) + 2)
                    except ValueError:
                        pass
                else:
                    match = re.search(r"try again in ([\d.]+)s", resp.text)
                    if match:
                        wait = max(wait, float(match.group(1)) + 2)
                print(f"Groq rate limit hit (attempt {i+1}/{attempts}), waiting {wait:.1f}s: {resp.text[:200]}")
                last_error = f"429 rate limited: {resp.text[:300]}"
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_error = e
            wait = base_delay * (i + 1)
            print(f"Groq direct call failed (attempt {i+1}/{attempts}): {e}\nRetrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Groq direct call failed after {attempts} attempts: {last_error}")


test = call_with_retry(llm, "Reply with exactly one word: OK")
print("LLM connection test:", test)

# ---------------------------------------------------------------------
# 1b. Timing instrumentation
# ---------------------------------------------------------------------
PIPELINE_START = time.time()
_last_checkpoint = {"t": PIPELINE_START}


def _timing_callback(task_label):
    def _cb(output):
        now = time.time()
        since_last = now - _last_checkpoint["t"]
        since_start = now - PIPELINE_START
        print(f"[TIMING] '{task_label}' finished — took {since_last:.1f}s "
              f"(total elapsed {since_start:.1f}s)")
        _last_checkpoint["t"] = now
    return _cb


def _step_callback(step):
    since_start = time.time() - PIPELINE_START
    print(f"[STEP] t+{since_start:.1f}s — {type(step).__name__}")


# ---------------------------------------------------------------------
# 2. Search tools
# ---------------------------------------------------------------------
SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8888")
WIKI_USER_AGENT = "MotiveUnknownBot/1.0 (automated video pipeline; contact: replace-with-your-email@example.com)"


@tool("Web Search")
def search_tool(query: str) -> str:
    """Searches the web via a self-hosted SearXNG instance and returns top results with titles, snippets, and links. Best for trending topics, angles, and general web context."""
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json"},
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])[:3]
            if not results:
                return "No results found for this query. Try a different angle."
            formatted = []
            for r in results:
                formatted.append(f"{r.get('title','')}\n{r.get('content','')}\n{r.get('url','')}")
            return "\n\n".join(formatted)
        except Exception as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    return f"Search failed after 3 attempts ({last_error}). Proceed using general knowledge instead."


@tool("Wikipedia Lookup")
def wikipedia_tool(query: str) -> str:
    """Looks up a topic, figure, or event on Wikipedia and returns summaries of the top matching pages. Best for verifying specific historical facts, figures, and events — use this before falling back to general web search."""
    last_error = None
    for attempt in range(3):
        try:
            search_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": 2,
                },
                headers={"User-Agent": WIKI_USER_AGENT},
                timeout=15,
            )
            search_resp.raise_for_status()
            hits = search_resp.json().get("query", {}).get("search", [])
            if not hits:
                return "No Wikipedia results found for this query. Try a different angle or use Web Search instead."

            formatted = []
            for hit in hits:
                title = hit["title"]
                summary_resp = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
                    headers={"User-Agent": WIKI_USER_AGENT},
                    timeout=15,
                )
                if summary_resp.status_code != 200:
                    continue
                summary = summary_resp.json()
                extract = summary.get("extract", "")[:800]
                url = summary.get("content_urls", {}).get("desktop", {}).get("page", "")
                formatted.append(f"{title}\n{extract}\n{url}")

            if not formatted:
                return "Found matching titles but couldn't fetch summaries. Try Web Search instead."
            return "\n\n".join(formatted)
        except Exception as e:
            last_error = e
            time.sleep(2 * (attempt + 1))
    return f"Wikipedia lookup failed after 3 attempts ({last_error}). Proceed using general knowledge instead."


# ---------------------------------------------------------------------
# 3. Agents
#
# SCOPE NOTE: still scoped to Ancient Egypt / Ancient Rome for now — that
# was originally because only those eras had character art, but since
# images are generated per-beat now, that constraint could be lifted
# whenever you want to broaden the topic pool. Left in place until you
# decide to widen it.
# ---------------------------------------------------------------------
topic_scout = Agent(
    role="Topic Scout",
    goal=(
        "Find a single, highly clickable video topic that is a specific historical "
        "story, event, or figure from Ancient Egypt or Ancient Rome."
    ),
    backstory=(
        "You track trending searches, Reddit threads, and history content that performs "
        "well on YouTube. Your job is to spot a specific angle (not a broad topic) — a "
        "particular event, figure, or moment — that has strong curiosity-gap potential. "
        "You only pick topics from Ancient Egypt or Ancient Rome. You pick ONE topic and "
        "justify why it will perform well."
    ),
    tools=[search_tool],
    llm=llm,
    max_iter=4,
    verbose=True,
)

researcher = Agent(
    role="Video Researcher",
    goal="Find the most surprising, well-sourced facts or story beats on the chosen historical topic",
    backstory=(
        "You're an obsessive researcher for a history storytelling channel. You dig up "
        "real, verifiable, surprising details and put them in the order they'd be told "
        "as a story. You avoid generic facts everyone already knows. You always check "
        "Wikipedia Lookup first for the topic and any figures/events it mentions — it's "
        "your primary, most reliable source. Only reach for Web Search when Wikipedia "
        "doesn't have enough detail on a specific angle."
    ),
    tools=[wikipedia_tool, search_tool],
    llm=llm,
    max_iter=6,
    verbose=True,
)

# NOTE: there is deliberately no CrewAI Agent for the Scriptwriter — the
# JSON-only output requirement was too strict for CrewAI's system-prompt
# templating to reliably satisfy, so script generation is done via direct
# API calls further down instead.

seo_specialist = Agent(
    role="YouTube SEO Specialist",
    goal="Generate a high-CTR title, description, and tag list for the video",
    backstory=(
        "You've studied thousands of high-performing history-channel uploads and know "
        "how to write curiosity-driven titles and keyword-rich descriptions."
    ),
    llm=llm,
    max_iter=5,
    verbose=True,
)

print("Agents ready.")

# ---------------------------------------------------------------------
# 4. Tasks — Topic Scout + Researcher
# ---------------------------------------------------------------------
topic_task = Task(
    description=(
        "Search for a specific, under-covered historical story, event, or figure from "
        "Ancient Egypt or Ancient Rome ONLY — no other eras, and no non-historical "
        "curiosity topics. Pick ONE specific angle. State the topic clearly, name which "
        "era it's from (ancient_egypt or ancient_rome), and give 2-3 sentences on why it "
        "will perform well."
    ),
    expected_output="One clearly stated topic, its era, and a short justification.",
    agent=topic_scout,
    callback=_timing_callback("Topic Scout"),
)

research_task = Task(
    description=(
        "Using the topic chosen by the Topic Scout, research 8-12 specific, surprising, "
        "and verifiable facts or story beats, each with a one-line source or context. "
        "Put them roughly in the order they'd be told as a story."
    ),
    expected_output="A bullet list of 8-12 facts/story beats in story order, each with a short source note.",
    agent=researcher,
    context=[topic_task],
    callback=_timing_callback("Researcher"),
)

print("Tasks ready.")

# ---------------------------------------------------------------------
# 5. Run Topic Scout and Researcher as TWO SEPARATE crew runs, not one —
# spreads their token usage across separate 60s TPM windows on Groq's free
# tier instead of stacking it in one.
# ---------------------------------------------------------------------
MAX_CREW_ATTEMPTS = 3


def _run_single_task_crew(agent, task, label):
    single_crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
        step_callback=_step_callback,
    )
    for attempt in range(1, MAX_CREW_ATTEMPTS + 1):
        try:
            return single_crew.kickoff()
        except Exception as e:
            wait = 70
            print(f"[{label} RETRY] Attempt {attempt}/{MAX_CREW_ATTEMPTS} failed: {e}\nWaiting {wait}s...")
            if attempt == MAX_CREW_ATTEMPTS:
                raise
            time.sleep(wait)


_run_single_task_crew(topic_scout, topic_task, "TOPIC SCOUT")
print("[PACING] Waiting 20s before Researcher to keep token usage spread across TPM windows...")
time.sleep(20)
result = _run_single_task_crew(researcher, research_task, "RESEARCHER")

print("\n\n===== TOPIC + RESEARCH OUTPUT =====\n")
print(result)

# ---------------------------------------------------------------------
# 6. Parse + validate the Scriptwriter's JSON output
# ---------------------------------------------------------------------
def _extract_json_object(raw_text: str, required_key: str = "segments") -> dict:
    """Some models write out a 'thinking process' before the real answer no
    matter how firmly you tell them not to — and that reasoning text can
    itself contain brace-like snippets that break a naive first-brace/
    last-brace slice. This scans for every *balanced* {...} block in the
    text and tries them from LAST to FIRST (the real answer comes after the
    reasoning, not before it), returning the first one that both parses as
    JSON and has the key the CALLER actually needs — 'segments' for the
    final script, 'beats' for the outline step."""
    candidates = []
    stack = []
    start = None
    for i, ch in enumerate(raw_text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(raw_text[start:i + 1])
                    start = None

    last_error = None
    for cand in reversed(candidates):
        try:
            data = json.loads(cand)
            if isinstance(data, dict) and required_key in data:
                return data
        except json.JSONDecodeError as e:
            last_error = e
            continue

    raise RuntimeError(
        f"No valid JSON object with a '{required_key}' key found anywhere in the output "
        f"({len(candidates)} brace-balanced candidate(s) tried, last parse error: {last_error}). "
        f"Raw output:\n{raw_text[:1500]}"
    )


VALID_ERAS = ("ancient_egypt", "ancient_rome")


def validate_script_dict(data: dict) -> dict:
    era = data.get("era")
    if era not in VALID_ERAS:
        raise RuntimeError(f"Script has invalid/missing era: {era!r}")

    segments = data.get("segments")
    if not isinstance(segments, list) or len(segments) < 4:
        raise RuntimeError("Script has too few segments (or 'segments' missing/not a list).")

    total_words = 0
    for i, seg in enumerate(segments):
        seg_type = seg.get("type")
        if seg_type == "narration":
            total_words += len(seg.get("text", "").split())
        elif seg_type == "dialogue":
            lines = seg.get("lines", [])
            if not lines:
                raise RuntimeError(f"Segment {i} is type 'dialogue' but has no lines.")
            for line in lines:
                total_words += len(line.get("text", "").split())
        else:
            raise RuntimeError(f"Segment {i} has invalid type: {seg_type!r}")
        if not seg.get("image_file"):
            raise RuntimeError(f"Segment {i} is missing its generated image_file.")

    if total_words < 1000 or total_words > 3200:
        raise RuntimeError(
            f"Script word count ({total_words}) is way outside the expected 1600-2400 "
            "word range — likely a truncated or malformed generation."
        )

    print(f"Script validated: era={era}, {len(segments)} segments, ~{total_words} words.")
    return data


def flatten_script_to_text(parsed_script: dict) -> str:
    parts = []
    for seg in parsed_script["segments"]:
        if seg["type"] == "narration":
            parts.append(seg["text"])
        else:
            for line in seg["lines"]:
                parts.append(f'{line["speaker"]}: {line["text"]}')
    return "\n".join(parts)


OUTLINE_TASK_DESCRIPTION = (
    "Using the research, plan a 10-15 minute narrated history video as a BEAT OUTLINE — "
    "structure only, not the full narration text yet.\n\n"
    "Output ONLY a single JSON object with this exact shape, nothing else:\n"
    "{\n"
    '  "era": "ancient_egypt" | "ancient_rome",\n'
    '  "beats": [\n'
    '    {"type": "narration", "gist": "<1 sentence: what happens in this beat, described '
    'concretely and visually enough to base an illustration on>"},\n'
    '    {"type": "dialogue", "gist": "<1 sentence: what this exchange is about>"}\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Produce 12-16 beats total, covering the full story arc from the research, in order.\n"
    "- Most beats should be type 'narration'.\n"
    "- Use type 'dialogue' only occasionally (roughly every 3-5 narration beats).\n"
    "- Each 'gist' must describe a concrete, visualizable moment (a place, an action, "
    "people doing something specific) — it's used to generate an illustration, so avoid "
    "vague or abstract gists like 'tensions rise'.\n"
    "- The first beat should be a hook.\n"
    "- Output must be ONLY the JSON object — first character '{', last character '}'."
)


def _generate_outline_direct(research_text: str, extra_note: str = "") -> dict:
    """Step 1 of 2: ask for a compact structural outline, not the full script.
    Chunking (outline, then one small generation per beat) is more reliable
    for hitting an aggregate word-count target than one big single-shot
    generation, regardless of model."""
    description = OUTLINE_TASK_DESCRIPTION
    if extra_note:
        description += f"\n\nIMPORTANT — this is a retry. Previous attempt was rejected: {extra_note}"
    messages = [
        {"role": "system", "content": "You are a precise JSON generator. Output ONLY the requested JSON object, nothing else."},
        {"role": "user", "content": f"Research to base the outline on:\n{research_text}\n\n{description}"},
    ]
    raw = _call_groq_direct(messages, max_tokens=2048, temperature=0.2)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    data = _extract_json_object(text, required_key="beats")

    era = data.get("era")
    if era not in VALID_ERAS:
        raise RuntimeError(f"Outline has invalid/missing era: {era!r}")
    beats = data.get("beats")
    if not isinstance(beats, list) or len(beats) < 10:
        raise RuntimeError(f"Outline has too few beats ({len(beats) if isinstance(beats, list) else 0}, need >=10).")
    for i, beat in enumerate(beats):
        if beat.get("type") not in ("narration", "dialogue"):
            raise RuntimeError(f"Outline beat {i} has invalid type: {beat.get('type')!r}")
        if not beat.get("gist"):
            raise RuntimeError(f"Outline beat {i} is missing a 'gist'.")

    return data


NARRATION_MIN_WORDS, NARRATION_MAX_WORDS = 110, 170


def _generate_narration_text(gist: str, era: str, attempts: int = 3) -> str:
    text = ""
    for attempt in range(1, attempts + 1):
        messages = [
            {"role": "system", "content": "You write narration for a history video. Output ONLY the narration text, nothing else."},
            {"role": "user", "content": (
                f"Write {NARRATION_MIN_WORDS}-{NARRATION_MAX_WORDS} words of narration for a "
                f"cartoon-illustrated history video (era: {era}). This is ONE beat of a larger "
                "script — write ONLY the narration text itself, nothing else: no JSON, no "
                "labels, no preamble, no markdown.\n\n"
                f"What happens in this beat: {gist}\n\n"
                "Keep sentences short — this is read aloud by AI voiceover. Write in an "
                "engaging storytelling narrator voice."
            )},
        ]
        text = _call_groq_direct(messages, max_tokens=800, temperature=0.6).strip()
        text = re.sub(r"^```\s*|\s*```$", "", text)
        if len(text.split()) >= NARRATION_MIN_WORDS * 0.7:
            return text
        print(f"[BEAT RETRY] narration beat too short ({len(text.split())} words), "
              f"retrying ({attempt}/{attempts})...")
    return text


DIALOGUE_SPEAKER_LABELS = ["Speaker 1", "Speaker 2"]


def _generate_dialogue_lines(gist: str, era: str, attempts: int = 3) -> list:
    """No named characters anymore — just two generic speaker labels. Voice
    variety comes from mapping these two labels to two fixed TTS voices."""
    lines = []
    for attempt in range(1, attempts + 1):
        messages = [
            {"role": "system", "content": "You write short dialogue for a history video. Output ONLY the requested lines, nothing else."},
            {"role": "user", "content": (
                "Write a short 2-4 line dialogue exchange between two people for a "
                f"cartoon-illustrated history video (era: {era}). Label the speakers exactly "
                "'Speaker 1' and 'Speaker 2'.\n\n"
                f"What this exchange is about: {gist}\n\n"
                "Output ONLY the lines, one per line, in this exact format:\n"
                "Speaker 1: line text\n"
                "Speaker 2: line text\n\n"
                "No JSON, no preamble, no extra commentary — just the alternating lines."
            )},
        ]
        raw = _call_groq_direct(messages, max_tokens=400, temperature=0.6).strip()
        lines = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            speaker, _, spoken_text = line.partition(":")
            speaker, spoken_text = speaker.strip(), spoken_text.strip()
            if speaker in DIALOGUE_SPEAKER_LABELS and spoken_text:
                lines.append({"speaker": speaker, "text": spoken_text})
        if len(lines) >= 2:
            return lines
        print(f"[BEAT RETRY] dialogue beat produced too few valid lines, retrying ({attempt}/{attempts})...")
    return lines


# ---------------------------------------------------------------------
# 6b. Image generation — Pollinations.ai (free, no API key)
#
# One generated cartoon-illustration image per beat, based on that beat's
# gist. A fixed style suffix keeps the look consistent across the whole
# video instead of each image looking like a different art style.
# ---------------------------------------------------------------------
STYLE_SUFFIX = (
    "flat vector cartoon illustration, bold clean outlines, warm vibrant colors, "
    "educational children's storybook art style, no text, no watermark, no logo"
)
ERA_LABELS = {"ancient_egypt": "Ancient Egypt", "ancient_rome": "Ancient Rome"}


def generate_beat_image(gist: str, era: str, filename: str, attempts: int = 4) -> str:
    """Pollinations.ai has no formal SLA and can be rate-limited/flaky like
    every other free service in this pipeline — same retry-with-backoff
    pattern as everywhere else. A too-small response is treated as a
    failure too, since that's usually an error page, not a real image."""
    era_label = ERA_LABELS.get(era, era)
    prompt = f"{era_label} scene: {gist}. {STYLE_SUFFIX}"
    encoded_prompt = requests.utils.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(
                url,
                params={"width": 1280, "height": 720, "nologo": "true"},
                timeout=90,
            )
            resp.raise_for_status()
            if len(resp.content) < 2000:
                raise RuntimeError(f"Response too small to be a real image ({len(resp.content)} bytes)")
            with open(filename, "wb") as f:
                f.write(resp.content)
            return filename
        except Exception as e:
            last_error = e
            wait = 8 * attempt
            print(f"[IMAGE RETRY] '{filename}' failed (attempt {attempt}/{attempts}): {e}\nRetrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Image generation failed for '{filename}' after {attempts} attempts: {last_error}")


def _assemble_script_from_outline(outline: dict) -> dict:
    era = outline["era"]
    os.makedirs("beat_images", exist_ok=True)
    segments = []
    for i, beat in enumerate(outline["beats"]):
        image_file = generate_beat_image(beat["gist"], era, f"beat_images/beat_{i:03d}.jpg")
        if beat["type"] == "narration":
            segments.append({
                "type": "narration",
                "text": _generate_narration_text(beat["gist"], era),
                "image_file": image_file,
            })
        else:
            lines = _generate_dialogue_lines(beat["gist"], era)
            if lines:
                segments.append({"type": "dialogue", "lines": lines, "image_file": image_file})
    return {"era": era, "segments": segments}


research_text = getattr(research_task.output, "raw", str(research_task.output))

MAX_OUTLINE_ATTEMPTS = 3
outline = None
last_error = None

for attempt in range(1, MAX_OUTLINE_ATTEMPTS + 1):
    print(f"[OUTLINE] Generating outline (attempt {attempt}/{MAX_OUTLINE_ATTEMPTS})...")
    t0 = time.time()
    try:
        outline = _generate_outline_direct(research_text, extra_note=str(last_error) if last_error else "")
        print(f"[TIMING] 'Outline (attempt {attempt})' finished — took {time.time() - t0:.1f}s, "
              f"{len(outline['beats'])} beats")
        break
    except RuntimeError as e:
        last_error = e
        print(f"[OUTLINE RETRY] Attempt {attempt}/{MAX_OUTLINE_ATTEMPTS} failed: {e}")

if outline is None:
    raise RuntimeError(f"Outline generation failed after {MAX_OUTLINE_ATTEMPTS} attempts. Last error: {last_error}")

print(f"[SCRIPT] Generating {len(outline['beats'])} beats (text + image) individually...")
t0 = time.time()
script_dict = _assemble_script_from_outline(outline)
print(f"[TIMING] 'All beats generated' finished — took {time.time() - t0:.1f}s")

parsed_script = validate_script_dict(script_dict)

# ---------------------------------------------------------------------
# 6c. SEO — runs AFTER the script is validated, using the actual finished
# script text embedded directly in the prompt.
# ---------------------------------------------------------------------
seo_task = Task(
    description=(
        "Based on the following finished script, write:\n"
        "1. Three title options (under 60 characters, curiosity-driven, no clickbait flags)\n"
        "2. A YouTube description (first 2 lines keyword-rich, then a short summary)\n"
        "3. A list of 15 relevant tags\n\n"
        f"SCRIPT:\n{flatten_script_to_text(parsed_script)}"
    ),
    expected_output="Titles, description, and tags clearly labeled.",
    agent=seo_specialist,
    callback=_timing_callback("SEO Specialist"),
)
_run_single_task_crew(seo_specialist, seo_task, "SEO")
seo_output = getattr(seo_task.output, "raw", str(seo_task.output))

# ---------------------------------------------------------------------
# 7. Voiceover — one file per narration block / dialogue line
# ---------------------------------------------------------------------
import edge_tts

NARRATOR_VOICE = "en-US-GuyNeural"
DIALOGUE_VOICES = {
    "Speaker 1": "en-US-DavisNeural",
    "Speaker 2": "en-US-JennyNeural",
}
TTS_FALLBACK_VOICE = "en-US-AriaNeural"

VIDEO_W, VIDEO_H = 1280, 720
VIDEO_FPS = 30


def tts_to_file(text: str, voice: str, filename: str, attempts: int = 4):
    """edge-tts's NoAudioReceived error is a known, still-unresolved issue
    upstream (it's an unofficial wrapper around Edge's internal "Read Aloud"
    service, not a real public API). Plain retries don't always help since
    it can be voice-specific throttling, so after 2 failures we also switch
    to a fallback voice."""
    async def _run(v):
        communicate = edge_tts.Communicate(text=text, voice=v)
        await communicate.save(filename)

    last_error = None
    for attempt in range(1, attempts + 1):
        use_voice = voice if attempt <= 2 else TTS_FALLBACK_VOICE
        try:
            asyncio.run(_run(use_voice))
            return
        except Exception as e:
            last_error = e
            wait = 5 * attempt
            print(f"[TTS RETRY] '{filename}' failed with voice '{use_voice}' "
                  f"(attempt {attempt}/{attempts}): {e}\nRetrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"TTS failed for '{filename}' after {attempts} attempts. Last error: {last_error}")


def generate_all_voiceovers(parsed_script: dict, out_dir: str = "audio"):
    os.makedirs(out_dir, exist_ok=True)
    clip_index = 0
    for seg in parsed_script["segments"]:
        if seg["type"] == "narration":
            fname = os.path.join(out_dir, f"clip_{clip_index:04d}_narration.mp3")
            tts_to_file(seg["text"], NARRATOR_VOICE, fname)
            seg["audio_file"] = fname
            clip_index += 1
        else:
            for line in seg["lines"]:
                voice = DIALOGUE_VOICES.get(line["speaker"], NARRATOR_VOICE)
                fname = os.path.join(out_dir, f"clip_{clip_index:04d}_dialogue.mp3")
                tts_to_file(line["text"], voice, fname)
                line["audio_file"] = fname
                clip_index += 1
    print(f"Generated {clip_index} voiceover clips in {out_dir}/")
    return parsed_script


def _get_audio_duration(path: str) -> float:
    """Uses ffmpeg itself (not ffprobe) to get duration — ffprobe isn't
    guaranteed to be installed alongside ffmpeg on every runner image."""
    result = subprocess.run(
        ["ffmpeg", "-i", path, "-f", "null", "-"],
        capture_output=True, text=True,
    )
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if not match:
        raise RuntimeError(f"Could not determine duration of '{path}' from ffmpeg output:\n{result.stderr[-500:]}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


# ---------------------------------------------------------------------
# 8. Ken Burns rendering — one generated image per beat, slow zoom/pan for
# the duration of that beat's voiceover, instead of a static frame.
# ---------------------------------------------------------------------
def _render_ken_burns_clip(image_path: str, duration: float, out_path: str, fps: int = VIDEO_FPS):
    total_frames = max(1, int(round(duration * fps)))
    zoom_per_frame = 0.15 / max(total_frames, 1)  # ~15% zoom over the clip's full duration
    vf = (
        f"scale=1600:-1,zoompan=z='min(zoom+{zoom_per_frame:.6f},1.2)':"
        f"d={total_frames}:s={VIDEO_W}x{VIDEO_H}:fps={fps}"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", image_path,
         "-vf", vf, "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path],
        check=True, capture_output=True,
    )


def _build_render_plan(parsed_script: dict):
    """Returns (image_path, duration) pairs (one per beat) plus the ordered
    list of audio files to concatenate. A dialogue beat holds on its one
    image for the combined duration of all its lines, while the audio
    plays each line in sequence underneath."""
    plan = []
    audio_files_in_order = []

    for seg in parsed_script["segments"]:
        if seg["type"] == "narration":
            duration = _get_audio_duration(seg["audio_file"])
            plan.append((seg["image_file"], duration))
            audio_files_in_order.append(seg["audio_file"])
        else:
            total_duration = 0.0
            for line in seg["lines"]:
                total_duration += _get_audio_duration(line["audio_file"])
                audio_files_in_order.append(line["audio_file"])
            plan.append((seg["image_file"], total_duration))

    return plan, audio_files_in_order


def build_video(parsed_script: dict, out_path: str = "final_video.mp4"):
    plan, audio_files_in_order = _build_render_plan(parsed_script)
    print(f"Render plan: {len(plan)} image segments.")

    os.makedirs("clips", exist_ok=True)
    clip_paths = []
    for i, (image_path, duration) in enumerate(plan):
        clip_path = f"clips/clip_{i:04d}.mp4"
        _render_ken_burns_clip(image_path, duration, clip_path)
        clip_paths.append(clip_path)

    video_concat_path = "video_concat_list.txt"
    with open(video_concat_path, "w") as f:
        for path in clip_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    audio_concat_path = "audio_concat_list.txt"
    with open(audio_concat_path, "w") as f:
        for path in audio_files_in_order:
            f.write(f"file '{os.path.abspath(path)}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", audio_concat_path,
         "-c", "copy", "full_audio.mp3"],
        check=True, capture_output=True,
    )

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", video_concat_path,
         "-i", "full_audio.mp3",
         "-c:v", "copy",
         "-c:a", "aac", "-shortest", out_path],
        check=True, capture_output=True,
    )
    total_duration = sum(d for _, d in plan)
    print(f"Video ready: {out_path} (~{total_duration:.0f}s of content, {len(plan)} image segments)")
    return out_path


# ---------------------------------------------------------------------
# 9. Thumbnail — reuse the first beat's generated image, with title text
# overlaid on top.
# ---------------------------------------------------------------------
def generate_thumbnail(parsed_script: dict, title: str, out_path: str = "thumbnail.jpg"):
    first_image_path = parsed_script["segments"][0]["image_file"]
    frame = Image.open(first_image_path).convert("RGB").resize((VIDEO_W, VIDEO_H))
    draw = ImageDraw.Draw(frame)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
    except Exception:
        font = ImageFont.load_default()

    words = title.upper().split(" ")
    y = 30
    for word in words:
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
            draw.text((30 + dx, y + dy), word, font=font, fill="black")
        draw.text((30, y), word, font=font, fill="white")
        y += 100

    frame.save(out_path, quality=90)
    print(f"Thumbnail ready: {out_path}")
    return out_path


# ---------------------------------------------------------------------
# 10. Upload
# ---------------------------------------------------------------------
def extract_seo_title(text):
    match = re.search(r"1\.\s*[\"\u201c]?(.+?)[\"\u201d]?\s*(?:\(|$)", text)
    return match.group(1).strip() if match else "Automated Video"


def upload_video(video_path: str, thumbnail_path: str, title: str, description: str, tags: list):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
    credentials = Credentials.from_authorized_user_file("youtube_token.json", SCOPES)
    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            raise RuntimeError(
                "YouTube token is invalid and can't be refreshed automatically. "
                "Re-run the one-time manual login step to generate a fresh youtube_token.json."
            )

    youtube = build("youtube", "v3", credentials=credentials)

    request_body = {
        "snippet": {
            "title": title[:100],
            "description": description[:4900],
            "tags": tags,
            "categoryId": "24",
        },
        "status": {
            "privacyStatus": "private",  # keep this until you've watched a few uploads and trust it
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
    response = youtube.videos().insert(part="snippet,status", body=request_body, media_body=media).execute()
    video_id = response["id"]
    print(f"Uploaded! https://youtu.be/{video_id}")

    try:
        youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_path)).execute()
        print("Thumbnail set.")
    except Exception as e:
        print(f"Thumbnail upload failed (likely phone verification not done yet): {e}")

    return video_id


# ---------------------------------------------------------------------
# 11. Run everything
# ---------------------------------------------------------------------
parsed_script = generate_all_voiceovers(parsed_script)
video_path = build_video(parsed_script)

video_title = extract_seo_title(seo_output)[:100]
thumb_path = generate_thumbnail(parsed_script, video_title)

upload_video(
    video_path, thumb_path, video_title, seo_output[:4900],
    tags=["history", parsed_script["era"].replace("_", " ")],
)

print("\nDone.")
