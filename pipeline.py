"""
Motive Unknown — automated video pipeline (Ken Burns / cartoon-illustration
format, fast-cut with burned-in subtitles). Runs unattended on GitHub
Actions. No interactive input anywhere.

Required GitHub Secrets (Settings -> Secrets and variables -> Actions):
  GROQ_API_KEY          - your Groq API key
  YOUTUBE_TOKEN_JSON    - the full contents of your youtube_token.json file
  YOUTUBE_CLIENT_SECRET_JSON - the full contents of your client_secret_....json file

Optional: drop a royalty-free music file at assets/music/background.mp3
(e.g. from YouTube Audio Library or Pixabay Music) to enable background
music — the pipeline skips music gracefully if that file isn't present.

No local character assets required — every clip's image is generated on
the fly (Pollinations.ai, free, no key) from a short phrase of narration,
so images change roughly every 2-3 seconds in step with the voiceover, and
that same phrase is burned in as a subtitle.
"""

import os
import re
import time
import json
import asyncio
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# images are generated per-clip now, that constraint could be lifted
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
        "particular event, figure, or moment — that has strong curiosity-gap potential, "
        "ideally with a twist, mystery, or unresolved question. You only pick topics from "
        "Ancient Egypt or Ancient Rome. You pick ONE topic and justify why it will perform well."
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
        "curiosity topics. Favor stories with genuine mystery, conflict, or a twist — not "
        "just a dry historical fact. Pick ONE specific angle. State the topic clearly, "
        "name which era it's from (ancient_egypt or ancient_rome), and give 2-3 sentences "
        "on why it will perform well."
    ),
    expected_output="One clearly stated topic, its era, and a short justification.",
    agent=topic_scout,
    callback=_timing_callback("Topic Scout"),
)

research_task = Task(
    description=(
        "Using the topic chosen by the Topic Scout, research 8-12 specific, surprising, "
        "and verifiable facts or story beats, each with a one-line source or context. "
        "Prioritize facts with real dramatic weight — betrayals, unsolved mysteries, "
        "shocking reversals, vivid physical detail — over dry background information. "
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
        if seg.get("type") != "narration":
            raise RuntimeError(f"Segment {i} has invalid type: {seg.get('type')!r} (only 'narration' is used now).")
        total_words += len(seg.get("text", "").split())
        if not seg.get("image_file"):
            raise RuntimeError(f"Segment {i} is missing its generated image_file.")

    if total_words < 1000 or total_words > 3200:
        raise RuntimeError(
            f"Script word count ({total_words}) is way outside the expected 1600-2400 "
            "word range — likely a truncated or malformed generation."
        )

    print(f"Script validated: era={era}, {len(segments)} clips, ~{total_words} words.")
    return data


def flatten_script_to_text(parsed_script: dict) -> str:
    return "\n".join(seg["text"] for seg in parsed_script["segments"])


OUTLINE_TASK_DESCRIPTION = (
    "Using the research, plan a 10-15 minute narrated history video as a BEAT OUTLINE — "
    "structure only, not the full narration text yet.\n\n"
    "Output ONLY a single JSON object with this exact shape, nothing else:\n"
    "{\n"
    '  "era": "ancient_egypt" | "ancient_rome",\n'
    '  "beats": [\n'
    '    {"type": "narration", "gist": "<1 sentence: what happens in this beat, described '
    'concretely and visually enough to base an illustration on>"}\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    "- Produce 12-16 beats total, covering the full story arc from the research, in order. "
    "Every beat is narration — there's just a narrator over illustrated scenes.\n"
    "- Frame this like a suspenseful mystery/true-crime documentary, not a dry timeline of "
    "facts. Build curiosity and tension — favor beats that reveal something surprising, "
    "unsettling, or that raise a new question, especially in the middle and toward the end.\n"
    "- Each 'gist' must describe a concrete, visualizable moment (a place, an action, "
    "people doing something specific) — it's used to generate an illustration, so avoid "
    "vague or abstract gists like 'tensions rise'.\n"
    "- The first beat should be a strong hook.\n"
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
    raw = _call_groq_direct(messages, max_tokens=2048, temperature=0.3)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    data = _extract_json_object(text, required_key="beats")

    era = data.get("era")
    if era not in VALID_ERAS:
        raise RuntimeError(f"Outline has invalid/missing era: {era!r}")
    beats = data.get("beats")
    if not isinstance(beats, list) or len(beats) < 10:
        raise RuntimeError(f"Outline has too few beats ({len(beats) if isinstance(beats, list) else 0}, need >=10).")
    for i, beat in enumerate(beats):
        if beat.get("type") != "narration":
            raise RuntimeError(f"Outline beat {i} has invalid type: {beat.get('type')!r}")
        if not beat.get("gist"):
            raise RuntimeError(f"Outline beat {i} is missing a 'gist'.")

    return data


NARRATION_MIN_WORDS, NARRATION_MAX_WORDS = 110, 170


def _generate_narration_text(gist: str, era: str, attempts: int = 3) -> str:
    text = ""
    for attempt in range(1, attempts + 1):
        messages = [
            {"role": "system", "content": (
                "You write narration for a suspenseful history documentary — think true-crime "
                "or mystery-show narrator, not a textbook. Output ONLY the narration text, "
                "nothing else."
            )},
            {"role": "user", "content": (
                f"Write {NARRATION_MIN_WORDS}-{NARRATION_MAX_WORDS} words of narration for a "
                f"cartoon-illustrated history video (era: {era}). This is ONE beat of a larger "
                "script — write ONLY the narration text itself, nothing else: no JSON, no "
                "labels, no preamble, no markdown.\n\n"
                f"What happens in this beat: {gist}\n\n"
                "Write in a punchy, suspenseful documentary-narrator voice. Use vivid, concrete "
                "sensory detail. Keep sentences SHORT — this is read aloud by AI voiceover, and "
                "short sentences also make for punchier on-screen captions. Where it fits "
                "naturally, use a rhetorical question or a line that raises tension. Avoid dry, "
                "encyclopedic phrasing — make the listener want to know what happens next."
            )},
        ]
        text = _call_groq_direct(messages, max_tokens=800, temperature=0.75).strip()
        text = re.sub(r"^```\s*|\s*```$", "", text)
        if len(text.split()) >= NARRATION_MIN_WORDS * 0.7:
            return text
        print(f"[BEAT RETRY] narration beat too short ({len(text.split())} words), "
              f"retrying ({attempt}/{attempts})...")
    return text


def _split_into_phrases(text: str, target_words: int = 8) -> list:
    """Splits narration into short phrases (~2-3 seconds of speech each at
    normal speaking pace) — each phrase gets its own image AND is the
    subtitle burned onto that image, so caption timing is automatically
    exact instead of needing separate alignment."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    phrases = []
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        if len(words) <= target_words + 4:
            phrases.append(sentence.strip())
            continue
        # Long sentence — split on comma boundaries first, then just chunk by
        # word count if there weren't enough commas to work with.
        parts = re.split(r"(?<=,)\s+", sentence)
        buffer = []
        for part in parts:
            buffer.extend(part.split())
            if len(buffer) >= target_words:
                phrases.append(" ".join(buffer))
                buffer = []
        if buffer:
            if phrases and len(buffer) < 3:
                phrases[-1] = phrases[-1] + " " + " ".join(buffer)
            else:
                phrases.append(" ".join(buffer))
    return [p for p in phrases if p]


# ---------------------------------------------------------------------
# 6b. Image generation — Pollinations.ai (free, no API key)
#
# One generated cartoon-illustration image per short phrase (not per whole
# beat) — this is what makes images change every 2-3 seconds in step with
# what's actually being said, instead of one image sitting on screen for a
# whole 30-45 second beat. A fixed style suffix keeps the look consistent
# across the whole video.
# ---------------------------------------------------------------------
STYLE_SUFFIX = (
    "flat vector cartoon illustration, bold clean outlines, warm vibrant colors, "
    "detailed and clear, educational storybook art style, no text, no watermark, no logo"
)
ERA_LABELS = {"ancient_egypt": "Ancient Egypt", "ancient_rome": "Ancient Rome"}
IMAGE_GEN_WORKERS = 5
TTS_GEN_WORKERS = 5


def generate_beat_image(phrase: str, era: str, filename: str, context_gist: str = "", attempts: int = 4) -> str:
    """Pollinations.ai has no formal SLA and can be rate-limited/flaky like
    every other free service in this pipeline — same retry-with-backoff
    pattern as everywhere else. A too-small response is treated as a
    failure too, since that's usually an error page, not a real image.
    Generated at a higher resolution than the final video so the Ken Burns
    zoom has detail to crop into instead of visibly pixelating."""
    era_label = ERA_LABELS.get(era, era)
    context_part = f" Broader scene context: {context_gist}." if context_gist else ""
    prompt = f"{era_label} scene, illustrating exactly this moment: {phrase}.{context_part} {STYLE_SUFFIX}"
    encoded_prompt = requests.utils.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(
                url,
                params={"width": 1600, "height": 900, "nologo": "true"},
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

    # Step 1: generate the full narration text per BIG beat (sequential —
    # these are real LLM calls against Groq's rate limit).
    beat_texts = []
    for beat in outline["beats"]:
        beat_texts.append((beat["gist"], _generate_narration_text(beat["gist"], era)))

    # Step 2: flatten into short phrase-level clips (~2-3s of speech each).
    clip_specs = []
    idx = 0
    for beat_gist, text in beat_texts:
        for phrase in _split_into_phrases(text, target_words=8):
            clip_specs.append((idx, phrase, beat_gist))
            idx += 1

    print(f"[SCRIPT] Split into {len(clip_specs)} short clips (~2-3s each) for image generation.")

    # Step 3: generate all images in parallel — Pollinations calls are
    # independent per clip, and with 150-250+ of them now, sequential would
    # be far too slow.
    segments = [None] * len(clip_specs)

    def _gen_one(spec):
        i, phrase, beat_gist = spec
        image_file = generate_beat_image(phrase, era, f"beat_images/clip_{i:04d}.jpg", context_gist=beat_gist)
        return i, {"type": "narration", "text": phrase, "image_file": image_file}

    with ThreadPoolExecutor(max_workers=IMAGE_GEN_WORKERS) as executor:
        futures = [executor.submit(_gen_one, spec) for spec in clip_specs]
        for future in as_completed(futures):
            i, seg = future.result()
            segments[i] = seg

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

print(f"[SCRIPT] Generating narration text + clips for {len(outline['beats'])} beats...")
t0 = time.time()
script_dict = _assemble_script_from_outline(outline)
print(f"[TIMING] 'All clips generated' finished — took {time.time() - t0:.1f}s")

parsed_script = validate_script_dict(script_dict)

# ---------------------------------------------------------------------
# 6c. SEO — runs AFTER the script is validated. Locked into a strict,
# parseable format instead of hoping to regex-match free-form natural
# language — that's what was causing the video title to fall back to
# "Automated Video" before.
# ---------------------------------------------------------------------
seo_task = Task(
    description=(
        "Based on the following finished script, write SEO metadata.\n\n"
        "Output in EXACTLY this format, nothing else — no extra commentary, no markdown, "
        "no headers:\n"
        "TITLE_1: <title, under 60 characters, curiosity-driven, no clickbait flags>\n"
        "TITLE_2: <title>\n"
        "TITLE_3: <title>\n"
        "DESCRIPTION: <YouTube description — first 2 lines keyword-rich, then a short summary>\n"
        "TAGS: <15 relevant tags, comma-separated>\n\n"
        f"SCRIPT:\n{flatten_script_to_text(parsed_script)}"
    ),
    expected_output="Exactly the TITLE_1/TITLE_2/TITLE_3/DESCRIPTION/TAGS format described above, nothing else.",
    agent=seo_specialist,
    callback=_timing_callback("SEO Specialist"),
)
_run_single_task_crew(seo_specialist, seo_task, "SEO")
seo_output = getattr(seo_task.output, "raw", str(seo_task.output))


def extract_seo_title(text: str) -> str:
    for pattern in [r"TITLE_1:\s*(.+)", r"Title\s*1[:\-]\s*(.+)", r"^1\.\s*(.+)"]:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("\"'\u201c\u201d").strip()
    return "Automated Video"


# ---------------------------------------------------------------------
# 7. Voiceover — one narration file per short clip
# ---------------------------------------------------------------------
import edge_tts

# Deeper, more mysterious voice for a documentary tone. Rate slowed and
# pitch lowered further on top of the voice's natural pitch, for weight.
NARRATOR_VOICE = "en-US-DavisNeural"
NARRATION_RATE = "-8%"
NARRATION_PITCH = "-15Hz"
TTS_FALLBACK_VOICE = "en-US-AriaNeural"

VIDEO_W, VIDEO_H = 1280, 720
VIDEO_FPS = 30


def tts_to_file(text: str, voice: str, filename: str, attempts: int = 4,
                 rate: str = NARRATION_RATE, pitch: str = NARRATION_PITCH):
    """edge-tts's NoAudioReceived error is a known, still-unresolved issue
    upstream (it's an unofficial wrapper around Edge's internal "Read Aloud"
    service, not a real public API). Plain retries don't always help since
    it can be voice-specific throttling, so after 2 failures we also switch
    to a fallback voice (at that point the rate/pitch tweak is dropped too,
    since it's more important to get SOME audio than the exact tone)."""
    async def _run(v, use_rate, use_pitch):
        communicate = edge_tts.Communicate(text=text, voice=v, rate=use_rate, pitch=use_pitch)
        await communicate.save(filename)

    last_error = None
    for attempt in range(1, attempts + 1):
        if attempt <= 2:
            use_voice, use_rate, use_pitch = voice, rate, pitch
        else:
            use_voice, use_rate, use_pitch = TTS_FALLBACK_VOICE, "+0%", "+0Hz"
        try:
            asyncio.run(_run(use_voice, use_rate, use_pitch))
            return
        except Exception as e:
            last_error = e
            wait = 5 * attempt
            print(f"[TTS RETRY] '{filename}' failed with voice '{use_voice}' "
                  f"(attempt {attempt}/{attempts}): {e}\nRetrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"TTS failed for '{filename}' after {attempts} attempts. Last error: {last_error}")


def generate_all_voiceovers(parsed_script: dict, out_dir: str = "audio"):
    """Parallelized — with 150-250+ short clips now instead of ~15 beats,
    sequential TTS generation would be far too slow."""
    os.makedirs(out_dir, exist_ok=True)
    segments = parsed_script["segments"]

    def _gen_one(i):
        fname = os.path.join(out_dir, f"clip_{i:04d}_narration.mp3")
        tts_to_file(segments[i]["text"], NARRATOR_VOICE, fname)
        return i, fname

    with ThreadPoolExecutor(max_workers=TTS_GEN_WORKERS) as executor:
        futures = [executor.submit(_gen_one, i) for i in range(len(segments))]
        for future in as_completed(futures):
            i, fname = future.result()
            segments[i]["audio_file"] = fname

    print(f"Generated {len(segments)} voiceover clips in {out_dir}/")
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
# 8. Ken Burns rendering + burned-in subtitles
#
# One generated image per short clip, a faster/more noticeable zoom given
# clips are now only ~2-3s each (the old slow zoom was tuned for 30-45s
# beats and would barely be visible now), plus the clip's own narration
# text burned in as a caption — timing is automatically exact since both
# the image AND the caption come from the same phrase-level split.
# ---------------------------------------------------------------------
CAPTION_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ZOOM_TARGET = 1.35  # ~35% zoom over each clip's duration — much faster/more
                     # noticeable than before, appropriate for a 2-3s clip
                     # instead of a 30-45s one


def _ffmpeg_escape_text(text: str) -> str:
    """Escapes characters that break ffmpeg's drawtext filter syntax.
    Apostrophes are swapped for a visually-identical unicode character
    instead of escaped, since backslash-escaping apostrophes inside
    drawtext's own quoting is unreliable across ffmpeg builds."""
    return (
        text.replace("\\", "\\\\")
            .replace(":", "\\:")
            .replace("'", "\u2019")
            .replace("%", "\\%")
    )


def _render_ken_burns_clip(image_path: str, duration: float, caption_text: str, out_path: str, fps: int = VIDEO_FPS):
    total_frames = max(1, int(round(duration * fps)))
    zoom_per_frame = (ZOOM_TARGET - 1.0) / max(total_frames, 1)
    escaped = _ffmpeg_escape_text(caption_text)
    vf = (
        f"scale=2000:-1,zoompan=z='min(zoom+{zoom_per_frame:.6f},{ZOOM_TARGET})':"
        f"d={total_frames}:s={VIDEO_W}x{VIDEO_H}:fps={fps},"
        f"drawtext=fontfile={CAPTION_FONT}:text='{escaped}':fontsize=46:fontcolor=white:"
        f"borderw=3:bordercolor=black:x=(w-text_w)/2:y=h-160"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", image_path,
         "-vf", vf, "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path],
        check=True, capture_output=True,
    )


def _build_render_plan(parsed_script: dict):
    """Returns (image_path, duration, caption_text) triples plus the
    ordered list of audio files to concatenate."""
    plan = []
    audio_files_in_order = []

    for seg in parsed_script["segments"]:
        duration = _get_audio_duration(seg["audio_file"])
        plan.append((seg["image_file"], duration, seg["text"]))
        audio_files_in_order.append(seg["audio_file"])

    return plan, audio_files_in_order


# ---------------------------------------------------------------------
# 8b. Background music (optional — skips gracefully if no file is present)
#
# Drop a royalty-free track (YouTube Audio Library, Pixabay Music, etc.)
# at the path below. Not something this pipeline can source on its own —
# music licensing needs a human picking a track they're actually cleared
# to use, not an automated download of whatever's easiest to find.
# ---------------------------------------------------------------------
BACKGROUND_MUSIC_PATH = "assets/music/background.mp3"
MUSIC_VOLUME_DB = -22


def _mix_background_music(narration_audio_path: str, out_path: str) -> str:
    if not os.path.exists(BACKGROUND_MUSIC_PATH):
        print(f"No background music found at '{BACKGROUND_MUSIC_PATH}' — using narration-only "
              f"audio. Drop a royalty-free track there to enable background music.")
        return narration_audio_path
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", narration_audio_path,
         "-stream_loop", "-1", "-i", BACKGROUND_MUSIC_PATH,
         "-filter_complex",
         f"[1:a]volume={MUSIC_VOLUME_DB}dB[music];[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
         "-map", "[aout]", out_path],
        check=True, capture_output=True,
    )
    print(f"Mixed background music from '{BACKGROUND_MUSIC_PATH}' under the narration.")
    return out_path


def build_video(parsed_script: dict, out_path: str = "final_video.mp4"):
    plan, audio_files_in_order = _build_render_plan(parsed_script)
    print(f"Render plan: {len(plan)} clips.")

    os.makedirs("clips", exist_ok=True)
    clip_paths = []
    for i, (image_path, duration, caption_text) in enumerate(plan):
        clip_path = f"clips/clip_{i:04d}.mp4"
        _render_ken_burns_clip(image_path, duration, caption_text, clip_path)
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

    final_audio_path = _mix_background_music("full_audio.mp3", "full_audio_mixed.mp3")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", video_concat_path,
         "-i", final_audio_path,
         "-c:v", "copy",
         "-c:a", "aac", "-shortest", out_path],
        check=True, capture_output=True,
    )
    total_duration = sum(d for _, d, _ in plan)
    print(f"Video ready: {out_path} (~{total_duration:.0f}s of content, {len(plan)} clips)")
    return out_path


# ---------------------------------------------------------------------
# 9. Thumbnail — reuse the first clip's generated image, with title text
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
