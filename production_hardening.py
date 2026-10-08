"""Last-mile production hardening for Relic Loop."""
from __future__ import annotations
import subprocess
from pathlib import Path
import pipeline as p

_original_research_topic = p.research_topic

def _fallback_research(topic: dict) -> str:
    question = str(topic.get("question") or "").strip()
    category = str(topic.get("category") or "everyday curiosity").strip()
    angles = topic.get("search_angles") or []
    prompt = f"""
Create an evidence-conscious research dossier for a Relic Loop episode.
Question: {question}
Category: {category}
Search angles: {angles}

The normal browser-search stage failed. Do not pretend you browsed the web and do not invent
citations, quotations, statistics, names, dates, or URLs. Use only knowledge you are reasonably
confident about and clearly label uncertainty.

Return plain text with these headings:
1. CORE ANSWER
2. STORY BEATS (8-12 numbered beats)
3. MECHANISM / CAUSE AND EFFECT
4. IMPORTANT EXAMPLES
5. MISCONCEPTIONS OR CONTRADICTIONS
6. DISPUTES OR UNCERTAINTY
7. VERIFIED SOURCES

For VERIFIED SOURCES write exactly: Browser search unavailable in fallback.
""".strip()
    return p.groq_call(
        p.GROQ_RESEARCH_MODEL,
        [{"role": "user", "content": prompt}],
        max_completion_tokens=3200,
        temperature=0.25,
        attempts=4,
    )

def _safe_research_topic(topic: dict) -> str:
    try:
        return _original_research_topic(topic)
    except Exception as exc:
        print(f"[RESEARCH FAILSAFE] Browser research failed: {exc}")
        print("[RESEARCH FAILSAFE] Switching to plain Groq research instead of killing the run.")
        return _fallback_research(topic)

p.research_topic = _safe_research_topic

# The workflow persists state/current_run at the end of every run. Those files are
# transient runner state. Mark already-tracked copies assume-unchanged so a concurrent
# dispatcher cannot make an otherwise successful production run fail on git push.
def _protect_transient_state() -> None:
    root = Path(__file__).resolve().parent
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "state/current_run"],
            cwd=root, text=True, capture_output=True, check=False,
        ).stdout.splitlines()
        if tracked:
            subprocess.run(
                ["git", "update-index", "--assume-unchanged", *tracked],
                cwd=root, check=False,
            )
            print(f"[STATE HARDENING] Protected {len(tracked)} transient run-state file(s).")
    except Exception as exc:
        print(f"[STATE HARDENING] Could not protect transient state: {exc}")

_protect_transient_state()
print("[PRODUCTION HARDENING] Research fallback + state-race protection installed.")
