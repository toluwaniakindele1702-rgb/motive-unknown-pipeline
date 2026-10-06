"""Relic Loop permanent topic gate entrypoint.

The original implementation is kept in pipeline_legacy.py. This file is the
actual executable pipeline entrypoint and enforces topic originality before
calling the expensive legacy pipeline stages.
"""
from __future__ import annotations

import json
import re
import sys
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
HISTORY_PATH = STATE_DIR / "content_history.json"
TOPIC_HISTORY_PATH = STATE_DIR / "topic_history.json"
CURRENT_TOPIC_PATH = STATE_DIR / "current_run" / "topic.json"

# Permanent families the channel has explicitly retired.
# Coffee/cup/mug/heat/liquid combinations are one retired family; zipper
# questions are another retired family regardless of wording.
BANNED_ZIPPER_RE = re.compile(
    r"\b(?:zipper|zippers|zip[- ]?fastener|clothing zip|jacket zip)\b|"
    r"\b(?:zip|snag|snagged|stuck)\b.*\b(?:zip|zipper|jacket|pants|clothing|fabric|seam)\b|"
    r"\b(?:zip|zipper|jacket|pants|clothing|fabric|seam)\b.*\b(?:zip|zipper|snag|snagged|stuck)\b",
    re.I,
)
BANNED_COFFEE_RE = re.compile(r"\bcoffee\b", re.I)
CUP_WORDS = {"cup", "cups", "mug", "mugs"}
HEAT_WORDS = {"hot", "warm", "heating", "heated", "heat", "liquid", "liquids"}

STOPWORDS = {
    "a","an","the","why","how","does","do","did","can","could","would","is","are","was","were",
    "to","of","in","on","for","with","from","and","or","that","this","these","those","it","its",
    "people","person","things","thing","really","actually","just","ever","you","your","we","our",
}
ALIASES = {
    "mugs":"mug","cups":"cup","zippers":"zipper","zip":"zipper","snags":"snag","snagged":"snag",
    "stuck":"stuck","warm":"hot","heating":"hot","heated":"hot","liquids":"liquid",
}


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def _norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    return {ALIASES.get(t, t) for t in _norm(text).split() if t not in STOPWORDS and len(t) > 2}


def _history_questions() -> list[str]:
    out: list[str] = []
    data = _load(HISTORY_PATH, {})
    if isinstance(data, dict):
        for item in data.get("videos", []):
            if isinstance(item, dict):
                value = item.get("question") or item.get("topic") or item.get("title")
                if value:
                    out.append(str(value))
    data = _load(TOPIC_HISTORY_PATH, {})
    if isinstance(data, dict):
        for item in data.get("reservations", []):
            if isinstance(item, dict) and item.get("question"):
                out.append(str(item["question"]))
    return list(dict.fromkeys(out))


def _banned_family(question: str, topic: str) -> str | None:
    text = f"{question} {topic}"
    if BANNED_ZIPPER_RE.search(text):
        return "retired zipper/zip/snags family"
    normalized = _tokens(text)
    if ("coffee" in normalized and normalized & (CUP_WORDS | HEAT_WORDS)):
        return "retired coffee/cup/mug/heat/liquid family"
    if normalized & CUP_WORDS and normalized & HEAT_WORDS:
        return "retired hot/warm cup/mug/liquid family"
    return None


def _duplicate_reason(candidate: str, history: list[str]) -> str | None:
    c_norm = _norm(candidate)
    c_tokens = _tokens(candidate)
    for old in history:
        o_norm = _norm(old)
        if not o_norm:
            continue
        if c_norm == o_norm:
            return f"exact historical match: {old}"
        ratio = SequenceMatcher(None, c_norm, o_norm, autojunk=False).ratio()
        if ratio >= 0.72:
            return f"near-duplicate ({ratio:.2f} similarity): {old}"
        o_tokens = _tokens(old)
        if c_tokens and o_tokens:
            overlap = len(c_tokens & o_tokens) / max(1, min(len(c_tokens), len(o_tokens)))
            if overlap >= 0.70 and len(c_tokens & o_tokens) >= 3:
                return f"same topic fingerprint: {old}"
    return None


def _reserve(topic: dict[str, Any]) -> None:
    data = _load(TOPIC_HISTORY_PATH, {"version": 1, "reservations": []})
    if not isinstance(data, dict):
        data = {"version": 1, "reservations": []}
    reservations = data.setdefault("reservations", [])
    reservations.append({
        "date": datetime.now(timezone.utc).isoformat(),
        "question": topic.get("question"),
        "topic": topic.get("topic"),
        "title": topic.get("title"),
        "reserved": True,
    })
    # Never discard old reservations. This is the permanent history layer.
    data["version"] = 2
    TOPIC_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOPIC_HISTORY_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _accept(topic: dict[str, Any], history: list[str]) -> None:
    question = str(topic.get("question") or "").strip()
    label = str(topic.get("topic") or "").strip()
    if not question:
        raise RuntimeError("TOPIC GATE: selector returned no question.")
    family = _banned_family(question, label)
    if family:
        raise RuntimeError(f"TOPIC GATE: REJECTED {family}: {question}")
    reason = _duplicate_reason(question, history)
    if reason:
        raise RuntimeError(f"TOPIC GATE: REJECTED duplicate: {reason}")
    _reserve(topic)
    print(f"[TOPIC GATE] PASS + PERMANENT RESERVATION: {question}")


def _main(mode: str = "full") -> None:
    import pipeline_legacy

    if mode != "full":
        pipeline_legacy.main(mode)
        return

    history = _history_questions()

    # A resumed topic must pass the same gate. If it is blocked, delete only
    # the saved topic checkpoint so the legacy pipeline is forced to choose a new one.
    if CURRENT_TOPIC_PATH.exists():
        saved = _load(CURRENT_TOPIC_PATH, {})
        if isinstance(saved, dict) and saved.get("question"):
            try:
                _accept(saved, history)
            except RuntimeError as exc:
                print(f"[{exc}]")
                CURRENT_TOPIC_PATH.unlink(missing_ok=True)
            else:
                pipeline_legacy.main(mode)
                return

    original_choose = pipeline_legacy.choose_topic

    def gated_choose_topic(legacy_history: dict[str, Any]) -> dict[str, Any]:
        # Ask the original selector for candidates, but do not let any candidate
        # reach research until it passes the permanent gate.
        for attempt in range(1, 8):
            candidate = original_choose(legacy_history)
            try:
                _accept(candidate, history)
                return candidate
            except RuntimeError as exc:
                print(f"[TOPIC GATE] candidate {attempt}/7 rejected: {exc}")
                history.append(str(candidate.get("question") or candidate.get("topic") or ""))
        raise RuntimeError("TOPIC GATE: 7 candidate attempts rejected; no safe original topic found.")

    pipeline_legacy.choose_topic = gated_choose_topic
    pipeline_legacy.main(mode)


if __name__ == "__main__":
    mode = "full"
    if "--mode" in sys.argv:
        i = sys.argv.index("--mode")
        if i + 1 < len(sys.argv):
            mode = sys.argv[i + 1]
    _main(mode)
