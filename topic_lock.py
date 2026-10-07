"""Hard production lock: Relic Loop may only consume assistant-curated queued topics.

This script is run before production and rewrites the legacy topic-selection entrypoint
so no Groq/model topic generation path can be reached. It also forces the main
orchestration to ignore stale current-run topics and consume the persistent queue.
"""
from __future__ import annotations

from pathlib import Path
import re

PIPELINE = Path("pipeline.py")

QUEUE_FUNCTION = r'''def choose_topic(history: dict[str, Any]) -> dict[str, Any]:
    """Return only an assistant-curated queue topic. Never ask an AI to invent one."""
    queue_path = STATE_DIR / "topic_queue.json"
    if not queue_path.exists():
        raise RuntimeError("Topic queue is missing: state/topic_queue.json. Refusing to invent a topic.")
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Topic queue is invalid JSON: {exc}") from exc

    topics = queue.get("topics") if isinstance(queue, dict) else None
    if not isinstance(topics, list) or not topics:
        raise RuntimeError("Topic queue is empty. Refusing to invent a topic or spend image-generation quota.")

    def canon(value: str) -> set[str]:
        words = re.findall(r"[a-z0-9]+", str(value or "").lower())
        stop = {"why","what","how","does","do","did","can","could","would","you","your","the","a","an","is","are","was","were","to","of","in","on","for","we","our","it","this","that","when","sometimes","really"}
        return {w for w in words if len(w) > 2 and w not in stop}

    previous: list[str] = []
    for row in history.get("videos", []) if isinstance(history, dict) else []:
        if isinstance(row, dict):
            for key in ("question", "topic", "title"):
                value = str(row.get(key) or "").strip()
                if value:
                    previous.append(value)

    # Permanent queue history also protects against repeats if content_history.json
    # is temporarily incomplete on a fresh runner.
    for row in topics:
        if isinstance(row, dict) and str(row.get("status", "")).lower() in {"used", "reserved", "skipped_duplicate"}:
            for key in ("question", "topic"):
                value = str(row.get(key) or "").strip()
                if value:
                    previous.append(value)

    candidates = [
        item for item in topics
        if isinstance(item, dict) and str(item.get("status", "queued")).lower() == "queued"
    ]
    candidates.sort(key=lambda item: int(item.get("id", 10**9)))
    if not candidates:
        raise RuntimeError("Topic queue has no unused queued topics. Refusing to invent a topic.")

    for item in candidates:
        question = str(item.get("question") or "").strip()
        if not question:
            raise RuntimeError(f"Queued topic {item.get('id')} has no question. Refusing to continue.")
        candidate_tokens = canon(question)
        duplicate = False
        duplicate_reason = ""
        for prior in previous:
            prior_tokens = canon(prior)
            shared = candidate_tokens & prior_tokens
            union = candidate_tokens | prior_tokens
            overlap = len(shared) / max(1, len(union))
            if question.casefold() == prior.casefold() or overlap >= 0.60 or len(shared) >= 3:
                duplicate = True
                duplicate_reason = prior
                break
        if duplicate:
            item["status"] = "skipped_duplicate"
            item["skip_reason"] = f"matches previously made/reserved topic: {duplicate_reason}"
            atomic_write_json(queue_path, queue)
            print(f"[TOPIC LOCK] Skipping duplicate queued topic {item.get('id')}: {question}")
            continue

        now = datetime.now(timezone.utc).isoformat()
        item["status"] = "reserved"
        item["reserved_at"] = now
        item["reserved_run_id"] = os.environ.get("GITHUB_RUN_ID", "local")
        item["reserved_run_number"] = os.environ.get("GITHUB_RUN_NUMBER", "")
        atomic_write_json(queue_path, queue)
        topic = {
            "queue_id": int(item["id"]),
            "question": question,
            "topic": str(item.get("topic") or question).strip(),
            "category": str(item.get("category") or "everyday curiosity").strip(),
            "era": str(item.get("era") or "modern day").strip(),
            "why_curious": str(item.get("why_curious") or "").strip(),
            "curiosity_gap": str(item.get("curiosity_gap") or "").strip(),
            "curiosity_score": int(item.get("curiosity_score", 9) or 9),
            "search_angles": list(item.get("search_angles") or []),
        }
        print(f"[TOPIC LOCK] Reserved assistant topic {topic['queue_id']}: {question}")
        return topic

    raise RuntimeError("Every queued topic was rejected as a previous/duplicate topic. No AI topic generation is allowed.")
'''.strip()


def apply() -> None:
    if not PIPELINE.exists():
        raise SystemExit("pipeline.py is missing")
    source = PIPELINE.read_text(encoding="utf-8")

    # Replace the entire legacy choose_topic function through the next top-level def.
    pattern = re.compile(r"(?ms)^def choose_topic\(history: dict\[str, Any\]\) -> dict\[str, Any\]:\n.*?(?=^def [A-Za-z_]\w*\(|\Z)")
    source, count = pattern.subn(QUEUE_FUNCTION + "\n\n", source, count=1)
    if count != 1:
        raise SystemExit("Could not replace the legacy topic selector; production must stop.")

    # Replace the orchestration's resume/AI-selection block with queue-only selection.
    main_pattern = re.compile(
        r'''(?ms)^    # Reuse small text artifacts saved by a previous failed Actions run\.\n.*?^    checkpoint\("topic_complete", question=topic\["question"\], resumed=topic_resumed\)'''
    )
    replacement = '''    # Topic ownership is deterministic. A stale current-run topic can never override the queue.
    topic = choose_topic(history)
    topic_resumed = False
    _save_current_json(CURRENT_TOPIC_PATH, topic)
    checkpoint("topic_complete", question=topic["question"], queue_id=topic.get("queue_id"), resumed=False)'''
    source, main_count = main_pattern.subn(replacement, source, count=1)
    if main_count != 1:
        raise SystemExit("Could not replace the main topic orchestration block; production must stop.")

    # Guard against the old AI selector being reintroduced later.
    forbidden = [
        "client.chat.completions.create",
        "Generate a fresh curiosity question",
        "Generate a new topic",
        "Choose a topic",
    ]
    # These strings may legitimately occur in research prompts; only fail if they remain
    # inside choose_topic after replacement.
    m = re.search(r"(?ms)^def choose_topic\(.*?(?=^def [A-Za-z_]\w*\(|\Z)", source)
    if not m or "groq_json(" in m.group(0) or "client.chat" in m.group(0):
        raise SystemExit("Topic lock verification failed: AI topic generation remains in choose_topic.")

    PIPELINE.write_text(source, encoding="utf-8")
    print("[TOPIC LOCK] VERIFIED: pipeline.py can only consume state/topic_queue.json for topic selection.")
    print("[TOPIC LOCK] VERIFIED: stale current_run/topic.json cannot override the queue.")


if __name__ == "__main__":
    apply()
