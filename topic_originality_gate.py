from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

PIPELINE = Path("pipeline.py")
MARKER = "# RELIC_LOOP_TOPIC_ORIGINALITY_GATE_V1"
text = PIPELINE.read_text(encoding="utf-8")
if MARKER in text:
    print("[TOPIC GATE] already installed")
    raise SystemExit(0)

anchor = '''        if all(data.get(k) for k in required) and isinstance(data.get("search_angles"), list) and score >= 8:\n'''
if anchor not in text:
    raise SystemExit("[TOPIC GATE] selector anchor not found")

replacement = anchor + '''            # RELIC_LOOP_TOPIC_ORIGINALITY_GATE_V1\n            # This is a HARD gate. The candidate cannot reach scripting unless it is materially\n            # different from every persisted episode in state/content_history.json.\n            history_path = Path(__file__).resolve().parent / "state" / "content_history.json"\n            try:\n                history_data = json.loads(history_path.read_text(encoding="utf-8"))\n            except Exception:\n                history_data = {"videos": []}\n            rows = history_data.get("videos", []) if isinstance(history_data, dict) else []\n\n            stop = {\n                "why", "what", "how", "does", "do", "did", "can", "could", "would",\n                "you", "your", "the", "a", "an", "is", "are", "was", "were",\n                "to", "of", "in", "on", "for", "we", "our", "it", "this", "that",\n                "not", "after", "before", "even", "really", "often", "sometimes",\n            }\n            def tokens(value):\n                return {w for w in re.findall(r"[a-z0-9]+", value.lower()) if len(w) > 2 and w not in stop}\n\n            candidate = str(data.get("question") or data.get("topic") or data.get("title") or "").strip().lower()\n            cand_tokens = tokens(candidate)\n            duplicate = False\n            reason = ""\n            for row in rows:\n                if not isinstance(row, dict):\n                    continue\n                previous = " | ".join(str(row.get(k, "")) for k in ("question", "topic", "title")).strip().lower()\n                if not previous or not candidate:\n                    continue\n                prev_tokens = tokens(previous)\n                shared = cand_tokens & prev_tokens\n                union = cand_tokens | prev_tokens\n                ratio = SequenceMatcher(None, candidate, previous).ratio()\n                overlap = len(shared) / max(1, len(union))\n\n                # Exact repeat / rewritten title / same subject or mechanism.\n                # These thresholds intentionally err on the side of rejecting a candidate.\n                if candidate == previous or ratio >= 0.72 or overlap >= 0.55 or len(shared) >= 3:\n                    duplicate = True\n                    reason = str(row.get("title") or row.get("question") or row.get("topic"))\n                    break\n\n            if duplicate:\n                print(f"[TOPIC HARD GATE] REJECTED: candidate overlaps previous episode: {reason}")\n                continue\n            print("[TOPIC HARD GATE] PASS: candidate is materially different from persisted episode history")\n'''
text = text.replace(anchor, replacement, 1)
PIPELINE.write_text(text, encoding="utf-8")
print("[TOPIC GATE] installed persisted-history hard gate")
