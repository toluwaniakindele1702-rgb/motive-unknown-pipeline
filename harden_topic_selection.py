from pathlib import Path

path = Path("pipeline.py")
text = path.read_text(encoding="utf-8")

MARKER = "# RELIC_LOOP_TOPIC_HARDENING_V2"
if MARKER in text:
    print("[TOPIC HARDENING] V2 already applied")
    raise SystemExit(0)

# Keep the model's topic brief focused on the channel's actual curiosity lane.
old = """Do NOT simply turn these examples into future videos. Generate fresh questions in the same\ncuriosity territory.\n"""
new = """Do NOT simply turn these examples into future videos. Generate fresh questions in the same\ncuriosity territory.\n\nRELIC LOOP TOPIC LANE:\nThe target is the viewer reaction: \"WAIT, I DO THAT TOO — WHY?\" Strong examples include questions\nlike \"Why can't you tickle yourself?\", \"Why does your voice sound different in a recording?\",\n\"Why do you sometimes feel your phone vibrate when it did not?\", and \"Why does time feel faster\nas you get older?\" These are examples of the FORMAT and curiosity level, not topics to copy.\nPrefer everyday sensations, brain predictions, illusions, habits, reflexes, memory glitches, social\nbehaviors, ordinary routines, and familiar objects with a surprising hidden reason. Prefer questions\nwith a strong visual experiment, demonstration, comparison, reveal, or before/after explanation.\nDo not choose generic object trivia just because it can be phrased as a \"why\" question.\n\nABSOLUTE ORIGINALITY RULE:\nA previous Relic Loop idea is permanently used. Never reuse it, even with different wording, a new\ntitle, a new angle, or a slightly different example. The underlying viewer experience, object,\nmechanism, phenomenon, or behavior must be genuinely different.\n"""
if old in text:
    text = text.replace(old, new, 1)

anchor = """        if all(data.get(k) for k in required) and isinstance(data.get(\"search_angles\"), list) and score >= 8:\n            return data\n"""
replacement = r'''        if all(data.get(k) for k in required) and isinstance(data.get("search_angles"), list) and score >= 8:
            # RELIC_LOOP_TOPIC_HARDENING_V2
            # Deterministic gate: history + permanent reservations + explicit known repeat families.
            import json
            import re
            from difflib import SequenceMatcher
            from pathlib import Path

            def _canon(value):
                s = str(value or "").lower()
                aliases = {
                    "mugs": "cup", "mug": "cup", "cups": "cup",
                    "warm": "hot", "heating": "hot", "heated": "hot",
                    "stuck": "snag", "snags": "snag", "snagged": "snag",
                    "fabrics": "fabric", "seams": "seam",
                    "zippers": "zipper", "zip": "zipper",
                    "liquids": "liquid",
                }
                words = re.findall(r"[a-z0-9]+", s)
                words = [aliases.get(w, w) for w in words]
                stop = {"why","what","how","does","do","did","can","could","would","you","your","the","a","an","is","are","was","were","to","of","in","on","for","we","our","it","this","that","after","before","even","really","often","sometimes","with","when"}
                return {w for w in words if len(w) > 2 and w not in stop}

            candidate = str(data.get("question") or data.get("topic") or data.get("title") or "").strip()
            cand = _canon(candidate)

            # Explicit permanent blocks requested after repeated production failures.
            forbidden_families = (
                {"zipper", "fabric"},
                {"zipper", "snag"},
                {"zipper", "stuck"},
                {"coffee", "cup"},
                {"coffee", "hot"},
                {"coffee", "liquid"},
            )
            blocked = False
            reason = ""
            for family in forbidden_families:
                if family.issubset(cand):
                    blocked = True
                    reason = "permanently blocked repeat family: " + ", ".join(sorted(family))
                    break

            # Load successful episodes AND previously reserved/attempted topics.
            history_path = Path("state/content_history.json")
            reserve_path = Path("state/topic_history.json")
            try:
                history_data = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {"videos": []}
            except Exception:
                history_data = {"videos": []}
            try:
                reserve_data = json.loads(reserve_path.read_text(encoding="utf-8")) if reserve_path.exists() else {"topics": []}
            except Exception:
                reserve_data = {"topics": []}

            previous = []
            for row in history_data.get("videos", []) if isinstance(history_data, dict) else []:
                if isinstance(row, dict):
                    previous.append(" | ".join(str(row.get(k, "")) for k in ("question", "topic", "title")))
            for row in reserve_data.get("topics", []) if isinstance(reserve_data, dict) else []:
                if isinstance(row, dict):
                    previous.append(str(row.get("question") or row.get("topic") or row.get("title") or ""))

            if not blocked:
                for prev in previous:
                    prev = str(prev).strip()
                    if not prev:
                        continue
                    p = _canon(prev)
                    shared = cand & p
                    union = cand | p
                    ratio = SequenceMatcher(None, candidate.lower(), prev.lower()).ratio()
                    overlap = len(shared) / max(1, len(union))
                    # Err on the side of rejection. Two+ meaningful shared tokens or a strong
                    # wording match means the same underlying episode is too likely.
                    if candidate.lower() == prev.lower() or ratio >= 0.70 or overlap >= 0.50 or len(shared) >= 2:
                        blocked = True
                        reason = f"previous/ reserved episode overlap: {prev}"
                        break

            if blocked:
                print(f"[TOPIC HARD GATE] REJECTED: {reason}")
                continue

            # Reserve the accepted idea BEFORE research/script generation. A failed run therefore
            # cannot recycle the same idea tomorrow. Reservations are permanent by design.
            reserve_path.parent.mkdir(parents=True, exist_ok=True)
            if not isinstance(reserve_data, dict):
                reserve_data = {"topics": []}
            topics = reserve_data.setdefault("topics", [])
            if not any(str(r.get("question", "")).strip().lower() == candidate.lower() for r in topics if isinstance(r, dict)):
                topics.append({"date": __import__("datetime").datetime.utcnow().isoformat() + "Z", "question": candidate, "topic": str(data.get("topic") or ""), "title": str(data.get("title") or ""), "reserved": True})
                reserve_path.write_text(json.dumps(reserve_data, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"[TOPIC HARD GATE] PASS + PERMANENT RESERVATION: {candidate}")
            return data
'''
if anchor not in text:
    raise SystemExit("[TOPIC HARDENING] selector anchor not found")
text = text.replace(anchor, replacement, 1)

path.write_text(text, encoding="utf-8")
print("[TOPIC HARDENING] V2 installed: deterministic duplicate rejection + permanent reservations")
