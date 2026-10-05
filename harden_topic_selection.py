from pathlib import Path

path = Path("pipeline.py")
text = path.read_text(encoding="utf-8")

MARKER = "# RELIC_LOOP_TOPIC_HARDENING_V1"
if MARKER in text:
    print("[TOPIC HARDENING] already applied")
    raise SystemExit(0)

old = """Do NOT simply turn these examples into future videos. Generate fresh questions in the same\ncuriosity territory.\n"""
new = """Do NOT simply turn these examples into future videos. Generate fresh questions in the same\ncuriosity territory.\n\nTRENDING-CURIOSITY QUESTION HUNT:\nThe target is NOT merely a topic that is scientifically interesting. Search for questions that make\nordinary viewers stop and think, \"WAIT, I DO THAT TOO — WHY?\" or \"I HAVE ALWAYS WONDERED ABOUT THAT.\"\nStrong examples include questions like \"Why can't you tickle yourself?\", \"Why do you sometimes\nfeel your phone vibrate when it did not?\", \"Why does your voice sound different in a recording?\",\nor \"Why does time feel faster as you get older?\" These are examples of the FORMAT and viewer\nreaction, not a list to copy.\n\nSEARCH FOR THESE HIGH-CURIOSITY PATTERNS:\n- impossible-to-explain everyday sensations and perceptions\n- things everyone does but almost nobody knows the reason for\n- weird brain predictions, illusions, habits, reflexes, memory glitches and social behaviors\n- ordinary objects or routines with a surprising hidden purpose\n- familiar situations where the obvious explanation is wrong or incomplete\n- everyday questions that have recently attracted strong discussion, articles, searches, or creator interest\n- questions with a strong visual experiment, demonstration, comparison, reveal, or before/after explanation\n\nDo not confuse \"trending\" with celebrity/news gossip. A strong Relic Loop trend can be an evergreen\nquestion that is suddenly getting attention because people are rediscovering it. Search broadly for\ncurrent interest, but prefer evergreen curiosity when it has stronger mass appeal.\n\nREJECT candidates that are merely adjacent to a previous episode. If the new question uses the same\ncore phenomenon, mechanism, object, behavior, or viewer experience as a recent episode, reject it even\nif the wording is different. We want genuinely different video subjects, not rewritten versions of\nold videos.\n"""
if old not in text:
    raise SystemExit("[TOPIC HARDENING] topic prompt anchor not found")
text = text.replace(old, new, 1)

anchor = """        if all(data.get(k) for k in required) and isinstance(data.get(\"search_angles\"), list) and score >= 8:\n            return data\n"""
replacement = """        if all(data.get(k) for k in required) and isinstance(data.get(\"search_angles\"), list) and score >= 8:\n            # RELIC_LOOP_TOPIC_HARDENING_V1\n            # Block exact and near-duplicate topics before they can reach scripting.\n            import re\n            from difflib import SequenceMatcher\n            candidate = str(data.get(\"question\") or data.get(\"topic\") or \"\").lower()\n            stop = {\"why\", \"what\", \"how\", \"does\", \"do\", \"can\", \"you\", \"your\", \"the\", \"a\", \"an\", \"is\", \"are\", \"to\", \"of\", \"in\", \"on\", \"for\", \"we\", \"our\", \"it\", \"this\", \"that\"}\n            def _topic_tokens(value):\n                return {w for w in re.findall(r\"[a-z0-9]+\", value.lower()) if len(w) > 2 and w not in stop}\n            cand_tokens = _topic_tokens(candidate)\n            duplicate = False\n            duplicate_reason = \"\"\n            for previous_q in previous:\n                prev = str(previous_q)\n                if candidate == prev.lower().strip():\n                    duplicate = True\n                    duplicate_reason = f\"exact match: {prev}\"\n                    break\n                ratio = SequenceMatcher(None, candidate, prev.lower()).ratio()\n                prev_tokens = _topic_tokens(prev)\n                overlap = (len(cand_tokens & prev_tokens) / max(1, len(cand_tokens | prev_tokens)))\n                if ratio >= 0.78 or overlap >= 0.68:\n                    duplicate = True\n                    duplicate_reason = f\"near duplicate: {prev}\"\n                    break\n            if duplicate:\n                print(f\"[TOPIC] rejected duplicate candidate: {duplicate_reason}\")\n                continue\n            return data\n"""
if anchor not in text:
    raise SystemExit("[TOPIC HARDENING] selector anchor not found")
text = text.replace(anchor, replacement, 1)

# Also expand the selector's own instructions so the model is rewarded for the desired question shape.
selector_anchor = """The strongest lane is: \"WAIT... I experience that all the time. Why does that happen?\"\n"""
selector_add = selector_anchor + """\nQUESTION-SHAPE PRIORITY:\nPrefer a short, instantly understandable question about a familiar experience, sensation, behavior,\nor object. The best candidate should feel like a question a normal person could ask while sitting on\nthe couch, using a phone, eating, travelling, getting ready, sleeping, talking to friends, or doing\na routine task. Give extra weight to \"Why can't I...\", \"Why does my brain/body...\", \"Why do we...\",\nand \"Why is X designed this way?\" patterns when the explanation contains a genuine surprising reveal.\n"""
if selector_anchor in text:
    text = text.replace(selector_anchor, selector_add, 1)

path.write_text(text, encoding="utf-8")
print("[TOPIC HARDENING] applied duplicate guard + high-curiosity search rules")
