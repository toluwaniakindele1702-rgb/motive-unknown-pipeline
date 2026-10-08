"""Runtime-aware script repair for the 5-7 minute Relic Loop policy."""
from __future__ import annotations
import json
import math
import time


def install(p):
    def repair(topic, research, plan, script):
        min_words = int(p.SCRIPT_MIN_WORDS)
        max_words = int(p.SCRIPT_MAX_WORDS)
        target_words = (min_words + max_words) // 2
        scenes = script.get("scenes") or []
        scene_count = len(scenes)
        if not scene_count:
            raise RuntimeError("Cannot repair a script with no scenes.")

        min_scene = max(25, math.floor(min_words / scene_count))
        max_scene = min(120, math.floor(max_words / scene_count))
        target_scene = max(min_scene, min(max_scene, round(target_words / scene_count)))
        repaired = json.loads(json.dumps(script, ensure_ascii=False))
        p.atomic_write_json(p.SCRIPT_PATH, repaired)
        p._save_current_json(p.CURRENT_SCRIPT_PATH, repaired)
        current_words = p._script_word_count(repaired)
        print(f"[SCRIPT] runtime-aware repair: {current_words} -> ~{target_words} words ({min_words}-{max_words} total; {min_scene}-{max_scene}/scene)")

        batch_size = 6
        for start in range(0, scene_count, batch_size):
            end = min(scene_count, start + batch_size)
            batch = repaired["scenes"][start:end]
            prompt = f"""
Repair these Relic Loop narration scenes for a SHORT, precise 5-7 minute explainer.
The FULL script must be {min_words}-{max_words} words, preferably about {target_words}.
For these {len(batch)} scenes, aim for about {target_scene} words each, between {min_scene} and {max_scene}.
If narration is short, add only useful explanation, mechanism, consequence, evidence, example, or transition.
If narration is long, remove repetition and filler. Do not pad the story. Do not change scene metadata.
Do not invent facts, dialogue, motives, events, statistics, or sources. Keep the energetic conversational voice.
Return JSON only: {{"narrations": ["one repaired narration per scene, in order"]}}
TOPIC: {json.dumps(topic, ensure_ascii=False)}
STORY PLAN: {json.dumps(plan, ensure_ascii=False)}
RESEARCH: {research}
SCENES: {json.dumps([{"id":s["id"],"narration":s["narration"],"setting":s["setting"],"characters":s["characters"],"action":s["action"],"props":s["props"],"mood":s["mood"]} for s in batch], ensure_ascii=False)}
""".strip()
            updated = False
            for attempt in range(1, 4):
                try:
                    result = p.groq_json(p.GROQ_WRITER_MODEL, [{"role":"user","content":prompt}], max_completion_tokens=2200, temperature=0.42, attempts=3)
                    narrations = result.get("narrations")
                    if not isinstance(narrations, list) or len(narrations) != len(batch):
                        raise RuntimeError(f"expected {len(batch)} narrations")
                    for scene, narration in zip(batch, narrations):
                        text = p.normalize_spaces(str(narration))
                        if p.count_words(text) < min_scene:
                            raise RuntimeError(f"scene {scene['id']} below {min_scene} words")
                        if p.count_words(text) > max_scene:
                            text = p._fit_narration_to_limit(text, max_scene)
                        scene["narration"] = text
                    updated = True
                    break
                except Exception as exc:
                    print(f"[SCRIPT] runtime repair batch {start+1}-{end}, attempt {attempt}/3 failed: {exc}")
                    if attempt < 3:
                        time.sleep(min(6.0, 1.5 * attempt))
            if not updated:
                for scene in batch:
                    text = str(scene.get("narration", "")).strip()
                    if p.count_words(text) > max_scene:
                        text = p._fit_narration_to_limit(text, max_scene)
                        scene["narration"] = text
                    if not min_scene <= p.count_words(text) <= max_scene:
                        raise RuntimeError(f"Could not repair scene {scene['id']} into runtime range")
                print(f"[SCRIPT] salvaged runtime repair batch {start+1}-{end}")
            p.atomic_write_json(p.SCRIPT_PATH, repaired)
            p._save_current_json(p.CURRENT_SCRIPT_PATH, repaired)
            print(f"[SCRIPT] saved runtime repair through scene {end} (~{p._script_word_count(repaired)} words)")

        final_words = p._script_word_count(repaired)
        if final_words < min_words:
            prompt = f"""
Lightly expand these narrations so the FULL Relic Loop script reaches {min_words}-{max_words} words.
Add only useful factual explanation, mechanism, consequence, evidence, examples, or transitions.
No filler, repetition, scenery, fake dialogue, or unsupported claims. Keep every scene at or below {max_scene} words.
Return JSON only: {{"narrations": ["one updated narration per scene, in order"]}}
RESEARCH: {research}
SCENES: {json.dumps([{"id":s["id"],"narration":s["narration"]} for s in repaired["scenes"]], ensure_ascii=False)}
""".strip()
            result = p.groq_json(p.GROQ_WRITER_MODEL, [{"role":"user","content":prompt}], max_completion_tokens=3200, temperature=0.30, attempts=3)
            narrations = result.get("narrations")
            if not isinstance(narrations, list) or len(narrations) != scene_count:
                raise RuntimeError("Runtime top-up returned the wrong number of narrations")
            for scene, narration in zip(repaired["scenes"], narrations):
                text = p.normalize_spaces(str(narration))
                if p.count_words(text) > max_scene:
                    text = p._fit_narration_to_limit(text, max_scene)
                if p.count_words(text) < 25:
                    raise RuntimeError(f"Runtime top-up made scene {scene['id']} too short")
                scene["narration"] = text
            final_words = p._script_word_count(repaired)

        while final_words > max_words:
            candidates = [s for s in repaired["scenes"] if p.count_words(str(s.get("narration", ""))) > 25]
            if not candidates:
                break
            scene = max(candidates, key=lambda s:p.count_words(str(s.get("narration", ""))))
            old = p.count_words(str(scene["narration"]))
            new_limit = max(25, old - max(8, min(20, final_words-max_words)))
            scene["narration"] = p._fit_narration_to_limit(str(scene["narration"]), new_limit)
            new = p.count_words(str(scene["narration"]))
            if new >= old:
                break
            final_words = p._script_word_count(repaired)

        if not min_words <= final_words <= max_words:
            raise RuntimeError(f"Runtime script repair finished at {final_words}; expected {min_words}-{max_words}")
        p.atomic_write_json(p.SCRIPT_PATH, repaired)
        p._save_current_json(p.CURRENT_SCRIPT_PATH, repaired)
        p.validate_script(repaired, allow_short=True)
        print(f"[SCRIPT] runtime repair successful: ~{final_words} words")
        return repaired

    p._repair_script_length = repair
    return repair
