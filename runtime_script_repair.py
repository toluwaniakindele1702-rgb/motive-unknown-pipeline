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

        # Keep this floor aligned with the downstream production validator.
        # A repaired script must never pass this stage and then fail later
        # because one narration is shorter than the validator permits.
        min_scene = 25
        max_scene = max(90, min(120, math.floor(max_words / max(1, scene_count)) + 25))
        target_scene = max(min_scene, min(max_scene, round(target_words / scene_count)))
        repaired = json.loads(json.dumps(script, ensure_ascii=False))
        p.atomic_write_json(p.SCRIPT_PATH, repaired)
        p._save_current_json(p.CURRENT_SCRIPT_PATH, repaired)
        current_words = p._script_word_count(repaired)
        print(f"[SCRIPT] runtime-aware repair: {current_words} -> ~{target_words} words ({min_words}-{max_words} total; {min_scene}-{max_scene}/scene)")

        # Small batches plus compact context prevent repair calls from repeatedly
        # consuming the entire Groq TPM window with the same research dossier.
        batch_size = 2
        topic_question = str(topic.get("question") or topic.get("topic") or "").strip()
        research_excerpt = str(research or "").strip()[:5000]
        plan_excerpt = json.dumps(plan, ensure_ascii=False)[:3500]

        for start in range(0, scene_count, batch_size):
            end = min(scene_count, start + batch_size)
            batch = repaired["scenes"][start:end]
            updated = False
            prompt = f"""
Repair these Relic Loop narration scenes for a short, precise 5-7 minute explainer.
The FULL script must be {min_words}-{max_words} words, preferably about {target_words}.
Each scene MUST contain at least {min_scene} words and MUST NOT exceed {max_scene} words.
Aim for about {target_scene} words per scene, but do not add filler or repetition.
If narration is short, add only useful explanation, mechanism, consequence, evidence, example, or transition.
If narration is long, remove repetition and filler.
Do not change scene metadata.
Do not invent facts, dialogue, motives, events, statistics, or sources.
Keep the energetic conversational voice.
Return JSON only: {{"narrations": ["one repaired narration per scene, in order"]}}
TOPIC QUESTION: {topic_question}
USEFUL RESEARCH EXCERPT: {research_excerpt}
STORY PLAN EXCERPT: {plan_excerpt}
SCENES: {json.dumps([{"id":s["id"],"narration":s["narration"]} for s in batch], ensure_ascii=False)}
""".strip()
            for attempt in range(1, 4):
                try:
                    result = p.groq_json(
                        p.GROQ_LIGHT_MODEL,
                        [{"role":"user","content":prompt}],
                        max_completion_tokens=900,
                        temperature=0.35,
                        attempts=2,
                    )
                    narrations = result.get("narrations")
                    if not isinstance(narrations, list) or len(narrations) != len(batch):
                        raise RuntimeError(f"expected {len(batch)} narrations")
                    for scene, narration in zip(batch, narrations):
                        text = p.normalize_spaces(str(narration))
                        words = p.count_words(text)
                        if words < min_scene:
                            raise RuntimeError(f"scene {scene['id']} below required floor of {min_scene} words")
                        if words > max_scene:
                            text = p._fit_narration_to_limit(text, max_scene)
                        scene["narration"] = text
                    updated = True
                    break
                except Exception as exc:
                    print(f"[SCRIPT] runtime repair batch {start+1}-{end}, attempt {attempt}/3 failed: {exc}")
                    if attempt < 3:
                        time.sleep(min(8.0, 2.0 * attempt))
            if not updated:
                # Never allow an invalid scene to escape this stage.
                for scene in batch:
                    text = p.normalize_spaces(str(scene.get("narration", "")))
                    words = p.count_words(text)
                    if words < min_scene:
                        raise RuntimeError(f"Could not preserve usable narration for scene {scene['id']}: {words} words")
                    if words > max_scene:
                        text = p._fit_narration_to_limit(text, max_scene)
                    scene["narration"] = text
                print(f"[SCRIPT] salvaged runtime repair batch {start+1}-{end}")
            p.atomic_write_json(p.SCRIPT_PATH, repaired)
            p._save_current_json(p.CURRENT_SCRIPT_PATH, repaired)
            print(f"[SCRIPT] saved runtime repair through scene {end} (~{p._script_word_count(repaired)} words)")

        # If the model produced a script below the total target, top up in tiny batches.
        final_words = p._script_word_count(repaired)
        while final_words < min_words:
            topup_per_scene = max(1, math.ceil((min_words - final_words) / scene_count) + 2)
            progress = False
            for start in range(0, scene_count, batch_size):
                if final_words >= min_words:
                    break
                end = min(scene_count, start + batch_size)
                batch = repaired["scenes"][start:end]
                prompt = f"""
Lightly expand these Relic Loop narrations by about {topup_per_scene} useful words each.
Preserve facts and wording. Add only mechanism, consequence, evidence, example, or transition.
No filler, repetition, scenery, fake dialogue, or unsupported claims.
Every scene must remain between {min_scene} and {max_scene} words.
Return JSON only: {{"narrations": ["one updated narration per scene, in order"]}}
SCENES: {json.dumps([{"id":s["id"],"narration":s["narration"]} for s in batch], ensure_ascii=False)}
""".strip()
                try:
                    result = p.groq_json(p.GROQ_LIGHT_MODEL, [{"role":"user","content":prompt}], max_completion_tokens=500, temperature=0.25, attempts=2)
                    narrations = result.get("narrations")
                    if isinstance(narrations, list) and len(narrations) == len(batch):
                        for scene, narration in zip(batch, narrations):
                            text = p.normalize_spaces(str(narration))
                            if p.count_words(text) >= min_scene:
                                new_text = p._fit_narration_to_limit(text, max_scene)
                                if p.count_words(new_text) > p.count_words(scene.get("narration", "")):
                                    scene["narration"] = new_text
                                    progress = True
                except Exception as exc:
                    print(f"[SCRIPT] small top-up batch {start+1}-{end} skipped: {exc}")
                final_words = p._script_word_count(repaired)
                p.atomic_write_json(p.SCRIPT_PATH, repaired)
                p._save_current_json(p.CURRENT_SCRIPT_PATH, repaired)
            if not progress:
                break
            final_words = p._script_word_count(repaired)

        final_words = p._script_word_count(repaired)
        while final_words > max_words:
            candidates = [s for s in repaired["scenes"] if p.count_words(str(s.get("narration", ""))) > min_scene]
            if not candidates:
                break
            scene = max(candidates, key=lambda s:p.count_words(str(s.get("narration", ""))))
            old = p.count_words(str(scene["narration"]))
            scene["narration"] = p._fit_narration_to_limit(str(scene["narration"]), max_scene)
            new = p.count_words(str(scene["narration"]))
            if new >= old:
                break
            final_words = p._script_word_count(repaired)

        final_words = p._script_word_count(repaired)
        if not min_words <= final_words <= max_words:
            raise RuntimeError(f"Runtime script repair finished at {final_words}; expected {min_words}-{max_words}")

        invalid = []
        for scene in repaired["scenes"]:
            words = p.count_words(str(scene.get("narration", "")))
            if words < min_scene or words > max_scene:
                invalid.append(f"scene {scene.get('id')}: {words} words")
        if invalid:
            raise RuntimeError("Runtime repair produced invalid scene lengths: " + ", ".join(invalid))

        p.atomic_write_json(p.SCRIPT_PATH, repaired)
        p._save_current_json(p.CURRENT_SCRIPT_PATH, repaired)
        p.validate_script(repaired, allow_short=True)
        print(f"[SCRIPT] runtime repair successful: ~{final_words} words")
        return repaired

    p._repair_script_length = repair
    return repair
