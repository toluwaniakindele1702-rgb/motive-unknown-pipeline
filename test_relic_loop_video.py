import unittest
from pathlib import Path
from unittest.mock import patch
from relic_loop_video import split_story_into_scenes, _duration, _safe_moderation_retry_prompt

class RelicLoopVideoTests(unittest.TestCase):
    def test_scene_split_preserves_all_words_and_order(self):
        narration = " ".join(f"Sentence number {i} ends here." for i in range(1, 21))
        beats = split_story_into_scenes(narration, 10)
        self.assertEqual(len(beats), 10)
        self.assertEqual(" ".join(beats), narration)
    def test_short_sentence_count_still_produces_ten_visual_beats(self):
        narration = " ".join(f"Sentence {i} has enough extra words to become a longer narration beat." for i in range(1, 8))
        beats = split_story_into_scenes(narration, 10)
        self.assertEqual(len(beats), 10)
        self.assertEqual(" ".join(beats), narration)
    def test_moderation_retry_keeps_character_beat_and_setting(self):
        prompt = ("CANONICAL CHARACTER LOCK: Milo in a red hoodie\\n"
                  "EXACT NARRATION MOMENT: Milo drops his lunchbox\\n"
                  "SETTING: a school hallway")
        safe = _safe_moderation_retry_prompt(prompt)
        self.assertIn("Milo in a red hoodie", safe)
        self.assertIn("Milo drops his lunchbox", safe)
        self.assertIn("a school hallway", safe)
        self.assertIn("family-friendly", safe)

    def test_duration_rejects_zero(self):
        fake = type("P", (), {"stdout":"0"})()
        with patch("relic_loop_video._run", return_value=fake):
            with self.assertRaises(RuntimeError):
                _duration(Path("unused"))
if __name__ == "__main__":
    unittest.main()
