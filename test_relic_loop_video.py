import unittest
from pathlib import Path
from unittest.mock import patch
from relic_loop_video import split_story_into_scenes, _duration, _safe_moderation_retry_prompt, _generate_valid_story
from creative_ai import QualityFeedback
from relic_loop_media import write_word_level_captions

class RelicLoopVideoTests(unittest.TestCase):
    def test_scene_split_preserves_all_words_and_order(self):
        narration = " ".join(f"Sentence number {i} ends here." for i in range(1, 21))
        beats = split_story_into_scenes(narration, 10)
        self.assertEqual(len(beats), 10)
        self.assertEqual(" ".join(beats), narration)
    def test_short_sentence_count_keeps_complete_sentence_boundaries(self):
        narration = " ".join(f"Sentence {i} has enough extra words to become a longer narration beat." for i in range(1, 8))
        beats = split_story_into_scenes(narration, 10)
        self.assertEqual(len(beats), 7)
        self.assertEqual(" ".join(beats), narration)
        self.assertTrue(all(beat.endswith(".") for beat in beats))
    def test_moderation_retry_keeps_character_beat_and_setting(self):
        prompt = ("CANONICAL CHARACTER LOCK: Milo in a red hoodie\n"
                  "EXACT NARRATION MOMENT: Milo drops his lunchbox\n"
                  "SETTING: a school hallway")
        safe = _safe_moderation_retry_prompt(prompt)
        self.assertIn("Milo in a red hoodie", safe)
        self.assertIn("Milo drops his lunchbox", safe)
        self.assertIn("a school hallway", safe)
        self.assertIn("family-friendly", safe)
        generic = _safe_moderation_retry_prompt(prompt, generic=True)
        self.assertIn("everyday situation", generic)
        self.assertNotIn("Milo drops his lunchbox", generic)

    def test_long_story_is_accepted_without_maximum(self):
        valid = {"title": "A Story", "narration": " ".join(["word"] * 249),
                 "protagonist": "Milo", "supporting_characters": [], "setting": "A shop"}
        with patch("relic_loop_video.generate_story", return_value=valid) as generate:
            result = _generate_valid_story(object(), "test-model", "topic", QualityFeedback())
        self.assertEqual(result["title"], valid["title"])
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(generate.call_args_list[0].args[2].target_words, (180, 0))

    def test_underlength_story_is_retried(self):
        valid = {"title": "A Story", "narration": " ".join(["word"] * 190),
                 "protagonist": "Milo", "supporting_characters": [], "setting": "A shop"}
        with patch("relic_loop_video.generate_story",
                   side_effect=[ValueError("Story has 170 words; minimum is 180"), valid]) as generate:
            result = _generate_valid_story(object(), "test-model", "topic", QualityFeedback())
        self.assertEqual(result["title"], valid["title"])
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(generate.call_args_list[0].args[2].target_words, (180, 0))


    def test_word_level_captions_are_sidecar_and_not_burned_in(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "captions.srt"
            result = write_word_level_captions([
                {"start": 0.0, "end": 0.2, "text": "Hello"},
                {"start": 0.2, "end": 0.45, "text": "there!"}
            ], target, audio_duration=1.0)
            content = target.read_text(encoding="utf-8")
            self.assertEqual(result["words_per_caption"], 1)
            self.assertFalse(result["burned_into_video"])
            self.assertIn("00:00:00,000 --> 00:00:00,200", content)
            self.assertIn("Hello", content)
            self.assertIn("there!", content)

    def test_duration_rejects_zero(self):
        fake = type("P", (), {"stdout":"0"})()
        with patch("relic_loop_video._run", return_value=fake):
            with self.assertRaises(RuntimeError):
                _duration(Path("unused"))
if __name__ == "__main__":
    unittest.main()
