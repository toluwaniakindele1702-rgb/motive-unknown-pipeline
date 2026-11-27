import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from creative_ai import (
    CreativeBrief, QualityFeedback, build_scene_continuity_directive,
    build_user_prompt, parse_json_object, validate_story, word_count,
)


class CreativeAITests(unittest.TestCase):
    def test_feedback_is_advisory_and_structured(self):
        feedback = QualityFeedback.from_mapping({
            "overall_score": 4,
            "strengths": ["Good hook"],
            "improvements": ["Improve payoff"],
        })
        prompt = feedback.as_prompt_context()
        self.assertIn("advisory only", prompt)
        self.assertIn("Improve payoff", prompt)

    def test_low_score_does_not_block_prompt(self):
        feedback = QualityFeedback.from_mapping({"overall_score": 1})
        prompt = build_user_prompt(CreativeBrief(topic="A broken elevator"), feedback)
        self.assertIn("A broken elevator", prompt)
        self.assertIn("1/10", prompt)

    def test_story_schema_and_word_count(self):
        narration = " ".join(f"word{i}" for i in range(190))
        story = validate_story({
            "title": "A test story",
            "narration": narration,
            "protagonist": "Milo",
            "supporting_characters": ["A neighbour", "", "Extra"],
            "setting": "A shop",
        })
        self.assertEqual(word_count(story["narration"]), 190)
        self.assertEqual(story["supporting_characters"], ["A neighbour", "Extra"])

    def test_accepts_long_story_without_any_maximum(self):
        narration = " ".join(f"word{i}" for i in range(249))
        story = validate_story({"title": "Long complete story", "narration": narration})
        self.assertEqual(word_count(story["narration"]), 249)

    def test_prompt_specifies_minimum_and_no_maximum(self):
        prompt = build_user_prompt(CreativeBrief())
        self.assertIn("at least 180 words", prompt)
        self.assertIn("NO maximum word count", prompt)

    def test_rejects_invalid_length(self):
        with self.assertRaises(ValueError):
            validate_story({"title": "Too short", "narration": "Only a few words."})

    def test_json_parser(self):
        self.assertEqual(parse_json_object('prefix {"score": 7} suffix'), {"score": 7})


    def test_generate_story_retries_when_groq_truncates_completion(self):
        story_json = {
            "title": "A complete story",
            "narration": " ".join(["word"] * 190),
            "protagonist": "Milo",
            "supporting_characters": [],
            "setting": "A shop",
        }
        truncated = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="length", message=SimpleNamespace(content='{"title":')
        )])
        complete = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(
                content=__import__("json").dumps(story_json)
            )
        )])
        client = Mock()
        client.chat.completions.create.side_effect = [truncated, complete]
        from creative_ai import generate_story
        result = generate_story(client, "test-model", CreativeBrief())
        self.assertEqual(result["title"], "A complete story")
        calls = client.chat.completions.create.call_args_list
        self.assertEqual(calls[0].kwargs["max_completion_tokens"], 8192)
        self.assertEqual(calls[1].kwargs["max_completion_tokens"], 16384)

    def test_generate_story_does_not_retry_unrelated_groq_errors(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError("authentication failed")
        from creative_ai import generate_story
        with self.assertRaisesRegex(RuntimeError, "authentication failed"):
            generate_story(client, "test-model", CreativeBrief())
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_scene_continuity_directive(self):
        directive = build_scene_continuity_directive("Milo, red scarf", "library", "book stays in left hand")
        self.assertIn("CHARACTER LOCK", directive)
        self.assertIn("book stays in left hand", directive)


if __name__ == "__main__":
    unittest.main()
