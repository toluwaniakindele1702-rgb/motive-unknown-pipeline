import unittest

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
        self.assertEqual(story["supporting_characters"], ["A neighbour"])

    def test_rejects_invalid_length(self):
        with self.assertRaises(ValueError):
            validate_story({"title": "Too short", "narration": "Only a few words."})

    def test_json_parser(self):
        self.assertEqual(parse_json_object('prefix {"score": 7} suffix'), {"score": 7})

    def test_scene_continuity_directive(self):
        directive = build_scene_continuity_directive("Milo, red scarf", "library", "book stays in left hand")
        self.assertIn("CHARACTER LOCK", directive)
        self.assertIn("book stays in left hand", directive)


if __name__ == "__main__":
    unittest.main()
