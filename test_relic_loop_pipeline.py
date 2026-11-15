import unittest
from unittest.mock import patch

from relic_loop_pipeline import run_pipeline


class _Message:
    content = '{"title":"The Queue That Would Not Move","narration":"' + (
        " ".join(["Milo tried to return a library book before closing."] * 30)
    ) + '","protagonist":"Milo","supporting_characters":[],"setting":"A library"}'


class _Choice:
    message = _Message()


class _Response:
    choices = [_Choice()]


class _Completions:
    def create(self, **kwargs):
        return _Response()


class _Chat:
    completions = _Completions()


class _Client:
    chat = _Chat()


class RelicLoopPipelineTests(unittest.TestCase):
    def test_creative_and_quality_are_connected_and_feedback_is_returned(self):
        with patch("relic_loop_pipeline.generate_story") as generate:
            generate.return_value = {
                "title": "The Queue That Would Not Move",
                "narration": " ".join(["Milo tried to return a library book before closing."] * 30),
                "protagonist": "Milo", "supporting_characters": [], "setting": "A library",
            }
            result = run_pipeline(
                {"topic": "a stubborn library queue"},
                creative_client=_Client(),
            )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stages"][0]["name"], "creative_ai")
        self.assertEqual(result["stages"][1]["name"], "quality_ai")
        self.assertTrue(result["quality_review"]["governance"]["advisory_only"])
        self.assertIn("improvements", result["next_creative_feedback"])
        self.assertFalse(result["published"])
        self.assertFalse(result["production_touched"])

    def test_quality_feedback_is_passed_into_creative_ai(self):
        previous = {"strengths": ["Strong visual opening"],
                    "improvements": ["Make the ending more specific"]}
        with patch("relic_loop_pipeline.generate_story") as generate:
            generate.return_value = {
                "title": "A Fresh Story",
                "narration": " ".join(["A character discovers an unusual problem."] * 30),
                "protagonist": "Milo", "supporting_characters": [], "setting": "A shop",
            }
            result = run_pipeline({"topic": "new premise",
                                  "previous_quality_feedback": previous},
                                 creative_client=_Client())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(generate.call_args.kwargs["feedback"].strengths,
                         ["Strong visual opening"])
        self.assertEqual(generate.call_args.kwargs["feedback"].improvements,
                         ["Make the ending more specific"])

    def test_technical_failure_is_reported_without_deployment(self):
        with patch("relic_loop_pipeline.generate_story", side_effect=RuntimeError("provider timeout")):
            result = run_pipeline({"topic": "test"}, creative_client=_Client())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["deployment"], "not_performed")
        self.assertFalse(result["production_touched"])
        self.assertFalse(result["published"])
        self.assertFalse(result["maintenance_ai"]["attempted"])

    def test_quality_scores_never_gate_completion(self):
        with patch("relic_loop_pipeline.generate_story") as generate:
            generate.return_value = {
                "title": "A Story",
                "narration": " ".join(["A character discovers an unusual problem."] * 30),
                "protagonist": "Milo", "supporting_characters": [], "setting": "A shop",
            }
            result = run_pipeline({"topic": "test"}, creative_client=_Client())
        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["quality_review"]["governance"]["production_decision"])


if __name__ == "__main__":
    unittest.main()
