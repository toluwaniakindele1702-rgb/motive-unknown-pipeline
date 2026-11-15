import unittest
from quality_ai import (
    DIMENSIONS, ReviewConfig, build_evaluator_prompt, review_submission,
    _normalise_evaluation, build_client_from_env,
)


class QualityAITests(unittest.TestCase):
    def test_deterministic_review_works_without_api(self):
        result = review_submission({
            "title": "Test", "narration": " ".join("word" + str(i) for i in range(190)),
            "duration_seconds": 60,
            "image_inspections": [{"valid": True}, {"valid": True}],
        })
        self.assertEqual(result["schema_version"], "1.0")
        self.assertTrue(result["governance"]["advisory_only"])
        self.assertFalse(result["governance"]["may_block_production"])
        self.assertFalse(result["governance"]["may_cancel_or_pause_run"])
        self.assertFalse(result["governance"]["may_authorize_or_trigger_upload"])
        self.assertIn("story_coherence", result["evidence_coverage"]["assessed_dimensions"])
        self.assertIn("visual_clarity", result["evidence_coverage"]["assessed_dimensions"])

    def test_missing_visual_audio_evidence_is_not_guessed(self):
        result = review_submission({"narration": "A short supplied transcript."})
        self.assertEqual(result["dimensions"]["visual_clarity"]["status"], "not_assessed")
        self.assertEqual(result["dimensions"]["audio_clarity"]["status"], "not_assessed")
        self.assertIn("visual_clarity", result["evidence_coverage"]["not_assessed_dimensions"])

    def test_duration_outside_target_is_advice_not_failure(self):
        result = review_submission({"duration_seconds": 120})
        self.assertEqual(result["dimensions"]["technical_polish"]["score"], 5.0)
        self.assertTrue(result["governance"]["advisory_only"])
        self.assertIsNone(result["governance"]["production_decision"])

    def test_bad_image_inspection_does_not_stop_review(self):
        result = review_submission({
            "image_inspections": [{"valid": True}, {"valid": False}],
            "duration_seconds": 60,
        })
        self.assertEqual(result["dimensions"]["visual_clarity"]["score"], 5.0)
        self.assertIsInstance(result["improvements"], list)

    def test_evaluation_schema_rejects_invalid_scores(self):
        result = _normalise_evaluation({
            "overall_score": 99,
            "dimensions": {"hook": {"score": 12, "status": "assessed", "evidence": ["x"]}},
        })
        self.assertIsNone(result["overall_score"])
        self.assertIsNone(result["dimensions"]["hook"]["score"])
        self.assertEqual(result["dimensions"]["hook"]["status"], "not_assessed")

    def test_feedback_adapter_matches_creative_ai_contract(self):
        feedback = review_submission({"narration": "word " * 190})["creative_ai_feedback"]
        self.assertEqual(set(feedback), {
            "overall_score", "strengths", "improvements", "continuity_notes",
            "pacing_notes", "story_notes",
        })

    def test_prompt_contains_evidence_but_not_unrelated_secrets(self):
        prompt = build_evaluator_prompt(
            {"narration": "hello", "title": "Story", "api_key": "do-not-send"},
            ["duration observed"],
        )
        self.assertIn("Story", prompt)
        self.assertNotIn("do-not-send", prompt)


    def test_cli_defaults_to_offline_without_secret(self):
        client, model = build_client_from_env({})
        self.assertIsNone(client)
        self.assertIsNone(model)

    def test_config_requires_model_when_api_key_is_present(self):
        with self.assertRaisesRegex(ValueError, "QUALITY_AI_MODEL"):
            build_client_from_env({"GROQ_API_KEY": "test-only-secret"})

    def test_all_dimensions_are_explicit(self):
        result = review_submission({})
        self.assertEqual(set(result["dimensions"]), set(DIMENSIONS))
        self.assertEqual(result["confidence"], "low")


if __name__ == "__main__":
    unittest.main()
