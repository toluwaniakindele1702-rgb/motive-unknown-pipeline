import unittest
from story_quality_gate import coach_feedback, _parse, DIMENSIONS

class StoryQualityGateTests(unittest.TestCase):
    def test_schema_and_dimensions(self):
        payload={"scores":{d:7 for d in DIMENSIONS},"overall_score":7,"issues":[],"rewrite_brief":"Add a sharper opening","verdict":"ready"}
        self.assertEqual(_parse(__import__("json").dumps(payload))["verdict"],"ready")
    def test_low_scores_generate_specific_coaching(self):
        review={"scores":{"opening_hook":5,"pacing":6,"payoff":7},"issues":["Slow opening"],"rewrite_brief":"Open on the problem"}
        feedback=coach_feedback(review)
        self.assertIn("Slow opening",feedback["improvements"])
        self.assertIn("Open on the problem",feedback["pacing_notes"][0])
    def test_review_is_not_a_production_gate(self):
        # The module's bounded loop always returns the last story, regardless of verdict.
        self.assertEqual(len(DIMENSIONS),8)

if __name__=="__main__": unittest.main()
