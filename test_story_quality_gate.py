import json
import unittest
from types import SimpleNamespace
from story_quality_gate import coach_feedback, _parse, DIMENSIONS, run_story_gate

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
        self.assertEqual(len(DIMENSIONS),11)
        self.assertIn("mid_story_rehook", DIMENSIONS)
        self.assertIn("voiceover_naturalness", DIMENSIONS)

    def test_two_candidates_select_the_higher_scoring_story(self):
        class FakeCompletions:
            def __init__(self):
                self.scores = [6.5, 9.0]
            def create(self, **kwargs):
                score = self.scores.pop(0)
                payload = {
                    "scores": {dimension: score for dimension in DIMENSIONS},
                    "overall_score": score, "issues": [], "rewrite_brief": "",
                    "verdict": "ready",
                }
                return SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload))
                )])
        client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        drafts = iter([{"title": "First"}, {"title": "Second"}])
        selected, history, _ = run_story_gate(
            client, "test-model", lambda feedback: next(drafts), {},
            max_rewrites=0, candidate_count=2,
        )
        self.assertEqual(selected["title"], "Second")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["overall_score"], 9.0)

if __name__=="__main__": unittest.main()
