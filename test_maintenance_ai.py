import tempfile, unittest, subprocess
from pathlib import Path
from maintenance_ai import redact, validate_patch_paths, validate_patch_in_sandbox, ALLOWED

class MaintenanceAITests(unittest.TestCase):
    def test_redacts_tokens(self):
        value=redact("Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345 token=supersecret")
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz",value)
        self.assertNotIn("supersecret",value)
    def test_rejects_workflow_and_path_traversal(self):
        bad="diff --git a/.github/workflows/deploy.yml b/.github/workflows/deploy.yml\n--- a/.github/workflows/deploy.yml\n+++ b/.github/workflows/deploy.yml\n@@ -1 +1 @@\n-a\n+b\n"
        with self.assertRaises(ValueError): validate_patch_paths(bad)
        traversal="diff --git a/../evil.py b/../evil.py\n--- a/../evil.py\n+++ b/../evil.py\n@@ -1 +1 @@\n-a\n+b\n"
        with self.assertRaises(ValueError): validate_patch_paths(traversal)
    def test_allows_only_existing_ai_component_paths(self):
        self.assertIn("maintenance_ai.py",ALLOWED)
    def test_sandbox_applies_patch_and_compiles_without_touching_checkout(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/"maintenance_ai.py").write_text("VALUE = 1\n")
            patch="diff --git a/maintenance_ai.py b/maintenance_ai.py\n--- a/maintenance_ai.py\n+++ b/maintenance_ai.py\n@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"
            report=validate_patch_in_sandbox(root,patch)
            self.assertTrue(report["valid"])
            self.assertTrue(report["sandbox_only"])
            self.assertEqual((root/"maintenance_ai.py").read_text(),"VALUE = 1\n")
    def test_rejects_non_ai_patch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/"maintenance_ai.py").write_text("VALUE = 1\n")
            patch="diff --git a/chispa_pifia_v2.py b/chispa_pifia_v2.py\n--- a/chispa_pifia_v2.py\n+++ b/chispa_pifia_v2.py\n@@ -1 +1 @@\n-a\n+b\n"
            with self.assertRaises(ValueError): validate_patch_in_sandbox(root,patch)

if __name__ == "__main__": unittest.main()
