# Relic Loop Maintenance AI

Standalone failure diagnosis and repair-proposal engine for the future separate AI pipeline.

- Uses Groq for optional root-cause analysis and minimal patch proposals.
- Redacts common credential patterns from diagnostic logs before model submission.
- Strict allowlist limits patches to the three AI components and their tests.
- Applies candidate patches only in a disposable Git sandbox, compiles changed Python files and runs changed tests.
- Never writes to the working checkout, pushes branches, opens PRs, changes GitHub settings, deploys, triggers production or publishes videos.
- No connection to the existing production pipeline.

Run offline validation:
python -m py_compile maintenance_ai.py test_maintenance_ai.py
python -m unittest -v test_maintenance_ai.py

Live diagnosis requires GROQ_API_KEY and a JSON input file with failure_context and source_files. A validated patch is a proposal for the future separate pipeline to review and apply under its own controls.
