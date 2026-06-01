# Agent instructions for rl-chess

## Research log is mandatory, but curated

The README research log is for structural architecture changes, methodology progress, meaningful training/evaluation experiments, and design decisions. Do not add routine test-only verification, mechanical cleanup, or transient debugging details unless they changed the methodology or exposed an important constraint.

For every entry:

- Add the current date and time, including timezone.
- State what changed or what experiment ran.
- Include the exact command for training/evaluation experiments and other methodology-relevant runs.
- Record observed metrics, pass/fail outcomes, and interpretation for experiments.
- If the run produced checkpoints or artifacts, record their paths.
- Keep entries concise and factual; omit routine RED/GREEN/full-suite test logs unless the test itself defines a new structural invariant.

## Project direction

- Keep the RL loop learning-first and inspectable.
- Use `python-chess` for board legality, state transitions, terminal detection, and result accounting.
- Prefer human-readable Unicode board diagrams at the RL/model boundary where practical.
- Keep Modal as a thin remote runner for the same core local training logic; do not hide core RL/search behavior in Modal-only code.
- Validate small, narrow learning properties before scaling broad training runs.

## Verification expectations

- Run `uv run pytest -q` before committing code changes.
- For training/validation changes, also run the smallest relevant smoke command before larger remote runs.
- Launch real/non-smoke Modal training through the spawn-based local entrypoint: `uv run modal run src/rl_chess/modal_app.py::main <training flags>`. The entrypoint calls `train_remote.spawn(...)`, prints a JSON payload with `function_call_id` and `dashboard_url`, then exits so the remote function continues independently. Add `--wait` only for tiny smoke runs where immediate feedback is more useful than durable handoff.
- Record all verification in the README research log. Always return final evaluation results to the user when Modal runs finish.
