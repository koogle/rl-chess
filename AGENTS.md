# Agent instructions for rl-chess

## Research log is mandatory

All code changes, validation runs, training experiments, and meaningful design decisions must be recorded in the README research log.

For every entry:

- Add the current date and time, including timezone.
- State what changed or what experiment ran.
- Include the exact command when applicable.
- Record the observed result, including key metrics and whether it passed or failed.
- If the run produced checkpoints or artifacts, record their paths.
- Keep entries concise and factual; do not replace the log with only commit messages.

## Project direction

- Keep the RL loop learning-first and inspectable.
- Use `python-chess` for board legality, state transitions, terminal detection, and result accounting.
- Prefer human-readable Unicode board diagrams at the RL/model boundary where practical.
- Keep Modal as a thin remote runner for the same core local training logic; do not hide core RL/search behavior in Modal-only code.
- Validate small, narrow learning properties before scaling broad training runs.

## Verification expectations

- Run `uv run pytest -q` before committing code changes.
- For training/validation changes, also run the smallest relevant smoke command before larger remote runs.
- Record all verification in the README research log.
