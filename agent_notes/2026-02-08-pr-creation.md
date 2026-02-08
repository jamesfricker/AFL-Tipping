# PR creation notes (2026-02-08)

## Context
- User requested opening a PR from the current working state.
- Current diff includes the sequential margin model rework plus four prior investigation notes.

## Decisions
- Keep all current tracked and untracked changes together in one commit to preserve the exact state the user approved.
- Run full tests before PR creation to confirm no regressions in this snapshot.
- Use a concise PR title focused on lineup + market pregame enhancements.

## Validation
- `uv run pytest -q` passed: 31 passed.
