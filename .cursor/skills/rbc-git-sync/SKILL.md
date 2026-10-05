---
name: rbc-git-sync
description: >-
  Runs the deterministic git sync script for this RBC project. Use at the
  start of a session (pull), after lab notebook or writeup note edits
  (push-lab), or when the user mentions git pull, push, sync, or keeping
  the clone current. Do not invent git command sequences.
---

# RBC git sync

Canonical git policy is in [AGENTS.md](../../../AGENTS.md). This skill only tells Cursor to **call the script**.

## Session start

Without asking, from the repo root:

- Windows: `scripts/sync.ps1 pull`
- Unix: `scripts/sync.sh pull`

If the script fails, stop. Do not rebase or `--force`.

## After lab notes

If you updated `lab/*.md`, `lab/evidence/`, or `writeup/` (not analysis Python), you may run `push-lab` on the same script.

Do not auto-commit `lab/tools/*.py` or other code until Sai approved that change.

Never add `*.avi`. If the script refuses, unstage and leave the AVI on disk only.
