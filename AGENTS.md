# Agent instructions (this project)

Canonical rules for **every** harness (Cursor, VS Code Copilot, Claude Code, Cline, Hermes, etc.). Cursor skills under `.cursor/skills/` are convenience overlays. Do not put unique policy only in a skill.

Humans: see [README.md](README.md) for the same map in setup language.

## Code changes require permission

Do **not** create, edit, delete, or run project code until Sai explicitly gives permission for that change.

This includes Python scripts, notebooks, helpers, configs that execute, and generated analysis code.

You **may** without extra permission:

- Read files
- Update `lab/` notebook files (`CURRENT.md`, `log.md`, `decisions.md`, `failures.md`, `gameplan.md`, `related-work.md`, evidence notes) when logging progress
- Update this `AGENTS.md` if Sai asks

You **must**:

1. Propose the change (what files, what it will do, what it will not do).
2. Wait for permission.
3. Only then implement exactly what was approved.

If permission is unclear, ask. Do not “just add a small script.”

## Git (deterministic script)

Session start: run the sync script **without asking**:

- Windows: `scripts/sync.ps1 pull`
- Unix: `scripts/sync.sh pull`

That is `git fetch` + `git pull --ff-only`. If it fails, **stop** and report. Do not rebase, merge inventively, or force.

After updating lab **notes** (markdown + `lab/evidence/` + `writeup/`): you may run `push-lab` on the same script. Do **not** auto-commit `lab/tools/*.py` or other analysis code until Sai approved that code change.

Never `git add` `*.avi` or other raw recordings. The script must refuse if an AVI would be staged.

Do not invent ad-hoc git command sequences; call the script.

## How we work

- Algorithmic detect and measure. Do not dump the video into an LLM/VLM as the method.
- Do not process the full AVI to try an idea. Use a still or a few seeked frames.
- Do not overwrite raw `.avi` / `.cihx` / `.mii`.
- Progress, evidence, and **failures** live in `lab/`. Read `lab/CURRENT.md` at the start of RBC work. Record misses in `lab/failures.md`. Follow `.cursor/skills/rbc-lab-log/SKILL.md` when in Cursor; the procedures there must match this file.
- Related work lives in `lab/related-work.md`. When a paper or tool is used or rejected, add a row. Do not start a literature-crawler loop.
- Paper/methods draft: **only when asked**. Compile from `lab/` into `writeup/draft.md`. Use `failures.md` for why an approach was taken or dropped. Do not invent results or physical speeds.

## Repo system map

| Path | Role |
|------|------|
| `AGENTS.md` | This file — agent rules |
| `README.md` | Human setup + map |
| `lab/CURRENT.md` | Now / next / blockers |
| `lab/log.md` | Dated history |
| `lab/decisions.md` | Durable choices |
| `lab/failures.md` | Misses and why |
| `lab/gameplan.md` | Current scientific plan |
| `lab/related-work.md` | Papers/tools table |
| `lab/evidence/` | Small proofs, never the AVI |
| `lab/tools/` | Experiment scripts (permission-gated) |
| `scripts/sync.ps1` / `sync.sh` | Pull / push-lab |
| `writeup/` | On-demand draft |
| `.cursor/skills/` | Cursor-only loaders pointing here |
