---
name: rbc-lab-log
description: >-
  Maintains the lightweight RBC lab notebook in this project (lab/CURRENT.md,
  lab/log.md, lab/decisions.md, lab/failures.md, lab/related-work.md,
  lab/evidence/). Use when finishing a session, recording progress, decisions,
  failures, blockers, evidence, file locations, or handoff; when the user says
  log, notebook, status, or what is next; and before claiming work is done on
  the IIsc RBC tracking project. Git pull/push: use rbc-git-sync / scripts/sync.
---

# RBC lab log

Project notebook lives in `lab/`. Keep it small. Do not add databases, extra tools, or a research platform.

## When to use

Read `lab/CURRENT.md` at the start of RBC work. If git is initialized, run `scripts/sync.ps1 pull` (or `sync.sh`) first — see [AGENTS.md](../../../AGENTS.md) and the rbc-git-sync skill. Do not invent git commands.

Update the notebook when any of these happen:

- a decision is made
- a file is added or its real name/path is discovered
- an experiment runs (success or failure)
- something did not work, including a dead-end we later fixed
- the user asks to log, share evidence, or recap status
- the session ends after meaningful work

Skip tiny chat clarifications that do not change the work.

## Layout (do not grow this unless asked)

| File | Role |
|---|---|
| [lab/CURRENT.md](../../../lab/CURRENT.md) | One page: now, next, blockers, key paths |
| [lab/log.md](../../../lab/log.md) | Append-only dated entries |
| [lab/decisions.md](../../../lab/decisions.md) | Durable choices (do not rewrite history in the log) |
| [lab/failures.md](../../../lab/failures.md) | What went wrong, why we think so, what we changed |
| [lab/gameplan.md](../../../lab/gameplan.md) | Current scientific/engineering game plan |
| [lab/related-work.md](../../../lab/related-work.md) | Papers/tools we use or reject |
| [lab/evidence/](../../../lab/evidence/) | Small proof artifacts only |

Raw recording files stay in the project root. Never copy the AVI into `lab/`.

## How to log

1. Append one entry to `lab/log.md` using the template below. Newest at the **bottom**.
2. Rewrite `lab/CURRENT.md` so it matches reality (it is not a history file).
3. If a choice should survive a recap, add a short item to `lab/decisions.md`.
4. If something **failed**, add an entry to `lab/failures.md` even if you later fixed it. Do not only keep the happy overlay. Say what we saw, why we think it happened, and what is still wrong.
5. If something **worked** or failed in a checkable way, put a small artifact in `lab/evidence/YYYYMMDD_short-name/` and link it from the log entry.
6. If a paper or tool is used or rejected, add a row to `lab/related-work.md`.
7. Do not edit the plan file in `.cursor/plans/` unless the user asks.

When iterating on the same evidence folder, later plots may overwrite earlier ones. If that happens, the failure write-up in `lab/failures.md` is the record; say that the PNGs are the after-state.

### Failure entry template

```markdown
## YYYY-MM-DD — short name

- **Tried:**
- **What went wrong:**
- **Why (we think):**
- **What we changed:**
- **Still wrong:**
- **Evidence:**
```

### Log entry template

```markdown
## YYYY-MM-DD — short title

- **Done:**
- **Evidence:** path or "none"
- **Open:**
- **Next:**
```

Keep bullets short. Link files with paths from the project root.

## Evidence rules

Store only what another person needs to verify a claim:

- a PNG, contact sheet, overlay, CSV of positions, short notes.txt
- a settings snippet or parameter list

Do not store:

- the original AVI / CIHX / MII
- lossy previews presented as analysis data
- annotated screenshots as if they were raw frames (label them as annotations)

If an artifact is too large, write a pointer in the log (`path`, frame index, what it shows) instead of copying it.

## Status recap

When the user asks where things stand, answer from `lab/CURRENT.md` first, then `lab/log.md` (last few entries). Do not restart a large planning exercise.

## Sharing

To share proof that something works, point at one `lab/evidence/...` folder plus the matching log entry. Do not paste the 23 GB video or dump frames into a chat by default.
