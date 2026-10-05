# IIsc RBC tracking

Track red blood cells in a high-speed channel video. Algorithmic detect-and-measure — not dumping the video into an LLM.

The ~23 GB AVI is **not** in git. Copy it next to this clone as
`2A002 2_C001H001S0001-001.avi`.

## Setup

1. `git clone` this repo and open the folder in Cursor, VS Code, or another harness.
2. Place the AVI (keep `.cihx` / `.mii` if they are not already in the clone).
3. At the start of a session, run the pull script (agents should do this without being asked):

```powershell
.\scripts\sync.ps1 pull
```

```bash
./scripts/sync.sh pull
```

Agent rules for every harness: read [`AGENTS.md`](AGENTS.md). Cursor skills under `.cursor/skills/` are overlays only.

## What each piece is for

| Path | Role |
|------|------|
| `AGENTS.md` | Rules for agents (permission, no full-video, logging, git pull). Source of truth. |
| `README.md` | Humans: setup and this map. |
| `lab/CURRENT.md` | One page: now / next / blockers. Read this first. |
| `lab/log.md` | Dated history of what we did. |
| `lab/decisions.md` | Choices that should stick. |
| `lab/failures.md` | What went wrong, why, what we changed. Source of the “why this method”. |
| `lab/gameplan.md` | Current detect/measure plan. |
| `lab/related-work.md` | Papers/tools we actually use or reject. |
| `lab/evidence/` | Small proof images/notes. Never the AVI. |
| `lab/tools/` | Experiment scripts (e.g. tube mask). Not auto-pushed. |
| `.cursor/skills/rbc-lab-log/` | Cursor: keep `lab/` updated. |
| `.cursor/skills/rbc-git-sync/` | Cursor: call the sync script. |
| `.cursor/skills/rbc-writeup/` | Cursor: on request, draft `writeup/draft.md` from `lab/`. |
| `scripts/sync.ps1` / `scripts/sync.sh` | Deterministic `git pull --ff-only`; optional lab-only push. |
| `writeup/` | Living methods draft, generated when asked. |
| `.github/copilot-instructions.md`, `CLAUDE.md`, `.clinerules` | One-liners: follow `AGENTS.md`. |

Do not add a second notes app. If it is not in this table, we probably should not add it.
