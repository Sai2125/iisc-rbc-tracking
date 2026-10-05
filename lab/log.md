# Log

Newest entries at the bottom.

## 2026-09-16 — Stored handoff and mapped local files

- **Done:** Wrote `RBC_handoff_context.txt` (sections 1–12 plus folder inventory). Did not change the notebook or raw companions.
- **Evidence:** none (inventory only)
- **Open:** Workspace search had missed large/ignored files; directory listing found them.
- **Next:** Use the handoff file instead of re-pasting context.

## 2026-09-16 — Local inventory

- **Done:** Listed `C:\Users\ASUS\Documents\IIsc`.
- **Evidence:** none. Facts: AVI is `2A002 2_C001H001S0001-001.avi` (22,901,777,870 bytes). CIHX 16,603 bytes. MII 1,139 bytes. Notebook `Untitled24.ipynb`. No `RBC Videos Longevity India` folder. No separate `RBC_01_Inspect.ipynb` filename (inspection cells live in `Untitled24.ipynb`).
- **Open:** AVI not decoded. Inspection notebook still points at Colab paths and the name without `-001`.
- **Next:** Either retarget inspection or, as later decided, start with single-frame detection on a still.

## 2026-09-16 — Sample frame and detection game plan

- **Done:** User added `image (3).png`. Confirmed diagonal tube, bath full of cell-like objects, target in the **right-hand flare** (red circle = annotation). Agreed the next problem is algorithmic single-frame detection, broken into: tube ROI → optional contrast → blobs → size/shape/location filters. Wrote `lab/gameplan.md`.
- **Evidence:** `image (3).png` (scouting still, annotated; not a raw AVI frame export)
- **Open:** No detector run yet. Mask not drawn. Narrow-section appearance unknown.
- **Next:** Try the four-layer recipe on this PNG without using the red circle as a feature.

## 2026-09-16 — Lab notebook + logging skill

- **Done:** Added `lab/` (CURRENT, log, decisions, gameplan, evidence/) and project skill `.cursor/skills/rbc-lab-log/SKILL.md` so agents keep this notebook updated.
- **Evidence:** none
- **Open:** Skill must be applied in later sessions; humans can still edit `lab/CURRENT.md` by hand.
- **Next:** First detection experiment on `image (3).png`.

## 2026-09-16 — Permission-before-code + tube-ROI proposal only

- **Done:** Wrote `AGENTS.md`: do not create or modify code until Sai permits; propose and wait. Logged that in `lab/decisions.md`. Wrote the tube-mask + probe-frame proposal into `lab/gameplan.md`. No scripts or notebook edits.
- **Evidence:** none
- **Open:** Awaiting permission to implement any tube-detection code.
- **Next:** Sai reviews the proposal (still → overlay; then ~3 seeked AVI frames if needed).

## 2026-09-16 — Tube mask on one random AVI frame

- **Done:** Seeked frame **24642 / 37274** (seed 20260916) from `2A002 2_C001H001S0001-001.avi`. Built wall + lumen masks. Did not scan the full video. Reused the saved still while tuning fill logic.
- **Evidence:** `lab/evidence/20260916_tube_mask_probe/` (`probe_still.png`, `overlay.png`, `lumen_mask.png`, `walls.png`). Green ribbon follows the bore and both flares; small leaks at mid-tube dirt.
- **Open:** Is this mask good enough to restrict cell search? Right flare tip slightly short.
- **Next:** User review of overlay; then blobs inside the lumen only.

## 2026-09-16 — Failure log added

- **Done:** Created `lab/failures.md` and wrote the tube-mask dead-ends (annotated PNG, fused walls / left-flare-only fill, thin bore erased by open + wrong gap rule, dirt leaks). Skill and `AGENTS.md` now require logging misses. Earlier failed overlays were overwritten; the write-up is the record of those attempts.
- **Evidence:** `lab/failures.md`; current after-state still in `lab/evidence/20260916_tube_mask_probe/`
- **Open:** Dirt leaks and short right tip still unfixed.
- **Next:** Overlay review, then cell blobs inside the lumen if the ribbon is accepted.

## 2026-10-05 — Repo, sync script, harness-agnostic rules

- **Done:** Added `.gitignore` (AVI excluded), `README.md` system map, extended `AGENTS.md`, `scripts/sync.ps1` + `sync.sh`, Copilot/CLAUDE/Cline pointers, Cursor skills `rbc-git-sync` and `rbc-writeup`, `lab/related-work.md`, `writeup/README.md`. Tube-mask code unchanged.
- **Evidence:** none (infra). Confirm `git status` does not list the AVI.
- **Open:** Need private GitHub remote and first push.
- **Next:** Overlay review for the lumen mask; other person clones and copies the AVI locally.




