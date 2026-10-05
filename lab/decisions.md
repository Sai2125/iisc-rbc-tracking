# Decisions

Add a line when a choice should outlive a chat recap. Do not delete old items; mark them superseded.

- **2026-09-16 — Method class:** Algorithmic detect and measure. Do not dump the video into an LLM/VLM to explain what is happening.
- **2026-09-16 — Next scientific step:** Single-frame identification of the cell **in the tube**, not whole-field RBC detection.
- **2026-09-16 — Scope of this lab notebook:** Lightweight files under `lab/`. No extra platform. Raw AVI stays in the project root.
- **2026-09-16 — Analysis data:** Original AVI / decoded pixels / lossless PNG. Lossy previews are for viewing only. Do not overwrite raw files.
- **2026-09-16 — Speeds:** No physical speed until capture timing and spatial calibration are verified. Pixel motion can be discussed after tracking works.
- **2026-09-16 — Annotation:** Red circle on `image (3).png` is not part of the experiment image. Do not use it as a detection feature. Screenshot coordinates may not match AVI coordinates.
- **2026-09-16 — Timing fields:** Prefer CIHX `recordInfo/recordRate` / MII film speed (10000). Do not use `deviceInfo/recordRate` (125) or AVI playback FPS as acquisition timing without verification.
- **2026-09-16 — Code permission:** Agents must not create or modify code until Sai gives permission. Propose first, wait, then implement. Rule lives in `AGENTS.md`.
- **2026-09-16 — Tube-mask prototype:** Sai approved seeking one random AVI still (bypassing the red-circle PNG) and building a lumen mask. Seed 20260916 → frame 24642 / 37274. Script: `lab/tools/probe_tube_mask.py`.
- **2026-09-16 — Log failures:** Misses go in `lab/failures.md` (what we saw, why, what changed, what remains). Do not keep only the successful overlay.
- **2026-10-05 — Git-shared notebook:** Private GitHub repo; AVI never committed. `AGENTS.md` is canonical for Cursor and other harnesses. Lab notes push via `scripts/sync.ps1 push-lab` / `sync.sh`; analysis code still needs Sai’s permission.
