# Current

Updated: 2026-10-05

## Now

First **tube/lumen mask** exists on one unannotated AVI frame. Bath is excluded. Ready to review the overlay, then next would be cell detection *inside* that mask.

Repo sharing is in place locally: git initialized, AVI gitignored, `AGENTS.md` is the harness-agnostic rule file, `scripts/sync.ps1` / `sync.sh` for pull and lab-only push. **Private GitHub remote not created yet** (`gh` is not installed).

## Next

1. Look at `lab/evidence/20260916_tube_mask_probe/overlay.png` and decide if the ribbon is good enough.
2. If yes: detect cell-like blobs only inside `lumen_mask.png` (still one frame).
3. Later: reuse the mask on a few other seeked frames (tube should be static). Not a full-video pass.

## Blockers / open

- Mask leaks a little at dirt stuck on the glass; far-right flare tip is slightly short. Write-up: `lab/failures.md`.
- AVI header count 37,274 on this decode — still not reconciled with CIHX saved range 0–17,471.
- No µm/pixel calibration. No physical speeds.
- Flow direction / occupancy / squeezed cell: not addressed.

## Key paths

| What | Path |
|---|---|
| Agent rules (all harnesses) | `AGENTS.md` |
| Human setup + map | `README.md` |
| Git sync | `scripts/sync.ps1`, `scripts/sync.sh` |
| Related work | `lab/related-work.md` |
| Failures | `lab/failures.md` |
| Tube-mask script | `lab/tools/probe_tube_mask.py` |
| Evidence (frame 24642) | `lab/evidence/20260916_tube_mask_probe/` |
| AVI (local only) | `2A002 2_C001H001S0001-001.avi` |
| Annotated screenshot (do not use for detection) | `image (3).png` |
