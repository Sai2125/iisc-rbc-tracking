# What did not go right

Newest at the bottom. Log misses with the same care as successes. A working overlay is not a reason to delete earlier failures.

Template:

```markdown
## YYYY-MM-DD — short name

- **Tried:**
- **What went wrong:** (what we saw, not a vibe)
- **Why (we think):**
- **What we changed:**
- **Still wrong:**
- **Evidence:**
```

---

## 2026-09-16 — Annotated screenshot as if it were a raw frame

- **Tried:** Use `image (3).png` to invent detection.
- **What went wrong:** Red circle sits on the right flare. It is not in the experiment image. Gray conversion would turn it into a dark ring that can look like a wall.
- **Why (we think):** The circle is a human annotation; screenshot coordinates also may not match the AVI.
- **What we changed:** Seeked an unannotated frame from the AVI instead (24642).
- **Still wrong:** `image (3).png` is still in the folder; do not feed it to the masker by accident.
- **Evidence:** `image (3).png` vs `lab/evidence/20260916_tube_mask_probe/probe_still.png`

## 2026-09-16 — First lumen fill only kept the left flare

- **Tried:** Adaptive-threshold walls, rotate tube horizontal, fill the **largest** gap between wall runs in each column; keep only the largest connected lumen component.
- **What went wrong:** `lumen_mask.png` was a white triangle in the left bulb only. Overlay was a solid red sausage; green almost absent. Lumen fraction ~0.37%.
- **Why (we think):**
  1. A 5×5 morphological close **fused** the two close walls in the narrow bore into one white band, so there was no gap to fill.
  2. “Largest gap” is the wrong rule: bath-side gaps can beat the bore.
  3. Keeping only the biggest blob dropped the right flare if it was disconnected.
  4. Overlay painted solid walls on top of any lumen, so even a thin bore would have been hidden.
- **What we changed:** Dropped the heavy close; thinner walls (two parallel lines). Fill between wall hits near the **centerline**, peel 2 px, punch walls out. Keep all sizable lumen pieces. Overlay uses wall **edges** plus green fill.
- **Still wrong:** See dirt leak and short right tip below.
- **Evidence:** first overlay/lumen pair was overwritten by later runs on the same still. The failure mode is described here; the saved files are the **after** state.

## 2026-09-16 — Thin bore deleted by cleanup / wrong gap pick

- **Tried:** After two walls were visible, still fill “largest valid gap,” then 5×5 morphological open on the lumen.
- **What went wrong:** Flares filled; most of the narrow channel stayed empty except a broken line on the right half. Mid-tube dirt became extra blobs.
- **Why (we think):** Open 5×5 is larger than a ~few-pixel bore, so it erases the channel. Largest-gap still prefers the wrong interval when debris adds extra runs. Max gap 0.45×height is wide enough to include bath.
- **What we changed:** Envelope between min/max wall rows within 0.18×height of the median wall row. Horizontal close (3×31) in rotated space. No 5×5 open.
- **Still wrong:** Dirt on the glass still inflates the envelope locally. Far-right flare tip underfilled (wall geometry is a Y / open tip).
- **Evidence:** `lab/evidence/20260916_tube_mask_probe/overlay.png` (current), `lumen_mask.png`

## 2026-09-16 — Dirt on the tube counted as lumen/walls

- **Tried:** Same wall detector; no extra “ignore compact blobs on the wall” rule.
- **What went wrong:** Mid-channel specks (visible on the still) produce extra red rings and small green islands off the true bore.
- **Why (we think):** Adaptive threshold does not know dirt from glass. Compact dark objects attached to the wall survive the “long component” filter if they connect to the tube, or they locally widen the min–max envelope.
- **What we changed:** Not fixed yet.
- **Still wrong:** Those leaks remain on the current overlay.
- **Evidence:** mid-tube blobs on `overlay.png` / `lumen_mask.png`
