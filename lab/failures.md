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

## 2026-11-07 — Per-label ROI contour optimization was slower

- **Tried:** Compute each watershed label's bounding box once, then call `findContours` on a cropped label mask.
- **What went wrong:** The cropped approach took about 0.58 s/frame vs about 0.45 s/frame for the original detector on three seeked frames. An initial centroid-offset mistake also changed detections; it was caught by comparison and not retained.
- **Why (we think):** The per-pixel coordinate grouping overhead outweighed avoiding repeated full-frame scans for this data.
- **What we changed:** Replaced it with a single contour extraction over the watershed foreground mask, mapping contours back to labels to preserve the original detection order.
- **Still wrong:** The new approach is only checked on three frames; full-pipeline runtime and broader output equivalence remain unverified.
- **Evidence:** Exact detection output comparison against the pre-change `tracking.py` on frames 0, 18637, and 37273; those checks are summarized in `lab/log.md`.

## 2026-11-07 — Stage 3 saves no tracks or trajectory rows

- **Tried:** Inspected Stage 3 output and traced `CellTracker` initialization and result collection.
- **What went wrong:** `trajectories.csv` and `biomechanics.csv` are zero-byte files; summary reports zero tracks and zero analyzed frames.
- **Why:** `add_new_tracks` created tracks with `state=None`, while `associate` excludes those tracks from matching. `update` only initialized Kalman state for matched tracks, so new tracks could never reach three hits. The Kalman filter also used invalid property names and vector shapes for the installed OpenCV API. Separately, `main` initialized `frame_count = 0` but never incremented it.
- **What we changed:** Initialize each track's Kalman filter at creation using OpenCV's `statePost`/`errorCovPost` and column-vector measurement/state shapes; update matched filters through that state; increment the successful decoded-frame count. Also pass the actual track ID into `compute_biomarker` instead of incorrectly using the first frame number as its ID.
- **Still wrong:** Existing empty Stage 3 output files are not regenerated automatically; rerun Stage 3 to replace them. The full video pipeline has not been rerun.
- **Evidence:** Before the fix, local `output/stage3/trajectories.csv` and `biomechanics.csv` were 0 bytes and `track_summary.json` reported zero. After the fix, a five-detection synthetic sequence maintained one track for five hits and produced its biomarker with ID 0. On five consecutive real frames, detector counts were 35–39 and the tracker emitted 37 tracks by the third frame.
