# Game plan: identify an RBC in one frame

Updated: 2026-09-16

This is **computer vision**, not “show the video to a language model.”

The machine only has a grid of brightness values. Detection means turning that grid into a **mask**, **box**, or **center `(x, y)`** with rules we can check.

The scientific object is not “any RBC.” The field contains many cell-like blobs. The object is **the cell inside the tube lumen** (in `image (3).png`, the cell in the right-hand flare). The red circle is annotation only.

## Four layers

```text
pixels
  → 1. only look where a tube cell could be
  → 2. optionally make it easier to separate from gray background
  → 3. mark candidate pixels / blobs
  → 4. keep the blob that looks like a cell, not a wall
```

### 1. Restrict search

Mask or crop to the channel + flares. Ignore the bath.

Ways to get the region: draw the tube once, or later detect the dark walls and fill the lumen.

Until this exists, every detector will report dozens of cells.

### 2. Preprocess (optional)

Increase cell-vs-background difference without moving the center much: flatten lighting, later subtract a static background from many frames, or light local contrast.

Skip on the first still if a simple threshold inside the flare already works.

### 3. Segment

Rule examples: pixels darker than local background, or compact blobs of expected size.

This yields many islands: cell, wall chips, dirt, leaked bath cells.

### 4. Filter

Keep islands that are cell-sized, compact (not long wall segments), and on the lumen / centerline.

## What one frame cannot give

Flow direction, occupancy of the tube, squeezed appearance in the constriction, speed.

Those wait until the same recipe produces `(x, y)` on consecutive frames. Tracking is a later step: associate the same blob across time.

## Immediate experiment

On `image (3).png` (then on decoded AVI frames): tube mask → blobs → filters → check that the flare cell survives **without** using the red circle.

If that fails, change the recipe. If it works, run it on the next frame.

## Proposed next: tube ROI + a few probe frames (not implemented)

Code is **not** written until Sai approves. Idea only:

The glass tube is essentially **fixed**. Finding it does not require the whole 23 GB file.

**Iterate on the still first:** `image (3).png` is enough to invent and visually check a tube mask. No AVI decode needed for that loop.

**Then confirm on a handful of AVI frames by seek, not by scanning:**

- Open the file, jump to chosen indices, decode those frames only, close.
- Suggested set: first, middle, last header frames (3 images). Optionally 2 more. That checks “does the tube stay put?”
- Do **not** average or process all 37k frames for this.

**What “detect the tube” should output:** a binary **lumen mask** (inside the glass, including flares), plus an overlay PNG so we can see it. Walls can be an intermediate mask.

**Simple recipe to try (in order):**

1. Dark-pixel threshold / morphology to pick the two wall lines.
2. Fill the space between them (lumen), or dilate a centerline into a band.
3. If that is messy, fall back to one manually drawn polygon and keep using it until automation is worth it.

A moving cell does not matter for a static-tube check, which is why frame choice can stay coarse here.
