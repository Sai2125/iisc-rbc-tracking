"""Seek one AVI frame and build a tube/lumen mask. Does not scan the full video."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "2A002 2_C001H001S0001-001.avi"
EVIDENCE = ROOT / "lab" / "evidence" / "20260916_tube_mask_probe"
STILL_NAME = "probe_still.png"
RNG_SEED = 20260916


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and np.array_equal(frame[:, :, 0], frame[:, :, 1]) and np.array_equal(
        frame[:, :, 1], frame[:, :, 2]
    ):
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def seek_random_frame(video_path: Path, seed: int) -> tuple[int, int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n < 1:
            raise RuntimeError(f"Unusable frame count: {n}")
        idx = int(np.random.default_rng(seed).integers(0, n))
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, idx):
            raise RuntimeError(f"Seek refused at {idx}")
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Decode failed at {idx}")
        return idx, n, frame
    finally:
        cap.release()


def wall_mask(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    dark = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = np.zeros_like(dark)
    h, w = gray.shape
    min_area = 0.002 * h * w
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        long_side = max(bw, bh)
        short_side = max(1, min(bw, bh))
        if area < min_area:
            continue
        if long_side < 0.35 * w:
            continue
        if long_side / short_side < 3.0 and area < 0.02 * h * w:
            continue
        keep[labels == i] = 255
    return keep


def lumen_from_walls(gray: np.ndarray, walls: np.ndarray) -> np.ndarray:
    """Fill between the two wall ridges along the tube's principal axis."""
    ys, xs = np.where(walls > 0)
    if xs.size < 200:
        raise RuntimeError("Too few wall pixels; adjust threshold rather than guessing a box.")
    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    axis = vt[0]
    if axis[0] < 0:
        axis = -axis
    angle = np.degrees(np.arctan2(axis[1], axis[0]))
    h, w = gray.shape
    rot = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])), angle, 1.0)
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    rotated_corners = cv2.transform(corners[None, :, :], rot)[0]
    x0, y0 = rotated_corners.min(axis=0)
    x1, y1 = rotated_corners.max(axis=0)
    rot[0, 2] -= x0
    rot[1, 2] -= y0
    out_w = int(np.ceil(x1 - x0))
    out_h = int(np.ceil(y1 - y0))
    walls_r = cv2.warpAffine(walls, rot, (out_w, out_h), flags=cv2.INTER_NEAREST)
    wall_rows = np.flatnonzero(walls_r.any(axis=1))
    if wall_rows.size == 0:
        raise RuntimeError("Walls disappeared after rotation.")
    center_y = float(np.median(wall_rows))
    max_span = int(0.18 * h)  # flare is wide; bath gaps are wider than this
    peel = 2
    lumen_r = np.zeros_like(walls_r)
    for x in range(out_w):
        ys_col = np.flatnonzero(walls_r[:, x] > 0)
        if ys_col.size == 0:
            continue
        near = ys_col[np.abs(ys_col - center_y) <= max_span]
        if near.size < 2:
            continue
        y0, y1 = int(near.min()), int(near.max())
        if y1 - y0 > max_span:
            continue
        if y1 - y0 <= 2 * peel:
            mid = (y0 + y1) // 2
            lumen_r[max(0, mid - 1) : mid + 2, x] = 255
        else:
            lumen_r[y0 + peel : y1 - peel + 1, x] = 255
        lumen_r[walls_r[:, x] > 0, x] = 0
    lumen_r = cv2.morphologyEx(lumen_r, cv2.MORPH_CLOSE, np.ones((3, 31), np.uint8))
    inv = cv2.invertAffineTransform(rot)
    lumen = cv2.warpAffine(lumen_r, inv, (w, h), flags=cv2.INTER_NEAREST)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lumen, connectivity=8)
    if n_labels <= 1:
        return lumen
    # Keep every sizable piece along the tube (flares can disconnect from the bore).
    min_keep = 0.0002 * h * w
    keep = np.zeros_like(lumen)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_keep:
            keep[labels == i] = 255
    return keep


def overlay(gray: np.ndarray, lumen: np.ndarray, walls: np.ndarray) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis[lumen > 0] = (vis[lumen > 0] * 0.4 + np.array([0, 200, 0]) * 0.6).astype(np.uint8)
    wall_edge = cv2.morphologyEx(
        walls, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    vis[wall_edge > 0] = (0, 0, 255)
    return vis


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    still_path = EVIDENCE / STILL_NAME
    meta_path = EVIDENCE / "notes.json"

    if still_path.is_file() and meta_path.is_file():
        gray = cv2.imread(str(still_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Could not read {still_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        idx = meta["avi_index_0based"]
        n = meta["avi_frame_count_header"]
        print(f"Reusing saved still AVI frame {idx} / {n} (no second seek)")
    else:
        if not VIDEO.is_file():
            raise FileNotFoundError(VIDEO)
        idx, n, frame = seek_random_frame(VIDEO, RNG_SEED)
        gray = to_gray(frame)
        if gray.dtype != np.uint8:
            raise RuntimeError(f"Unexpected dtype {gray.dtype}")
        if not cv2.imwrite(str(still_path), gray):
            raise RuntimeError("Failed to write still")
        meta = {
            "video": str(VIDEO.name),
            "avi_index_0based": idx,
            "avi_frame_count_header": n,
            "rng_seed": RNG_SEED,
            "still": STILL_NAME,
            "annotation": "none; decoded from AVI, not the red-circle screenshot",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved still AVI frame {idx} / {n}")

    walls = wall_mask(gray)
    lumen = lumen_from_walls(gray, walls)
    vis = overlay(gray, lumen, walls)
    cv2.imwrite(str(EVIDENCE / "walls.png"), walls)
    cv2.imwrite(str(EVIDENCE / "lumen_mask.png"), lumen)
    cv2.imwrite(str(EVIDENCE / "overlay.png"), vis)
    (EVIDENCE / "notes.txt").write_text(
        (
            f"Random AVI frame {idx} of {n} (seed {RNG_SEED}). "
            "Red = detected walls, green = lumen mask. "
            "Prototype only; not used for tracking yet.\n"
        ),
        encoding="utf-8",
    )
    print("Wrote", EVIDENCE)
    print("lumen fraction", float((lumen > 0).mean()))
    print("wall fraction", float((walls > 0).mean()))


if __name__ == "__main__":
    main()
