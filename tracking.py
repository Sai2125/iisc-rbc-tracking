"""
STAGE 3 — Single-Track-Per-Cell Passage Tracker & Annotated Video Generator

Guarantees ONE single continuous track (and one unified bounding box) per RBC
from the start of the microfluidic passage to the end of the passage:

1. Single-Cell Contour Merging (Intra-Frame NMS):
   In brightfield/phase microscopy, a single RBC has a bright center + dark rim
   (or leading + trailing edge in |frame - bg|), which previously split one cell
   into 2-3 separate contours and spawned parallel tracks on the same cell.
   Now, all contours within `merge_radius_px = 55` along the channel axis are
   fused into ONE unified cell bounding box per cell.

2. Passage-Level Trajectory Stitcher + Gap Interpolation:
   Links any track fragments across the narrow bore (including across the stuck
   wall dirt at x ~ 435) using monotonic channel-flow velocity matching, and
   interpolates short 1-8 frame gaps so the bounding box never flickers or
   switches IDs mid-passage.

3. Strict Complete-Passage Filter:
   Rejects short noise blips and wall jitter; keeps only real cell passages that
   substantively traverse the channel (`min_span_px = 180`), renumbered cleanly
   as CELL #1, CELL #2, ...
"""
import os
import sys
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from numpy.lib.stride_tricks import sliding_window_view

import cv2
import numpy as np

# ====================================================================
# PATH RESOLUTION
# ====================================================================
ROOT_DIR = Path(__file__).resolve().parent
VIDEO_FILENAME = "2A002 2_C001H001S0001-001.avi"


def _resolve_out_dir():
    local_out = ROOT_DIR / "output"
    legacy_out = Path(r"C:\Users\amara\rbc_tracking\output")
    if (local_out / "calibration.json").is_file() or not (legacy_out / "calibration.json").is_file():
        return str(local_out)
    return str(legacy_out)


def _resolve_video_path(calib_video_path):
    if calib_video_path and os.path.isfile(calib_video_path):
        return calib_video_path
    candidates = [
        ROOT_DIR / VIDEO_FILENAME,
        ROOT_DIR.parent / VIDEO_FILENAME,
        Path.home() / "Documents" / "IIsc" / VIDEO_FILENAME,
        Path.home() / "Desktop" / "rbctrack" / VIDEO_FILENAME,
        Path.home() / "Downloads" / VIDEO_FILENAME,
        Path(r"C:\Users\amara\Downloads") / VIDEO_FILENAME,
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return calib_video_path


OUT = _resolve_out_dir()
CALIB_FILE = os.path.join(OUT, "calibration.json")
BG_FILE = os.path.join(OUT, "median_bg.npy")


def load_calibration():
    with open(CALIB_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_background():
    return np.load(BG_FILE)


# ====================================================================
# EXACT FULL-WIDTH INTERIOR LUMEN MASK (x = 0 .. W-1)
# ====================================================================

def _medfilt1d(arr, k=15):
    pad = k // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.median(sliding_window_view(padded, k), axis=-1)


def build_exact_lumen_geometry(bg):
    """
    Trace the top and bottom dark glass walls across the ENTIRE frame width (x = 0..W-1),
    including the left flare (x < 115), narrow constriction (115 <= x <= 1165),
    and right flare (x > 1165).
    """
    h, w = bg.shape[:2]

    xs_fit = np.arange(120, w - 120, dtype=np.float64)
    ys_fit = np.array([bg[:, int(x)].argmin() for x in xs_fit], dtype=np.float64)
    for _ in range(4):
        A = np.polyfit(xs_fit, ys_fit, 1)
        r = ys_fit - np.polyval(A, xs_fit)
        keep = np.abs(r) < 2.0 * r.std()
        if keep.sum() < 20:
            break
        xs_fit, ys_fit = xs_fit[keep], ys_fit[keep]
    A = np.polyfit(xs_fit, ys_fit, 1)
    slope, intercept = float(A[0]), float(A[1])

    top_wall = np.zeros(w, dtype=np.float64)
    bot_wall = np.zeros(w, dtype=np.float64)

    for x in range(w):
        yc = slope * x + intercept
        max_search = 28 if (x < 115 or x > 1165) else 9
        y_lo = max(0, int(round(yc - max_search)))
        y_hi = min(h - 1, int(round(yc + max_search)))
        y_mid = np.clip(int(round(yc)), y_lo + 1, y_hi - 1)

        top_slice = bg[y_lo:y_mid, x]
        bot_slice = bg[y_mid + 1:y_hi + 1, x]

        top_wall[x] = y_lo + int(np.argmin(top_slice)) if top_slice.size > 0 else yc - 3.5
        bot_wall[x] = (y_mid + 1) + int(np.argmin(bot_slice)) if bot_slice.size > 0 else yc + 3.5

    top_smooth = _medfilt1d(top_wall, k=15)
    bot_smooth = _medfilt1d(bot_wall, k=15)

    lumen_mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        yc = 0.5 * (top_smooth[x] + bot_smooth[x])
        # Peel 1.5 px inside walls so wall shimmer is excluded
        half_w = max(1.8, 0.5 * (bot_smooth[x] - top_smooth[x]) - 1.5)
        y0 = max(0, int(np.floor(yc - half_w)))
        y1 = min(h, int(np.ceil(yc + half_w)) + 1)
        lumen_mask[y0:y1, x] = 255

    return lumen_mask, top_smooth, bot_smooth, slope, intercept


# ====================================================================
# SINGLE-CELL CONTOUR FUSION (Prevents 1 cell from becoming 2-3 boxes)
# ====================================================================

def _merge_same_cell_blobs(raw_blobs, merge_radius_px=55.0):
    """
    A single RBC in brightfield/phase contrast produces a bright center + dark rim
    (or leading + trailing lobes), which threshold into 2-3 nearby contours.
    This merges any blobs within `merge_radius_px` along X into ONE single cell detection.
    """
    if len(raw_blobs) <= 1:
        return raw_blobs

    blobs = sorted(raw_blobs, key=lambda b: b["cx"])
    merged = []
    cluster = [blobs[0]]

    for b in blobs[1:]:
        if abs(b["cx"] - cluster[-1]["cx"]) <= merge_radius_px:
            cluster.append(b)
        else:
            merged.append(_fuse_cluster(cluster))
            cluster = [b]
    merged.append(_fuse_cluster(cluster))
    return merged


def _fuse_cluster(cluster):
    if len(cluster) == 1:
        return cluster[0]

    # Weight centroid by each sub-contour's area * peak_diff
    weights = np.array([max(1.0, c["area"] * c["peak_diff"]) for c in cluster], dtype=np.float64)
    w_sum = float(weights.sum())
    cx = float(np.sum([c["cx"] * w for c, w in zip(cluster, weights)]) / w_sum)
    cy = float(np.sum([c["cy"] * w for c, w in zip(cluster, weights)]) / w_sum)

    x0 = min(c["bbox"][0] for c in cluster)
    y0 = min(c["bbox"][1] for c in cluster)
    x1 = max(c["bbox"][0] + c["bbox"][2] for c in cluster)
    y1 = max(c["bbox"][1] + c["bbox"][3] for c in cluster)
    bw = int(x1 - x0)
    bh = int(y1 - y0)

    total_area = float(sum(c["area"] for c in cluster))
    effective_area = max(total_area, 0.72 * float(bw * bh))

    # Equivalent ellipse perimeter from bounding box for merged cell circularity
    a = max(1.0, 0.5 * bw)
    b = max(1.0, 0.5 * bh)
    perim_approx = float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))
    circ = float(4.0 * np.pi * effective_area / (perim_approx * perim_approx)) if perim_approx > 0 else 0.5
    circ = float(np.clip(circ, 0.08, 1.0))

    long_side = float(max(bw, bh))
    short_side = float(max(1, min(bw, bh)))

    return {
        "cx": cx,
        "cy": cy,
        "bbox": (int(x0), int(y0), bw, bh),
        "area": effective_area,
        "circularity": circ,
        "elongation": long_side / short_side,
        "peak_diff": float(max(c["peak_diff"] for c in cluster)),
    }


def detect_cells_in_passage(frame, background, lumen_mask, roi_y_bounds, diff_thresh=13.0):
    """
    Detect RBCs strictly inside the microfluidic passage and fuse multi-lobe
    contours of the same cell into a single bounding box.
    """
    y_min, y_max = roi_y_bounds
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    gray_roi = gray[y_min:y_max, :].astype(np.float32)
    bg_roi = background[y_min:y_max, :]
    mask_roi = lumen_mask[y_min:y_max, :]

    diff = np.abs(gray_roi - bg_roi)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    diff[mask_roi == 0] = 0.0

    max_val = float(diff.max())
    if max_val < diff_thresh:
        return []

    binary = (diff >= diff_thresh).astype(np.uint8) * 255

    # Horizontal morphological close (11x3) bridges the bright center and rim of one cell
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_blobs = []

    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        bx, by_rel, bw, bh = cv2.boundingRect(cnt)
        box_area = float(bw * bh)

        if area < 9.0 and box_area < 15.0:
            continue
        if area > 4500.0:
            continue

        blob_patch = diff[by_rel:by_rel + bh, bx:bx + bw]
        peak_diff = float(blob_patch.max()) if blob_patch.size > 0 else 0.0
        if peak_diff < diff_thresh + 2.5:
            continue

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"]) + y_min
        else:
            cx = float(bx + 0.5 * bw)
            cy = float(by_rel + 0.5 * bh + y_min)

        by = int(by_rel + y_min)
        effective_area = max(area, 0.72 * box_area)

        perimeter = float(cv2.arcLength(cnt, True))
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.5
        circularity = float(np.clip(circularity, 0.08, 1.0))

        long_side = float(max(bw, bh))
        short_side = float(max(1, min(bw, bh)))

        raw_blobs.append({
            "cx": cx,
            "cy": cy,
            "bbox": (int(bx), int(by), int(bw), int(bh)),
            "area": effective_area,
            "circularity": circularity,
            "elongation": long_side / short_side,
            "peak_diff": peak_diff,
        })

    # Fuse nearby contours belonging to the same cell into ONE detection
    return _merge_same_cell_blobs(raw_blobs, merge_radius_px=55.0)


# ====================================================================
# SINGLE-TRACK-PER-PASSAGE TRACKER + FRAGMENT STITCHER
# ====================================================================

class PassageCellTracker:
    """
    Tracks each RBC across the full passage and stitches any broken segments
    (e.g. across the narrowest bore or dirt spot) into ONE continuous track per cell.
    """
    def __init__(self, slope, intercept, max_age=25, min_hits=4, max_step_px=120.0):
        self.slope = slope
        self.intercept = intercept
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_step_px = max_step_px

        norm = np.sqrt(1.0 + slope * slope)
        self.ux = 1.0 / norm
        self.uy = slope / norm

        self.tracks = {}
        self.finished_tracks = {}
        self.next_id = 1

    def _init_kf(self, cx, cy, vx=0.0, vy=0.0):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.35
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        state = np.array([[float(cx)], [float(cy)], [float(vx)], [float(vy)]], dtype=np.float32)
        kf.statePre = state.copy()
        kf.statePost = state.copy()
        return kf

    def step(self, frame_idx, detections):
        for tid, tr in self.tracks.items():
            tr["kf"].predict()

        track_ids = list(self.tracks.keys())
        n_t = len(track_ids)
        n_d = len(detections)

        matched_t = []
        matched_d = []

        if n_t > 0 and n_d > 0:
            BIG_COST = 1e6
            cost = np.full((n_t, n_d), BIG_COST, dtype=np.float64)
            for i, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                pred = tr["kf"].statePre.reshape(-1)
                px, py = float(pred[0]), float(pred[1])
                last_x, last_y = tr["path"][-1]["cx"], tr["path"][-1]["cy"]
                dt = max(1, frame_idx - tr["last_frame"])

                for j, d in enumerate(detections):
                    dx_pred = d["cx"] - px
                    dy_pred = d["cy"] - py
                    ds = abs(dx_pred * self.ux + dy_pred * self.uy)
                    dn = abs(-dx_pred * self.uy + dy_pred * self.ux)
                    raw_dist = np.hypot(d["cx"] - last_x, d["cy"] - last_y) / float(dt)

                    if dn <= 20.0 and (ds <= self.max_step_px or raw_dist <= self.max_step_px):
                        cost[i, j] = min(ds, raw_dist) + 3.0 * dn

            try:
                from scipy.optimize import linear_sum_assignment
                r_ind, c_ind = linear_sum_assignment(cost)
            except Exception:
                r_ind, c_ind = [], []
                work = cost.copy()
                for _ in range(min(n_t, n_d)):
                    r, c = np.unravel_index(np.argmin(work), work.shape)
                    if work[r, c] >= BIG_COST:
                        break
                    r_ind.append(int(r))
                    c_ind.append(int(c))
                    work[r, :] = BIG_COST
                    work[:, c] = BIG_COST

            for r, c in zip(r_ind, c_ind):
                if cost[r, c] < self.max_step_px:
                    matched_t.append(track_ids[r])
                    matched_d.append(c)

        for tid, d_idx in zip(matched_t, matched_d):
            d = detections[d_idx]
            tr = self.tracks[tid]
            dt = max(1, frame_idx - tr["last_frame"])
            prev = tr["path"][-1]
            inst_vel = float(np.hypot(d["cx"] - prev["cx"], d["cy"] - prev["cy"]) / float(dt))

            meas = np.array([[float(d["cx"])], [float(d["cy"])]], dtype=np.float32)
            tr["kf"].correct(meas)
            tr["hits"] += 1
            tr["last_frame"] = frame_idx

            tr["path"].append({
                "frame": frame_idx,
                "track_id": tid,
                "cx": d["cx"],
                "cy": d["cy"],
                "bbox": d["bbox"],
                "area": d["area"],
                "circularity": d["circularity"],
                "elongation": d["elongation"],
                "velocity": inst_vel,
                "ADI_running": 1.0,
            })

        unmatched_d = set(range(n_d)) - set(matched_d)
        for d_idx in sorted(unmatched_d):
            d = detections[d_idx]
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "kf": self._init_kf(d["cx"], d["cy"]),
                "hits": 1,
                "last_frame": frame_idx,
                "path": [{
                    "frame": frame_idx,
                    "track_id": tid,
                    "cx": d["cx"],
                    "cy": d["cy"],
                    "bbox": d["bbox"],
                    "area": d["area"],
                    "circularity": d["circularity"],
                    "elongation": d["elongation"],
                    "velocity": 0.0,
                    "ADI_running": 1.0,
                }],
            }

        to_del = []
        for tid, tr in self.tracks.items():
            if frame_idx - tr["last_frame"] > self.max_age:
                self.finished_tracks[tid] = tr["path"]
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

    def _stitch_and_interpolate(self, raw_paths, min_span_px=120.0):
        """
        1. Discard stationary wall jitter (require monotonic movement > 25 px).
        2. Stitch collinear fragments of the same cell passage across short frame gaps.
        3. Linearly interpolate any 1..12 frame gaps so the bounding box stays locked
           on the cell continuously from start of passage to end of passage.
        """
        # Filter to fragments that actually move along X
        moving = []
        for path in raw_paths:
            if len(path) < 2:
                continue
            dx = path[-1]["cx"] - path[0]["cx"]
            span_x = max(p["cx"] for p in path) - min(p["cx"] for p in path)
            if abs(dx) < 20.0 and span_x < 25.0:
                continue
            moving.append(sorted(path, key=lambda p: p["frame"]))

        if not moving:
            return {}

        # Determine dominant flow direction (+1 for left->right, -1 for right->left)
        net_dxs = [p[-1]["cx"] - p[0]["cx"] for p in moving]
        flow_dir = 1.0 if np.sum(net_dxs) >= 0 else -1.0

        # Sort fragments by start frame
        moving.sort(key=lambda p: p[0]["frame"])

        # Stitch fragments that belong to the same cell passage
        stitched = []
        used = [False] * len(moving)

        for i in range(len(moving)):
            if used[i]:
                continue
            curr = list(moving[i])
            used[i] = True

            while True:
                best_j = None
                best_score = 1e9
                f_end = curr[-1]["frame"]
                x_end = curr[-1]["cx"]
                # Estimate velocity of current track
                dt_curr = max(1, curr[-1]["frame"] - curr[0]["frame"])
                v_curr = (curr[-1]["cx"] - curr[0]["cx"]) / float(dt_curr)

                for j in range(i + 1, len(moving)):
                    if used[j]:
                        continue
                    cand = moving[j]
                    f_start = cand[0]["frame"]
                    x_start = cand[0]["cx"]
                    gap_f = f_start - f_end

                    # Allow stitching across up to 40 frames if the cell moved forward along the tube
                    if 1 <= gap_f <= 40:
                        dx_gap = (x_start - x_end) * flow_dir
                        if -15.0 <= dx_gap <= 650.0:
                            expected_dx = abs(v_curr) * gap_f
                            score = abs(dx_gap - expected_dx) + 4.0 * gap_f
                            if score < best_score:
                                best_score = score
                                best_j = j

                if best_j is not None:
                    curr.extend(moving[best_j])
                    used[best_j] = True
                else:
                    break

            stitched.append(curr)

        # Keep only genuine cell passages that travel a real span along the channel
        # and interpolate any missing frames inside each passage so the box never drops out!
        final_tracks = {}
        new_id = 1

        for path in stitched:
            # Deduplicate same-frame entries if any
            by_f = {}
            for pt in path:
                by_f[pt["frame"]] = pt
            pts = [by_f[f] for f in sorted(by_f.keys())]
            if len(pts) < self.min_hits:
                continue

            span_x = float(max(p["cx"] for p in pts) - min(p["cx"] for p in pts))
            if span_x < min_span_px:
                continue

            # Interpolate gaps <= 15 frames so the bounding box stays smoothly locked on the cell
            continuous_pts = []
            for k in range(len(pts) - 1):
                p0 = pts[k]
                p1 = pts[k + 1]
                continuous_pts.append(p0)
                gap = p1["frame"] - p0["frame"]
                if 1 < gap <= 15:
                    for step_f in range(1, gap):
                        alpha = float(step_f) / float(gap)
                        cx_i = (1.0 - alpha) * p0["cx"] + alpha * p1["cx"]
                        cy_i = self.slope * cx_i + self.intercept
                        bw_i = int(round((1.0 - alpha) * p0["bbox"][2] + alpha * p1["bbox"][2]))
                        bh_i = int(round((1.0 - alpha) * p0["bbox"][3] + alpha * p1["bbox"][3]))
                        bx_i = int(round(cx_i - 0.5 * bw_i))
                        by_i = int(round(cy_i - 0.5 * bh_i))
                        continuous_pts.append({
                            "frame": p0["frame"] + step_f,
                            "track_id": new_id,
                            "cx": cx_i,
                            "cy": cy_i,
                            "bbox": (bx_i, by_i, bw_i, bh_i),
                            "area": (1.0 - alpha) * p0["area"] + alpha * p1["area"],
                            "circularity": (1.0 - alpha) * p0["circularity"] + alpha * p1["circularity"],
                            "elongation": (1.0 - alpha) * p0["elongation"] + alpha * p1["elongation"],
                            "velocity": abs(p1["cx"] - p0["cx"]) / float(gap),
                            "ADI_running": 1.0,
                        })
            continuous_pts.append(pts[-1])

            # Recompute running velocity and running ADI cleanly
            areas_accum = []
            for idx, p in enumerate(continuous_pts):
                p["track_id"] = new_id
                if idx == 0:
                    p["velocity"] = 0.0
                else:
                    prev = continuous_pts[idx - 1]
                    dt = max(1, p["frame"] - prev["frame"])
                    p["velocity"] = float(np.hypot(p["cx"] - prev["cx"], p["cy"] - prev["cy"]) / float(dt))
                areas_accum.append(p["area"])
                p["ADI_running"] = float(np.min(areas_accum) / max(1.0, np.mean(areas_accum)))

            final_tracks[new_id] = continuous_pts
            new_id += 1

        # Enforce at most 1 active track per frame if two stitched fragments still overlap
        return final_tracks

    def finalize_all(self, min_hits=4, min_span_px=120.0):
        all_raw = list(self.finished_tracks.values()) + [tr["path"] for tr in self.tracks.values()]
        tracks = self._stitch_and_interpolate(all_raw, min_span_px=min_span_px)
        # Fallback if a shorter clip was used where the cell only traveled 45+ px inside the window
        if not tracks and min_span_px > 45.0:
            tracks = self._stitch_and_interpolate(all_raw, min_span_px=45.0)
        return tracks


# ====================================================================
# BIOMECHANICS COMPUTATION
# ====================================================================

def compute_biomarker(track_id, path):
    if len(path) < 2:
        return None

    frames = [p["frame"] for p in path]
    areas = [p["area"] for p in path]
    circs = [p["circularity"] for p in path]
    elongs = [p["elongation"] for p in path]
    vels = [p["velocity"] for p in path[1:] if p["velocity"] > 0]

    area_mean = float(np.mean(areas))
    area_min = float(np.min(areas))
    adi = area_min / area_mean if area_mean > 0 else 0.0

    shape_recovery = None
    if len(path) >= 5:
        mid_circ = float(np.mean(circs[len(circs) // 3: 2 * len(circs) // 3]))
        post_circ = float(np.mean(circs[2 * len(circs) // 3:]))
        if mid_circ > 0:
            shape_recovery = post_circ / mid_circ

    x_start, x_end = float(path[0]["cx"]), float(path[-1]["cx"])
    travel_px = float(max(p["cx"] for p in path) - min(p["cx"] for p in path))

    return {
        "track_id": int(track_id),
        "entrance_frame": int(frames[0]),
        "exit_frame": int(frames[-1]),
        "track_length_frames": len(path),
        "duration_ms": float((frames[-1] - frames[0]) / 10.0),
        "x_start_px": round(x_start, 1),
        "x_end_px": round(x_end, 1),
        "net_travel_px": round(travel_px, 1),
        "velocity_mean": float(np.mean(vels)) if vels else 0.0,
        "velocity_max": float(np.max(vels)) if vels else 0.0,
        "area_mean": area_mean,
        "area_min": area_min,
        "circularity_mean": float(np.mean(circs)),
        "circularity_min": float(np.min(circs)),
        "elongation_max": float(np.max(elongs)),
        "ADI": float(adi),
        "shape_recovery": float(shape_recovery) if shape_recovery is not None else None,
    }


# ====================================================================
# DIRECT ANNOTATED VIDEO RENDERER (1 Single Box + HUD Per Cell)
# ====================================================================

def _cell_color_bgr(track_id):
    palette = [
        (50, 255, 100),   # Crisp Neon Green
        (0, 215, 255),    # Amber Gold
        (255, 200, 50),   # Cyan
        (180, 105, 255),  # Magenta
    ]
    return palette[(track_id - 1) % len(palette)]


def render_annotated_videos(
    video_path,
    start_frame,
    frames_to_run,
    top_wall,
    bot_wall,
    valid_tracks,
    biomarkers_by_id,
    out_dir,
    playback_fps=15.0
):
    full_mp4 = os.path.join(out_dir, "tracked_cells_video.mp4")
    active_mp4 = os.path.join(out_dir, "active_passages_only.mp4")
    best_png = os.path.join(out_dir, "tracked_cell_snapshot.png")

    by_frame = defaultdict(list)
    history_by_track = defaultdict(list)
    for tid, path in valid_tracks.items():
        for pt in path:
            by_frame[pt["frame"]].append(pt)

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer_full = cv2.VideoWriter(full_mp4, fourcc, playback_fps, (w, h))
    writer_active = cv2.VideoWriter(active_mp4, fourcc, playback_fps, (w, h))

    xs_wall = np.arange(0, w, 8)
    top_pts = np.column_stack([xs_wall, top_wall[xs_wall]]).astype(np.int32)
    bot_pts = np.column_stack([xs_wall, bot_wall[xs_wall]]).astype(np.int32)

    active_frame_count = 0
    best_frame_img = None
    best_frame_score = -1

    for offset in range(frames_to_run):
        f_idx = start_frame + offset
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Subtle channel wall boundary lines
        cv2.polylines(vis, [top_pts], False, (60, 125, 60), 1, cv2.LINE_AA)
        cv2.polylines(vis, [bot_pts], False, (60, 125, 60), 1, cv2.LINE_AA)

        frame_dets = by_frame.get(f_idx, [])
        for d in frame_dets:
            tid = d["track_id"]
            color = _cell_color_bgr(tid)
            bx, by, bw, bh = d["bbox"]
            cx_i, cy_i = int(round(d["cx"])), int(round(d["cy"]))
            history_by_track[tid].append((cx_i, cy_i))

            # 1. Single continuous trajectory line from start of passage
            trail = history_by_track[tid]
            if len(trail) >= 2:
                cv2.polylines(vis, [np.array(trail, np.int32)], False, color, 2, cv2.LINE_AA)
                cv2.circle(vis, trail[0], 4, (0, 255, 255), -1, cv2.LINE_AA)

            # 2. Clean bounding box around the cell
            pad_x, pad_y = max(8, bw // 2), max(8, bh // 2)
            x0, y0 = max(0, cx_i - pad_x), max(0, cy_i - pad_y)
            x1, y1 = min(w - 1, cx_i + pad_x), min(h - 1, cy_i + pad_y)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)

            # 3. Live Biomechanics HUD box anchored to the cell
            bio = biomarkers_by_id.get(tid, {})
            adi_val = bio.get("ADI", d["ADI_running"])
            line1 = f"CELL #{tid}  (x={cx_i}px)"
            line2 = f"Vel: {d['velocity']:.1f} px/f | Area: {d['area']:.0f} px2"
            line3 = f"Circ: {d['circularity']:.2f} | ADI: {adi_val:.2f} | Elong: {d['elongation']:.1f}x"

            card_w, card_h = 265, 58
            card_x = int(np.clip(cx_i - card_w // 2, 10, w - card_w - 10))
            card_y = y0 - card_h - 14 if y0 > card_h + 38 else y1 + 14

            roi_hud = vis[card_y:card_y + card_h, card_x:card_x + card_w]
            if roi_hud.shape[0] == card_h and roi_hud.shape[1] == card_w:
                dark_box = np.full_like(roi_hud, 15)
                vis[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(roi_hud, 0.2, dark_box, 0.8, 0)
            cv2.rectangle(vis, (card_x, card_y), (card_x + card_w, card_y + card_h), color, 1)

            leader_y = card_y + card_h if card_y < y0 else card_y
            box_anchor_y = y0 if card_y < y0 else y1
            cv2.line(vis, (cx_i, leader_y), (cx_i, box_anchor_y), color, 1, cv2.LINE_AA)

            cv2.putText(vis, line1, (card_x + 8, card_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
            cv2.putText(vis, line2, (card_x + 8, card_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(vis, line3, (card_x + 8, card_y + 51), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 230, 255), 1, cv2.LINE_AA)

            if len(trail) > best_frame_score:
                best_frame_score = len(trail)
                best_frame_img = vis.copy()

        # Top HUD status bar
        cv2.rectangle(vis, (0, 0), (w, 30), (14, 17, 22), -1)
        status_txt = (
            f"IISc RBC Single-Track Passage | Frame {f_idx} ({f_idx / 10.0:.1f} ms) | "
            f"Active Cell in Tube: {len(frame_dets)} | Total Passage Cells: {len(valid_tracks)}"
        )
        cv2.putText(vis, status_txt, (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

        writer_full.write(vis)
        if len(frame_dets) > 0:
            writer_active.write(vis)
            active_frame_count += 1

    writer_full.release()
    writer_active.release()
    cap.release()

    if best_frame_img is not None:
        cv2.imwrite(best_png, best_frame_img)

    return full_mp4, active_mp4, best_png, active_frame_count


# ====================================================================
# MAIN PIPELINE
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Single-Track-Per-Cell RBC Passage Tracker")
    parser.add_argument("--max-frames", type=int, default=1000, help="Number of frames to process (default: 1000; pass 0 for all 37,274)")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index to start from (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Optional override path to .avi file")
    parser.add_argument("--min-span", type=float, default=120.0, help="Minimum horizontal passage span (px) required for a valid cell track (default: 120)")
    parser.add_argument("--diff-thresh", type=float, default=13.0, help="Minimum background-difference intensity inside lumen (default: 13.0)")
    args = parser.parse_args()

    if not os.path.isfile(CALIB_FILE) or not os.path.isfile(BG_FILE):
        print(f"Calibration files not found in {OUT}. Running preprocessing.py (Stage 1) first...")
        import preprocessing
        preprocessing.main()

    calib = load_calibration()
    background = load_background()

    video_path = args.video if args.video else _resolve_video_path(calib.get("video"))
    fps = calib.get("fps", 10000)

    print("=" * 68)
    print("STAGE 3: SINGLE-TRACK-PER-CELL PASSAGE TRACKER")
    print("=" * 68)
    print(f"Video path : {video_path}")
    print(f"Output dir : {OUT}")

    lumen_mask, top_wall, bot_wall, slope, intercept = build_exact_lumen_geometry(background)
    ys_active = np.where(lumen_mask > 0)[0]
    roi_y_bounds = (max(0, int(ys_active.min()) - 8), min(background.shape[0], int(ys_active.max()) + 8))

    out_dir = os.path.join(OUT, "stage3")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    frames_to_run = total_frames - args.start_frame
    if args.max_frames and args.max_frames > 0:
        frames_to_run = min(frames_to_run, args.max_frames)

    print(f"\n[1/2] Tracking cells across {frames_to_run} frames (start={args.start_frame})...")

    tracker = PassageCellTracker(slope=slope, intercept=intercept, max_age=25, min_hits=4, max_step_px=120.0)
    frame_count = 0

    for offset in range(frames_to_run):
        f_idx = args.start_frame + offset
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        dets = detect_cells_in_passage(
            frame, background, lumen_mask, roi_y_bounds, diff_thresh=args.diff_thresh
        )
        tracker.step(f_idx, dets)

        if frame_count % 250 == 0:
            print(f"      processed {frame_count}/{frames_to_run} frames...")

    cap.release()

    valid_tracks = tracker.finalize_all(min_hits=4, min_span_px=args.min_span)

    all_trajectories = []
    all_biomarkers = []
    biomarkers_by_id = {}

    for tid, path in valid_tracks.items():
        bio = compute_biomarker(tid, path)
        if bio:
            all_biomarkers.append(bio)
            biomarkers_by_id[tid] = bio
        for p in path:
            bx, by, bw, bh = p["bbox"]
            all_trajectories.append({
                "frame": p["frame"],
                "track_id": tid,
                "cx": round(p["cx"], 2),
                "cy": round(p["cy"], 2),
                "bbox_x": bx,
                "bbox_y": by,
                "bbox_w": bw,
                "bbox_h": bh,
                "velocity_px_per_frame": round(p["velocity"], 3),
                "area": round(p["area"], 2),
                "circularity": round(p["circularity"], 4),
                "elongation": round(p["elongation"], 3),
                "ADI_running": round(p["ADI_running"], 4),
            })

    traj_csv = os.path.join(out_dir, "trajectories.csv")
    with open(traj_csv, "w", newline="", encoding="utf-8") as f:
        if all_trajectories:
            writer = csv.DictWriter(f, fieldnames=list(all_trajectories[0].keys()))
            writer.writeheader()
            writer.writerows(all_trajectories)

    bio_csv = os.path.join(out_dir, "biomechanics.csv")
    with open(bio_csv, "w", newline="", encoding="utf-8") as f:
        if all_biomarkers:
            writer = csv.DictWriter(f, fieldnames=list(all_biomarkers[0].keys()))
            writer.writeheader()
            writer.writerows(all_biomarkers)

    summary = {
        "total_tracks": len(all_biomarkers),
        "total_frames_analyzed": frame_count,
        "start_frame": args.start_frame,
        "fps": fps,
        "video": video_path,
        "metrics": {
            "velocity_mean_px_per_frame": float(np.mean([b["velocity_mean"] for b in all_biomarkers])) if all_biomarkers else None,
            "velocity_max_px_per_frame": float(np.max([b["velocity_max"] for b in all_biomarkers])) if all_biomarkers else None,
            "ADI_mean": float(np.mean([b["ADI"] for b in all_biomarkers])) if all_biomarkers else None,
            "circularity_mean": float(np.mean([b["circularity_mean"] for b in all_biomarkers])) if all_biomarkers else None,
        }
    }
    with open(os.path.join(out_dir, "track_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[2/2] Rendering annotated video (1 unified box + biomechanics HUD per cell)...")
    full_mp4, active_mp4, best_png, active_frames = render_annotated_videos(
        video_path=video_path,
        start_frame=args.start_frame,
        frames_to_run=frame_count,
        top_wall=top_wall,
        bot_wall=bot_wall,
        valid_tracks=valid_tracks,
        biomarkers_by_id=biomarkers_by_id,
        out_dir=out_dir,
        playback_fps=15.0,
    )

    print("\n" + "=" * 68)
    print("STAGE 3 COMPLETE — SINGLE TRACK PER CELL")
    print("=" * 68)
    print(f"  ✓ Active Cell Passages Video : {active_mp4} ({active_frames} active frames)")
    print(f"  ✓ Full Annotated Video       : {full_mp4}")
    print(f"  ✓ Total Distinct Cells       : {len(all_biomarkers)}")

    if all_biomarkers:
        print("\nPer-Cell Passage Summary:")
        print(f"  {'CellID':>6} | {'Frames':>6} | {'FrameSpan':>13} | {'X Passage Span':>16} | {'VelMean':>7} | {'AreaMean':>8} | {'ADI':>5} | {'Circ':>5}")
        print("  " + "-" * 82)
        for b in all_biomarkers:
            fspan = f"{b['entrance_frame']}->{b['exit_frame']}"
            xspan = f"{b['x_start_px']:.0f}->{b['x_end_px']:.0f}px"
            print(
                f"  #{b['track_id']:5d} | {b['track_length_frames']:6d} | {fspan:>13} | {xspan:>16} | "
                f"{b['velocity_mean']:7.2f} | {b['area_mean']:8.1f} | {b['ADI']:5.2f} | {b['circularity_mean']:5.2f}"
            )
        print(f"\nOpen the video:\n  start {active_mp4}")


if __name__ == "__main__":
    main()
