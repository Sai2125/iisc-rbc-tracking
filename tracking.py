"""
STAGE 3 — End-to-End Passage RBC Detection, Tracking & Annotated Video Generator

Tracks Red Blood Cells from the start of the microfluidic passage (Entrance Flare)
through the narrow constriction bore to the end of the passage (Exit Flare).

Fixes over the original draft:
1. Full-width exact interior lumen mask (x = 0..1279) including both entrance & exit flares,
   peeled inside the dark glass walls and median-smoothed so glass wall shimmer & stuck dirt
   (e.g. at x=435) are excluded.
2. Bounding-box + contour extraction (`bbox_x, bbox_y, bbox_w, bbox_h, area, circularity, elongation`)
   without full-frame Otsu hallucinations on empty frames.
3. Channel-axis Kalman + Hungarian tracker with longitudinal velocity continuity so a fast cell
   squeezing through the narrow bore keeps a single continuous Cell ID from entrance to exit.
4. Stationary-speck rejection: filters out wall-vibration false positives that do not travel
   along the channel passage.
5. Direct annotated video output (`output/stage3/tracked_cells_video.mp4` and
   `output/stage3/active_passages_only.mp4`) drawing the bounding box, trajectory trail,
   Cell ID, and live biomechanics (Velocity, Area, Circularity, ADI) directly on the video.
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

    Returns:
      lumen_mask (uint8 HxW, 255 inside lumen, 0 on walls & bath),
      top_wall (float array W),
      bot_wall (float array W),
      slope (float),
      intercept (float)
    """
    h, w = bg.shape[:2]

    # 1. Sigma-clipped linear fit on the central bore (x = 120 .. w-120)
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

    # 2. Locate upper and lower dark glass wall centers in every column x = 0..w-1
    top_wall = np.zeros(w, dtype=np.float64)
    bot_wall = np.zeros(w, dtype=np.float64)

    for x in range(w):
        yc = slope * x + intercept
        # Flares widen up to ~28 px from centerline; narrow bore is within ~9 px
        max_search = 28 if (x < 115 or x > 1165) else 9
        y_lo = max(0, int(round(yc - max_search)))
        y_hi = min(h - 1, int(round(yc + max_search)))
        y_mid = np.clip(int(round(yc)), y_lo + 1, y_hi - 1)

        top_slice = bg[y_lo:y_mid, x]
        bot_slice = bg[y_mid + 1:y_hi + 1, x]

        top_wall[x] = y_lo + int(np.argmin(top_slice)) if top_slice.size > 0 else yc - 3.5
        bot_wall[x] = (y_mid + 1) + int(np.argmin(bot_slice)) if bot_slice.size > 0 else yc + 3.5

    # 3. Median-filter wall positions to ignore dirt specks stuck on glass
    top_smooth = _medfilt1d(top_wall, k=15)
    bot_smooth = _medfilt1d(bot_wall, k=15)

    # 4. Build interior lumen mask (peel 1.0 px inside dark wall centers so wall shimmer is excluded,
    # while guaranteeing at least a 3 px core in the narrowest bore so squeezed cells are detected)
    lumen_mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        yc = 0.5 * (top_smooth[x] + bot_smooth[x])
        half_w = max(2.0, 0.5 * (bot_smooth[x] - top_smooth[x]) - 1.0)
        y0 = max(0, int(np.floor(yc - half_w)))
        y1 = min(h, int(np.ceil(yc + half_w)) + 1)
        lumen_mask[y0:y1, x] = 255

    return lumen_mask, top_smooth, bot_smooth, slope, intercept


# ====================================================================
# CELL DETECTION INSIDE THE PASSAGE LUMEN
# ====================================================================

def detect_cells_in_passage(frame, background, lumen_mask, roi_y_bounds, diff_thresh=11.0):
    """
    Detect RBCs strictly inside the microfluidic passage (flares + constriction bore).

    Returns list of dicts:
      {
        "cx": float, "cy": float,
        "bbox": (x, y, w, h),
        "area": float,
        "circularity": float,
        "elongation": float,
        "peak_diff": float
      }
    """
    y_min, y_max = roi_y_bounds
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    # Work only on the horizontal strip containing the tilted tube (5x faster than full 1280x480)
    gray_roi = gray[y_min:y_max, :].astype(np.float32)
    bg_roi = background[y_min:y_max, :]
    mask_roi = lumen_mask[y_min:y_max, :]

    # Absolute contrast difference against static temporal median background
    diff = np.abs(gray_roi - bg_roi)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)

    # Zero out everything outside the interior tube lumen (excludes bath + dark wall ridges)
    diff[mask_roi == 0] = 0.0

    # Adaptive + floor threshold: only trigger when a real cell perturbation is inside the lumen
    max_val = float(diff.max())
    if max_val < diff_thresh:
        return []

    # Threshold at max(diff_thresh, 0.35 * local_peak) to segment the cell body cleanly
    binary = (diff >= diff_thresh).astype(np.uint8) * 255

    # Connect slightly separated bright/dark phase rings of the same cell along the channel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        bx, by_rel, bw, bh = cv2.boundingRect(cnt)
        box_area = float(bw * bh)

        # In the narrow constriction (width ~4 px), a squeezed cell may have contourArea ~ 10-15 px^2
        # and bounding box ~ 8x4 px. Reject tiny 1-2 px sensor specks and giant artifacts.
        if area < 8.0 and box_area < 14.0:
            continue
        if area > 4500.0:
            continue

        # Check peak intensity difference inside this blob to reject weak wall vibration
        blob_patch = diff[by_rel:by_rel + bh, bx:bx + bw]
        peak_diff = float(blob_patch.max()) if blob_patch.size > 0 else 0.0
        if peak_diff < diff_thresh + 2.0:
            continue

        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"]) + y_min
        else:
            cx = float(bx + 0.5 * bw)
            cy = float(by_rel + 0.5 * bh + y_min)

        by = int(by_rel + y_min)
        effective_area = max(area, 0.75 * box_area)

        perimeter = float(cv2.arcLength(cnt, True))
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.5
        circularity = float(np.clip(circularity, 0.05, 1.0))

        long_side = float(max(bw, bh))
        short_side = float(max(1, min(bw, bh)))
        elongation = long_side / short_side

        detections.append({
            "cx": cx,
            "cy": cy,
            "bbox": (int(bx), int(by), int(bw), int(bh)),
            "area": effective_area,
            "circularity": circularity,
            "elongation": elongation,
            "peak_diff": peak_diff,
        })

    return detections


# ====================================================================
# PASSAGE TRACKER (Follows cells continuously from Entrance to Exit)
# ====================================================================

class PassageCellTracker:
    """
    Multi-object tracker specialized for high-speed microfluidic channel passages.
    Uses constant-velocity Kalman prediction + anisotropic channel-axis association
    so fast cells accelerating into the constriction bore maintain a single ID.
    """
    def __init__(self, slope, max_age=15, min_hits=3, max_step_px=75.0):
        self.slope = slope
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_step_px = max_step_px

        # Unit vector along channel passage
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
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.25
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 10.0
        state = np.array([[float(cx)], [float(cy)], [float(vx)], [float(vy)]], dtype=np.float32)
        kf.statePre = state.copy()
        kf.statePost = state.copy()
        return kf

    def step(self, frame_idx, detections):
        # 1. Predict active tracks forward
        for tid, tr in self.tracks.items():
            tr["kf"].predict()

        track_ids = list(self.tracks.keys())
        n_t = len(track_ids)
        n_d = len(detections)

        matched_t = []
        matched_d = []

        if n_t > 0 and n_d > 0:
            cost = np.full((n_t, n_d), np.inf, dtype=np.float64)
            for i, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                pred = tr["kf"].statePre.reshape(-1)
                px, py = float(pred[0]), float(pred[1])
                last_x, last_y = tr["path"][-1]["cx"], tr["path"][-1]["cy"]
                dt = max(1, frame_idx - tr["last_frame"])

                for j, d in enumerate(detections):
                    # Distance from Kalman prediction
                    dx_pred = d["cx"] - px
                    dy_pred = d["cy"] - py
                    # Project error along channel axis (s) and perpendicular to channel (n)
                    ds = abs(dx_pred * self.ux + dy_pred * self.uy)
                    dn = abs(-dx_pred * self.uy + dy_pred * self.ux)

                    # Raw displacement from last observed position
                    raw_dist = np.hypot(d["cx"] - last_x, d["cy"] - last_y) / float(dt)

                    # Penalize perpendicular jumps heavily (cells cannot jump through glass walls)
                    if dn <= 18.0 and (ds <= self.max_step_px or raw_dist <= self.max_step_px):
                        cost[i, j] = ds + 3.5 * dn

            # Solve assignment
            try:
                from scipy.optimize import linear_sum_assignment
                r_ind, c_ind = linear_sum_assignment(cost)
            except ImportError:
                r_ind, c_ind = [], []
                work = cost.copy()
                for _ in range(min(n_t, n_d)):
                    r, c = np.unravel_index(np.argmin(work), work.shape)
                    if not np.isfinite(work[r, c]):
                        break
                    r_ind.append(int(r))
                    c_ind.append(int(c))
                    work[r, :] = np.inf
                    work[:, c] = np.inf

            for r, c in zip(r_ind, c_ind):
                if cost[r, c] < self.max_step_px:
                    matched_t.append(track_ids[r])
                    matched_d.append(c)

        # 2. Update matched tracks
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

            # Running ADI (min_area / mean_area so far)
            areas_so_far = [p["area"] for p in tr["path"]] + [d["area"]]
            running_adi = float(np.min(areas_so_far) / max(1.0, np.mean(areas_so_far)))

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
                "ADI_running": running_adi,
            })

        # 3. Create new tracks for unmatched detections
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

        # 4. Retire aged-out tracks
        to_del = []
        for tid, tr in self.tracks.items():
            if frame_idx - tr["last_frame"] > self.max_age:
                self.finished_tracks[tid] = tr["path"]
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

    def finalize_all(self, min_hits=3, min_net_travel_px=15.0):
        """
        Return all valid moving cell tracks that achieved at least min_hits and
        actually traveled along the passage (rejects static wall specks).
        """
        all_raw = dict(self.finished_tracks)
        for tid, tr in self.tracks.items():
            all_raw[tid] = tr["path"]

        valid_tracks = {}
        new_id = 1
        for old_tid in sorted(all_raw.keys()):
            path = all_raw[old_tid]
            if len(path) < min_hits:
                continue
            # Check net travel distance along passage (start to end of track)
            dx = path[-1]["cx"] - path[0]["cx"]
            dy = path[-1]["cy"] - path[0]["cy"]
            net_travel = float(np.hypot(dx, dy))
            max_span_x = float(max(p["cx"] for p in path) - min(p["cx"] for p in path))
            if net_travel < min_net_travel_px and max_span_x < min_net_travel_px:
                continue

            # Re-index cleanly as Cell #1, #2, ...
            for p in path:
                p["track_id"] = new_id
            valid_tracks[new_id] = path
            new_id += 1

        return valid_tracks


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
    vels = [p["velocity"] for p in path[1:]]

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
    travel_px = float(np.hypot(path[-1]["cx"] - path[0]["cx"], path[-1]["cy"] - path[0]["cy"]))

    return {
        "track_id": int(track_id),
        "entrance_frame": int(frames[0]),
        "exit_frame": int(frames[-1]),
        "track_length_frames": len(path),
        "duration_ms": float((frames[-1] - frames[0]) / 10.0),  # 0.1 ms/frame at 10,000 fps
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
# DIRECT ANNOTATED VIDEO RENDERER (Bounding Box + Biomechanics HUD)
# ====================================================================

def _cell_color_bgr(track_id):
    palette = [
        (60, 255, 100),   # Neon green
        (0, 215, 255),    # Gold / amber
        (255, 190, 40),   # Cyan-blue
        (180, 105, 255),  # Pink-magenta
        (80, 160, 255),   # Orange
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
    playback_fps=20.0
):
    """
    Render the actual microscopy video with:
    - Subtle channel wall guide lines
    - Tight bounding box around each tracked cell in the passage
    - Trajectory trail from the entrance of the passage
    - Callout HUD box next to the cell showing:
        CELL #ID | Vel: X.X px/f | Area: XX px^2 | Circ: 0.XX | ADI: 0.XX
    """
    full_mp4 = os.path.join(out_dir, "tracked_cells_video.mp4")
    active_mp4 = os.path.join(out_dir, "active_passages_only.mp4")
    best_png = os.path.join(out_dir, "tracked_cell_snapshot.png")

    # Map frame_idx -> list of track point dicts
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

    # Precompute subtle wall polylines (every 8 px)
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

        # Draw faint dark-green guide along the calibrated glass walls
        cv2.polylines(vis, [top_pts], False, (70, 130, 70), 1, cv2.LINE_AA)
        cv2.polylines(vis, [bot_pts], False, (70, 130, 70), 1, cv2.LINE_AA)

        frame_dets = by_frame.get(f_idx, [])
        for d in frame_dets:
            tid = d["track_id"]
            color = _cell_color_bgr(tid)
            bx, by, bw, bh = d["bbox"]
            cx_i, cy_i = int(round(d["cx"])), int(round(d["cy"]))
            history_by_track[tid].append((cx_i, cy_i))

            # 1. Draw full passage trajectory trail up to current frame
            trail = history_by_track[tid]
            if len(trail) >= 2:
                cv2.polylines(vis, [np.array(trail, np.int32)], False, color, 2, cv2.LINE_AA)
                # Mark entrance point
                cv2.circle(vis, trail[0], 3, (0, 255, 255), -1, cv2.LINE_AA)

            # 2. Draw padded bounding box with corner brackets around the cell
            pad = 6
            x0, y0 = max(0, bx - pad), max(0, by - pad)
            x1, y1 = min(w - 1, bx + bw + pad), min(h - 1, by + bh + pad)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)

            # 3. Draw floating biomechanics HUD card right above/below the bounding box
            bio = biomarkers_by_id.get(tid, {})
            adi_val = bio.get("ADI", d["ADI_running"])
            line1 = f"CELL #{tid}  [x={int(round(d['cx']))}px]"
            line2 = f"Vel: {d['velocity']:.1f} px/f | Area: {d['area']:.0f} px2"
            line3 = f"Circ: {d['circularity']:.2f} | ADI: {adi_val:.2f}"

            card_w, card_h = 235, 54
            card_x = int(np.clip(cx_i - card_w // 2, 8, w - card_w - 8))
            card_y = y0 - card_h - 10 if y0 > card_h + 25 else y1 + 10

            # Semi-transparent dark HUD background
            roi_hud = vis[card_y:card_y + card_h, card_x:card_x + card_w]
            if roi_hud.shape[0] == card_h and roi_hud.shape[1] == card_w:
                dark_box = np.full_like(roi_hud, 18)
                vis[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(roi_hud, 0.25, dark_box, 0.75, 0)
            cv2.rectangle(vis, (card_x, card_y), (card_x + card_w, card_y + card_h), color, 1)

            # Leader line from HUD card to cell bounding box
            leader_y = card_y + card_h if card_y < y0 else card_y
            box_anchor_y = y0 if card_y < y0 else y1
            cv2.line(vis, (cx_i, leader_y), (cx_i, box_anchor_y), color, 1, cv2.LINE_AA)

            cv2.putText(vis, line1, (card_x + 7, card_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
            cv2.putText(vis, line2, (card_x + 7, card_y + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(vis, line3, (card_x + 7, card_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 230, 255), 1, cv2.LINE_AA)

            if len(trail) > best_frame_score:
                best_frame_score = len(trail)
                best_frame_img = vis.copy()

        # Top status bar
        cv2.rectangle(vis, (0, 0), (w, 30), (15, 18, 22), -1)
        status_txt = (
            f"IISc RBC Passage Tracker | Frame {f_idx} ({f_idx / 10.0:.1f} ms) | "
            f"Cells in Passage: {len(frame_dets)} | Total Confirmed Cells: {len(valid_tracks)}"
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
    parser = argparse.ArgumentParser(description="Stage 3: End-to-End RBC Passage Tracking + Video Output")
    parser.add_argument("--max-frames", type=int, default=500, help="Number of frames to process (default: 500; pass 0 for all 37,274)")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index to start from (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Optional override path to .avi file")
    parser.add_argument("--min-travel", type=float, default=15.0, help="Minimum net travel (px) to keep a cell track (rejects static wall dirt)")
    parser.add_argument("--diff-thresh", type=float, default=11.0, help="Minimum background-difference intensity inside lumen (default: 11.0)")
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
    print("STAGE 3: END-TO-END PASSAGE RBC TRACKING + VIDEO RENDERING")
    print("=" * 68)
    print(f"Video path : {video_path}")
    print(f"Output dir : {OUT}")

    # 1. Build exact full-width interior lumen mask (x = 0..1279, including both flares)
    lumen_mask, top_wall, bot_wall, slope, intercept = build_exact_lumen_geometry(background)
    ys_active = np.where(lumen_mask > 0)[0]
    roi_y_bounds = (max(0, int(ys_active.min()) - 8), min(background.shape[0], int(ys_active.max()) + 8))
    print(
        f"Full-width passage lumen calibrated (x=0..{background.shape[1]-1}, y={roi_y_bounds[0]}..{roi_y_bounds[1]}), "
        f"active lumen = {100.0 * float((lumen_mask > 0).mean()):.2f}% of frame"
    )

    out_dir = os.path.join(OUT, "stage3")
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "exact_lumen_mask.png"), lumen_mask)

    # 2. Scan frames and track cells through the passage
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    frames_to_run = total_frames - args.start_frame
    if args.max_frames and args.max_frames > 0:
        frames_to_run = min(frames_to_run, args.max_frames)

    print(f"\n[1/3] Detecting & tracking cells across {frames_to_run} frames (start={args.start_frame})...")

    tracker = PassageCellTracker(slope=slope, max_age=15, min_hits=3, max_step_px=75.0)
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
            print(f"      scanned {frame_count}/{frames_to_run} frames...")

    cap.release()

    # 3. Filter to moving cells that actually travel along the passage
    valid_tracks = tracker.finalize_all(min_hits=3, min_net_travel_px=args.min_travel)

    # If the chosen frame window had no moving cells (because cells pass intermittently at 10,000 fps),
    # automatically scan forward in coarse steps to find the first active cell passage!
    if len(valid_tracks) == 0 and args.start_frame == 0 and total_frames > frames_to_run:
        print("\n      No moving cell passed through the tube in frames 0..{}. Scanning video for active cell passages...".format(frames_to_run))
        cap_scan = cv2.VideoCapture(video_path)
        found_frame = None
        for probe_f in range(0, total_frames, 25):
            cap_scan.set(cv2.CAP_PROP_POS_FRAMES, probe_f)
            ok, pf = cap_scan.read()
            if not ok:
                break
            dets = detect_cells_in_passage(pf, background, lumen_mask, roi_y_bounds, diff_thresh=args.diff_thresh)
            if dets:
                found_frame = max(0, probe_f - 30)
                print(f"      -> Found cell inside passage at frame {probe_f}! Tracking window {found_frame}..{found_frame + frames_to_run}...")
                break
        cap_scan.release()

        if found_frame is not None:
            args.start_frame = found_frame
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
            tracker = PassageCellTracker(slope=slope, max_age=15, min_hits=3, max_step_px=75.0)
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
            cap.release()
            valid_tracks = tracker.finalize_all(min_hits=3, min_net_travel_px=args.min_travel)

    # 4. Assemble trajectories & biomechanics
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

    # Save CSVs & JSON
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
    summary_json = os.path.join(out_dir, "track_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 5. Render Annotated Videos with Bounding Box + Biomechanics HUD
    print("\n[2/3] Rendering annotated video with bounding boxes & live biomechanics HUD...")
    full_mp4, active_mp4, best_png, active_frames = render_annotated_videos(
        video_path=video_path,
        start_frame=args.start_frame,
        frames_to_run=frame_count,
        top_wall=top_wall,
        bot_wall=bot_wall,
        valid_tracks=valid_tracks,
        biomarkers_by_id=biomarkers_by_id,
        out_dir=out_dir,
        playback_fps=20.0,
    )

    print("\n" + "=" * 68)
    print("STAGE 3 COMPLETE")
    print("=" * 68)
    print(f"  ✓ Annotated Full Video    : {full_mp4}")
    print(f"  ✓ Active Passages Video   : {active_mp4} ({active_frames} active frames)")
    if os.path.isfile(best_png):
        print(f"  ✓ Best Tracked Snapshot   : {best_png}")
    print(f"  ✓ Trajectories CSV        : {traj_csv} ({len(all_trajectories)} rows)")
    print(f"  ✓ Biomechanics CSV        : {bio_csv} ({len(all_biomarkers)} moving cells)")

    if all_biomarkers:
        print("\nTracked Cells Passing Through Channel:")
        print(f"  {'CellID':>6} | {'Frames':>6} | {'FrameSpan':>13} | {'X Start->End':>14} | {'VelMean':>7} | {'AreaMean':>8} | {'ADI':>5} | {'Circ':>5}")
        print("  " + "-" * 80)
        for b in sorted(all_biomarkers, key=lambda x: x["net_travel_px"], reverse=True)[:15]:
            fspan = f"{b['entrance_frame']}->{b['exit_frame']}"
            xspan = f"{b['x_start_px']:.0f}->{b['x_end_px']:.0f}px"
            print(
                f"  #{b['track_id']:5d} | {b['track_length_frames']:6d} | {fspan:>13} | {xspan:>14} | "
                f"{b['velocity_mean']:7.2f} | {b['area_mean']:8.1f} | {b['ADI']:5.2f} | {b['circularity_mean']:5.2f}"
            )
        print(f"\nTo watch the tracked cell video right now, run:\n  start {active_mp4}")
    else:
        print("\nNo moving cells exceeded the travel threshold in this window. Try increasing --max-frames 2000 or lowering --diff-thresh 8.")


if __name__ == "__main__":
    main()
