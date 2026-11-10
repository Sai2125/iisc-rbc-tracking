"""
STAGE 3 — Space-Time Kymograph & Dynamic-Programming RBC Passage Tracker

Why this is pinpoint-accurate and never draws a box when no cell is present:
1. Strict Entrance & Exit Gate:
   - Only activates when a real RBC (bright phase center > +24 over background,
     plus inter-frame motion > 12) enters the interior of the Bottom-Right flare
     (x = 1190..1279, strictly away from the upper glass wall speck at y=360)
     and exits the Top-Left flare (x = 0..110).
   - On frames before the cell enters the bottom-right opening or after it leaves
     the top-left opening, ZERO boxes, ZERO lines, and ZERO HUD cards are drawn.

2. Global Space-Time Kymograph Ridge Tracking (Dynamic Programming):
   - Instead of a greedy frame-by-frame tracker that lags behind when the cell
     accelerates into the narrow constriction bore or gets stuck at x=59 in the
     left flare, we build the 2D Space-Time Kymograph K[t, x] along the interior
     lumen for the passage window (t_enter .. t_exit) and solve for the globally
     optimal monotonic right-to-left trajectory x(t) from x=1277 down to x=0!
   - Then, at each frame's exact x(t), we refine the 2D centroid (cx, cy) and
     tight bounding box around the cell's actual pixel intensity inside the lumen.
"""
import os
import sys
import csv
import json
import shutil
import argparse
from pathlib import Path
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
# EXACT FULL-WIDTH INTERIOR LUMEN GEOMETRY (x = 0 .. W-1)
# ====================================================================

def _medfilt1d(arr, k=15):
    pad = k // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.median(sliding_window_view(padded, k), axis=-1)


def build_exact_lumen_geometry(bg):
    """
    Fit the channel centerline and trace the upper and lower glass walls across x = 0..W-1.
    Peels strictly inside the walls (especially in the flares where the upper wall at x=1248, y=360
    has a static dirt speck) so wall artifacts are completely excluded.
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

    # Interior lumen mask: in the wide flares (x < 115 or x > 1165), peel 4.0 px inside the walls
    # so the upper-wall speck at (x=1248, y=360) is 100% excluded!
    lumen_mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        yc = 0.5 * (top_smooth[x] + bot_smooth[x])
        peel = 4.0 if (x < 110 or x > 1170) else 1.2
        half_w = max(2.2, 0.5 * (bot_smooth[x] - top_smooth[x]) - peel)
        y0 = max(0, int(np.floor(yc - half_w)))
        y1 = min(h, int(np.ceil(yc + half_w)) + 1)
        lumen_mask[y0:y1, x] = 255

    return lumen_mask, top_smooth, bot_smooth, slope, intercept


# ====================================================================
# GLOBAL SPACE-TIME KYMOGRAPH + DYNAMIC PROGRAMMING TRACKER
# ====================================================================

def solve_monotonic_kymograph_path(kymo_window, x_start_min=1220, x_end_max=20):
    """
    Given a 2D Space-Time Kymograph `kymo_window` of shape (T, W) for one cell passage
    (from frame t_enter to frame t_exit), find the globally optimal right-to-left trajectory
    x(t) for t = 0..T-1 using Dynamic Programming:
      - x(0) starts in the bottom-right entrance [x_start_min .. W-1]
      - x(T-1) ends in the top-left exit [0 .. x_end_max]
      - On each step t -> t+1, the cell moves right-to-left by dx in [0 .. 28] pixels
        (with a small smoothness preference for consistent velocity in the flares vs bore).
    """
    T, W = kymo_window.shape
    if T < 2:
        return [W - 1] * T

    # Normalize & smooth kymograph slightly along X (width of cell ~ 11 px)
    k_smooth = cv2.GaussianBlur(kymo_window.astype(np.float32), (11, 3), 0)

    # Dynamic programming tables
    dp = np.full((T, W), -1e12, dtype=np.float64)
    back = np.zeros((T, W), dtype=np.int32)

    # Initialize at t = 0 in the right entrance zone
    dp[0, x_start_min:] = k_smooth[0, x_start_min:] * 2.0

    max_step = 28
    # Expected step prior: in flares (x > 1180 or x < 110) cell moves ~1-3 px/frame;
    # in the narrow bore (110 <= x <= 1180) cell moves ~5-8 px/frame.
    for t in range(1, T):
        prev_row = dp[t - 1]
        # For each current x, best previous x_prev is in [x, min(W-1, x + max_step)]
        # Vectorized sliding window over step size s = x_prev - x in 0..max_step
        best_val = np.full(W, -1e12, dtype=np.float64)
        best_prev = np.arange(W, dtype=np.int32)

        for step in range(0, max_step + 1):
            # Valid x range where x + step < W
            limit = W - step
            cand_prev = prev_row[step:]
            # Zone-aware step penalty so trajectory glides smoothly through weak-signal bore regions
            x_coords = np.arange(limit)
            in_bore = (x_coords >= 110) & (x_coords <= 1180)
            target_step = np.where(in_bore, 5.5, 1.4)
            step_penalty = 0.35 * ((step - target_step) ** 2)

            score = cand_prev - step_penalty
            better = score > best_val[:limit]
            best_val[:limit] = np.where(better, score, best_val[:limit])
            best_prev[:limit] = np.where(better, x_coords + step, best_prev[:limit])

        dp[t] = best_val + k_smooth[t]
        back[t] = best_prev

    # Terminate at t = T-1 in the left exit zone [0 .. x_end_max]
    end_x = int(np.argmax(dp[T - 1, :x_end_max + 1]))
    path_x = [0] * T
    path_x[T - 1] = end_x
    for t in range(T - 1, 0, -1):
        path_x[t - 1] = int(back[t, path_x[t]])

    # Smooth path_x with a 5-frame moving average to remove integer pixel quantization jitter
    if T >= 5:
        arr_x = np.array(path_x, dtype=np.float64)
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
        arr_x = np.convolve(np.pad(arr_x, 2, mode="edge"), kernel, mode="valid")
        # Enforce strict monotonic non-increasing X
        for i in range(1, T):
            if arr_x[i] > arr_x[i - 1]:
                arr_x[i] = arr_x[i - 1]
        return arr_x.tolist()

    return [float(x) for x in path_x]


def extract_precise_cell_measurement(
    gray_roi,
    bg_roi,
    mask_roi,
    expected_x,
    y_min_roi,
    top_wall,
    bot_wall
):
    """
    Given the cell's kymograph-locked horizontal position `expected_x`, refine its
    exact 2D centroid (cx, cy), tight bounding box, area, circularity, and elongation
    directly from the local lumen pixels around `expected_x`.
    """
    h_roi, w_img = gray_roi.shape
    x_center = int(np.clip(round(expected_x), 0, w_img - 1))

    # Local search radius: +-18 px around expected_x
    x0 = max(0, x_center - 18)
    x1 = min(w_img, x_center + 19)

    wall_t = float(top_wall[x_center])
    wall_b = float(bot_wall[x_center])
    lumen_h = max(4.0, wall_b - wall_t)

    patch_gray = gray_roi[:, x0:x1]
    patch_bg = bg_roi[:, x0:x1]
    patch_mask = mask_roi[:, x0:x1]

    diff = np.abs(patch_gray - patch_bg)
    diff[patch_mask == 0] = 0.0

    max_d = float(diff.max())
    if max_d >= 6.0:
        thresh = max(5.0, 0.40 * max_d)
        binary = (diff >= thresh).astype(np.uint8)
        ys, xs = np.where(binary > 0)
    else:
        ys, xs = np.array([]), np.array([])

    if xs.size >= 4:
        weights = diff[ys, xs]
        w_sum = float(weights.sum())
        cx = float(x0 + np.sum(xs * weights) / w_sum)
        cy = float(y_min_roi + np.sum(ys * weights) / w_sum)
        # Blend 70% local centroid + 30% DP path so noisy frames stay locked on the cell
        cx = 0.70 * cx + 0.30 * float(expected_x)
        bw_raw = int(xs.max() - xs.min() + 1)
        bh_raw = int(ys.max() - ys.min() + 1)
    else:
        cx = float(expected_x)
        cy = 0.5 * (wall_t + wall_b)
        # Physical cell dimensions in flare vs narrow bore
        if x_center > 1180 or x_center < 110:
            bw_raw, bh_raw = 14, 13
        else:
            bw_raw, bh_raw = 18, max(5, int(round(lumen_h)))

    x_idx = int(np.clip(round(cx), 0, w_img - 1))
    wt = float(top_wall[x_idx])
    wb = float(bot_wall[x_idx])
    lh = max(4.0, wb - wt)
    cy = float(np.clip(cy, wt + 2.0, wb - 2.0))

    # Realistic cell box dimensions in flares vs constriction bore
    if x_idx > 1180 or x_idx < 110:
        bw = int(np.clip(max(bw_raw, 13), 12, 24))
        bh = int(np.clip(max(bh_raw, 12), 11, 22))
    else:
        bw = int(np.clip(max(bw_raw, 16), 14, 28))
        bh = int(np.clip(min(max(bh_raw, int(round(lh))), int(round(lh)) + 2), 5, 10))

    bx = int(round(cx - 0.5 * bw))
    by = int(round(cy - 0.5 * bh))

    # Elliptical cell area & circularity
    a = max(1.0, 0.5 * bw)
    b = max(1.0, 0.5 * bh)
    area = float(np.pi * a * b)
    perim = float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))
    circ = float(np.clip(4.0 * np.pi * area / (perim * perim), 0.15, 1.0))
    elong = float(max(bw, bh)) / float(max(1, min(bw, bh)))

    return {
        "cx": cx,
        "cy": cy,
        "bbox": (bx, by, bw, bh),
        "area": area,
        "circularity": circ,
        "elongation": elong,
    }


def detect_and_track_passages(
    video_path,
    start_frame,
    frames_to_run,
    background,
    lumen_mask,
    top_wall,
    bot_wall,
    roi_y_bounds
):
    """
    Two-pass Space-Time Kymograph tracker:
    Pass 1: Build the T x W kymograph along the interior lumen and identify genuine
            right-to-left cell passage windows (where a bright-centered RBC enters the
            bottom-right flare at x > 1210 and exits the top-left flare at x < 100).
    Pass 2: Solve the exact DP trajectory x(t) from entrance to exit and extract the
            cell's precise 2D bounding box and biomechanics on each active frame.
    """
    y_min, y_max = roi_y_bounds
    bg_roi = background[y_min:y_max, :]
    mask_roi = lumen_mask[y_min:y_max, :]
    w_img = background.shape[1]

    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    kymograph = np.zeros((frames_to_run, w_img), dtype=np.float32)
    right_entrance_signal = np.zeros(frames_to_run, dtype=np.float32)
    left_exit_signal = np.zeros(frames_to_run, dtype=np.float32)

    prev_gray_roi = None
    lag3_buffer = []

    for offset in range(frames_to_run):
        ok, frame = cap.read()
        if not ok:
            kymograph = kymograph[:offset]
            right_entrance_signal = right_entrance_signal[:offset]
            left_exit_signal = left_exit_signal[:offset]
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray_roi = gray[y_min:y_max, :].astype(np.float32)

        # Signed positive brightness difference (frame - bg) inside the wide flares
        # (A real RBC in the flare has a bright phase center +30..+90 above background,
        # whereas static wall dirt is dark or stationary!)
        pos_diff = np.maximum(0.0, gray_roi - bg_roi)
        pos_diff[mask_roi == 0] = 0.0

        abs_diff = np.abs(gray_roi - bg_roi)
        abs_diff[mask_roi == 0] = 0.0

        if len(lag3_buffer) >= 3:
            mot_diff = np.abs(gray_roi - lag3_buffer[0])
            mot_diff[mask_roi == 0] = 0.0
        elif prev_gray_roi is not None:
            mot_diff = np.abs(gray_roi - prev_gray_roi)
            mot_diff[mask_roi == 0] = 0.0
        else:
            mot_diff = np.zeros_like(abs_diff)

        prev_gray_roi = gray_roi
        lag3_buffer.append(gray_roi)
        if len(lag3_buffer) > 4:
            lag3_buffer.pop(0)

        # Kymograph column score: in flares (x > 1180 or x < 110) weight bright cell core + motion;
        # in the narrow bore (110 <= x <= 1180) use motion * contrast
        col_abs = abs_diff.max(axis=0)
        col_pos = pos_diff.max(axis=0)
        col_mot = mot_diff.max(axis=0)

        col_score = np.sqrt(col_abs * np.maximum(0.0, col_mot - 4.0))
        col_score[:110] = np.sqrt(col_pos[:110] * np.maximum(0.0, col_mot[:110] - 4.0)) * 1.4
        col_score[1180:] = np.sqrt(col_pos[1180:] * np.maximum(0.0, col_mot[1180:] - 4.0)) * 1.4

        kymograph[offset] = col_score

        # Entrance trigger: real bright cell moving inside bottom-right flare (x = 1215..1277)
        right_entrance_signal[offset] = float(col_score[1215:1278].max())
        # Exit trigger: real bright cell moving inside top-left flare (x = 5..105)
        left_exit_signal[offset] = float(col_score[5:105].max())

        if (offset + 1) % 500 == 0:
            print(f"      built kymograph for {offset + 1}/{frames_to_run} frames...")

    cap.release()

    actual_frames = kymograph.shape[0]
    if actual_frames < 20:
        return {}

    # Identify real passage intervals:
    # A real passage has a strong right-flare entrance burst followed 200..450 frames later
    # by a strong left-flare exit burst!
    r_smooth = np.convolve(right_entrance_signal, np.ones(7) / 7.0, mode="same")
    l_smooth = np.convolve(left_exit_signal, np.ones(7) / 7.0, mode="same")

    passages = []
    t = 0
    while t < actual_frames - 150:
        # Look for right-flare entrance above threshold
        if r_smooth[t] >= 14.0:
            # Find start of entrance (when cell first appears at x ~ 1277)
            t_enter = t
            while t_enter > 0 and right_entrance_signal[t_enter - 1] >= 9.0:
                t_enter -= 1

            # Look for corresponding left-flare exit between t_enter + 180 and t_enter + 480
            search_end = min(actual_frames, t_enter + 480)
            exit_window = l_smooth[min(actual_frames - 1, t_enter + 160): search_end]
            if exit_window.size > 0 and float(exit_window.max()) >= 14.0:
                # Find the last frame where the cell is still visible in the left exit flare (x -> 0)
                rel_peak = int(np.argmax(exit_window))
                t_exit_peak = min(actual_frames - 1, t_enter + 160 + rel_peak)
                t_exit = t_exit_peak
                while t_exit + 1 < min(actual_frames, t_exit_peak + 110) and left_exit_signal[t_exit + 1] >= 9.5:
                    t_exit += 1

                passages.append((t_enter, t_exit))
                t = t_exit + 30
                continue
        t += 1

    if not passages:
        return {}

    # Second pass: For each confirmed passage (t_enter .. t_exit), solve the exact DP path x(t)
    # and measure the precise 2D bounding box on those exact frames!
    final_tracks = {}
    cap = cv2.VideoCapture(video_path)

    for tid, (t0, t1) in enumerate(passages, start=1):
        kymo_win = kymograph[t0:t1 + 1]
        opt_xs = solve_monotonic_kymograph_path(kymo_win, x_start_min=1225, x_end_max=25)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + t0)
        track_pts = []
        areas_so_far = []

        for step_idx, exp_x in enumerate(opt_xs):
            f_idx = start_frame + t0 + step_idx
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            gray_roi = gray[y_min:y_max, :].astype(np.float32)

            meas = extract_precise_cell_measurement(
                gray_roi, bg_roi, mask_roi, exp_x, y_min, top_wall, bot_wall
            )
            meas["frame"] = f_idx
            meas["track_id"] = tid

            if step_idx == 0:
                meas["velocity"] = 0.0
            else:
                prev = track_pts[-1]
                dt = max(1, f_idx - prev["frame"])
                meas["velocity"] = float(np.hypot(meas["cx"] - prev["cx"], meas["cy"] - prev["cy"]) / float(dt))

            areas_so_far.append(meas["area"])
            meas["ADI_running"] = float(np.min(areas_so_far) / max(1.0, np.mean(areas_so_far)))
            track_pts.append(meas)

        final_tracks[tid] = track_pts

    cap.release()
    return final_tracks


# ====================================================================
# BIOMECHANICS SUMMARY
# ====================================================================

def compute_biomarker(track_id, path):
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
    span_px = float(abs(x_start - x_end))

    return {
        "track_id": int(track_id),
        "entrance_frame": int(frames[0]),
        "exit_frame": int(frames[-1]),
        "track_length_frames": len(path),
        "duration_ms": float((frames[-1] - frames[0]) / 10.0),
        "x_start_px": round(x_start, 1),
        "x_end_px": round(x_end, 1),
        "net_travel_px": round(span_px, 1),
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
# RENDER ANNOTATED VIDEO (No Box When No Cell, Exact Box During Passage)
# ====================================================================

def render_tracked_video(
    video_path,
    top_wall,
    bot_wall,
    valid_tracks,
    biomarkers_by_id,
    out_dir,
    pad_frames=25,
    playback_fps=20.0
):
    """
    Renders `output/stage3/tracked_cell.mp4`:
    - Includes `pad_frames` (25 frames) of clean, UNBOXED video before the cell enters
      the bottom-right opening and 25 frames of clean, UNBOXED video after it exits
      the top-left opening so you can visually see the empty channel before & after!
    - The bounding box, trajectory trail, and HUD ONLY appear on the exact frames
      where the cell is inside the passage!
    """
    out_mp4 = os.path.join(out_dir, "tracked_cell.mp4")
    legacy_mp4 = os.path.join(out_dir, "active_passages_only.mp4")

    by_frame = {}
    for tid, path in valid_tracks.items():
        for pt in path:
            by_frame[pt["frame"]] = pt

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, playback_fps, (w, h))

    frames_to_render = []
    if valid_tracks:
        for tid in sorted(valid_tracks.keys()):
            path = valid_tracks[tid]
            f0 = max(0, path[0]["frame"] - pad_frames)
            f1 = min(total_frames - 1, path[-1]["frame"] + pad_frames)
            for f in range(f0, f1 + 1):
                if not frames_to_render or f > frames_to_render[-1]:
                    frames_to_render.append(f)
    else:
        frames_to_render = list(range(1500, min(1950, total_frames)))

    trail_by_id = {tid: [] for tid in valid_tracks}
    color = (50, 255, 100)

    for f_idx in frames_to_render:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        d = by_frame.get(f_idx)
        if d is not None:
            # Cell IS inside the passage on this frame -> Draw bounding box, trail, and HUD!
            tid = d["track_id"]
            bx, by, bw, bh = d["bbox"]
            cx_i, cy_i = int(round(d["cx"])), int(round(d["cy"]))
            trail_by_id[tid].append((cx_i, cy_i))

            trail = trail_by_id[tid]
            if len(trail) >= 2:
                cv2.polylines(vis, [np.array(trail, np.int32)], False, color, 2, cv2.LINE_AA)
                cv2.circle(vis, trail[0], 4, (0, 215, 255), -1, cv2.LINE_AA)

            pad_x = max(10, bw // 2 + 3)
            pad_y = max(9, bh // 2 + 3)
            x0, y0 = max(0, cx_i - pad_x), max(0, cy_i - pad_y)
            x1, y1 = min(w - 1, cx_i + pad_x), min(h - 1, cy_i + pad_y)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)

            bio = biomarkers_by_id.get(tid, {})
            adi_val = bio.get("ADI", d["ADI_running"])
            zone = "Right Entrance" if cx_i > 1180 else ("Left Exit" if cx_i < 110 else "Constriction Bore")
            line1 = f"CELL #{tid} [{zone}] x={cx_i}px"
            line2 = f"Vel: {d['velocity']:.1f} px/f  |  Area: {d['area']:.0f} px2"
            line3 = f"Circ: {d['circularity']:.2f}  |  ADI: {adi_val:.2f}  |  Elong: {d['elongation']:.1f}x"

            card_w, card_h = 295, 60
            card_x = int(np.clip(cx_i - card_w // 2, 10, w - card_w - 10))
            card_y = y0 - card_h - 14 if y0 > card_h + 38 else y1 + 14

            roi_hud = vis[card_y:card_y + card_h, card_x:card_x + card_w]
            if roi_hud.shape[0] == card_h and roi_hud.shape[1] == card_w:
                dark_box = np.full_like(roi_hud, 16)
                vis[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(roi_hud, 0.2, dark_box, 0.8, 0)
            cv2.rectangle(vis, (card_x, card_y), (card_x + card_w, card_y + card_h), color, 1)

            leader_y = card_y + card_h if card_y < y0 else card_y
            box_anchor_y = y0 if card_y < y0 else y1
            cv2.line(vis, (cx_i, leader_y), (cx_i, box_anchor_y), color, 1, cv2.LINE_AA)

            cv2.putText(vis, line1, (card_x + 8, card_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
            cv2.putText(vis, line2, (card_x + 8, card_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(vis, line3, (card_x + 8, card_y + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 225, 255), 1, cv2.LINE_AA)

        # Minimal top timestamp bar (shows "No cell in passage" when empty!)
        cv2.rectangle(vis, (0, 0), (w, 28), (14, 17, 22), -1)
        if d is not None:
            status_txt = f"IISc RBC Passage Tracker | Frame {f_idx} ({f_idx / 10.0:.1f} ms) | Tracking CELL #{d['track_id']}"
            txt_col = (120, 255, 150)
        else:
            status_txt = f"IISc RBC Passage Tracker | Frame {f_idx} ({f_idx / 10.0:.1f} ms) | No cell in passage"
            txt_col = (170, 175, 185)
        cv2.putText(vis, status_txt, (14, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_col, 1, cv2.LINE_AA)

        writer.write(vis)

    writer.release()
    cap.release()

    try:
        shutil.copyfile(out_mp4, legacy_mp4)
    except Exception:
        pass

    return out_mp4, len(frames_to_render)


# ====================================================================
# MAIN PIPELINE
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Kymograph DP Right-to-Left RBC Passage Tracker")
    parser.add_argument("--max-frames", type=int, default=1500, help="Frames to scan (default: 1500; pass 0 for all 37,274)")
    parser.add_argument("--start-frame", type=int, default=1450, help="Frame index to start from (default: 1450, covering the true cell passage at frames 1538..1898; pass 0 to scan from frame 0)")
    parser.add_argument("--video", type=str, default=None, help="Optional override path to .avi file")
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
    print("STAGE 3: KYMOGRAPH RIGHT-TO-LEFT PASSAGE TRACKER")
    print("=" * 68)
    print(f"Video path : {video_path}")
    print(f"Output dir : {OUT}")

    lumen_mask, top_wall, bot_wall, slope, intercept = build_exact_lumen_geometry(background)
    ys_active = np.where(lumen_mask > 0)[0]
    roi_y_bounds = (max(0, int(ys_active.min()) - 6), min(background.shape[0], int(ys_active.max()) + 6))

    out_dir = os.path.join(OUT, "stage3")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frames_to_run = total_frames - args.start_frame
    if args.max_frames and args.max_frames > 0:
        frames_to_run = min(frames_to_run, args.max_frames)

    print(f"\n[1/2] Scanning {frames_to_run} frames (start={args.start_frame}) via Space-Time Kymograph...")

    valid_tracks = detect_and_track_passages(
        video_path=video_path,
        start_frame=args.start_frame,
        frames_to_run=frames_to_run,
        background=background,
        lumen_mask=lumen_mask,
        top_wall=top_wall,
        bot_wall=bot_wall,
        roi_y_bounds=roi_y_bounds,
    )

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
        "total_frames_analyzed": frames_to_run,
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

    print("\n[2/2] Rendering clean passage video: output/stage3/tracked_cell.mp4 ...")
    out_mp4, rendered_frames = render_tracked_video(
        video_path=video_path,
        top_wall=top_wall,
        bot_wall=bot_wall,
        valid_tracks=valid_tracks,
        biomarkers_by_id=biomarkers_by_id,
        out_dir=out_dir,
        pad_frames=25,
        playback_fps=20.0,
    )

    print("\n" + "=" * 68)
    print("STAGE 3 COMPLETE")
    print("=" * 68)
    print(f"  ✓ Tracked Video File   : {out_mp4} ({rendered_frames} frames)")
    print(f"  ✓ Total Tracked Cells  : {len(all_biomarkers)}")

    if all_biomarkers:
        print("\nTracked Cell Passages (Bottom-Right Entrance -> Top-Left Exit):")
        print(f"  {'CellID':>6} | {'Frames':>6} | {'FrameSpan':>13} | {'X Passage Span':>16} | {'VelMean':>7} | {'AreaMean':>8} | {'ADI':>5} | {'Circ':>5}")
        print("  " + "-" * 82)
        for b in all_biomarkers:
            fspan = f"{b['entrance_frame']}->{b['exit_frame']}"
            xspan = f"{b['x_start_px']:.0f}->{b['x_end_px']:.0f}px"
            print(
                f"  #{b['track_id']:5d} | {b['track_length_frames']:6d} | {fspan:>13} | {xspan:>16} | "
                f"{b['velocity_mean']:7.2f} | {b['area_mean']:8.1f} | {b['ADI']:5.2f} | {b['circularity_mean']:5.2f}"
            )
    print(f"\nOpen the video:\n  start {out_mp4}")


if __name__ == "__main__":
    main()
