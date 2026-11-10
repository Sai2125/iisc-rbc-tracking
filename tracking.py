"""
STAGE 3 — Multi-Cell Successive Kymograph Peeling + RTS Kalman Smoother

Fixes both:
1. Back-to-Back / Simultaneous Cells in the Tube (e.g. Cell #2 at Frame 2335 and
   Cell #3 right behind it at Frame 2402):
   - Detects all distinct cell entrance peaks in the bottom-right flare separated by
     >= 45 frames (even when an earlier cell is still inside the bore!).
   - Solves each cell's monotonic right-to-left trajectory x(t) via Dynamic Programming
     and immediately TERMINATES the track the instant the cell reaches the left exit
     boundary (x <= 12 px), then PEELS (zeros out) that cell's ridge from the kymograph
     so a trailing cell (Cell #3) tracks its own ridge cleanly without merging!
   - Renders simultaneous bounding boxes + color-coded HUD cards when Cell #2 and
     Cell #3 are inside the channel at the same time.

2. Clean Exit & Entrance Geometry (No End-of-Track Wiggle):
   - Terminates each track immediately when the cell reaches x <= 12 px at the left
     exit opening (never lingering on empty frames at x = 3..9 px).
   - Smoothly tapers vertical position `cy` to the calibrated channel centerline so
     there are zero vertical hooks or stair-step wiggles at entrance or exit.
"""
import os
import sys
import csv
import json
import shutil
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
# EXACT FULL-WIDTH INTERIOR LUMEN GEOMETRY (x = 0 .. W-1)
# ====================================================================

def _medfilt1d(arr, k=15):
    pad = k // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.median(sliding_window_view(padded, k), axis=-1)


def build_exact_lumen_geometry(bg):
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
        peel = 4.5 if (x < 110 or x > 1170) else 1.2
        half_w = max(2.2, 0.5 * (bot_smooth[x] - top_smooth[x]) - peel)
        y0 = max(0, int(np.floor(yc - half_w)))
        y1 = min(h, int(np.ceil(yc + half_w)) + 1)
        lumen_mask[y0:y1, x] = 255

    return lumen_mask, top_smooth, bot_smooth, slope, intercept


# ====================================================================
# STAGE 3A: SUCCESSIVE KYMOGRAPH DP WITH IMMEDIATE EXIT CUTOFF
# ====================================================================

def find_all_entrance_peaks(r_smooth, r_raw, r_thresh, min_distance=48):
    """
    Find every distinct cell entrance peak in the bottom-right flare, even when two
    cells enter only ~55-75 frames apart (like Cell #2 at Frame 2374 and Cell #3 at Frame 2434).
    """
    T = len(r_smooth)
    candidates = []
    for t in range(2, T - 2):
        if r_smooth[t] >= r_thresh and r_smooth[t] >= r_smooth[t - 1] and r_smooth[t] >= r_smooth[t + 1]:
            candidates.append((float(r_smooth[t]), t))

    # Greedy non-maximum suppression by peak height with min_distance=48 frames
    candidates.sort(reverse=True)
    selected_peaks = []
    for val, t_pk in candidates:
        if all(abs(t_pk - existing) >= min_distance for existing in selected_peaks):
            selected_peaks.append(t_pk)
    selected_peaks.sort()

    # For each entrance peak, walk back at most 28 frames to find the exact entrance frame at x ~ 1272
    entrances = []
    for i, t_pk in enumerate(selected_peaks):
        t_enter = t_pk
        # Do not walk back into a previous cell's entrance
        prev_limit = selected_peaks[i - 1] + 25 if i > 0 else 0
        min_back = max(prev_limit, t_pk - 28)
        while t_enter > min_back and r_raw[t_enter - 1] >= 0.45 * r_thresh:
            t_enter -= 1
        entrances.append((t_enter, t_pk))

    return entrances


def trace_single_cell_dp_ridge(kymo_full, t_enter, l_smooth, l_thresh, x_start_min=1230, x_exit_cutoff=14):
    """
    Given a cell's entrance frame `t_enter`, solve for its individual right-to-left
    trajectory ridge across the kymograph and find its OWN exit frame `t_exit` (rather
    than blindly taking argmax over a 460-frame window, which previously confused
    Cell #2's exit with Cell #3's exit!).

    Once solved, cuts off the trajectory immediately as soon as the cell reaches
    `x <= x_exit_cutoff` (14 px) so the box never lingers or goes haywire at x = 3..9 px!
    """
    T_total, W = kymo_full.shape
    max_duration = min(430, T_total - t_enter)
    if max_duration < 160:
        return None, None

    k_win = kymo_full[t_enter: t_enter + max_duration]
    k_smooth = cv2.GaussianBlur(k_win.astype(np.float32), (9, 3), 0)

    dp = np.full((max_duration, W), -1e12, dtype=np.float64)
    back = np.zeros((max_duration, W), dtype=np.int32)

    # Start in the bottom-right entrance zone [x_start_min .. W-1]
    dp[0, x_start_min:] = k_smooth[0, x_start_min:] * 2.5

    max_step = 24
    for t in range(1, max_duration):
        prev_row = dp[t - 1]
        best_val = np.full(W, -1e12, dtype=np.float64)
        best_prev = np.arange(W, dtype=np.int32)

        for step in range(0, max_step + 1):
            limit = W - step
            cand_prev = prev_row[step:]
            x_coords = np.arange(limit)
            in_bore = (x_coords >= 110) & (x_coords <= 1180)
            target_step = np.where(in_bore, 5.4, 1.35)
            step_penalty = 0.22 * ((step - target_step) ** 2)

            score = cand_prev - step_penalty
            better = score > best_val[:limit]
            best_val[:limit] = np.where(better, score, best_val[:limit])
            best_prev[:limit] = np.where(better, x_coords + step, best_prev[:limit])

        dp[t] = best_val + k_smooth[t]
        back[t] = best_prev

    # Find the earliest valid exit time `t_rel` in [180 .. max_duration-1] where the DP path
    # reaches the left exit zone [0 .. 22] with strong average per-frame score and a local left-flare exit signal
    best_t_rel = None
    best_exit_score = -1e12

    for t_rel in range(185, max_duration):
        f_global = t_enter + t_rel
        # Normalize path score by length so earlier (true) exits aren't beaten by lingering at x=0
        avg_dp_score = float(dp[t_rel, :22].max()) / float(t_rel)
        left_bonus = 0.6 * float(l_smooth[f_global])
        total_score = avg_dp_score + left_bonus
        if l_smooth[f_global] >= 0.65 * l_thresh and total_score > best_exit_score:
            best_exit_score = total_score
            best_t_rel = t_rel
            # Once we pass a clear local peak in l_smooth that starts dropping, stop so we don't jump to the NEXT cell's exit!
            if t_rel > 220 and l_smooth[f_global] < 0.75 * l_smooth[max(t_enter + 185, f_global - 12): f_global + 1].max():
                break

    if best_t_rel is None:
        # Fallback to best normalized score in [200 .. max_duration-1]
        scores = [float(dp[tr, :22].max()) / float(tr) for tr in range(200, max_duration)]
        if not scores:
            return None, None
        best_t_rel = 200 + int(np.argmax(scores))

    # Backtrack from best_t_rel
    T_len = best_t_rel + 1
    end_x = int(np.argmax(dp[best_t_rel, :22]))
    path_x = [0] * T_len
    path_x[T_len - 1] = end_x
    for t in range(T_len - 1, 0, -1):
        path_x[t - 1] = int(back[t, path_x[t]])

    # Trim any leading stationary frames where the DP waited at x > 1250 for a later cell
    start_trim = 0
    while start_trim + 10 < T_len and (path_x[start_trim] - path_x[start_trim + 8]) < 3:
        start_trim += 1

    # Trim any trailing frames once the cell has reached x <= x_exit_cutoff (14 px)
    # so the ending NEVER goes haywire at x = 3..9 px after the cell exits!
    end_trim = T_len - 1
    for idx in range(start_trim, T_len):
        if path_x[idx] <= x_exit_cutoff:
            end_trim = idx
            break

    if end_trim - start_trim < 120:
        return None, None

    trimmed_x = [float(x) for x in path_x[start_trim: end_trim + 1]]
    actual_t_enter = t_enter + start_trim
    return actual_t_enter, trimmed_x


def peel_ridge_from_kymograph(kymograph, t_enter, path_x, half_width_x=38):
    """
    Erase (zero out) a tracked cell's space-time ridge from the kymograph so a second
    cell right behind it (like Cell #3 behind Cell #2) is tracked cleanly without interference.
    """
    T_total, W = kymograph.shape
    for idx, x_val in enumerate(path_x):
        t = t_enter + idx
        if 0 <= t < T_total:
            x0 = max(0, int(round(x_val)) - half_width_x)
            x1 = min(W, int(round(x_val)) + half_width_x + 1)
            kymograph[t, x0:x1] = 0.0


# ====================================================================
# STAGE 3B: SUB-PIXEL 2D MATCHED CENTROID + CONFIDENCE EXTRACTION
# ====================================================================

def extract_subpixel_observation(
    gray_roi,
    prev_gray_roi,
    bg_roi,
    mask_roi,
    guide_x,
    y_min_roi,
    top_wall,
    bot_wall
):
    h_roi, w_img = gray_roi.shape
    x_c = int(np.clip(round(guide_x), 0, w_img - 1))

    x0 = max(0, x_c - 20)
    x1 = min(w_img, x_c + 21)

    patch_gray = gray_roi[:, x0:x1]
    patch_bg = bg_roi[:, x0:x1]
    patch_mask = mask_roi[:, x0:x1]

    if prev_gray_roi is not None:
        patch_mot = np.abs(patch_gray - prev_gray_roi[:, x0:x1])
    else:
        patch_mot = np.abs(patch_gray - patch_bg)

    if x_c > 1180 or x_c < 110:
        pos_diff = np.maximum(0.0, patch_gray - patch_bg)
        energy = pos_diff * np.sqrt(np.maximum(1.0, patch_mot))
    else:
        abs_diff = np.abs(patch_gray - patch_bg)
        energy = abs_diff * np.sqrt(np.maximum(1.0, patch_mot))

    energy = cv2.GaussianBlur(energy.astype(np.float32), (5, 3), 0)
    energy[patch_mask == 0] = 0.0

    peak_e = float(energy.max())
    if peak_e >= 12.0:
        thresh = 0.48 * peak_e
        ys, xs = np.where(energy >= thresh)
    else:
        ys, xs = np.array([]), np.array([])

    center_y_wall = float(0.5 * (top_wall[x_c] + bot_wall[x_c]))

    if xs.size >= 3:
        weights = energy[ys, xs] ** 1.5
        w_sum = float(weights.sum())
        zx = float(x0 + np.sum(xs * weights) / w_sum)
        raw_zy = float(y_min_roi + np.sum(ys * weights) / w_sum)
        # Regularize vertical coordinate gently toward channel centerline so flare entrance/exit never hooks vertically
        zy = 0.45 * raw_zy + 0.55 * center_y_wall
        bw_raw = int(xs.max() - xs.min() + 1)
        bh_raw = int(ys.max() - ys.min() + 1)
        conf = float(np.clip(peak_e / 80.0, 0.15, 1.0))
    else:
        zx = float(guide_x)
        zy = center_y_wall
        bw_raw, bh_raw = (14, 13) if (x_c > 1180 or x_c < 110) else (18, 6)
        conf = 0.05

    return {
        "zx": zx,
        "zy": zy,
        "bw_raw": bw_raw,
        "bh_raw": bh_raw,
        "conf": conf,
    }


# ====================================================================
# STAGE 3C: OPENCV KALMAN FILTER + FORWARD-BACKWARD RTS SMOOTHER
# ====================================================================

def kalman_rts_smooth_passage(observations, top_wall, bot_wall):
    N = len(observations)
    if N == 0:
        return []

    w_max = len(top_wall) - 1

    F = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],
        dtype=np.float64
    )
    H = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]],
        dtype=np.float64
    )
    Q = np.diag([0.35, 0.08, 0.30, 0.02]).astype(np.float64)

    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = F.astype(np.float32)
    kf.measurementMatrix = H.astype(np.float32)
    kf.processNoiseCov = Q.astype(np.float32)

    x_pred_list = np.zeros((N, 4), dtype=np.float64)
    P_pred_list = np.zeros((N, 4, 4), dtype=np.float64)
    x_filt_list = np.zeros((N, 4), dtype=np.float64)
    P_filt_list = np.zeros((N, 4, 4), dtype=np.float64)

    z0 = observations[0]
    vx0 = (observations[min(N - 1, 5)]["zx"] - z0["zx"]) / float(max(1, min(N - 1, 5)))
    vy0 = (observations[min(N - 1, 5)]["zy"] - z0["zy"]) / float(max(1, min(N - 1, 5)))
    x_curr = np.array([z0["zx"], z0["zy"], vx0, vy0], dtype=np.float64)
    P_curr = np.diag([4.0, 1.5, 4.0, 0.5]).astype(np.float64)

    kf.statePost = x_curr.reshape(4, 1).astype(np.float32)
    kf.errorCovPost = P_curr.astype(np.float32)

    for t in range(N):
        obs = observations[t]
        if t == 0:
            x_pred = x_curr.copy()
            P_pred = P_curr.copy()
        else:
            kf.predict()
            x_pred = F @ x_curr
            P_pred = F @ P_curr @ F.T + Q

        x_pred_list[t] = x_pred
        P_pred_list[t] = P_pred

        conf = max(0.05, obs["conf"])
        sigma_z = 2.2 / conf
        R_t = np.diag([sigma_z ** 2, (sigma_z * 0.5) ** 2])

        z_t = np.array([obs["zx"], obs["zy"]], dtype=np.float64)
        y_innov = z_t - (H @ x_pred)
        S = H @ P_pred @ H.T + R_t
        K = P_pred @ H.T @ np.linalg.inv(S)

        x_curr = x_pred + K @ y_innov
        P_curr = (np.eye(4) - K @ H) @ P_pred

        kf.measurementNoiseCov = R_t.astype(np.float32)
        kf.correct(z_t.reshape(2, 1).astype(np.float32))

        x_filt_list[t] = x_curr
        P_filt_list[t] = P_curr

    # Backward RTS Smoother Pass
    x_smooth = np.zeros((N, 4), dtype=np.float64)
    P_smooth = np.zeros((N, 4, 4), dtype=np.float64)
    x_smooth[-1] = x_filt_list[-1]
    P_smooth[-1] = P_filt_list[-1]

    for t in range(N - 2, -1, -1):
        P_pred_next = P_pred_list[t + 1]
        C_t = P_filt_list[t] @ F.T @ np.linalg.inv(P_pred_next + 1e-6 * np.eye(4))
        x_smooth[t] = x_filt_list[t] + C_t @ (x_smooth[t + 1] - x_pred_list[t + 1])
        P_smooth[t] = P_filt_list[t] + C_t @ (P_smooth[t + 1] - P_pred_next) @ C_t.T

    # Build smooth, monotonically decreasing right-to-left trajectory with zero end-wiggle
    smoothed_track = []
    areas_so_far = []

    for t in range(N):
        obs = observations[t]
        cx = float(np.clip(x_smooth[t, 0], 0.0, w_max))
        if t > 0 and cx > smoothed_track[-1]["cx"]:
            cx = smoothed_track[-1]["cx"]

        x_idx = int(np.clip(round(cx), 0, w_max))
        wt = float(top_wall[x_idx])
        wb = float(bot_wall[x_idx])
        yc_wall = 0.5 * (wt + wb)
        lumen_h = max(4.0, wb - wt)

        # Smoothly blend cy with channel centerline so there is zero vertical hook at x=1180 or x=110
        if 110 <= x_idx <= 1180:
            cy = yc_wall
        else:
            dev = np.clip(x_smooth[t, 1] - yc_wall, -3.5, 3.5)
            # Taper vertical deviation to 0 near the extreme entrance/exit edges
            edge_taper = min(1.0, min(x_idx / 40.0, (w_max - x_idx) / 40.0))
            cy = float(yc_wall + edge_taper * dev)

        bw_raw, bh_raw = obs["bw_raw"], obs["bh_raw"]
        if x_idx > 1180 or x_idx < 110:
            bw = int(np.clip(max(bw_raw, 14), 13, 22))
            bh = int(np.clip(max(bh_raw, 13), 12, 20))
        else:
            bw = int(np.clip(max(bw_raw, 17), 15, 26))
            bh = int(np.clip(min(max(bh_raw, int(round(lumen_h))), int(round(lumen_h)) + 2), 5, 9))

        bx = int(round(cx - 0.5 * bw))
        by = int(round(cy - 0.5 * bh))

        a = max(1.0, 0.5 * bw)
        b = max(1.0, 0.5 * bh)
        area = float(np.pi * a * b)
        perim = float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))
        circ = float(np.clip(4.0 * np.pi * area / (perim * perim), 0.15, 1.0))
        elong = float(max(bw, bh)) / float(max(1, min(bw, bh)))

        if t == 0:
            vel = float(np.hypot(x_smooth[0, 2], x_smooth[0, 3]))
        else:
            prev = smoothed_track[-1]
            vel = float(np.hypot(cx - prev["cx"], cy - prev["cy"]))

        areas_so_far.append(area)
        adi_run = float(np.min(areas_so_far) / max(1.0, np.mean(areas_so_far)))

        smoothed_track.append({
            "frame": obs["frame"],
            "track_id": obs["track_id"],
            "cx": cx,
            "cy": cy,
            "bbox": (bx, by, bw, bh),
            "area": area,
            "circularity": circ,
            "elongation": elong,
            "velocity": vel,
            "ADI_running": adi_run,
        })

    return smoothed_track


# ====================================================================
# FULL HYBRID PIPELINE (Multi-Cell Kymograph Peeling + RTS Kalman)
# ====================================================================

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
    y_min, y_max = roi_y_bounds
    bg_roi = background[y_min:y_max, :]
    mask_roi = lumen_mask[y_min:y_max, :]
    w_img = background.shape[1]

    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    kymograph = np.zeros((frames_to_run, w_img), dtype=np.float32)
    roi_cache = []

    prev_gray_roi = None
    lag3_buffer = []

    for offset in range(frames_to_run):
        ok, frame = cap.read()
        if not ok:
            kymograph = kymograph[:offset]
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        roi_u8 = gray[y_min:y_max, :].copy()
        roi_cache.append(roi_u8)
        gray_roi = roi_u8.astype(np.float32)

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

        col_abs = abs_diff.max(axis=0)
        col_pos = pos_diff.max(axis=0)
        col_mot = mot_diff.max(axis=0)

        col_score = np.sqrt(col_abs * np.maximum(0.0, col_mot - 4.0))
        col_score[:110] = np.sqrt(col_pos[:110] * np.maximum(0.0, col_mot[:110] - 4.0)) * 1.4
        col_score[1180:] = np.sqrt(col_pos[1180:] * np.maximum(0.0, col_mot[1180:] - 4.0)) * 1.4

        kymograph[offset] = col_score

        if (offset + 1) % 500 == 0:
            print(f"      [Stage 3A] Kymograph built for {offset + 1}/{frames_to_run} frames...", flush=True)

    cap.release()

    actual_frames = kymograph.shape[0]
    if actual_frames < 20:
        return {}

    # Subtract temporal median baseline per column
    col_baseline = np.median(kymograph, axis=0, keepdims=True)
    kymograph = np.maximum(0.0, kymograph - col_baseline)

    right_entrance_signal = kymograph[:, 1215:1278].max(axis=1)
    left_exit_signal = kymograph[:, 5:105].max(axis=1)

    r_smooth = np.convolve(right_entrance_signal, np.ones(7) / 7.0, mode="same")
    l_smooth = np.convolve(left_exit_signal, np.ones(7) / 7.0, mode="same")

    r_med = float(np.median(r_smooth))
    r_mad = float(np.median(np.abs(r_smooth - r_med))) + 1e-3
    r_thresh = max(8.0, r_med + 3.2 * r_mad)

    l_med = float(np.median(l_smooth))
    l_mad = float(np.median(np.abs(l_smooth - l_med))) + 1e-3
    l_thresh = max(8.0, l_med + 3.0 * l_mad)

    # Detect ALL distinct entrance peaks (including back-to-back cells ~55 frames apart!)
    entrances = find_all_entrance_peaks(r_smooth, right_entrance_signal, r_thresh, min_distance=48)

    work_kymo = kymograph.copy()
    final_tracks = {}
    tid = 1

    for t_enter_cand, _ in entrances:
        actual_t_enter, dp_xs = trace_single_cell_dp_ridge(
            work_kymo, t_enter_cand, l_smooth, l_thresh, x_start_min=1225, x_exit_cutoff=14
        )
        if actual_t_enter is None or not dp_xs:
            continue

        # Peel this cell's ridge out of work_kymo so a trailing cell right behind it (Cell #3)
        # tracks its own ridge cleanly!
        peel_ridge_from_kymograph(work_kymo, actual_t_enter, dp_xs, half_width_x=38)

        t_end = actual_t_enter + len(dp_xs) - 1
        print(
            f"      [Stage 3B/3C] Refining CELL #{tid} (Frames {start_frame + actual_t_enter} -> {start_frame + t_end}) "
            f"via Kymograph Peeling + RTS Kalman Smoother...",
            flush=True
        )

        observations = []
        prev_g_roi = None

        for step_idx, gx in enumerate(dp_xs):
            cache_idx = actual_t_enter + step_idx
            if cache_idx >= len(roi_cache):
                break
            f_idx = start_frame + cache_idx
            gray_roi = roi_cache[cache_idx].astype(np.float32)

            obs = extract_subpixel_observation(
                gray_roi, prev_g_roi, bg_roi, mask_roi, gx, y_min, top_wall, bot_wall
            )
            obs["frame"] = f_idx
            obs["track_id"] = tid
            observations.append(obs)
            prev_g_roi = gray_roi

        smoothed = kalman_rts_smooth_passage(observations, top_wall, bot_wall)
        if smoothed:
            final_tracks[tid] = smoothed
            tid += 1

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
# RENDER ANNOTATED VIDEO (Supports Back-to-Back Simultaneous Cells!)
# ====================================================================

def _cell_color_bgr(track_id):
    palette = [
        (50, 255, 100),   # Cell #1: Neon Green
        (0, 215, 255),    # Cell #2: Gold / Amber
        (255, 200, 50),   # Cell #3: Cyan-Blue
        (200, 110, 255),  # Cell #4: Magenta
    ]
    return palette[(track_id - 1) % len(palette)]


def render_tracked_video(
    video_path,
    top_wall,
    bot_wall,
    valid_tracks,
    biomarkers_by_id,
    out_dir,
    pad_frames=20,
    playback_fps=20.0
):
    out_mp4 = os.path.join(out_dir, "tracked_cell.mp4")
    legacy_mp4 = os.path.join(out_dir, "active_passages_only.mp4")

    # Map frame_idx -> LIST of active cell points (so back-to-back cells #2 and #3 both render!)
    by_frame = defaultdict(list)
    for tid, path in valid_tracks.items():
        for pt in path:
            by_frame[pt["frame"]].append(pt)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, playback_fps, (w, h))

    frames_set = set()
    if valid_tracks:
        for tid in sorted(valid_tracks.keys()):
            path = valid_tracks[tid]
            f0 = max(0, path[0]["frame"] - pad_frames)
            f1 = min(total_frames - 1, path[-1]["frame"] + pad_frames)
            for f in range(f0, f1 + 1):
                frames_set.add(f)
        frames_to_render = sorted(frames_set)
    else:
        frames_to_render = list(range(1500, min(1950, total_frames)))

    trail_by_id = {tid: [] for tid in valid_tracks}
    current_pos = -999

    for f_idx in frames_to_render:
        if f_idx != current_pos:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok:
            break
        current_pos = f_idx + 1

        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        active_cells = by_frame.get(f_idx, [])
        for idx_in_frame, d in enumerate(active_cells):
            tid = d["track_id"]
            color = _cell_color_bgr(tid)
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
            # Alternate HUD placement above/below if two cells are close horizontally
            place_above = (idx_in_frame % 2 == 0) and (y0 > card_h + 38)
            card_y = y0 - card_h - 14 if place_above else min(h - card_h - 6, y1 + 14)

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

        cv2.rectangle(vis, (0, 0), (w, 28), (14, 17, 22), -1)
        if active_cells:
            ids_str = ", ".join(f"CELL #{c['track_id']}" for c in active_cells)
            status_txt = f"IISc RBC Passage Tracker | Frame {f_idx} ({f_idx / 10.0:.1f} ms) | Tracking {ids_str}"
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
    parser = argparse.ArgumentParser(description="Stage 3: Multi-Cell Kymograph Peeling + RTS Kalman Smoother")
    parser.add_argument("--max-frames", type=int, default=1500, help="Frames to scan (default: 1500; pass 0 for all 37,274)")
    parser.add_argument("--start-frame", type=int, default=1450, help="Frame index to start from (default: 1450)")
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
    print("STAGE 3: MULTI-CELL KYMOGRAPH PEELING + RTS KALMAN SMOOTHER")
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

    print(f"\n[1/2] Running Multi-Cell Kymograph Peeling + RTS Kalman Smoother on {frames_to_run} frames (start={args.start_frame})...")

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
        pad_frames=20,
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
