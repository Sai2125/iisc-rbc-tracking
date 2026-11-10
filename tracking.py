"""
STAGE 3 — Right-to-Left Passage RBC Tracker (Bottom-Right Entrance -> Top-Left Exit)

Designed specifically for the IISc microfluidic channel geometry:
- Cells enter at the BOTTOM-RIGHT flare opening (x ~ 1279, y ~ 385)
- Squeeze right-to-left through the narrow constriction bore (x: 1180 -> 110)
- Exit out of the TOP-LEFT flare opening (x ~ 0, y ~ 170)

How this eliminates the false jumps (e.g. x=72) completely:
1. True Motion Gate: Requires BOTH static background contrast |frame - median_bg| AND
   temporal change |frame_t - frame_{t-5}| so static wall hotspots (like x=72) have
   zero response when no cell is moving there.
2. Causal Right-to-Left State Machine:
   - Locks onto a cell when it enters the bottom-right flare (x > 1050).
   - Tracks the cell within a local causal search window moving right-to-left
     (x_t <= x_{t-1} + 5 px), so it is physically impossible for the box to teleport
     to the left flare (x=72) while the cell is in the middle of the bore!
   - Follows the cell continuously until it exits the top-left opening (x < 35).
"""
import os
import sys
import csv
import json
import shutil
import argparse
from pathlib import Path
from collections import deque
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
        half_w = max(1.8, 0.5 * (bot_smooth[x] - top_smooth[x]) - 1.2)
        y0 = max(0, int(np.floor(yc - half_w)))
        y1 = min(h, int(np.ceil(yc + half_w)) + 1)
        lumen_mask[y0:y1, x] = 255

    return lumen_mask, top_smooth, bot_smooth, slope, intercept


# ====================================================================
# CAUSAL RIGHT-TO-LEFT PASSAGE TRACKER
# ====================================================================

def measure_cell_at_window(
    bg_diff_roi,
    motion_roi,
    x_min_search,
    x_max_search,
    y_min_roi,
    top_wall,
    bot_wall,
    min_motion=6.0,
    min_bg_diff=10.0
):
    """
    Search for the cell strictly within horizontal window [x_min_search, x_max_search].
    Requires genuine temporal motion (`motion_roi`) AND background difference (`bg_diff_roi`)
    so static wall hotspots (like x=72 or x=732) cannot be picked when nothing is moving there.
    """
    w_img = bg_diff_roi.shape[1]
    x0 = int(np.clip(round(x_min_search), 0, w_img - 1))
    x1 = int(np.clip(round(x_max_search), 0, w_img))
    if x1 <= x0 + 2:
        return None

    # Combined motion * contrast score inside the search window
    sub_bg = bg_diff_roi[:, x0:x1]
    sub_mot = motion_roi[:, x0:x1]

    # Column score requires BOTH background contrast and temporal motion
    col_bg = sub_bg.max(axis=0)
    col_mot = sub_mot.max(axis=0)
    col_score = np.where((col_bg >= min_bg_diff) & (col_mot >= min_motion), col_bg * np.sqrt(col_mot), 0.0)
    if col_score.size >= 7:
        col_score = np.convolve(col_score, np.ones(7, dtype=np.float32) / 7.0, mode="same")

    if float(col_score.max()) <= 0.0:
        return None

    rel_peak_x = int(np.argmax(col_score))
    peak_x = x0 + rel_peak_x

    # Extract tight 44-px patch around peak_x to measure bounding box & shape
    px0 = max(0, peak_x - 22)
    px1 = min(w_img, peak_x + 22)
    patch = bg_diff_roi[:, px0:px1]
    patch_max = float(patch.max())
    if patch_max < min_bg_diff:
        return None

    thresh_val = max(min_bg_diff * 0.75, 0.42 * patch_max)
    binary = (patch >= thresh_val).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    ys, xs = np.where(binary > 0)
    if xs.size < 4:
        return None

    weights = patch[ys, xs]
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return None

    cx = float(px0 + np.sum(xs * weights) / w_sum)
    x_idx = int(np.clip(round(cx), 0, w_img - 1))

    # Lock Y center and vertical bounds to the calibrated lumen walls at cx
    wall_t = float(top_wall[x_idx])
    wall_b = float(bot_wall[x_idx])
    lumen_h = max(4.0, wall_b - wall_t)
    cy = float(y_min_roi + np.sum(ys * weights) / w_sum)
    cy = float(np.clip(cy, wall_t + 1.0, wall_b - 1.0))

    bx0 = int(px0 + xs.min())
    bx1 = int(px0 + xs.max()) + 1
    bw = int(np.clip(bx1 - bx0, 8, 42))
    bh = int(np.clip(max(ys.max() - ys.min() + 1, int(round(lumen_h))), 5, 38))
    bx = int(round(cx - 0.5 * bw))
    by = int(round(cy - 0.5 * bh))

    pixel_area = float(xs.size)
    effective_area = max(pixel_area, 0.75 * float(bw * bh))

    a = max(1.0, 0.5 * bw)
    b = max(1.0, 0.5 * bh)
    perim = float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))
    circ = float(np.clip(4.0 * np.pi * effective_area / (perim * perim), 0.12, 1.0))
    elongation = float(max(bw, bh)) / float(max(1, min(bw, bh)))

    return {
        "cx": cx,
        "cy": cy,
        "bbox": (bx, by, bw, bh),
        "area": effective_area,
        "circularity": circ,
        "elongation": elongation,
    }


def track_right_to_left_passages(
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
    Track each RBC entering from the Bottom-Right opening (x ~ 1279) and moving
    monotonically right-to-left (decreasing x) until it exits the Top-Left opening (x ~ 0).
    """
    y_min, y_max = roi_y_bounds
    bg_roi = background[y_min:y_max, :]
    mask_roi = lumen_mask[y_min:y_max, :]
    w_img = background.shape[1]

    cap = cv2.VideoCapture(video_path)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Keep a 5-frame ring buffer so slow movement in the wide flare and fast movement in the bore
    # both produce strong temporal difference |frame_t - frame_{t-lag}|
    gray_history = deque(maxlen=6)

    completed_passages = []
    active_path = []
    missed_count = 0
    vx_est = -4.0  # cells move right-to-left (negative vx)

    for offset in range(frames_to_run):
        f_idx = start_frame + offset
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray_roi = gray[y_min:y_max, :].astype(np.float32)

        bg_diff = np.abs(gray_roi - bg_roi)
        bg_diff = cv2.GaussianBlur(bg_diff, (5, 3), 0)
        bg_diff[mask_roi == 0] = 0.0

        if len(gray_history) > 0:
            # Max temporal change vs 1 frame ago and 5 frames ago
            mot1 = np.abs(gray_roi - gray_history[-1])
            mot5 = np.abs(gray_roi - gray_history[0])
            motion_roi = np.maximum(mot1, mot5)
            motion_roi = cv2.GaussianBlur(motion_roi, (5, 3), 0)
            motion_roi[mask_roi == 0] = 0.0
        else:
            motion_roi = np.zeros_like(bg_diff)

        gray_history.append(gray_roi)

        if not active_path:
            # Waiting for a new cell to enter the Bottom-Right flare (x in 1120 .. 1279)
            det = measure_cell_at_window(
                bg_diff, motion_roi,
                x_min_search=1120,
                x_max_search=w_img,
                y_min_roi=y_min,
                top_wall=top_wall,
                bot_wall=bot_wall,
                min_motion=8.0,
                min_bg_diff=14.0,
            )
            if det is not None:
                det["frame"] = f_idx
                active_path = [det]
                missed_count = 0
                vx_est = -3.0
        else:
            # We are actively tracking a cell moving right-to-left!
            last_pt = active_path[-1]
            dt = max(1, f_idx - last_pt["frame"])
            last_x = last_pt["cx"]

            # Predict next X position: in the right flare (x > 1165) cells move ~1-5 px/frame;
            # inside the narrow constriction (x < 1165) cells accelerate to ~6-25 px/frame.
            pred_x = last_x + vx_est * dt
            # Search window strictly around and to the LEFT of last_x (cannot jump rightwards or teleport to x=72!)
            search_hi = min(w_img, last_x + 6.0)
            max_left_jump = 35.0 * dt if last_x > 1150 else max(60.0, 32.0 * dt)
            search_lo = max(0.0, min(pred_x - 45.0, last_x - max_left_jump))

            det = measure_cell_at_window(
                bg_diff, motion_roi,
                x_min_search=search_lo,
                x_max_search=search_hi,
                y_min_roi=y_min,
                top_wall=top_wall,
                bot_wall=bot_wall,
                min_motion=4.5,
                min_bg_diff=8.5,
            )

            if det is not None:
                # Enforce monotonic right-to-left progression (clamp tiny backward jitter)
                if det["cx"] > last_x:
                    det["cx"] = last_x
                det["frame"] = f_idx
                step_vx = (det["cx"] - last_x) / float(dt)
                vx_est = float(np.clip(0.65 * vx_est + 0.35 * step_vx, -35.0, -0.5))
                active_path.append(det)
                missed_count = 0

                # If cell reached the top-left exit opening (x <= 25), complete this passage!
                if det["cx"] <= 25.0:
                    completed_passages.append(active_path)
                    active_path = []
                    missed_count = 0
            else:
                missed_count += 1
                # If the cell was already near the left exit (x < 180) and disappeared, it exited!
                if last_x < 180.0 and missed_count >= 5:
                    completed_passages.append(active_path)
                    active_path = []
                    missed_count = 0
                elif missed_count > 25:
                    # Save if it covered a substantial part of the channel
                    if last_x < 950.0 and (active_path[0]["cx"] - last_x) >= 200.0:
                        completed_passages.append(active_path)
                    active_path = []
                    missed_count = 0

        if (offset + 1) % 500 == 0:
            print(f"      scanned {offset + 1}/{frames_to_run} frames...")

    cap.release()

    if active_path and (active_path[0]["cx"] - active_path[-1]["cx"]) >= 120.0:
        completed_passages.append(active_path)

    # Interpolate any 1..25 frame micro-gaps inside each passage so the bounding box
    # moves smoothly from bottom-right entrance (x~1278) to top-left exit (x~0) on EVERY frame
    final_tracks = {}
    w_max = len(top_wall) - 1

    for tid, seg in enumerate(completed_passages, start=1):
        if len(seg) < 5:
            continue

        # Ensure monotonically non-increasing cx (right-to-left flow)
        for k in range(1, len(seg)):
            if seg[k]["cx"] > seg[k - 1]["cx"]:
                seg[k]["cx"] = seg[k - 1]["cx"]

        continuous = []
        for k in range(len(seg) - 1):
            p0 = seg[k]
            p1 = seg[k + 1]
            continuous.append(dict(p0))
            gap = p1["frame"] - p0["frame"]
            if gap > 1:
                for step_f in range(1, gap):
                    alpha = float(step_f) / float(gap)
                    cx_i = (1.0 - alpha) * p0["cx"] + alpha * p1["cx"]
                    x_idx = int(np.clip(round(cx_i), 0, w_max))
                    cy_i = float(0.5 * (top_wall[x_idx] + bot_wall[x_idx]))
                    bw_i = int(round((1.0 - alpha) * p0["bbox"][2] + alpha * p1["bbox"][2]))
                    bh_i = max(5, int(round(bot_wall[x_idx] - top_wall[x_idx])))
                    continuous.append({
                        "frame": p0["frame"] + step_f,
                        "cx": cx_i,
                        "cy": cy_i,
                        "bbox": (int(round(cx_i - 0.5 * bw_i)), int(round(cy_i - 0.5 * bh_i)), bw_i, bh_i),
                        "area": (1.0 - alpha) * p0["area"] + alpha * p1["area"],
                        "circularity": (1.0 - alpha) * p0["circularity"] + alpha * p1["circularity"],
                        "elongation": (1.0 - alpha) * p0["elongation"] + alpha * p1["elongation"],
                    })
        continuous.append(dict(seg[-1]))

        # Smooth cx with a 5-point moving average so the box glides without pixel stair-stepping
        if len(continuous) >= 5:
            cxs = np.array([p["cx"] for p in continuous], dtype=np.float64)
            kernel5 = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
            cxs_s = np.convolve(np.pad(cxs, 2, mode="edge"), kernel5, mode="valid")
            for p, sx in zip(continuous, cxs_s):
                p["cx"] = float(sx)
                x_idx = int(np.clip(round(sx), 0, w_max))
                p["cy"] = float(0.5 * (top_wall[x_idx] + bot_wall[x_idx]))
                bw, bh = p["bbox"][2], p["bbox"][3]
                p["bbox"] = (int(round(p["cx"] - 0.5 * bw)), int(round(p["cy"] - 0.5 * bh)), bw, bh)

        # Compute instantaneous velocity and running ADI
        areas_so_far = []
        for idx, p in enumerate(continuous):
            p["track_id"] = tid
            if idx == 0:
                p["velocity"] = 0.0
            else:
                prev = continuous[idx - 1]
                dt = max(1, p["frame"] - prev["frame"])
                p["velocity"] = float(np.hypot(p["cx"] - prev["cx"], p["cy"] - prev["cy"]) / float(dt))
            areas_so_far.append(p["area"])
            p["ADI_running"] = float(np.min(areas_so_far) / max(1.0, np.mean(areas_so_far)))

        final_tracks[tid] = continuous

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
# RENDER ANNOTATED VIDEO (tracked_cell.mp4)
# ====================================================================

def render_tracked_video(
    video_path,
    top_wall,
    bot_wall,
    valid_tracks,
    biomarkers_by_id,
    out_dir,
    pad_frames=12,
    playback_fps=20.0
):
    """
    Renders ONLY the real passage frames (from 12 frames before the cell enters the
    bottom-right flare to 12 frames after it exits the top-left flare) into
    `output/stage3/tracked_cell.mp4`.
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

    xs_wall = np.arange(0, w, 8)
    top_pts = np.column_stack([xs_wall, top_wall[xs_wall]]).astype(np.int32)
    bot_pts = np.column_stack([xs_wall, bot_wall[xs_wall]]).astype(np.int32)

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
        frames_to_render = list(range(785, min(1250, total_frames)))

    trail_by_id = {tid: [] for tid in valid_tracks}
    color = (50, 255, 100)

    for f_idx in frames_to_render:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Subtle channel wall boundary lines
        cv2.polylines(vis, [top_pts], False, (60, 120, 60), 1, cv2.LINE_AA)
        cv2.polylines(vis, [bot_pts], False, (60, 120, 60), 1, cv2.LINE_AA)

        d = by_frame.get(f_idx)
        if d is not None:
            tid = d["track_id"]
            bx, by, bw, bh = d["bbox"]
            cx_i, cy_i = int(round(d["cx"])), int(round(d["cy"]))
            trail_by_id[tid].append((cx_i, cy_i))

            # 1. Trajectory trail starting from the BOTTOM-RIGHT opening (trail[0])
            trail = trail_by_id[tid]
            if len(trail) >= 2:
                cv2.polylines(vis, [np.array(trail, np.int32)], False, color, 2, cv2.LINE_AA)
                cv2.circle(vis, trail[0], 4, (0, 215, 255), -1, cv2.LINE_AA)

            # 2. Bounding box around the cell
            pad_x = max(11, bw // 2 + 4)
            pad_y = max(10, bh // 2 + 4)
            x0, y0 = max(0, cx_i - pad_x), max(0, cy_i - pad_y)
            x1, y1 = min(w - 1, cx_i + pad_x), min(h - 1, cy_i + pad_y)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)

            # 3. Live Biomechanics HUD card anchored above the cell
            bio = biomarkers_by_id.get(tid, {})
            adi_val = bio.get("ADI", d["ADI_running"])
            zone = "Right Entrance" if cx_i > 1165 else ("Left Exit" if cx_i < 115 else "Constriction Bore")
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

        # Top HUD header
        cv2.rectangle(vis, (0, 0), (w, 30), (14, 17, 22), -1)
        active_str = f"Tracking CELL #{d['track_id']} (Right -> Left)" if d is not None else "Passage Window"
        status_txt = f"IISc RBC Passage Tracker | Frame {f_idx} ({f_idx / 10.0:.1f} ms) | {active_str}"
        cv2.putText(vis, status_txt, (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

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
    parser = argparse.ArgumentParser(description="Stage 3: Right-to-Left Passage RBC Tracker")
    # Note: In 2A002 2_C001H001S0001-001.avi, the first cell enters the bottom-right opening
    # at Frame ~795 and exits the top-left opening around Frame ~1150, so scanning 1500 frames
    # by default captures the complete entrance-to-exit passage!
    parser.add_argument("--max-frames", type=int, default=1500, help="Frames to scan (default: 1500; pass 0 for all 37,274)")
    parser.add_argument("--start-frame", type=int, default=750, help="Frame index to start from (default: 750, just before first cell enters at frame ~795; pass 0 to scan from frame 0)")
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
    print("STAGE 3: RIGHT-TO-LEFT PASSAGE TRACKER (BOTTOM-RIGHT -> TOP-LEFT)")
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
    cap.release()

    frames_to_run = total_frames - args.start_frame
    if args.max_frames and args.max_frames > 0:
        frames_to_run = min(frames_to_run, args.max_frames)

    print(f"\n[1/2] Tracking cells from bottom-right entrance to top-left exit across {frames_to_run} frames (start={args.start_frame})...")

    valid_tracks = track_right_to_left_passages(
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

    print("\n[2/2] Rendering annotated passage video: output/stage3/tracked_cell.mp4 ...")
    out_mp4, rendered_frames = render_tracked_video(
        video_path=video_path,
        top_wall=top_wall,
        bot_wall=bot_wall,
        valid_tracks=valid_tracks,
        biomarkers_by_id=biomarkers_by_id,
        out_dir=out_dir,
        pad_frames=12,
        playback_fps=20.0,
    )

    print("\n" + "=" * 68)
    print("STAGE 3 COMPLETE")
    print("=" * 68)
    print(f"  ✓ Tracked Video File   : {out_mp4} ({rendered_frames} frames)")
    print(f"  ✓ Total Tracked Cells  : {len(all_biomarkers)}")

    if all_biomarkers:
        print("\nTracked Cell Passages (Bottom-Right -> Top-Left):")
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
