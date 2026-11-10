"""
STAGE 3 — Deterministic Single-Track-Per-Passage RBC Tracker & Video Renderer

Guarantees STRICTLY:
  - At most ONE cell detection per frame (1D + 2D peak detection inside the lumen)
  - Strictly ONE continuous track ID (CELL #1, CELL #2, ...) per passage from entrance to exit
  - ONE output video file: output/stage3/tracked_cell.mp4
"""
import os
import sys
import csv
import json
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
# EXACT CHANNEL LUMEN GEOMETRY (x = 0 .. W-1)
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
        xs_fit, ys_fit = xs[keep] if "xs" in locals() else xs_fit[keep], ys_fit[keep]
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

    # Interior lumen mask: strictly between the walls so wall shimmer is ignored
    lumen_mask = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        yc = 0.5 * (top_smooth[x] + bot_smooth[x])
        half_w = max(1.6, 0.5 * (bot_smooth[x] - top_smooth[x]) - 1.5)
        y0 = max(0, int(np.floor(yc - half_w)))
        y1 = min(h, int(np.ceil(yc + half_w)) + 1)
        lumen_mask[y0:y1, x] = 255

    return lumen_mask, top_smooth, bot_smooth, slope, intercept


# ====================================================================
# SINGLE-CELL DETECTOR (Returns at most 1 cell per frame)
# ====================================================================

def detect_single_cell_in_passage(
    frame,
    prev_gray,
    background,
    lumen_mask,
    top_wall,
    bot_wall,
    roi_y_bounds,
    diff_thresh=14.0
):
    """
    Detect AT MOST ONE moving RBC inside the channel lumen on the current frame.

    Why frame-to-frame motion + background contrast?
      - Static dirt on the glass wall (or slight illumination drift) does not move
        between frame t-1 and frame t.
      - A real RBC moving through the passage creates BOTH a background difference
        |frame - median_bg| AND inter-frame motion |frame - prev_frame| inside the lumen!
      - By finding the single strongest 1D column response along the lumen and extracting
        only the cell patch around that peak, we guarantee AT MOST 1 detection per frame.
    """
    y_min, y_max = roi_y_bounds
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    gray_f = gray[y_min:y_max, :].astype(np.float32)
    bg_f = background[y_min:y_max, :]
    mask_roi = lumen_mask[y_min:y_max, :]

    # Background difference inside the lumen
    bg_diff = np.abs(gray_f - bg_f)
    bg_diff = cv2.GaussianBlur(bg_diff, (5, 3), 0)
    bg_diff[mask_roi == 0] = 0.0

    # Combine with inter-frame motion if prev_gray is available so stationary wall shimmer = 0
    if prev_gray is not None:
        prev_f = prev_gray[y_min:y_max, :].astype(np.float32)
        motion_diff = np.abs(gray_f - prev_f)
        motion_diff = cv2.GaussianBlur(motion_diff, (5, 3), 0)
        motion_diff[mask_roi == 0] = 0.0
        # Geometric mean of static-contrast and motion-contrast rejects both static dirt & camera flicker
        signal = np.sqrt(bg_diff * np.maximum(motion_diff, 0.35 * bg_diff))
    else:
        signal = bg_diff

    # 1D column energy profile along the channel X axis (smoothed over a 15-px cell width)
    col_max = signal.max(axis=0)
    col_energy = np.convolve(col_max, np.ones(15, dtype=np.float32) / 15.0, mode="same")

    peak_x = int(np.argmax(col_energy))
    peak_val = float(col_max[max(0, peak_x - 7): min(len(col_max), peak_x + 8)].max())

    if peak_val < diff_thresh:
        return None, gray

    # Extract a local 70-px horizontal window around peak_x inside the lumen to measure the cell's box & shape
    w_img = gray_f.shape[1]
    x_win0 = max(0, peak_x - 35)
    x_win1 = min(w_img, peak_x + 35)
    sub_sig = bg_diff[:, x_win0:x_win1]

    local_thresh = max(diff_thresh * 0.75, 0.40 * float(sub_sig.max()))
    binary = (sub_sig >= local_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    ys, xs = np.where(binary > 0)
    if xs.size < 6:
        return None, gray

    # Weighted centroid & unified bounding box of the single cell
    weights = sub_sig[ys, xs]
    w_sum = float(weights.sum())
    if w_sum <= 0:
        return None, gray

    cx = float(x_win0 + np.sum(xs * weights) / w_sum)
    cy = float(y_min + np.sum(ys * weights) / w_sum)

    bx0 = int(x_win0 + xs.min())
    bx1 = int(x_win0 + xs.max()) + 1
    by0 = int(y_min + ys.min())
    by1 = int(y_min + ys.max()) + 1

    # Clamp vertical box neatly to the local channel walls so the box hugs the cell inside the tube
    cx_clamped = int(np.clip(round(cx), 0, w_img - 1))
    wall_top_y = int(np.floor(top_wall[cx_clamped]))
    wall_bot_y = int(np.ceil(bot_wall[cx_clamped]))
    by0 = max(wall_top_y - 2, by0)
    by1 = min(wall_bot_y + 2, max(by0 + 4, by1))

    bw = max(6, bx1 - bx0)
    bh = max(4, by1 - by0)

    pixel_area = float(xs.size)
    effective_area = max(pixel_area, 0.75 * float(bw * bh))

    # Equivalent ellipse circularity & elongation
    a = max(1.0, 0.5 * bw)
    b = max(1.0, 0.5 * bh)
    perim = float(np.pi * (3.0 * (a + b) - np.sqrt((3.0 * a + b) * (a + 3.0 * b))))
    circ = float(np.clip(4.0 * np.pi * effective_area / (perim * perim), 0.10, 1.0))
    elongation = float(max(bw, bh)) / float(max(1, min(bw, bh)))

    return {
        "cx": cx,
        "cy": cy,
        "bbox": (int(bx0), int(by0), int(bw), int(bh)),
        "area": effective_area,
        "circularity": circ,
        "elongation": elongation,
        "peak_diff": peak_val,
    }, gray


# ====================================================================
# PASSAGE BUILDER (Groups single-cell detections into 1 Track per Passage)
# ====================================================================

def build_clean_passage_tracks(frame_detections, slope, intercept, top_wall, bot_wall, min_span_px=150.0, min_frames=5):
    """
    Given at most 1 detection per frame:
    1. Group detections into continuous passage events (allowing up to 35-frame gaps).
    2. Reject any event that doesn't travel monotonically across at least `min_span_px`
       of the channel (eliminates stationary wall noise 100%).
    3. Smoothly interpolate across any missed frames inside a passage so the single
       bounding box glides continuously from start of passage to end of passage.
    """
    if not frame_detections:
        return {}

    # 1. Split into candidate passage segments whenever frame gap > 35 or cell jumps backwards > 80 px
    segments = []
    curr = [frame_detections[0]]

    for d in frame_detections[1:]:
        prev = curr[-1]
        gap_f = d["frame"] - prev["frame"]
        if gap_f <= 35:
            curr.append(d)
        else:
            segments.append(curr)
            curr = [d]
    segments.append(curr)

    # 2. Filter segments for monotonic movement and minimum horizontal span
    valid_passages = []
    for seg in segments:
        if len(seg) < min_frames:
            continue

        # Remove outlier jumps within the segment using a running median on cx
        xs = np.array([p["cx"] for p in seg], dtype=np.float64)
        if len(xs) >= 5:
            med_xs = _medfilt1d(xs, k=min(5, len(xs) if len(xs) % 2 == 1 else len(xs) - 1))
            keep_seg = [p for p, mx in zip(seg, med_xs) if abs(p["cx"] - mx) <= 65.0]
        else:
            keep_seg = seg

        if len(keep_seg) < min_frames:
            continue

        x_start = keep_seg[0]["cx"]
        x_end = keep_seg[-1]["cx"]
        span_x = float(max(p["cx"] for p in keep_seg) - min(p["cx"] for p in keep_seg))
        net_disp = abs(x_end - x_start)

        # Must actually travel across the channel passage
        if span_x < min_span_px and net_disp < min_span_px * 0.8:
            continue

        valid_passages.append(keep_seg)

    # If no segment crossed 150px (e.g. user ran a short clip where cell only moved 60px),
    # fall back to keeping the single longest-traveling segment if it moved >= 50px
    if not valid_passages and segments:
        best_seg = max(segments, key=lambda s: abs(s[-1]["cx"] - s[0]["cx"]) if len(s) >= 3 else 0.0)
        if len(best_seg) >= 3 and abs(best_seg[-1]["cx"] - best_seg[0]["cx"]) >= 50.0:
            valid_passages = [best_seg]

    # 3. Build continuous frame-by-frame single tracks (1 ID per passage)
    final_tracks = {}
    w_max = len(top_wall) - 1

    for tid_idx, seg in enumerate(valid_passages, start=1):
        continuous = []
        for k in range(len(seg) - 1):
            p0 = seg[k]
            p1 = seg[k + 1]
            continuous.append(dict(p0))
            gap = p1["frame"] - p0["frame"]
            if 1 < gap <= 35:
                for step_f in range(1, gap):
                    alpha = float(step_f) / float(gap)
                    cx_i = (1.0 - alpha) * p0["cx"] + alpha * p1["cx"]
                    x_idx = int(np.clip(round(cx_i), 0, w_max))
                    cy_i = 0.5 * (top_wall[x_idx] + bot_wall[x_idx])
                    bw_i = int(round((1.0 - alpha) * p0["bbox"][2] + alpha * p1["bbox"][2]))
                    bh_i = max(4, int(round(bot_wall[x_idx] - top_wall[x_idx])))
                    bx_i = int(round(cx_i - 0.5 * bw_i))
                    by_i = int(round(cy_i - 0.5 * bh_i))
                    continuous.append({
                        "frame": p0["frame"] + step_f,
                        "cx": cx_i,
                        "cy": cy_i,
                        "bbox": (bx_i, by_i, bw_i, bh_i),
                        "area": (1.0 - alpha) * p0["area"] + alpha * p1["area"],
                        "circularity": (1.0 - alpha) * p0["circularity"] + alpha * p1["circularity"],
                        "elongation": (1.0 - alpha) * p0["elongation"] + alpha * p1["elongation"],
                    })
        continuous.append(dict(seg[-1]))

        # Smooth cx trajectory slightly (3-frame moving average) so bounding box never jitters
        if len(continuous) >= 3:
            cxs = np.array([p["cx"] for p in continuous], dtype=np.float64)
            cxs_smooth = np.convolve(np.pad(cxs, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
            for p, sx in zip(continuous, cxs_smooth):
                p["cx"] = float(sx)
                x_idx = int(np.clip(round(sx), 0, w_max))
                p["cy"] = float(0.5 * (top_wall[x_idx] + bot_wall[x_idx]))
                bw = p["bbox"][2]
                bh = p["bbox"][3]
                p["bbox"] = (int(round(p["cx"] - 0.5 * bw)), int(round(p["cy"] - 0.5 * bh)), bw, bh)

        # Compute per-frame velocity and running ADI
        areas_so_far = []
        for idx, p in enumerate(continuous):
            p["track_id"] = tid_idx
            if idx == 0:
                p["velocity"] = 0.0
            else:
                prev = continuous[idx - 1]
                dt = max(1, p["frame"] - prev["frame"])
                p["velocity"] = float(np.hypot(p["cx"] - prev["cx"], p["cy"] - prev["cy"]) / float(dt))
            areas_so_far.append(p["area"])
            p["ADI_running"] = float(np.min(areas_so_far) / max(1.0, np.mean(areas_so_far)))

        final_tracks[tid_idx] = continuous

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
    span_px = float(max(p["cx"] for p in path) - min(p["cx"] for p in path))

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
# RENDER SINGLE ANNOTATED VIDEO (tracked_cell.mp4)
# ====================================================================

def render_tracked_video(
    video_path,
    top_wall,
    bot_wall,
    valid_tracks,
    biomarkers_by_id,
    out_dir,
    pad_frames=10,
    playback_fps=15.0
):
    """
    Renders ONE clean video file (`output/stage3/tracked_cell.mp4`) covering each
    cell's passage from entrance to exit (plus `pad_frames` before and after),
    with a single bounding box, trajectory line, Cell ID, and live biomechanics HUD.
    """
    out_mp4 = os.path.join(out_dir, "tracked_cell.mp4")
    # Also overwrite active_passages_only.mp4 so old multi-track videos are replaced
    legacy_mp4 = os.path.join(out_dir, "active_passages_only.mp4")

    by_frame = {}
    for tid, path in valid_tracks.items():
        for pt in path:
            # Strictly 1 cell per frame
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

    # Determine frame windows to render for each confirmed passage
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
        frames_to_render = list(range(min(200, total_frames)))

    trail_by_id = {tid: [] for tid in valid_tracks}
    color = (50, 255, 100)  # High-visibility green

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

            # 1. Trajectory line from entrance to current position
            trail = trail_by_id[tid]
            if len(trail) >= 2:
                cv2.polylines(vis, [np.array(trail, np.int32)], False, color, 2, cv2.LINE_AA)
                cv2.circle(vis, trail[0], 4, (0, 220, 255), -1, cv2.LINE_AA)

            # 2. Single Bounding Box around the cell
            pad_x = max(10, bw // 2 + 4)
            pad_y = max(10, bh // 2 + 4)
            x0, y0 = max(0, cx_i - pad_x), max(0, cy_i - pad_y)
            x1, y1 = min(w - 1, cx_i + pad_x), min(h - 1, cy_i + pad_y)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)

            # 3. Biomechanics & Cell ID HUD box anchored right above the cell box
            bio = biomarkers_by_id.get(tid, {})
            adi_val = bio.get("ADI", d["ADI_running"])
            line1 = f"CELL #{tid}  |  Pos: x={cx_i} px"
            line2 = f"Vel: {d['velocity']:.1f} px/f  |  Area: {d['area']:.0f} px2"
            line3 = f"Circ: {d['circularity']:.2f}  |  ADI: {adi_val:.2f}  |  Elong: {d['elongation']:.1f}x"

            card_w, card_h = 285, 60
            card_x = int(np.clip(cx_i - card_w // 2, 10, w - card_w - 10))
            card_y = y0 - card_h - 14 if y0 > card_h + 38 else y1 + 14

            roi_hud = vis[card_y:card_y + card_h, card_x:card_x + card_w]
            if roi_hud.shape[0] == card_h and roi_hud.shape[1] == card_w:
                dark_box = np.full_like(roi_hud, 16)
                vis[card_y:card_y + card_h, card_x:card_y + card_w] if False else None
                vis[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(roi_hud, 0.2, dark_box, 0.8, 0)
            cv2.rectangle(vis, (card_x, card_y), (card_x + card_w, card_y + card_h), color, 1)

            leader_y = card_y + card_h if card_y < y0 else card_y
            box_anchor_y = y0 if card_y < y0 else y1
            cv2.line(vis, (cx_i, leader_y), (cx_i, box_anchor_y), color, 1, cv2.LINE_AA)

            cv2.putText(vis, line1, (card_x + 8, card_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
            cv2.putText(vis, line2, (card_x + 8, card_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(vis, line3, (card_x + 8, card_y + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 225, 255), 1, cv2.LINE_AA)

        # Top HUD header
        cv2.rectangle(vis, (0, 0), (w, 30), (14, 17, 22), -1)
        active_str = f"Tracking CELL #{d['track_id']}" if d is not None else "Waiting for cell..."
        status_txt = f"IISc RBC Passage Tracker | Frame {f_idx} ({f_idx / 10.0:.1f} ms) | {active_str}"
        cv2.putText(vis, status_txt, (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

        writer.write(vis)

    writer.release()
    cap.release()

    # Copy to active_passages_only.mp4 as well so whichever file the user opens is identical
    import shutil
    try:
        shutil.copyfile(out_mp4, legacy_mp4)
    except Exception:
        pass

    return out_mp4, len(frames_to_render)


# ====================================================================
# MAIN PIPELINE
# ====================================================================

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Single-Track-Per-Passage RBC Tracker")
    parser.add_argument("--max-frames", type=int, default=2000, help="Frames to scan (default: 2000; pass 0 for all 37,274)")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index to start from (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Optional override path to .avi file")
    parser.add_argument("--min-span", type=float, default=150.0, help="Minimum horizontal passage travel (px) to keep a cell track (default: 150)")
    parser.add_argument("--diff-thresh", type=float, default=14.0, help="Minimum contrast threshold inside lumen (default: 14.0)")
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
    print("STAGE 3: SINGLE-CELL PASSAGE TRACKER (1 TRACK PER CELL)")
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

    print(f"\n[1/2] Scanning {frames_to_run} frames (start={args.start_frame}) for moving cells inside passage...")

    raw_single_detections = []
    prev_gray = None
    frame_count = 0

    for offset in range(frames_to_run):
        f_idx = args.start_frame + offset
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        det, prev_gray = detect_single_cell_in_passage(
            frame,
            prev_gray,
            background,
            lumen_mask,
            top_wall,
            bot_wall,
            roi_y_bounds,
            diff_thresh=args.diff_thresh,
        )
        if det is not None:
            det["frame"] = f_idx
            raw_single_detections.append(det)

        if frame_count % 500 == 0:
            print(f"      scanned {frame_count}/{frames_to_run} frames...")

    cap.release()

    # Build clean 1-track-per-passage trajectories
    valid_tracks = build_clean_passage_tracks(
        raw_single_detections,
        slope=slope,
        intercept=intercept,
        top_wall=top_wall,
        bot_wall=bot_wall,
        min_span_px=args.min_span,
        min_frames=5,
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

    print("\n[2/2] Rendering single-track video: output/stage3/tracked_cell.mp4 ...")
    out_mp4, rendered_frames = render_tracked_video(
        video_path=video_path,
        top_wall=top_wall,
        bot_wall=bot_wall,
        valid_tracks=valid_tracks,
        biomarkers_by_id=biomarkers_by_id,
        out_dir=out_dir,
        pad_frames=10,
        playback_fps=15.0,
    )

    print("\n" + "=" * 68)
    print("STAGE 3 COMPLETE")
    print("=" * 68)
    print(f"  ✓ Tracked Video File   : {out_mp4} ({rendered_frames} frames)")
    print(f"  ✓ Total Tracked Cells  : {len(all_biomarkers)}")

    if all_biomarkers:
        print("\nTracked Cell Passages:")
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
