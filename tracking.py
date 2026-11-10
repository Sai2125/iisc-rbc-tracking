"""
STAGE 3 — Detection + Tracking for deterministic RBC tracking

Uses Stage 1 calibration to:
1. Detect cells per frame (background subtraction + Otsu + watershed)
2. Track trajectories (Kalman filter + Hungarian assignment)
3. Compute biomechanics (velocity, ADI, circularity, elongation)

Output files:
- output/stage3/trajectories.csv (per-frame cell positions)
- output/stage3/biomechanics.csv (per-cell metrics)
- output/stage3/track_summary.json (per-track statistics)
"""
import cv2
import numpy as np
import json, os, csv, multiprocessing
from multiprocessing import Pool
from collections import defaultdict

# ====================================================================
# PATHS
# ====================================================================
OUT = r"C:\Users\amara\rbc_tracking\output"
CALIB_FILE = os.path.join(OUT, "calibration.json")
BG_FILE = os.path.join(OUT, "median_bg.npy")
VIDEO = None  # Will be read from calibration.json

# ====================================================================
# LOAD CALIBRATION
# ====================================================================

def load_calibration():
    with open(CALIB_FILE, "r") as f:
        return json.load(f)

def load_background():
    return np.load(BG_FILE)

# ====================================================================
# DETECTION: Background Subtraction + Otsu + Watershed
# ====================================================================

def detect_cells(frame, background, width_profile_dict, verbose=False):
    """
    Detect cells in a single frame using:
    1. Background subtraction
    2. Otsu thresholding (adaptive per frame)
    3. Connected components with watershed for touching cells
    4. Morphological filtering

    Returns: list of (center_x, center_y, area, circularity)
    """
    # 1. Background subtraction (static contrast)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = gray - background
    diff = np.clip(diff, -50, 50)  # Limit outliers
    diff = np.abs(diff)

    # 2. Edge-preserving blur
    diff_uint8 = cv2.convertScaleAbs(diff.astype(np.uint8), alpha=1.5, beta=0)
    diff_blur = cv2.bilateralFilter(diff_uint8, d=9, sigmaColor=75, sigmaSpace=3)

    # 3. Otsu thresholding (auto-threshold per frame)
    _, thresh = cv2.threshold(
        diff_blur.astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 4. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 5. Distance transform + watershed for touching cells
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    _, dist_thresh = cv2.threshold(
        dist, 0.5 * dist.max(), 255, 0
    )
    dist_thresh = dist_thresh.astype(np.uint8)

    # Mark sure background (outside channel)
    sure_bg = cv2.distanceTransform(255 - thresh, cv2.DIST_L2, 5)
    _, sure_bg_thresh = cv2.threshold(
        sure_bg, 2, 255, 0
    )
    sure_bg_thresh = sure_bg_thresh.astype(np.uint8)

    # Mark sure foreground (cells)
    sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, 0)[1].astype(np.uint8)

    # Mark regions of interest (unknown)
    unknown = cv2.subtract(sure_bg_thresh, sure_fg)
    unknown = cv2.dilate(unknown, kernel, iterations=1)
    unknown = cv2.erode(unknown, kernel, iterations=1)

    # Watershed markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.watershed(thresh_color, markers)

    # Extract cell contours in a single pass over the foreground mask
    # (Watershed boundary pixels have markers == -1 and background is <= 0,
    # so positive label islands are 8-disconnected and can be extracted together,
    # then sorted by label to preserve exact detection ordering.)
    fg_mask = (markers > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_to_cnt = {}
    for cnt in cnts:
        x0, y0 = cnt[0, 0]
        lbl = int(markers[y0, x0])
        if lbl > 0 and lbl not in label_to_cnt:
            label_to_cnt[lbl] = cnt

    cell_data = []
    for label in sorted(label_to_cnt.keys()):
        cnt = label_to_cnt[label]

        # Filter by area (remove noise)
        area = cv2.contourArea(cnt)
        if area < 20 or area > 5000:  # Adjust thresholds based on your data
            continue

        # Compute centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Check if inside channel (verify with width profile)
        # Get local channel width at this x position
        local_width = get_local_width(width_profile_dict, float(cx))
        if local_width is None:
            continue

        # Compute circularity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        cell_data.append((cx, cy, area, circularity))

    return cell_data


def get_local_width(width_profile, x_coord):
    """Interpolate channel width at x position."""
    if not width_profile:
        return None

    xs = [p["x"] for p in width_profile]
    widths = [p["width_px"] for p in width_profile]

    if x_coord < xs[0] or x_coord > xs[-1]:
        return None

    # Linear interpolation
    for i in range(len(xs) - 1):
        if xs[i] <= x_coord <= xs[i + 1]:
            t = (x_coord - xs[i]) / (xs[i + 1] - xs[i])
            w = widths[i] + t * (widths[i + 1] - widths[i])
            return w

    return None


# ====================================================================
# TRACKING: Kalman Filter + Hungarian Algorithm
# ====================================================================

def _create_kalman_filter(cx, cy):
    """Initialize a 4-state (x, y, vx, vy), 2-measurement (x, y) OpenCV Kalman filter."""
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.float32
    )
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    state = np.array([[float(cx)], [float(cy)], [0.0], [0.0]], dtype=np.float32)
    kf.statePre = state.copy()
    kf.statePost = state.copy()
    return kf


class CellTracker:
    def __init__(self, max_age=30, min_hits=3):
        """
        Track cells across frames using Kalman filter.

        Args:
            max_age: Maximum frames to keep a track without detection
            min_hits: Minimum detections to create a new track
        """
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks = {}  # track_id -> {"last_seen": frame, "state": Kalman, "hits": 1, "path": [...]}
        self.completed_paths = {}  # track_id -> full path [(frame, cx, cy, area, circ), ...] for confirmed tracks
        self.next_track_id = 0
        self.frame_num = 0

    def predict(self):
        """Predict all active tracks."""
        for tid, track in self.tracks.items():
            if track["state"] is None:
                continue
            track["state"].predict()

    def associate(self, detections):
        """
        Assign detections to tracks using Hungarian algorithm.

        Args:
            detections: list of (cx, cy, area, circularity)

        Returns:
            (matched_tracks, matched_dets, untracked_detections, untracked_tracks)
        """
        track_indices = list(self.tracks.keys())
        det_indices = list(range(len(detections)))
        n_tracks = len(track_indices)
        n_dets = len(detections)

        # Handle edge cases
        if n_tracks == 0 or n_dets == 0:
            return [], [], list(det_indices), list(track_indices)

        # Filter to tracks with valid Kalman states only
        valid_tracks = [tid for tid in track_indices if self.tracks[tid]["state"] is not None]
        if not valid_tracks:
            return [], [], list(det_indices), list(track_indices)

        # Compute distances (squared Euclidean)
        n_valid = len(valid_tracks)
        cost_matrix = np.full((n_valid, n_dets), np.inf)

        for i, tid in enumerate(valid_tracks):
            track = self.tracks[tid]
            pred_pos = track["state"].statePost[:2].reshape(-1)
            for j, (cx, cy, area, circ) in enumerate(detections):
                dist = (cx - float(pred_pos[0]))**2 + (cy - float(pred_pos[1]))**2
                cost_matrix[i, j] = dist

        # Hungarian algorithm (with pure-NumPy fallback if scipy is absent)
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except ImportError:
            row_ind, col_ind = [], []
            work = cost_matrix.copy()
            for _ in range(min(n_valid, n_dets)):
                r, c = np.unravel_index(np.argmin(work), work.shape)
                if not np.isfinite(work[r, c]):
                    break
                row_ind.append(int(r))
                col_ind.append(int(c))
                work[r, :] = np.inf
                work[:, c] = np.inf

        # Filter matches (ignore high cost)
        max_cost = 100  # pixels^2 (10 px radius)
        matched_tracks = []
        matched_dets = []

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < max_cost:
                tid = valid_tracks[i]
                matched_tracks.append(tid)
                matched_dets.append(det_indices[j])

        return (
            matched_tracks,
            matched_dets,
            list(set(det_indices) - set(matched_dets)),
            list(set(track_indices) - set(matched_tracks))
        )

    def update(self, detections, matched_tracks, matched_dets):
        """Update tracks with new detections."""
        for i, tid in enumerate(matched_tracks):
            det_idx = matched_dets[i]
            cx, cy, area, circ = detections[det_idx]

            track = self.tracks[tid]
            if track["state"] is None:
                track["state"] = _create_kalman_filter(cx, cy)
                track["hits"] = 1
                track["last_seen"] = self.frame_num
                track["path"] = [(self.frame_num, cx, cy, area, circ)]
            else:
                measurement = np.array([[float(cx)], [float(cy)]], dtype=np.float32)
                track["state"].correct(measurement)
                track["hits"] += 1
                track["last_seen"] = self.frame_num
                track["path"].append((self.frame_num, cx, cy, area, circ))
                if track["hits"] >= self.min_hits:
                    self.completed_paths[tid] = track["path"]

        # Remove tracks that have aged out
        tracks_to_remove = []
        for tid, track in self.tracks.items():
            if track["last_seen"] >= self.frame_num - self.max_age:
                track["age"] = self.frame_num - track["last_seen"]
            else:
                tracks_to_remove.append(tid)

        for tid in tracks_to_remove:
            del self.tracks[tid]

    def add_new_tracks(self, untracked_dets, detections):
        """Create new tracks with initialized Kalman filters for untracked detections."""
        for det_idx in untracked_dets:
            cx, cy, area, circ = detections[det_idx]

            self.tracks[self.next_track_id] = {
                "state": _create_kalman_filter(cx, cy),
                "hits": 1,
                "last_seen": self.frame_num,
                "path": [(self.frame_num, cx, cy, area, circ)]
            }
            self.next_track_id += 1

    def process_frame(self, detections):
        """
        Process a single frame: predict, associate, update.

        Args:
            detections: list of (cx, cy, area, circularity)

        Returns:
            list of (track_id, cx, cy, area, circularity) for all active tracks
        """
        self.frame_num += 1

        # Predict all tracks
        self.predict()

        # Associate detections to tracks
        matched_tracks, matched_dets, unmatched_dets, unmatched_tracks = self.associate(detections)

        # Update tracks
        self.update(detections, matched_tracks, matched_dets)

        # Add new tracks
        self.add_new_tracks(unmatched_dets, detections)

        # Collect active tracks that were updated on this frame and reached min_hits
        results = []
        for tid, track in self.tracks.items():
            if track["hits"] < self.min_hits or track["last_seen"] != self.frame_num:
                continue
            last = track["path"][-1]
            frame, cx, cy, area, circ = last
            results.append((tid, cx, cy, area, circ))

        return results


# ====================================================================
# BIOMECHANICS CALCULATION
# ====================================================================

def compute_biomarker(track_path, width_profile, track_id=None):
    """
    Compute biomechanical metrics for a single cell track.

    Returns:
        dict with:
          - velocity_mean (px/frame)
          - velocity_max (px/frame)
          - area_mean (px²)
          - area_min (px²)
          - circularity_mean
          - ADI (Area Deformation Index)
          - elongation_ratio
    """
    if len(track_path) < 2:
        return None

    # Extract time series
    frames = [p[0] for p in track_path]
    areas = [p[3] for p in track_path]
    circs = [p[4] for p in track_path]
    positions = [(p[1], p[2]) for p in track_path]

    # 1. Velocity (per frame displacement)
    velocities = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        v = np.sqrt(dx**2 + dy**2)
        velocities.append(v)

    if not velocities:
        return None

    velocity_mean = np.mean(velocities)
    velocity_max = np.max(velocities)

    # 2. Area metrics
    area_mean = np.mean(areas)
    area_min = np.min(areas)

    # 3. Circularity metrics
    circ_mean = np.mean(circs)
    circ_min = np.min(circs)

    # 4. ADI (Area Deformation Index)
    # ADI = area_min / area_mean (normalized to [0,1])
    # Lower = more deformation
    ADI = area_min / area_mean if area_mean > 0 else 0

    # 5. Shape recovery (post-constriction)
    # Check if cell recovers shape after constriction
    shape_recovery = None
    if len(track_path) >= 5:
        mid_circ = np.mean(circs[len(circs)//3:2*len(circs)//3])
        post_circ = np.mean(circs[2*len(circs)//3:])
        shape_recovery = post_circ / mid_circ if mid_circ > 0 else None

    return {
        "track_id": int(track_id) if track_id is not None else int(track_path[0][0]),
        "velocity_mean": float(velocity_mean),
        "velocity_max": float(velocity_max),
        "area_mean": float(area_mean),
        "area_min": float(area_min),
        "circularity_mean": float(circ_mean),
        "circularity_min": float(circ_min),
        "ADI": float(ADI),
        "shape_recovery": float(shape_recovery) if shape_recovery is not None else None,
        "track_length_frames": len(track_path),
        "entrance_frame": frames[0],
        "exit_frame": frames[-1],
        "duration_ms": (frames[-1] - frames[0]) / 10  # 0.1ms/frame at 10kHz
    }


# ====================================================================
# MAIN PIPELINE
# ====================================================================

def main():
    calib = load_calibration()
    background = load_background()

    VIDEO = calib["video"]
    fps = calib["fps"]
    width_profile = calib["width_profile"]

    print("=" * 60)
    print("STAGE 3: DETECTION + TRACKING")
    print("=" * 60)

    # Initialize tracker
    tracker = CellTracker(max_age=100, min_hits=3)

    # Output directories
    out_dir = os.path.join(OUT, "stage3")
    os.makedirs(out_dir, exist_ok=True)

    # Trajectory data
    all_trajectories = []
    all_biomarkers = []

    # Video capture
    cap = cv2.VideoCapture(VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing {total_frames} frames at {fps} fps...")

    frame_count = 0
    for i in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        # Detect cells
        detections = detect_cells(frame, background, width_profile, verbose=(i % 1000 == 0))

        # Process with tracker
        results = tracker.process_frame(detections)

        # Store trajectories
        for tid, cx, cy, area, circ in results:
            all_trajectories.append({
                "frame": i,
                "track_id": tid,
                "cx": cx,
                "cy": cy,
                "area": area,
                "circularity": circ
            })

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{total_frames} frames")

    cap.release()

    # Compute biomechanics from tracks
    print("\nComputing biomechanical metrics...")
    track_paths = defaultdict(list)
    for traj in all_trajectories:
        key = traj["track_id"]
        track_paths[key].append((
            traj["frame"],
            traj["cx"],
            traj["cy"],
            traj["area"],
            traj["circularity"]
        ))
    # Use full path (including warm-up frames 1..min_hits-1) when available
    for tid, full_path in tracker.completed_paths.items():
        if len(full_path) >= len(track_paths.get(tid, [])):
            track_paths[tid] = list(full_path)

    for tid, path in track_paths.items():
        biomarker = compute_biomarker(path, width_profile, track_id=tid)
        if biomarker:
            all_biomarkers.append(biomarker)

    # Save results
    print(f"\nSaving results ({len(all_biomarkers)} tracks, {len(all_trajectories)} frames)...")

    # Trajectories CSV
    traj_csv = os.path.join(out_dir, "trajectories.csv")
    with open(traj_csv, "w", newline="") as f:
        if all_trajectories:
            writer = csv.DictWriter(f, fieldnames=all_trajectories[0].keys())
            writer.writeheader()
            writer.writerows(all_trajectories)
    print(f"  ✓ {traj_csv}")

    # Biomechanics CSV
    bio_csv = os.path.join(out_dir, "biomechanics.csv")
    with open(bio_csv, "w", newline="") as f:
        if all_biomarkers:
            writer = csv.DictWriter(f, fieldnames=all_biomarkers[0].keys())
            writer.writeheader()
            writer.writerows(all_biomarkers)
    print(f"  ✓ {bio_csv}")

    # Track summary JSON
    summary = {
        "total_tracks": len(all_biomarkers),
        "total_frames_analyzed": frame_count,
        "fps": fps,
        "video": VIDEO,
        "metrics": {
            "velocity_mean_px_per_frame": float(np.mean([b["velocity_mean"] for b in all_biomarkers])) if all_biomarkers else None,
            "velocity_max_px_per_frame": float(np.max([b["velocity_max"] for b in all_biomarkers])) if all_biomarkers else None,
            "ADI_mean": float(np.mean([b["ADI"] for b in all_biomarkers])) if all_biomarkers else None,
            "circularity_mean": float(np.mean([b["circularity_mean"] for b in all_biomarkers])) if all_biomarkers else None
        }
    }
    summary_json = os.path.join(out_dir, "tracking_summary.json" if False else "track_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ {summary_json}")

    print("\n" + "=" * 60)
    print("STAGE 3 COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {out_dir}")
    print(f"  - {len(all_biomarkers)} cell tracks")
    print(f"  - {len(all_trajectories)} frame-wise detections")
    print(f"\nNext: Stage 4 (Testing + Deployment)")


if __name__ == "__main__":
    main()
