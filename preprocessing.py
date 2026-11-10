"""
STAGE 1 — Preprocessing + Calibration for deterministic RBC tracking

Run this once to generate:
- calibration.json (centerline, width profile, background model)
- output/preprocessed/ (sample preprocessed frames for QA)
"""
import cv2
import numpy as np
import json, os, sys
from pathlib import Path

# ====================================================================
# PATHS (Auto-detected across machines, with CLI / env override)
# ====================================================================
ROOT_DIR = Path(__file__).resolve().parent
VIDEO_FILENAME = "2A002 2_C001H001S0001-001.avi"


def resolve_video_path():
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        return str(Path(sys.argv[1]).resolve())
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
    return str(ROOT_DIR / VIDEO_FILENAME)


VIDEO = resolve_video_path()
OUT = str(ROOT_DIR / "output")
FPS = 10000  # from .cihx metadata


# ====================================================================
# CALIBRATION FUNCTIONS
# ====================================================================

def build_median_background(video_path, n_samples=150):
    """Temporal median across evenly spaced frames."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Could not find AVI file at: {video_path}\n"
            f"Place '{VIDEO_FILENAME}' in {ROOT_DIR} or pass its path: python preprocessing.py <path_to_avi>"
        )
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"OpenCV could not read frames from {video_path}")
    step = max(1, total // n_samples)
    frames = []
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    arr = np.array(frames)
    return np.median(arr, axis=0), len(frames), total


def fit_centerline(bg, x_start=90, x_end=None, n_iter=4):
    """Sigma-clipped least-squares fit of the dark channel line."""
    h, w = bg.shape
    if x_end is None:
        x_end = w - 90
    xs = np.arange(x_start, x_end, dtype=np.float64)
    ys = np.array([bg[:, int(x)].argmin() for x in xs], dtype=np.float64)

    for _ in range(n_iter):
        A = np.polyfit(xs, ys, 1)
        r = ys - np.polyval(A, xs)
        keep = np.abs(r) < 2.0 * r.std()
        if keep.sum() < 20:
            break
        xs, ys = xs[keep], ys[keep]

    A = np.polyfit(xs, ys, 1)
    resid = ys - np.polyval(A, xs)
    return float(A[0]), float(A[1]), float(resid.std()), int(len(xs))


def measure_width(bg, slope, intercept, x0, half_len=40, smooth=5):
    """Channel width at position x0, measured perpendicular to the centreline."""
    h, w = bg.shape
    ang = np.arctan(slope)
    nx, ny = -np.sin(ang), np.cos(ang)

    y0 = slope * x0 + intercept
    ts = np.arange(-half_len, half_len + 1)
    xs = np.clip(x0 + ts * nx, 0, w - 1)
    ys = np.clip(y0 + ts * ny, 0, h - 1)

    prof = cv2.remap(bg.astype(np.float32),
                     xs.astype(np.float32).reshape(1, -1),
                     ys.astype(np.float32).reshape(1, -1),
                     cv2.INTER_LINEAR).ravel()

    if smooth > 1:
        prof = np.convolve(prof, np.ones(smooth) / smooth, mode='same')

    c = half_len
    lo, hi = c - 25, c + 25
    lo, hi = max(3, lo), min(len(prof) - 4, hi)

    win = prof[lo:hi]
    if win.size < 10:
        return None

    left_region = prof[max(0, lo):c]
    right_region = prof[c:hi]
    if left_region.size < 3 or right_region.size < 3:
        return None

    i_left = np.argmin(left_region) + max(0, lo)
    i_right = np.argmin(right_region) + c

    width = float(ts[i_right] - ts[i_left])
    if width <= 0 or width > 2 * half_len:
        return None
    return width, float(ts[i_left]), float(ts[i_right])


def width_profile(bg, slope, intercept, step=15, half_len=40):
    h, w = bg.shape
    xs = np.arange(120, w - 120, step)
    out = []
    for x0 in xs:
        m = measure_width(bg, slope, intercept, float(x0), half_len=half_len)
        if m is None:
            continue
        width, l, r = m
        out.append({"x": float(x0), "y": float(slope * x0 + intercept),
                    "width_px": width, "n_left": l, "n_right": r})
    return out


# ====================================================================
# PREPROCESSING FUNCTIONS
# ====================================================================

def bilateral_filter(image, sigma_color=75, sigma_space=3):
    """Edge-preserving blur for microscopy."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def clahe_apply(img, clip_limit=2.0, grid=8):
    """CLAHE on a single channel image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    return clahe.apply(img.astype(np.uint8))


def straighten_frame(frame, slope, angle_deg):
    """Rotate frame to straighten the channel, then crop the band."""
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), -angle_deg, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    gray_rot = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    y_line = int(np.median([gray_rot[:, x].argmin() for x in range(0, w, 50)]))
    y0, y1 = max(0, y_line - 40), min(h, y_line + 40)
    return rotated[y0:y1, :]


def preprocess_frame(gray, bg):
    """Generate static-contrast signal."""
    bs = gray.astype(np.float32) - bg
    bs = np.clip(bs, -50, 50)
    bs = bilateral_filter(bs.astype(np.uint8), sigma_color=80, sigma_space=5)
    bs_clahe = clahe_apply(bs)
    return bs_clahe


# ====================================================================
# MAIN
# ====================================================================

def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "preprocessed"), exist_ok=True)

    print("=" * 60)
    print("STAGE 1: PREPROCESSING + CALIBRATION")
    print("=" * 60)
    print(f"Video path : {VIDEO}")
    print(f"Output dir : {OUT}")

    print("\n[1/3] Building median background ...", flush=True)
    bg, n_used, total_frames = build_median_background(VIDEO, n_samples=150)
    np.save(os.path.join(OUT, "median_bg.npy"), bg)
    cv2.imwrite(os.path.join(OUT, "median_bg.png"), bg.astype(np.uint8))
    print(f"      {n_used} frames sampled, bg range {bg.min():.0f}-{bg.max():.0f}")

    print("[2/3] Fitting channel centerline ...", flush=True)
    slope, intercept, rstd, npts = fit_centerline(bg)
    angle_deg = np.degrees(np.arctan(slope))
    print(f"      y = {slope:.6f}*x + {intercept:.3f}   resid={rstd:.2f}px ({npts} cols)")

    print("[3/3] Measuring width profile ...", flush=True)
    wp = width_profile(bg, slope, intercept)
    widths = np.array([p["width_px"] for p in wp])
    print(f"      {len(wp)} positions: min={widths.min():.1f} px, "
          f"median={np.median(widths):.1f} px, max={widths.max():.1f} px")

    calib = {
        "video": VIDEO,
        "fps": FPS,
        "frame_count": total_frames,
        "resolution": [int(bg.shape[1]), int(bg.shape[0])],
        "background": {"n_samples": n_used, "file": "median_bg.npy"},
        "channel_line": {"slope": slope, "intercept": intercept, "resid_std_px": rstd},
        "angle_deg": angle_deg,
        "width_profile": wp,
        "width_summary": {"min_px": float(widths.min()), "median_px": float(np.median(widths)), "max_px": float(widths.max())},
        "notes": "Pixel units only - physical calibration not available in .cihx."
    }
    with open(os.path.join(OUT, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\n      ✓ Saved calibration.json")

    print("\n[4/4] Generating preprocessed signals (first 100 frames)...", flush=True)
    cap = cv2.VideoCapture(VIDEO)
    sample_count = min(100, total_frames)

    for i in range(sample_count):
        ok, f = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        signal = preprocess_frame(gray, bg)
        straight = straighten_frame(f, slope, angle_deg)

        np.save(os.path.join(OUT, "preprocessed", f"frame_{i:04d}_signal.npy"), signal)
        cv2.imwrite(os.path.join(OUT, "preprocessed", f"frame_{i:04d}_straight.png"), straight)

        if (i + 1) % 25 == 0:
            print(f"      processed {i+1}/{sample_count} frames")

    cap.release()
    print(f"\n      ✓ Saved {sample_count} preprocessed frames to output/preprocessed/")

    print("\n" + "=" * 60)
    print("STAGE 1 COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Inspect output/preprocessed/ to verify signal quality")
    print("  2. Proceed to Stage 3: python tracking.py")


if __name__ == "__main__":
    main()
