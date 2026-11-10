"""
VISUALIZE TRACKS — Stage 3 Visual QA & Interactive Track Explorer

Reads:
  - output/calibration.json
  - output/stage3/trajectories.csv
  - output/stage3/biomechanics.csv
  - The raw .avi video (for background frame & annotated MP4 playback)

Generates in output/stage3/:
  1. tracks_overview.png  — Composite trajectory map overlaid on the microscope frame
  2. tracks_overlay.mp4   — Slow-motion annotated video showing live track IDs + trails
  3. tracks_viewer.html   — Interactive frame-by-frame browser viewer of your real tracks
"""
import os
import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
VIDEO_FILENAME = "2A002 2_C001H001S0001-001.avi"


def resolve_out_dir():
    local_out = ROOT_DIR / "output"
    legacy_out = Path(r"C:\Users\amara\rbc_tracking\output")
    if (local_out / "calibration.json").is_file() or not (legacy_out / "calibration.json").is_file():
        return local_out
    return legacy_out


def resolve_video_path(calib_video):
    if calib_video and os.path.isfile(calib_video):
        return calib_video
    candidates = [
        ROOT_DIR / VIDEO_FILENAME,
        ROOT_DIR.parent / VIDEO_FILENAME,
        Path.home() / "Documents" / "IIsc" / VIDEO_FILENAME,
        Path.home() / "Desktop" / "rbctrack" / VIDEO_FILENAME,
        Path.home() / "Downloads" / VIDEO_FILENAME,
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return calib_video


def track_color_bgr(track_id):
    """Deterministic vivid color per track_id in BGR."""
    golden_ratio = 0.618033988749895
    hue = int(((track_id * golden_ratio) % 1.0) * 180)
    hsv = np.uint8([[[hue, 230, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_channel_corridor(img, width_profile, y_margin=12.0):
    """Draw the calibrated channel centerline and corridor bounds."""
    if not width_profile:
        return
    xs = [p["x"] for p in width_profile]
    ys = [p["y"] for p in width_profile]
    ws = [p["width_px"] for p in width_profile]

    upper_pts = []
    lower_pts = []
    center_pts = []
    for x, y, w in zip(xs, ys, ws):
        half = 0.5 * w + y_margin
        center_pts.append((int(round(x)), int(round(y))))
        upper_pts.append((int(round(x)), int(round(y - half))))
        lower_pts.append((int(round(x)), int(round(y + half))))

    cv2.polylines(img, [np.array(upper_pts, np.int32)], False, (120, 180, 80), 1, cv2.LINE_AA)
    cv2.polylines(img, [np.array(lower_pts, np.int32)], False, (120, 180, 80), 1, cv2.LINE_AA)
    cv2.polylines(img, [np.array(center_pts, np.int32)], False, (80, 80, 200), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Visualize Stage 3 RBC Tracks")
    parser.add_argument("--max-frames", type=int, default=200, help="Frames to render in MP4 video (default: 200)")
    parser.add_argument("--y-margin", type=float, default=12.0, help="Corridor Y margin shown on overlay")
    args = parser.parse_args()

    out_dir = resolve_out_dir()
    stage3_dir = out_dir / "stage3"
    calib_file = out_dir / "calibration.json"
    bg_png = out_dir / "median_bg.png"
    traj_csv = stage3_dir / "trajectories.csv"
    bio_csv = stage3_dir / "biomechanics.csv"

    if not traj_csv.is_file():
        print(f"Error: {traj_csv} not found. Run 'python tracking.py --max-frames 200' first.")
        return

    with open(calib_file, "r", encoding="utf-8") as f:
        calib = json.load(f)

    width_profile = calib.get("width_profile", [])
    video_path = resolve_video_path(calib.get("video"))

    # Load trajectories
    trajectories = []
    by_frame = defaultdict(list)
    by_track = defaultdict(list)
    with open(traj_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "frame": int(row["frame"]),
                "track_id": int(row["track_id"]),
                "cx": float(row["cx"]),
                "cy": float(row["cy"]),
                "area": float(row["area"]),
                "circularity": float(row["circularity"]),
            }
            trajectories.append(item)
            by_frame[item["frame"]].append(item)
            by_track[item["track_id"]].append(item)

    # Load biomechanics
    biomarkers = []
    if bio_csv.is_file():
        with open(bio_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                biomarkers.append({
                    "track_id": int(row["track_id"]),
                    "velocity_mean": float(row["velocity_mean"]),
                    "velocity_max": float(row["velocity_max"]),
                    "area_mean": float(row["area_mean"]),
                    "area_min": float(row["area_min"]),
                    "circularity_mean": float(row["circularity_mean"]),
                    "circularity_min": float(row["circularity_min"]),
                    "ADI": float(row["ADI"]),
                    "shape_recovery": float(row["shape_recovery"]) if row.get("shape_recovery") not in (None, "", "None") else None,
                    "track_length_frames": int(row["track_length_frames"]),
                    "entrance_frame": int(row["entrance_frame"]),
                    "exit_frame": int(row["exit_frame"]),
                    "duration_ms": float(row["duration_ms"]),
                })

    print("=" * 60)
    print("STAGE 3 TRACK VISUALIZER")
    print("=" * 60)
    print(f"Loaded {len(trajectories)} trajectory detections across {len(by_track)} tracks.")

    # 1. Build composite overview image (tracks_overview.png)
    if bg_png.is_file():
        base_img = cv2.imread(str(bg_png), cv2.IMREAD_COLOR)
    else:
        base_img = np.full((480, 1280, 3), 40, dtype=np.uint8)

    overview = base_img.copy()
    draw_channel_corridor(overview, width_profile, y_margin=args.y_margin)

    for tid, pts in by_track.items():
        pts_sorted = sorted(pts, key=lambda p: p["frame"])
        color = track_color_bgr(tid)
        coords = [(int(round(p["cx"])), int(round(p["cy"]))) for p in pts_sorted]
        if len(coords) >= 2:
            cv2.polylines(overview, [np.array(coords, np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.circle(overview, coords[-1], 3, color, -1, cv2.LINE_AA)
        if len(coords) >= 5:
            cv2.putText(
                overview, f"#{tid}", (coords[0][0] + 4, coords[0][1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA
            )

    overview_path = stage3_dir / "tracks_overview.png"
    cv2.imwrite(str(overview_path), overview)
    print(f"  ✓ Saved composite trajectory map : {overview_path}")

    # 2. Build annotated slow-motion MP4 video (tracks_overlay.mp4)
    mp4_path = stage3_dir / "tracks_overlay.mp4"
    if video_path and os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, 20.0, (w, h))

        max_f = max(by_frame.keys()) if by_frame else 0
        n_render = min(args.max_frames, max_f + 1)
        history = defaultdict(list)

        for f_idx in range(n_render):
            ok, frame = cap.read()
            if not ok:
                break
            vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            draw_channel_corridor(vis, width_profile, y_margin=args.y_margin)

            active_dets = by_frame.get(f_idx, [])
            for d in active_dets:
                tid = d["track_id"]
                pt = (int(round(d["cx"])), int(round(d["cy"])))
                history[tid].append(pt)
                color = track_color_bgr(tid)

                # Draw trail
                trail = history[tid][-25:]
                if len(trail) >= 2:
                    cv2.polylines(vis, [np.array(trail, np.int32)], False, color, 2, cv2.LINE_AA)

                # Draw cell circle + ID label
                radius = max(4, int(round(np.sqrt(d["area"] / np.pi))))
                cv2.circle(vis, pt, radius + 2, color, 1, cv2.LINE_AA)
                cv2.putText(
                    vis, f"ID {tid}", (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA
                )

            cv2.putText(
                vis, f"Frame {f_idx} | Active Tracked Cells: {len(active_dets)}",
                (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
            )
            writer.write(vis)

        writer.release()
        cap.release()
        print(f"  ✓ Saved annotated playback video : {mp4_path}")

    # 3. Build self-contained interactive HTML viewer (tracks_viewer.html)
    viewer_html_path = stage3_dir / "tracks_viewer.html"
    data_payload = {
        "width_profile": width_profile,
        "trajectories": trajectories,
        "biomarkers": sorted(biomarkers, key=lambda b: b["track_length_frames"], reverse=True),
    }
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IISc RBC Track Explorer — Stage 3 Output</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f1419; color: #e2e8f0; margin: 0; padding: 24px; }}
  .wrap {{ max-width: 1280px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 6px; }}
  .sub {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 18px; }}
  .canvas-card {{ background: #161d26; border: 1px solid #2d3748; border-radius: 8px; padding: 16px; margin-bottom: 20px; }}
  canvas {{ width: 100%; height: auto; background: #090d11; border: 1px solid #2d3748; border-radius: 6px; display: block; }}
  .controls {{ display: flex; align-items: center; gap: 14px; margin-top: 12px; font-family: monospace; }}
  input[type=range] {{ flex: 1; accent-color: #10b981; }}
  button {{ background: #10b981; color: #06281e; border: none; padding: 8px 14px; border-radius: 5px; font-weight: 700; cursor: pointer; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-family: monospace; font-size: 0.82rem; background: #161d26; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #263140; text-align: left; }}
  th {{ background: #1e293b; color: #94a3b8; }}
  tr:hover {{ background: #243042; cursor: pointer; }}
  tr.selected {{ background: #064e3b; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>IISc RBC Track Explorer (Real Stage 3 Output)</h1>
  <div class="sub">Click any row in the table below to highlight that cell's trajectory, or press Play / drag the frame slider.</div>
  <div class="canvas-card">
    <canvas id="c" width="1280" height="480"></canvas>
    <div class="controls">
      <button id="playBtn" onclick="togglePlay()">Play</button>
      <span>Frame: <strong id="fLabel">0</strong></span>
      <input type="range" id="fSlider" min="0" max="199" value="0" oninput="setFrame(this.value)">
      <button onclick="selectedTrack=null; draw();" style="background:#334155; color:#fff;">Show All Tracks</button>
    </div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Track ID</th>
        <th>Length (Frames)</th>
        <th>Frame Span</th>
        <th>Mean Vel (px/f)</th>
        <th>Mean Area (px²)</th>
        <th>Min Area (px²)</th>
        <th>ADI</th>
        <th>Mean Circ</th>
      </tr>
    </thead>
    <tbody id="tb"></tbody>
  </table>
</div>
<script>
const DATA = {json.dumps(data_payload)};
let currentFrame = 0;
let playing = false;
let timer = null;
let selectedTrack = null;

const byFrame = {{}};
const byTrack = {{}};
let maxFrame = 0;
DATA.trajectories.forEach(t => {{
  if (!byFrame[t.frame]) byFrame[t.frame] = [];
  byFrame[t.frame].push(t);
  if (!byTrack[t.track_id]) byTrack[t.track_id] = [];
  byTrack[t.track_id].push(t);
  if (t.frame > maxFrame) maxFrame = t.frame;
}});
document.getElementById('fSlider').max = maxFrame;

function colorFor(tid) {{
  const h = Math.floor(((tid * 0.6180339887) % 1.0) * 360);
  return `hsl(${{h}}, 85%, 60%)`;
}}

function setFrame(v) {{
  currentFrame = Number(v);
  document.getElementById('fLabel').textContent = currentFrame;
  document.getElementById('fSlider').value = currentFrame;
  draw();
}}

function togglePlay() {{
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
  if (playing) {{
    timer = setInterval(() => {{
      setFrame((currentFrame + 1) % (maxFrame + 1));
    }}, 60);
  }} else {{
    clearInterval(timer);
  }}
}}

function draw() {{
  const c = document.getElementById('c');
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#090d11';
  ctx.fillRect(0, 0, c.width, c.height);

  // Draw calibrated channel corridor
  const wp = DATA.width_profile;
  if (wp.length > 1) {{
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.45)';
    ctx.lineWidth = 1.5;
    [-1, 1].forEach(sign => {{
      ctx.beginPath();
      wp.forEach((p, idx) => {{
        const y = p.y + sign * (0.5 * p.width_px + 12);
        if (idx === 0) ctx.moveTo(p.x, y);
        else ctx.lineTo(p.x, y);
      }});
      ctx.stroke();
    }});
  }}

  // Draw faint full trajectories
  Object.keys(byTrack).forEach(tidStr => {{
    const tid = Number(tidStr);
    if (selectedTrack !== null && selectedTrack !== tid) return;
    const pts = byTrack[tid];
    ctx.strokeStyle = selectedTrack === tid ? '#fff' : colorFor(tid);
    ctx.globalAlpha = selectedTrack === tid ? 0.95 : 0.35;
    ctx.lineWidth = selectedTrack === tid ? 3 : 1.5;
    ctx.beginPath();
    pts.forEach((p, i) => {{
      if (i === 0) ctx.moveTo(p.cx, p.cy);
      else ctx.lineTo(p.cx, p.cy);
    }});
    ctx.stroke();
  }});
  ctx.globalAlpha = 1.0;

  // Draw active detections on currentFrame
  const dets = byFrame[currentFrame] || [];
  dets.forEach(d => {{
    if (selectedTrack !== null && selectedTrack !== d.track_id) return;
    const col = colorFor(d.track_id);
    const r = Math.max(5, Math.sqrt(d.area / Math.PI));
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(d.cx, d.cy, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = '12px monospace';
    ctx.fillText('#' + d.track_id, d.cx + 8, d.cy - 6);
  }});
}}

const tb = document.getElementById('tb');
DATA.biomarkers.slice(0, 50).forEach(b => {{
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td><strong style="color:${{colorFor(b.track_id)}}">#${{b.track_id}}</strong></td>
    <td>${{b.track_length_frames}}</td>
    <td>${{b.entrance_frame}} &rarr; ${{b.exit_frame}}</td>
    <td>${{b.velocity_mean.toFixed(2)}}</td>
    <td>${{b.area_mean.toFixed(1)}}</td>
    <td>${{b.area_min.toFixed(1)}}</td>
    <td>${{b.ADI.toFixed(3)}}</td>
    <td>${{b.circularity_mean.toFixed(3)}}</td>
  `;
  tr.onclick = () => {{
    selectedTrack = b.track_id;
    setFrame(b.entrance_frame);
    document.querySelectorAll('#tb tr').forEach(r => r.classList.remove('selected'));
    tr.classList.add('selected');
  }};
  tb.appendChild(tr);
}});
draw();
</script>
</body>
</html>"""
    with open(viewer_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  ✓ Saved interactive HTML explorer: {viewer_html_path}")
    print("\nTo view right now in Windows:")
    print(f"  start {viewer_html_path}")
    print(f"  start {overview_path}")


if __name__ == "__main__":
    main()
