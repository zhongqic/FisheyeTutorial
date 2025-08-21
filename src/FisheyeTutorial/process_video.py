from ultralytics import YOLO
import cv2
import time
import os
import numpy as np
import pandas as pd
import supervision as sv
from pathlib import Path
from collections import defaultdict
from datetime import timedelta, datetime
from PIL import Image
import time
import io
import base64
from IPython.display import display, HTML
from IPython.display import DisplayHandle
import torch


# Colab-friendly display helpers
try:
    from google.colab.patches import cv2_imshow  # Colab preferred
    HAS_COLAB_IMSHOW = True
except Exception:
    from IPython.display import display, clear_output
    from PIL import Image
    HAS_COLAB_IMSHOW = False

def ensure_dirs(out_dir: Path):
    rh_dir = out_dir / "RH"
    non_rh_dir = out_dir / "Non_RH"
    rh_dir.mkdir(parents=True, exist_ok=True)
    non_rh_dir.mkdir(parents=True, exist_ok=True)
    return rh_dir, non_rh_dir

def init_line_counter(frame_width: int, frame_height: int, line_pos=0.5,
                      move_right: str = "Up", move_left: str = "Down"):
    # define the start and end of the line
    start = sv.Point(int(frame_width * line_pos), -250)
    end   = sv.Point(int(frame_width * line_pos), frame_height)
    line_counter = sv.LineZone(start=start, end=end, triggering_anchors=[sv.Position.CENTER])
    line_annot   = sv.LineZoneAnnotator(
        thickness=1, text_thickness=1, text_scale=0.6,
        custom_in_text = move_right, custom_out_text = move_left,
        display_in_count = False, display_out_count = False
    )
    box_annot = sv.BoxAnnotator(thickness=1)
    return line_counter, line_annot, box_annot

class _FrameDisplayer:
    """Robust in-notebook image displayer with FPS pacing."""
    def __init__(self, target_fps=15, scale=0.75, jpeg_quality=85):
        self.target_fps = target_fps
        self.scale = scale
        self.jpeg_quality = jpeg_quality
        self.handle = None
        self._last_t = None
        self._min_dt = 1.0 / max(1, target_fps)

    def _ensure_handle(self):
        if self.handle is None:
            # Create a container so we can update it without flicker
            self.handle = display(HTML('<div id="video-container"></div>'), display_id=True)

    def _pace(self):
        now = time.perf_counter()
        if self._last_t is not None:
            dt = now - self._last_t
            if dt < self._min_dt:
                time.sleep(self._min_dt - dt)
        self._last_t = time.perf_counter()

    def show(self, frame_bgr, hud_text=None):
        # Pace output
        self._pace()

        overlay = frame_bgr
        if self.scale != 1.0:
            overlay = cv2.resize(overlay, None, fx=self.scale, fy=self.scale)

        if hud_text:
            cv2.putText(overlay, hud_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # BGR -> RGB -> JPEG (base64)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=self.jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode()

        self._ensure_handle()
        # Update the existing display with a new <img>; this survives interrupts/re-runs
        self.handle.update(HTML(
            f'<img src="data:image/jpeg;base64,{b64}" style="max-width:100%;height:auto;">'
        ))

    def reset(self):
        """Optional: call between runs to clear pacing state."""
        self._last_t = None

# Create one global displayer you can reuse across cells/runs.
try:
    _GLOBAL_DISPLAY
except NameError:
    _GLOBAL_DISPLAY = _FrameDisplayer(target_fps=15, scale=0.75, jpeg_quality=85)

def _show_frame(frame_bgr, line_counter, scale=0.75, fps=None, target_fps=20, move_right: str = "Up", move_left: str = "Down"):
    """Display using a robust DisplayHandle with pacing."""
    # Keep global displayer in sync with desired settings
    _GLOBAL_DISPLAY.scale = scale
    _GLOBAL_DISPLAY.target_fps = target_fps
    _GLOBAL_DISPLAY._min_dt = 1.0 / max(1, target_fps)

    hud = f"{move_right}: {line_counter.in_count}   {move_left}: {line_counter.out_count}"
    if fps is not None:
        hud += f"   FPS: {fps:.1f}"

    _GLOBAL_DISPLAY.show(frame_bgr, hud_text=hud)

def process_video(
    video_path: str,
    weights: str,
    out_dir: str,
    class_id: int = 0,
    conf_thresh: float = 0.7,
    line_pos: float = 0.5,
    imgsz=(480, 320),
    tracker_cfg="bytetrack.yaml",
    show: bool = False,
    show_every: int = 3,
    display_scale: float = 0.75,
    move_right: str = "Up",
    move_left: str = "Down"
):

    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rh_dir, non_rh_dir = ensure_dirs(out_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(weights).to(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_counter, line_annot, box_annot = init_line_counter(frame_w, frame_h, line_pos, move_right, move_left)

    conf_hist, time_hist, pos_hist, size_hist = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    events = []
    count_csv_path = out_dir / f"{video_path.stem}_count.csv"

    print(f"{datetime.now()}  start: {video_path.name}")

    frame_idx = 0
    t0 = time.time()
    fps_smooth = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Inference + tracking
        t_infer0 = time.time()
        results = model.track(frame, persist=True, tracker=tracker_cfg, imgsz=list(imgsz), verbose=False)
        t_infer1 = time.time()

        r0 = results[0]
        frame_annot = line_annot.annotate(frame=frame, line_counter=line_counter)
        
        if r0.boxes is None or r0.boxes.xywh is None:
            # Optionally still display the raw frame
            if show and frame_idx % show_every == 0:
                _show_frame(frame_annot, line_counter, scale=display_scale, fps=fps_smooth,
                           move_right=move_right, move_left=move_left)
            continue

        det = sv.Detections.from_ultralytics(r0)
        if r0.boxes.id is None:
            if show and frame_idx % show_every == 0:
                _show_frame(frame, line_counter, scale=display_scale, fps=fps_smooth,
                           move_right=move_right, move_left=move_left)
            continue
        det.track_id = r0.boxes.id.cpu().numpy().astype(int)

        # Filter for counting
        keep = (det.class_id == class_id) & (det.confidence >= conf_thresh)
        det_f = det[keep] if keep.any() else det[:0]

        # Annotate (boxes and line)
        frame_annot = box_annot.annotate(scene=frame.copy(), detections=det)
        frame_annot = line_annot.annotate(frame=frame_annot, line_counter=line_counter)

        crossed_in, crossed_out = line_counter.trigger(detections=det_f)

        # Histories for stats
        boxes_xywh = r0.boxes.xywh.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy() if r0.boxes.conf is not None else np.zeros(len(boxes_xywh))
        clss = r0.boxes.cls.cpu().numpy().astype(int).tolist()
        tids = r0.boxes.id.int().cpu().tolist()
        t_now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        for i, tid in enumerate(tids):
            x, y, w, h = boxes_xywh[i]
            conf_hist[tid].append(float(confs[i]))
            time_hist[tid].append(float(t_now))
            pos_hist[tid].append((float(x), float(y)))
            size_hist[tid].append((float(w), float(h)))

        def add_events(mask, label):
            idxs = np.where(mask)[0]
            for j in idxs:
                tid = int(det_f.tracker_id[j])
                conf = float(det_f.confidence[j])
                cls = int(det_f.class_id[j])
                events.append({
                    "frame": frame_idx,
                    "time": str(timedelta(seconds=t_now)),
                    "track_id": tid,
                    "species": cls,
                    "label": label,
                    "confidence": conf
                })
                out_subdir = rh_dir if cls == class_id else non_rh_dir
                cv2.imwrite(str(out_subdir / f"{tid}_{cls}_{label}_{frame_idx}.png"), frame_annot)

        add_events(crossed_in, move_right)
        add_events(crossed_out, move_left)

        # Simple FPS estimate (EMA)
        dt = max(1e-6, t_infer1 - t_infer0)
        cur_fps = 1.0 / dt
        fps_smooth = cur_fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * cur_fps)

        # Show preview occasionally
        if show and (frame_idx % show_every == 0):
            _show_frame(frame_annot, line_counter, scale=display_scale, fps=fps_smooth,
                       move_right=move_right, move_left=move_left)

    cap.release()

    df = pd.DataFrame(events)
    df.to_csv(count_csv_path, index=False)
    print(f"{datetime.now()}  done: {video_path.name} -> {count_csv_path}")


