"""
Step 4 — Live Inference
YOLOv8 fastener detection + identification plugged into your webcam pipeline.

Usage:
    python step4_inference.py

Controls:
    Q  — quit
    S  — save current frame snapshot
    +  — increase confidence threshold
    -  — decrease confidence threshold
"""

import cv2
import numpy as np
import time
import os
from ultralytics import YOLO


# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = "runs/detect/fasteners/weights/best.pt"
CAMERA_INDEX = 0
FRAME_W      = 640
FRAME_H      = 480
CONF_INIT    = 0.45      # Initial confidence threshold (adjustable live with +/-)
CONF_STEP    = 0.05
IOU_THRESH   = 0.45      # NMS IoU threshold
IMG_SIZE     = 416       # Must match training img size


# ── Color palette per class (BGR) ──────────────────────────────────────────────
CLASS_COLORS = [
    (0,   200, 255),   # bolt    — cyan
    (0,   255, 120),   # nut     — green
    (255, 100,   0),   # screw   — blue
    (180,   0, 255),   # rivet   — purple
    (0,   180, 255),   # washer  — yellow-ish
]

def get_color(class_id: int) -> tuple:
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


# ── Preprocessing (reused from your existing pipeline) ────────────────────────
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE contrast enhancement before passing to YOLO.
    Improves detection on reflective metallic surfaces.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ── Drawing helpers ────────────────────────────────────────────────────────────
def draw_detection(frame, box, class_name, class_id, confidence):
    """Draw bounding box + label for a single detection."""
    x1, y1, x2, y2 = map(int, box)
    color = get_color(class_id)

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner accents (cleaner look than a plain rectangle)
    corner_len = min(12, (x2 - x1) // 4, (y2 - y1) // 4)
    for cx, cy, dx, dy in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, 3)

    # Label background + text
    label = f"{class_name}  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    lx, ly = x1, max(y1 - 6, th + 4)
    cv2.rectangle(frame, (lx, ly - th - 4), (lx + tw + 8, ly + 2), color, -1)
    cv2.putText(frame, label, (lx + 4, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(frame, detections, conf_thresh, fps):
    """Top-left HUD: FPS, threshold, count summary."""
    lines = [
        f"FPS: {fps:.1f}   conf >= {conf_thresh:.2f}   +/- to adjust",
        f"Detections: {len(detections)}",
    ]
    # Count per class
    counts = {}
    for det in detections:
        name = det['name']
        counts[name] = counts.get(name, 0) + 1
    for name, n in counts.items():
        lines.append(f"  {name}: {n}")

    for i, line in enumerate(lines):
        y = 20 + i * 20
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 230), 1, cv2.LINE_AA)

    cv2.putText(frame, "Q=quit  S=save snapshot",
                (10, FRAME_H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (150, 150, 150), 1, cv2.LINE_AA)


# ── Main inference loop ────────────────────────────────────────────────────────
def run():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Train first with step3_train.py")
        return

    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.fuse()   # fuse Conv+BN layers — speeds up CPU inference

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    conf_thresh = CONF_INIT
    snap_count  = 0
    fps         = 0.0
    t_prev      = time.time()

    print("[INFO] Running. Q=quit  S=save  +/- adjust confidence\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Preprocessing ────────────────────────────────────────────────────
        enhanced = preprocess_frame(frame)

        # ── YOLO inference ───────────────────────────────────────────────────
        results = model.predict(
            source    = enhanced,
            imgsz     = IMG_SIZE,
            conf      = conf_thresh,
            iou       = IOU_THRESH,
            device    = "cpu",
            verbose   = False,
        )

        # ── Parse detections ─────────────────────────────────────────────────
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "box"   : box.xyxy[0].tolist(),
                    "conf"  : float(box.conf[0]),
                    "cls"   : int(box.cls[0]),
                    "name"  : model.names[int(box.cls[0])],
                })

        # ── Draw detections ──────────────────────────────────────────────────
        display = frame.copy()
        for det in detections:
            draw_detection(display, det["box"], det["name"], det["cls"], det["conf"])

        # ── FPS ──────────────────────────────────────────────────────────────
        t_now = time.time()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        draw_hud(display, detections, conf_thresh, fps)
        cv2.imshow("Fastener Detection — YOLOv8", display)

        # ── Keys ─────────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('s'):
            fname = f"snapshot_{snap_count:04d}.jpg"
            cv2.imwrite(fname, display)
            snap_count += 1
            print(f"[INFO] Saved {fname}")

        elif key == ord('+') or key == ord('='):
            conf_thresh = min(conf_thresh + CONF_STEP, 0.95)
            print(f"[INFO] Confidence threshold: {conf_thresh:.2f}")

        elif key == ord('-'):
            conf_thresh = max(conf_thresh - CONF_STEP, 0.10)
            print(f"[INFO] Confidence threshold: {conf_thresh:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stopped.")


if __name__ == "__main__":
    run()
