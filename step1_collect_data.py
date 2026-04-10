"""
Step 1 — Data Collection
Capture fastener images from your webcam and save them organized by class.

Usage:
    python step1_collect_data.py

Controls:
    SPACE  — save current frame as a sample
    N      — move to next fastener class
    Q      — quit
"""

import cv2
import os

# ── Define your fastener classes here ─────────────────────────────────────────
# Add or remove class names to match what you're detecting
CLASSES = ["bolt", "nut", "screw", "rivet", "washer"]

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR    = "dataset/images/raw"   # Where images are saved
TARGET_COUNT  = 100                    # Images to collect per class
CAMERA_INDEX  = 0
FRAME_W, FRAME_H = 640, 480


def collect():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    class_idx = 0
    counts = {c: 0 for c in CLASSES}

    print("\n=== DATA COLLECTION ===")
    print(f"Classes: {CLASSES}")
    print(f"Target : {TARGET_COUNT} images per class")
    print("Place ONE fastener type in view, then press SPACE to capture.\n")

    while class_idx < len(CLASSES):
        current_class = CLASSES[class_idx]
        class_dir = os.path.join(OUTPUT_DIR, current_class)
        os.makedirs(class_dir, exist_ok=True)

        ret, frame = cap.read()
        if not ret:
            break

        n = counts[current_class]
        remaining = TARGET_COUNT - n

        # HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_W, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"Class: {current_class}  [{class_idx+1}/{len(CLASSES)}]",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Captured: {n}/{TARGET_COUNT}   SPACE=save  N=next class  Q=quit",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Progress bar
        bar_w = int((n / TARGET_COUNT) * (FRAME_W - 20))
        cv2.rectangle(frame, (10, FRAME_H - 20), (FRAME_W - 10, FRAME_H - 8), (60, 60, 60), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (10, FRAME_H - 20), (10 + bar_w, FRAME_H - 8), (0, 200, 100), -1)

        cv2.imshow("Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\n[INFO] Quit early.")
            break

        elif key == ord(' '):
            filename = os.path.join(class_dir, f"{current_class}_{n:04d}.jpg")
            cv2.imwrite(filename, frame)
            counts[current_class] += 1
            print(f"  Saved {filename}  ({counts[current_class]}/{TARGET_COUNT})")

            if counts[current_class] >= TARGET_COUNT:
                print(f"\n[DONE] {current_class} complete. Press N to move to next class.")

        elif key == ord('n'):
            if counts[current_class] < 20:
                print(f"[WARN] Only {counts[current_class]} images for '{current_class}'. "
                      "Recommended minimum is 20. Press N again to skip anyway.")
            print(f"\n>>> Moving to next class...")
            class_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== COLLECTION SUMMARY ===")
    for c, n in counts.items():
        status = "OK" if n >= 20 else "LOW"
        print(f"  {c:12s}: {n:3d} images  [{status}]")
    print(f"\nImages saved to: {OUTPUT_DIR}/")
    print("Next step: label the images using LabelImg, then run step3_train.py")


if __name__ == "__main__":
    collect()
