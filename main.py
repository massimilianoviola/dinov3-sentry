import time
from datetime import datetime

import cv2
import numpy as np
import yaml

from src.backbone import DinoFeatureExtractor
from src.detector import CosineSimilarityDetector, MahalanobisDetector
from src.video_source import VideoSource


def format_time(seconds, is_live=False):
    # Wall clock (HH:MM:SS) for live and video time (MM:SS.ss) for recorded
    if is_live:
        return datetime.now().strftime("%H:%M:%S")
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def main():
    with open("configs/settings.yaml") as f:
        cfg = yaml.safe_load(f)

    backbone = DinoFeatureExtractor(
        model_repo=cfg["model"]["model_repo"],
        device=cfg["model"]["device"],
    )

    # detector = MahalanobisDetector()
    detector = CosineSimilarityDetector()
    source = VideoSource(cfg["stream"]["path"], cfg["stream"]["quality"])

    n_train = cfg["detector"]["min_train_frames"]
    min_anomaly_fraction = cfg["detector"]["min_anomaly_fraction"]

    print(f"[Analyzer] Calibrating on first {n_train} frames...")
    buffer = []
    frame_idx = 0

    while len(buffer) < n_train:
        frame = source.read()
        if frame is None:
            continue

        display = frame.copy()
        cv2.putText(
            display,
            f"CALIBRATING: {len(buffer)}/{n_train}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imshow("DINOv3 Sentry", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            source.release()
            cv2.destroyAllWindows()
            return

        feats, _ = backbone.extract(frame)
        buffer.append(feats)
        frame_idx += 1

    detector.fit(buffer)
    del buffer
    print("[Analyzer] Calibration complete.")

    print("[Analyzer] Monitoring for anomalies... (press q to stop)\n")
    in_anomaly = False
    start_time = None
    anomaly_count = 0

    while True:
        frame = source.read()
        if frame is None:
            break

        feats, _ = backbone.extract(frame)
        scores, binary_mask = detector.predict(feats)
        total_patches = binary_mask.size
        patch_count = int(np.sum(binary_mask))
        patch_fraction = patch_count / total_patches

        # Use wall clock for live duration, frame index for recorded
        if source.is_live:
            current_time = time.time()
        else:
            current_time = frame_idx / source.fps

        is_triggered = patch_fraction >= min_anomaly_fraction

        if is_triggered and not in_anomaly:
            in_anomaly = True
            start_time = current_time
            anomaly_count += 1
            print(
                f"\n[{format_time(current_time, source.is_live)}] Anomaly #{anomaly_count} started"
            )

        elif not is_triggered and in_anomaly:
            in_anomaly = False
            duration = current_time - start_time
            print(
                f"\n[{format_time(current_time, source.is_live)}] Anomaly #{anomaly_count} ended ({duration:.1f}s)"
            )

        # Real-time progress bar
        total_patches = binary_mask.size
        pct = patch_count / total_patches * 100
        bar_width = 40
        filled = int(bar_width * patch_count / total_patches)
        bar = "█" * filled + "░" * (bar_width - filled)
        status = "ANOMALY" if in_anomaly else "normal "
        ts = format_time(current_time, source.is_live)
        print(
            f"\r[{ts}] [{status}] {bar} {pct:5.1f}% ({patch_count}/{total_patches})",
            end="",
            flush=True,
        )

        disp_h, disp_w = frame.shape[:2]
        left_view = frame

        # Binary mask overlay on right view
        mask_big = cv2.resize(
            binary_mask.astype(np.uint8) * 255, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST
        )
        colored = np.zeros_like(frame)
        colored[mask_big > 0] = (0, 0, 255)  # Red for anomalous patches
        right_view = cv2.addWeighted(frame, 0.7, colored, 0.3, 0)

        combined = np.hstack((left_view, right_view))

        status = "ANOMALY" if in_anomaly else "NORMAL"
        color = (0, 0, 255) if in_anomaly else (0, 255, 0)
        cv2.putText(
            combined,
            f"{status} | Patches: {patch_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        cv2.imshow("DINOv3 Sentry", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    source.release()
    cv2.destroyAllWindows()
    print(f"\n[Analyzer] Processed {frame_idx} frames. Detected {anomaly_count} anomaly event(s).")


if __name__ == "__main__":
    main()
