import re
import sys
import time

import cv2
import yt_dlp


class VideoSource:
    """Frame extractor for YouTube streams (live/recorded), local video files, and webcams.
    Implements frame skipping for live content to prevent latency accumulation.
    """

    def __init__(self, path, quality, max_retries=3):
        self.path = path
        self.quality = quality
        self.max_retries = max_retries
        self.retry_count = 0
        self.cap = None
        self.fps = 30
        self.is_live = False
        self.max_height = self._parse_max_height(quality)
        self._connect()

    def _parse_max_height(self, quality):
        match = re.search(r"height<=(\d+)", quality)
        if match:
            return int(match.group(1))
        return None

    def _connect(self):
        # Stop execution when retry limit is reached
        if self.retry_count >= self.max_retries:
            print(f"[VideoSource] Max retries ({self.max_retries}) reached. Exiting.")
            sys.exit(1)
        self.retry_count += 1
        print(f"[VideoSource] Connection attempt {self.retry_count}/{self.max_retries}...")

        is_webcam = False
        is_local = False

        # Try to treat path as an integer index for webcam
        try:
            webcam_idx = int(self.path)
            is_webcam = True
        except (ValueError, TypeError):
            is_webcam = False
            # Check if the path is a local file or a yt video
            is_local = not str(self.path).startswith(("http://", "https://"))

        try:
            if is_webcam:
                print(f"[VideoSource] Opening webcam index: {webcam_idx}")
                self.cap = cv2.VideoCapture(webcam_idx)
                self.is_live = True
            elif is_local:
                print(f"[VideoSource] Opening local file: {self.path}")
                self.cap = cv2.VideoCapture(self.path)
                self.is_live = False
            else:
                ydl_opts = {"format": self.quality, "quiet": True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.path, download=False)
                    # Check if the yt video is a live stream
                    self.is_live = info.get("is_live", False)
                    self.cap = cv2.VideoCapture(info["url"])

            if self.cap is None or not self.cap.isOpened():
                print("[VideoSource] Failed to open video source.")
                self.cap = None
                return

            if self.is_live:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Reset the count after a successful open
            self.retry_count = 0

            # Get the actual FPS
            source_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if source_fps > 0:
                self.fps = source_fps

            print("[VideoSource] Mode: LIVE" if self.is_live else "[VideoSource] Mode: RECORDED")
            print(f"[VideoSource] Native FPS: {self.fps}")
            print("[VideoSource] Connected!")

        except Exception as e:
            print(f"[VideoSource] Error: {e}")
            if self.cap:
                self.cap.release()
            self.cap = None

    def _resize_frame(self, frame):
        if self.max_height is None or frame is None:
            return frame
        h, w = frame.shape[:2]
        if h <= self.max_height:
            return frame
        scale = self.max_height / h
        new_w = int(w * scale)
        return cv2.resize(frame, (new_w, self.max_height), interpolation=cv2.INTER_LANCZOS4)

    def read(self):
        if self.cap is None:
            # Exponential backoff calculation
            wait = min(2**self.retry_count, 16)
            print(f"[VideoSource] Waiting {wait} seconds before next attempt...")
            time.sleep(wait)
            self._connect()
            return None

        # For live streams, try to stay up to date
        if self.is_live:
            latest_frame = None
            # Skip up to 2 buffered frames to catch up over time without hitting a loop
            for _ in range(2):
                if not self.cap.grab():
                    break
                ret, frame = self.cap.retrieve()
                if ret:
                    latest_frame = frame

            if latest_frame is not None:
                return self._resize_frame(latest_frame)

        # Default behavior for recorded videos
        ret, frame = self.cap.read()

        if not ret:
            print("[VideoSource] Frame read failed. Reconnecting...")
            self.cap.release()
            self.cap = None  # Trigger the reconnection logic on next read
            return None
        return self._resize_frame(frame)

    def release(self):
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    source = VideoSource(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "bestvideo[height<=720]/best[height<=720]"
    )
    while True:
        frame = source.read()
        if frame is not None:
            status_text = ("LIVE" if source.is_live else "RECORDED") + f" | FPS: {source.fps:.1f}"
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("VideoSource", frame)
            safe_fps = source.fps if source.fps > 0 else 30
            wait_time = int(1000 / safe_fps)
            if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                break
    source.release()
    cv2.destroyAllWindows()
