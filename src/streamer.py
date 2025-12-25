import sys
import time

import cv2
import yt_dlp


class LiveStream:
    def __init__(self, url, quality, max_retries=3):
        self.url = url
        self.quality = quality
        self.max_retries = max_retries
        self.retry_count = 0
        self.cap = None
        self.fps = 30
        self.is_live = False
        self._connect()

    def _connect(self):
        # Stop execution when retry limit is reached
        if self.retry_count >= self.max_retries:
            print(f"[Stream] Max retries ({self.max_retries}) reached. Exiting.")
            sys.exit(1)
        self.retry_count += 1
        print(f"[Stream] Connection attempt {self.retry_count}/{self.max_retries}...")

        ydl_opts = {"format": self.quality, "quiet": True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)

                # Check if the source is a live stream
                self.is_live = info.get("is_live", False)

                # Open the video stream
                self.cap = cv2.VideoCapture(info["url"])
                if self.is_live:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not self.cap.isOpened():
                    print("[Stream] Failed to open video stream.")
                    self.cap = None
                    return

                # Reset the count after a successful open
                self.retry_count = 0

                # Get the actual FPS from the stream
                source_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if source_fps > 0:
                    self.fps = source_fps
                print("[Stream] Mode: LIVE" if self.is_live else "RECORDED")
                print(f"[Stream] Native FPS: {self.fps}")
                print("[Stream] Connected!")

        except Exception as e:
            print(f"[Stream] Error: {e}")
            if self.cap:
                self.cap.release()
            self.cap = None

    def read(self):
        if self.cap is None:
            # Exponential backoff calculation
            wait = min(2**self.retry_count, 16)
            print(f"[Stream] Waiting {wait} seconds before next attempt...")
            time.sleep(wait)
            self._connect()
            return None

        # For live streams, skip all frames currently waiting in the buffer
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
                return latest_frame

        # Default behavior for recorded videos
        ret, frame = self.cap.read()

        if not ret:
            print("[Stream] Frame read failed. Reconnecting...")
            self.cap.release()
            self.cap = None  # Trigger the reconnection logic on next read
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    stream = LiveStream("https://www.youtube.com/watch?v=EvsLqQS_80E", "95/best[height<=720]")
    while True:
        frame = stream.read()
        if frame is not None:
            cv2.imshow("Stream", frame)
        # For saved videos, use the calculated FPS delay. For live streams, use a 1ms delay.
        wait_time = 1 if stream.is_live else int(1000 / stream.fps)
        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
            break
    stream.release()
    cv2.destroyAllWindows()
