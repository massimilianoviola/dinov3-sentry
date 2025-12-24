import time

import cv2
import yt_dlp


class LiveStream:
    def __init__(self, url, quality):
        self.url = url
        self.quality = quality
        self.cap = None
        self.fps = 30
        self._connect()

    def _connect(self):
        print(f"[Stream] Connecting...")
        ydl_opts = {"format": self.quality, "quiet": True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.url, download=False)
                self.cap = cv2.VideoCapture(info["url"])
                # Get the actual FPS from the stream
                source_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if source_fps > 0:
                    print(f"[Stream] Connected. Native FPS: {source_fps}")
                    self.fps = source_fps
        except Exception as e:
            print(f"[Stream] Error: {e}")

    def read(self):
        if not self.cap:
            return None
        ret, frame = self.cap.read()
        if not ret:
            print("[Stream] Reconnecting...")
            self.cap.release()
            time.sleep(1)
            self._connect()
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()


if __name__ == "__main__":
    stream = LiveStream("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "bestvideo[height<=720]")
    wait_time = int(1000 / stream.fps)
    while True:
        frame = stream.read()
        if frame is not None:
            cv2.imshow("Stream", frame)
        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
            break
    stream.release()
    cv2.destroyAllWindows()
