import threading
import time
from pathlib import Path

import cv2
import playsound


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.alert_sound = Path(__file__).parent / "includes" / "alert.mp3"
        self.alert_cooldown_seconds = 3
        self.last_alert_time = 0
        self.alert_is_playing = False

        # ⏱️ NEW: no-eyes timer
        self.no_eyes_start_time = None
        self.no_eyes_threshold = 2  # seconds

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("Could not load OpenCV face/eye detectors.")

    def detect_face(self):
        if not self.cap.isOpened():
            raise RuntimeError("Could not open the camera.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            looking_at_screen, status, eyes_visible = self._is_looking_at_screen(frame)

            # ⏱️ NEW: 2-second rule for no eyes
            now = time.monotonic()

            if not eyes_visible:
                if self.no_eyes_start_time is None:
                    self.no_eyes_start_time = now
                elif now - self.no_eyes_start_time >= self.no_eyes_threshold:
                    self._play_alert()
            else:
                self.no_eyes_start_time = None

            self._draw_status(frame, status, looking_at_screen)
            cv2.imshow("Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _is_looking_at_screen(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
        )

        if len(faces) == 0:
            return False, "No face detected", False

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        self._draw_box(frame, x, y, w, h)

        face_gray = gray[y:y + h, x:x + w]

        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20),
        )

        frame_h, frame_w = frame.shape[:2]

        face_cx = x + w / 2
        face_cy = y + h / 2

        centered_x = abs(face_cx - frame_w / 2) < frame_w * 0.22
        centered_y = abs(face_cy - frame_h / 2) < frame_h * 0.28

        eyes_visible = len(eyes) >= 2

        if centered_x and centered_y and eyes_visible:
            return True, "Looking at screen", True

        if not eyes_visible:
            return False, "Eyes not visible", False

        return False, "Face turned away", True

    def _play_alert(self):
        now = time.monotonic()

        if now - self.last_alert_time < self.alert_cooldown_seconds:
            return
        if self.alert_is_playing:
            return

        self.last_alert_time = now

        if not self.alert_sound.exists():
            print("Alert: look at the screen (no audio file found).")
            return

        self.alert_is_playing = True
        threading.Thread(target=self._play_alert_sound, daemon=True).start()

    def _play_alert_sound(self):
        try:
            playsound.playsound(str(self.alert_sound))
        finally:
            self.alert_is_playing = False

    def _draw_box(self, frame, x, y, w, h):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 2)

    def _draw_status(self, frame, status, looking):
        color = (0, 255, 0) if looking else (0, 0, 255)

        cv2.putText(
            frame,
            status,
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    Camera().detect_face()