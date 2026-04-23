import threading
import time
from pathlib import Path

import cv2
import playsound3


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.alert_sound = Path("alert.mp3")
        self.alert_cooldown_seconds = 3
        self.last_alert_time = 0
        self.alert_is_playing = False

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
            looking_at_screen, status = self._is_looking_at_screen(frame)

            if not looking_at_screen:
                self._play_alert()

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
            return False, "No face detected"

        x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
        self._draw_box(frame, x, y, width, height)

        face_gray = gray[y : y + height, x : x + width]
        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20),
        )

        frame_height, frame_width = frame.shape[:2]
        face_center_x = x + width / 2
        face_center_y = y + height / 2
        centered_x = abs(face_center_x - frame_width / 2) < frame_width * 0.22
        centered_y = abs(face_center_y - frame_height / 2) < frame_height * 0.28
        enough_eyes_visible = len(eyes) >= 2

        if centered_x and centered_y and enough_eyes_visible:
            return True, "Looking at screen"

        if not enough_eyes_visible:
            return False, "Look at screen - eyes not visible"

        return False, "Look at screen - face turned away"

    def _play_alert(self):
        now = time.monotonic()
        if now - self.last_alert_time < self.alert_cooldown_seconds:
            return
        if self.alert_is_playing:
            return

        self.last_alert_time = now

        if not self.alert_sound.exists():
            print("Alert: look at the screen. Add alert.mp3 to play a sound.")
            return

        self.alert_is_playing = True
        thread = threading.Thread(target=self._play_alert_sound, daemon=True)
        thread.start()

    def _play_alert_sound(self):
        try:
            playsound3.playsound(str(self.alert_sound))
        finally:
            self.alert_is_playing = False

    def _draw_box(self, frame, x, y, width, height):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 180, 255), 2)

    def _draw_status(self, frame, status, looking_at_screen):
        color = (0, 180, 0) if looking_at_screen else (0, 0, 255)
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
    camera = Camera()
    camera.detect_face()
