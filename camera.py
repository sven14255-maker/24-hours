import threading
import time
from collections import deque
from pathlib import Path

import cv2
import pygame


class Camera:
    # --- Configuration ---
    ALERT_COOLDOWN_SECONDS = 3
    NO_EYES_THRESHOLD = 2       # seconds before alert triggers
    STABILITY_BUFFER_SIZE = 6   # frames — smooths out detection noise
    STABILITY_REQUIRED = 4      # frames that must agree before state changes

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.alert_sound = Path(__file__).parent / "includes" / "alert.mp3"

        self._last_alert_time = 0
        self._alert_is_playing = False
        self._alert_lock = threading.Lock()

        self._no_eyes_start_time = None

        # Rolling buffer of recent eye-visible booleans for stable state
        self._eye_buffer: deque[bool] = deque(maxlen=self.STABILITY_BUFFER_SIZE)

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("Could not load OpenCV face/eye detectors.")

        pygame.mixer.init()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open camera and run the attention-detection loop."""
        if not self.cap.isOpened():
            raise RuntimeError("Could not open the camera.")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                looking, status, eyes_visible = self._analyse_frame(frame)

                self._update_alert_state(eyes_visible)
                self._draw_status(frame, status, looking)
                cv2.imshow("Attention Monitor", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self._stop_alert()
            self.cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()

    # ------------------------------------------------------------------
    # Alert logic
    # ------------------------------------------------------------------

    def _update_alert_state(self, eyes_visible: bool) -> None:
        """Trigger or stop alerts based on stabilised eye-presence state."""
        self._eye_buffer.append(eyes_visible)

        # Only act when the buffer has enough agreement
        if len(self._eye_buffer) < self.STABILITY_BUFFER_SIZE:
            return

        stable_visible = sum(self._eye_buffer) >= self.STABILITY_REQUIRED
        now = time.monotonic()

        if not stable_visible:
            if self._no_eyes_start_time is None:
                self._no_eyes_start_time = now
            elif now - self._no_eyes_start_time >= self.NO_EYES_THRESHOLD:
                self._play_alert()
        else:
            self._no_eyes_start_time = None
            self._stop_alert()

    def _play_alert(self) -> None:
        """Play the alert sound, respecting cooldown and deduplication."""
        now = time.monotonic()
        with self._alert_lock:
            if self._alert_is_playing:
                return
            if now - self._last_alert_time < self.ALERT_COOLDOWN_SECONDS:
                return
            if not self.alert_sound.exists():
                print(f"Alert sound not found: {self.alert_sound}")
                return

            self._last_alert_time = now
            self._alert_is_playing = True

        thread = threading.Thread(target=self._run_alert_sound, daemon=True)
        thread.start()

    def _run_alert_sound(self) -> None:
        """Background thread: play sound via pygame (interruptible)."""
        try:
            pygame.mixer.music.load(str(self.alert_sound))
            pygame.mixer.music.play()
            # Wait until playback finishes or is stopped externally
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
        except Exception as exc:
            print(f"Audio error: {exc}")
        finally:
            with self._alert_lock:
                self._alert_is_playing = False

    def _stop_alert(self) -> None:
        """Immediately stop any playing alert sound."""
        with self._alert_lock:
            if self._alert_is_playing:
                pygame.mixer.music.stop()
                self._alert_is_playing = False

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------

    def _analyse_frame(self, frame) -> tuple[bool, str, bool]:
        """Return (looking_at_screen, status_label, eyes_visible)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(100, 100),
        )

        if len(faces) == 0:
            return False, "No face detected", False

        # Use the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        self._draw_face_box(frame, x, y, w, h)

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

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_face_box(self, frame, x: int, y: int, w: int, h: int) -> None:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 2)

    def _draw_status(self, frame, status: str, looking: bool) -> None:
        color = (0, 220, 80) if looking else (0, 60, 220)
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
    Camera().run()