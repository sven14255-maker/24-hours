import threading
import time
from pathlib import Path
import shutil
import subprocess

import cv2
import mediapipe as mp
import numpy as np
import playsound3


class Camera:
    # Punten van ogen en neus.
    LEFT_EYE_CORNERS = (33, 133)
    RIGHT_EYE_CORNERS = (362, 263)
    LEFT_EYE_LIDS = (159, 145)
    RIGHT_EYE_LIDS = (386, 374)
    LEFT_IRIS = (468, 469, 470, 471, 472)
    RIGHT_IRIS = (473, 474, 475, 476, 477)
    NOSE_TIP = 1

    def __init__(self):
        self.project_dir = Path(__file__).resolve().parent
        self.cap = cv2.VideoCapture(0)
        self.alert_sounds = [
            self._resolve_existing_path("alert.wav", "includes/alert.wav"),
            self._resolve_existing_path("alert.mp3", "includes/alert.mp3"),
        ]
        self.model_path = self._resolve_existing_path("face_landmarker.task")
        self.object_model_path = self._resolve_existing_path(
            "efficientdet_lite0.tflite",
            "includes/efficientdet_lite0.tflite",
        )
        self.window_name = "Camera"

        # Instellingen voor detectie.
        self.fullscreen_enabled = True
        self.alert_cooldown_seconds = 3
        self.blink_grace_period = 0.35
        self.distracted_grace_seconds = 5
        self.gaze_tolerance_multiplier = 1.0
        self.head_turn_threshold = 0.18
        self.closed_eye_frame_threshold = 3
        self.distraction_confirmation_frames = 3
        self.phone_detection_stride = 3
        self.calibration_step_seconds = 1.5
        self.calibration_steps = [
            ("midden", "Kijk naar het midden van je scherm"),
            ("links", "Kijk naar de linkerkant van je scherm"),
            ("rechts", "Kijk naar de rechterkant van je scherm"),
            ("boven", "Kijk naar de bovenkant van je scherm"),
            ("onder", "Kijk naar de onderkant van je scherm"),
        ]

        # Huidige status.
        self.stage = "Calibreren"
        self.status_message = "Start calibratie"
        self.lost_focus_since = None
        self.study_started_at = None
        self.focus_accumulated_seconds = 0.0
        self.current_focus_block_started_at = None
        self.last_blink_time = None
        self.eyes_closed_since = None
        self.is_paused = False
        self.stage_before_pause = None

        # Data van calibratie.
        self.calibration_started_at = None
        self.current_calibration_index = 0
        self.calibration_samples = {
            name: [] for name, _ in self.calibration_steps
        }
        self.calibration_center = None
        self.calibration_tolerance = None

        # Tellers voor stabiele detectie.
        self.focus_frames = 0
        self.closed_eye_frames = 0
        self.non_focus_frames = 0
        self.non_focus_reason = None

        # Tijden en statistieken.
        self.app_started_at = time.monotonic()
        self.pause_started_at = None
        self.paused_accumulated_seconds = 0.0
        self.distraction_count = 0
        self.longest_focus_block_seconds = 0.0
        self.last_alert_time = 0
        self.alert_is_playing = False

        self.frame_index = 0
        self.last_phone_detections = []

        if not self.model_path.exists():
            raise RuntimeError("Missing face_landmarker.task in the project folder.")
        if not self.object_model_path.exists():
            raise RuntimeError("Missing efficientdet_lite0.tflite in the project folder.")

        self.base_options = mp.tasks.BaseOptions(model_asset_path=str(self.model_path))
        self.landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=self.base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            self.landmarker_options
        )
        self.object_detector = mp.tasks.vision.ObjectDetector.create_from_options(
            mp.tasks.vision.ObjectDetectorOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(self.object_model_path)
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                max_results=3,
                score_threshold=0.35,
                category_allowlist=["cell phone"],
            )
        )

    def _resolve_existing_path(self, *relative_paths):
        # Pak het eerste pad dat bestaat.
        for relative_path in relative_paths:
            candidate = self.project_dir / relative_path
            if candidate.exists():
                return candidate
        return self.project_dir / relative_paths[0]

    def detect_face(self):
        if not self.cap.isOpened():
            raise RuntimeError("Could not open the camera.")

        # Open het cameravenster.
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._apply_window_mode()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Spiegel het beeld.
                self.frame_index += 1
                frame = cv2.flip(frame, 1)

                if self.is_paused:
                    status = "Pauze"
                    distracted_seconds = 0.0
                else:
                    looking_at_screen, status = self._analyze_frame(frame)
                    distracted_seconds = self._update_stage(looking_at_screen)
                    if self.stage == "Niet aan het studeren":
                        self._play_alert()

                self.status_message = status
                self._draw_status(frame, status, distracted_seconds)
                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord("r"):
                    self._reset_calibration()
                if key == ord("p"):
                    self._toggle_pause()
                if key == ord("f"):
                    self._toggle_fullscreen()

        finally:
            self.face_landmarker.close()
            self.object_detector.close()
            self.cap.release()
            cv2.destroyAllWindows()

    def _analyze_frame(self, frame):
        # Analyseer 1 frame.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
        phone_detected = self._detect_phone(mp_image, timestamp_ms)

        if phone_detected:
            self._reset_focus_tracking()
            return False, "Telefoon"

        if len(result.face_landmarks) > 1:
            self._reset_focus_tracking()
            return False, "Meerdere gezichten"

        # Zonder gezicht geen blikmeting.
        if not result.face_landmarks:
            return self._handle_non_focus(
                "geen_gezicht",
                self._status_bij_geen_gezicht(),
            )

        landmarks = result.face_landmarks[0]
        gaze_features = self._extract_gaze_features(frame, landmarks)

        if gaze_features is None:
            return self._handle_closed_eyes()

        # Reset oogsluiting.
        self.eyes_closed_since = None
        self.closed_eye_frames = 0

        if self.calibration_center is None:
            return self._handle_calibration(gaze_features)

        if abs(gaze_features["head_turn"]) > self.head_turn_threshold:
            return self._handle_non_focus(
                "hoofd_gedraaid",
                "Hoofd gedraaid",
            )

        centered = self._is_gaze_centered(gaze_features)
        if centered:
            self._reset_non_focus_tracking()
            self.focus_frames += 1
            return True, "Op scherm"

        direction = self._describe_gaze_direction(gaze_features)
        return self._handle_non_focus(
            "wegkijken",
            direction,
        )

    def _handle_closed_eyes(self):
        # Behandel dichte ogen.
        now = time.monotonic()
        if self.eyes_closed_since is None:
            self.eyes_closed_since = now

        self.closed_eye_frames += 1
        eyes_closed_duration = now - self.eyes_closed_since

        if self.calibration_center is None:
            return False, "Ogen dicht"

        if (
            eyes_closed_duration <= self.blink_grace_period
            or self.closed_eye_frames < self.closed_eye_frame_threshold
        ):
            self.last_blink_time = now
            if self.stage in ("Studeren", "Afgeleid"):
                return True, "Op scherm"

        return self._handle_non_focus("ogen_dicht", "Ogen dicht")

    def _handle_non_focus(self, reason, status):
        # Bevestig afleiding pas na meer frames.
        self.focus_frames = 0
        if self.non_focus_reason == reason:
            self.non_focus_frames += 1
        else:
            self.non_focus_reason = reason
            self.non_focus_frames = 1

        if (
            self.stage == "Studeren"
            and self.non_focus_frames < self.distraction_confirmation_frames
        ):
            return True, "Op scherm"

        return False, status

    def _detect_phone(self, mp_image, timestamp_ms):
        # Check soms op telefoon.
        if self.frame_index % self.phone_detection_stride == 0:
            result = self.object_detector.detect_for_video(mp_image, timestamp_ms)
            self.last_phone_detections = []

            for detection in result.detections:
                categories = detection.categories or []
                if not categories:
                    continue

                category = categories[0]
                category_name = (category.category_name or "").lower()
                if category_name != "cell phone":
                    continue

                self.last_phone_detections.append(detection)

        return bool(self.last_phone_detections)

    def _extract_gaze_features(self, frame, landmarks):
        # Haal blikdata uit het gezicht.
        height, width = frame.shape[:2]
        points = np.array([(landmark.x * width, landmark.y * height) for landmark in landmarks])

        left_eye = self._single_eye_features(
            points,
            self.LEFT_EYE_CORNERS,
            self.LEFT_EYE_LIDS,
            self.LEFT_IRIS,
        )
        right_eye = self._single_eye_features(
            points,
            self.RIGHT_EYE_CORNERS,
            self.RIGHT_EYE_LIDS,
            self.RIGHT_IRIS,
        )

        if left_eye is None or right_eye is None:
            return None

        horizontal = (left_eye["horizontal"] + right_eye["horizontal"]) / 2.0
        vertical = (left_eye["vertical"] + right_eye["vertical"]) / 2.0
        openness = (left_eye["openness"] + right_eye["openness"]) / 2.0

        if openness < 0.16:
            return None

        left_eye_center = (left_eye["outer_corner"] + left_eye["inner_corner"]) / 2.0
        right_eye_center = (right_eye["outer_corner"] + right_eye["inner_corner"]) / 2.0
        eye_midpoint = (left_eye_center + right_eye_center) / 2.0
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        if eye_distance < 1e-6:
            return None

        nose_tip = points[self.NOSE_TIP]
        head_turn = (nose_tip[0] - eye_midpoint[0]) / eye_distance

        return {
            "horizontal": float(horizontal),
            "vertical": float(vertical),
            "openness": float(openness),
            "head_turn": float(head_turn),
        }

    def _single_eye_features(self, points, corner_indices, lid_indices, iris_indices):
        # Bereken data van 1 oog.
        outer_corner = points[corner_indices[0]]
        inner_corner = points[corner_indices[1]]
        top_lid = points[lid_indices[0]]
        bottom_lid = points[lid_indices[1]]
        iris_points = points[list(iris_indices)]
        iris_center = iris_points.mean(axis=0)

        horizontal_span = np.linalg.norm(inner_corner - outer_corner)
        vertical_span = np.linalg.norm(bottom_lid - top_lid)
        if horizontal_span < 1e-6 or vertical_span < 1e-6:
            return None

        horizontal_axis = inner_corner - outer_corner
        vertical_axis = bottom_lid - top_lid

        horizontal = (
            np.dot(iris_center - outer_corner, horizontal_axis) / (horizontal_span ** 2)
        ) - 0.5
        vertical = (
            np.dot(iris_center - top_lid, vertical_axis) / (vertical_span ** 2)
        ) - 0.5
        openness = vertical_span / horizontal_span

        return {
            "outer_corner": outer_corner,
            "inner_corner": inner_corner,
            "horizontal": float(horizontal),
            "vertical": float(vertical),
            "openness": float(openness),
        }

    def _handle_calibration(self, gaze_features):
        # Verzamel calibratie-data.
        now = time.monotonic()
        step_name, _ = self.calibration_steps[self.current_calibration_index]

        if self.calibration_started_at is None:
            self.calibration_started_at = now

        self.calibration_samples[step_name].append(
            [gaze_features["horizontal"], gaze_features["vertical"]]
        )

        elapsed = now - self.calibration_started_at
        if elapsed >= self.calibration_step_seconds:
            self.current_calibration_index += 1
            self.calibration_started_at = now

            if self.current_calibration_index >= len(self.calibration_steps):
                self._finish_calibration(now)
                return True, "Klaar"

        return False, self._calibration_status()

    def _finish_calibration(self, now):
        # Bereken midden en toleranties.
        means = {}
        spreads = {}
        for step_name, _ in self.calibration_steps:
            samples = np.array(self.calibration_samples[step_name], dtype=np.float64)
            means[step_name] = samples.mean(axis=0)
            spreads[step_name] = samples.std(axis=0)

        center = means["midden"]
        left = means["links"]
        right = means["rechts"]
        up = means["boven"]
        down = means["onder"]

        left_gap = abs(center[0] - left[0])
        right_gap = abs(right[0] - center[0])
        up_gap = abs(center[1] - up[1])
        down_gap = abs(down[1] - center[1])
        usable_horizontal_gap = min(left_gap, right_gap)
        usable_vertical_gap = min(up_gap, down_gap)
        horizontal_tolerance = max(
            usable_horizontal_gap * 0.45 * self.gaze_tolerance_multiplier,
            0.08 * self.gaze_tolerance_multiplier,
        )
        vertical_tolerance = max(
            usable_vertical_gap * 0.45 * self.gaze_tolerance_multiplier,
            spreads["midden"][1] * 2.5 * self.gaze_tolerance_multiplier,
            0.06 * self.gaze_tolerance_multiplier,
        )

        self.calibration_center = center
        self.calibration_tolerance = np.array(
            [horizontal_tolerance, vertical_tolerance],
            dtype=np.float64,
        )
        self.stage = "Studeren"
        self.study_started_at = now
        self.current_focus_block_started_at = now
        self.focus_accumulated_seconds = 0.0
        self.lost_focus_since = None
        self._reset_non_focus_tracking()

    def _calibration_status(self):
        # Tekst voor calibratie.
        step_name, prompt = self.calibration_steps[self.current_calibration_index]
        if self.calibration_started_at is None:
            remaining = self.calibration_step_seconds
        else:
            elapsed = time.monotonic() - self.calibration_started_at
            remaining = max(0.0, self.calibration_step_seconds - elapsed)

        return f"{prompt} ({remaining:.1f}s)"

    def _is_gaze_centered(self, gaze_features):
        # Check of blik in het midden zit.
        centered = np.array(
            [
                gaze_features["horizontal"] - self.calibration_center[0],
                gaze_features["vertical"] - self.calibration_center[1],
            ]
        )
        return bool(np.all(np.abs(centered) <= self.calibration_tolerance))

    def _describe_gaze_direction(self, gaze_features):
        # Maak tekst van kijkrichting.
        horizontal = gaze_features["horizontal"] - self.calibration_center[0]
        vertical = gaze_features["vertical"] - self.calibration_center[1]
        tolerance_x, tolerance_y = self.calibration_tolerance

        parts = []
        if vertical < -tolerance_y:
            parts.append("omhoog")
        elif vertical > tolerance_y:
            parts.append("omlaag")

        if horizontal < -tolerance_x:
            parts.append("naar links")
        elif horizontal > tolerance_x:
            parts.append("naar rechts")

        if not parts:
            return "weg"
        return " en ".join(parts)

    def _status_bij_geen_gezicht(self):
        # Status zonder gezicht.
        if self.calibration_center is None:
            return "Geen gezicht"
        return "Geen gezicht"

    def _reset_calibration(self):
        # Reset calibratie.
        self.stage = "Calibreren"
        self.status_message = "Calibratie opnieuw gestart"
        self.lost_focus_since = None
        self.study_started_at = None
        self.current_focus_block_started_at = None
        self.focus_accumulated_seconds = 0.0
        self.calibration_started_at = None
        self.current_calibration_index = 0
        self.calibration_samples = {
            name: [] for name, _ in self.calibration_steps
        }
        self.calibration_center = None
        self.calibration_tolerance = None
        self.last_alert_time = 0
        self.last_blink_time = None
        self.eyes_closed_since = None
        self._reset_focus_tracking()

    def _store_current_study_segment(self, now):
        # Sla studietijd op.
        if self.study_started_at is None:
            return

        segment_seconds = max(0.0, now - self.study_started_at)
        self.focus_accumulated_seconds += segment_seconds
        self.study_started_at = None

    def _finalize_focus_block(self, now):
        # Update langste focusblok.
        if self.current_focus_block_started_at is None:
            return

        block_seconds = max(0.0, now - self.current_focus_block_started_at)
        self.longest_focus_block_seconds = max(
            self.longest_focus_block_seconds,
            block_seconds,
        )
        self.current_focus_block_started_at = None

    def _reset_study_timer(self):
        # Reset studietijd.
        self.study_started_at = None
        self.focus_accumulated_seconds = 0.0
        self.current_focus_block_started_at = None

    def _update_stage(self, looking_at_screen):
        # Update hoofdstatus.
        now = time.monotonic()
        distracted_seconds = 0.0

        if self.calibration_center is None:
            self.stage = "Calibreren"
            return distracted_seconds

        if looking_at_screen:
            if self.stage == "Niet aan het studeren":
                self.study_started_at = now
                self.current_focus_block_started_at = now
            elif self.study_started_at is None:
                self.study_started_at = now
                self.current_focus_block_started_at = now
            elif self.stage == "Afgeleid" and self.lost_focus_since is not None:
                self.study_started_at = now
                self.current_focus_block_started_at = now

            self.lost_focus_since = None
            self.stage = "Studeren"
            return distracted_seconds

        if self.stage == "Studeren":
            self.distraction_count += 1
            self._store_current_study_segment(now)
            self._finalize_focus_block(now)
            self.lost_focus_since = now
            self.stage = "Afgeleid"

        if self.stage == "Afgeleid" and self.lost_focus_since is not None:
            distracted_seconds = now - self.lost_focus_since
            if distracted_seconds >= self.distracted_grace_seconds:
                self.stage = "Niet aan het studeren"
                self._reset_study_timer()
                self._play_alert(force=True)

        return distracted_seconds

    def _get_study_seconds(self):
        # Totale studietijd.
        if self.calibration_center is None:
            return 0.0

        total = self.focus_accumulated_seconds
        if self.stage == "Studeren" and self.study_started_at is not None:
            total += time.monotonic() - self.study_started_at

        return total

    def _get_screen_seconds(self):
        # Totale schermtijd zonder pauze.
        paused_seconds = self.paused_accumulated_seconds
        if self.is_paused and self.pause_started_at is not None:
            paused_seconds += time.monotonic() - self.pause_started_at
        return max(0.0, time.monotonic() - self.app_started_at - paused_seconds)

    def _format_seconds(self, total_seconds):
        # Maak tijd leesbaar.
        total_seconds = max(0, int(total_seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _play_alert(self, force=False):
        # Start alarm.
        now = time.monotonic()
        if not force and now - self.last_alert_time < self.alert_cooldown_seconds:
            return
        if self.alert_is_playing:
            return

        self.last_alert_time = now

        if not any(sound.exists() for sound in self.alert_sounds):
            return

        self.alert_is_playing = True
        thread = threading.Thread(target=self._play_alert_sound, daemon=True)
        thread.start()

    def _play_alert_sound(self):
        # Speel alarm in aparte thread.
        try:
            for alert_sound in self.alert_sounds:
                if not alert_sound.exists():
                    continue
                if self._play_alert_with_system_player(alert_sound):
                    return
                playsound3.playsound(str(alert_sound))
                return
        finally:
            self.alert_is_playing = False

    def _play_alert_with_system_player(self, alert_sound):
        # Probeer lokale audio-spelers.
        alert_path = str(alert_sound.resolve())
        player_commands = [
            ["paplay", alert_path],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", alert_path],
            ["mpv", "--really-quiet", "--no-video", alert_path],
            ["mpg123", "-q", alert_path],
        ]

        for command in player_commands:
            if shutil.which(command[0]) is None:
                continue

            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                continue

            if completed.returncode == 0:
                return True

        return False

    def _toggle_pause(self):
        # Zet pauze aan of uit.
        now = time.monotonic()
        if not self.is_paused:
            self.is_paused = True
            self.pause_started_at = now
            self.stage_before_pause = self.stage
            if self.stage == "Studeren":
                self._store_current_study_segment(now)
                self._finalize_focus_block(now)
            return

        self.is_paused = False
        if self.pause_started_at is not None:
            self.paused_accumulated_seconds += now - self.pause_started_at
        self.pause_started_at = None

        if self.calibration_center is None:
            self.stage = "Calibreren"
            return

        if self.stage_before_pause == "Studeren":
            self.stage = "Studeren"
            self.study_started_at = now
            self.current_focus_block_started_at = now
        elif self.stage_before_pause == "Afgeleid":
            self.stage = "Afgeleid"
            self.lost_focus_since = now
        else:
            self.stage = self.stage_before_pause or "Afgeleid"
            self.lost_focus_since = None

        self._reset_focus_tracking()

    def _toggle_fullscreen(self):
        # Wissel schermmodus.
        self.fullscreen_enabled = not self.fullscreen_enabled
        self._apply_window_mode()

    def _apply_window_mode(self):
        # Zet window mode.
        mode = cv2.WINDOW_FULLSCREEN if self.fullscreen_enabled else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(
            self.window_name,
            cv2.WND_PROP_FULLSCREEN,
            mode,
        )

    def _reset_non_focus_tracking(self):
        # Reset afleidingsteller.
        self.non_focus_frames = 0
        self.non_focus_reason = None

    def _reset_focus_tracking(self):
        # Reset korte tellers.
        self.focus_frames = 0
        self.closed_eye_frames = 0
        self._reset_non_focus_tracking()

    def _draw_text(
        self,
        frame,
        text,
        origin,
        color,
        scale=0.7,
        thickness=2,
        font=cv2.FONT_HERSHEY_SIMPLEX,
    ):
        # Teken tekst.
        cv2.putText(
            frame,
            text,
            origin,
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_panel(self, frame, top_left, bottom_right, color, alpha=0.45):
        # Teken een paneel.
        overlay = frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_progress_bar(
        self,
        frame,
        top_left,
        width,
        height,
        progress,
        fill_color,
        background_color=(65, 72, 90),
    ):
        # Teken een balk.
        x, y = top_left
        radius = max(1, height // 2)
        progress = max(0.0, min(1.0, progress))

        cv2.rectangle(frame, (x, y), (x + width, y + height), background_color, -1)
        if progress > 0:
            fill_width = max(radius, int(width * progress))
            cv2.rectangle(frame, (x, y), (x + fill_width, y + height), fill_color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_red_warning(self, frame):
        # Rode waarschuwing zonder tekst.
        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (30, 30, 180), -1)
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        border_color = (40, 40, 255)
        cv2.rectangle(frame, (10, 10), (width - 10, height - 10), border_color, 10, cv2.LINE_AA)
        cv2.rectangle(frame, (28, 28), (width - 28, height - 28), (120, 120, 255), 3, cv2.LINE_AA)

    def _draw_status(self, frame, status, distracted_seconds):
        # Teken de interface.
        stage_to_show = "Gepauzeerd" if self.is_paused else self.stage
        stage_color = {
            "Calibreren": (80, 194, 255),
            "Studeren": (95, 214, 126),
            "Afgeleid": (0, 196, 255),
            "Niet aan het studeren": (87, 87, 255),
            "Gepauzeerd": (170, 176, 188),
        }[stage_to_show]
        accent_color = {
            "Calibreren": (255, 226, 140),
            "Studeren": (159, 255, 208),
            "Afgeleid": (115, 235, 255),
            "Niet aan het studeren": (140, 150, 255),
            "Gepauzeerd": (214, 214, 214),
        }[stage_to_show]

        study_time = self._format_seconds(self._get_study_seconds())
        height, width = frame.shape[:2]

        if self.stage == "Niet aan het studeren":
            self._draw_red_warning(frame)

        # Donkere overlay voor leesbaarheid.
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 130), (20, 20, 28), -1)
        cv2.rectangle(overlay, (0, height - 90), (width, height), (20, 20, 28), -1)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        self._draw_panel(frame, (22, 22), (width - 22, 152), (28, 30, 40), alpha=0.58)

        self._draw_text(frame, "FOCUS MONITOR", (42, 58), (245, 245, 245), scale=0.95, thickness=2)
        self._draw_text(frame, status, (42, 92), (210, 219, 235), scale=0.62, thickness=2)

        status_text = f"STATUS  {stage_to_show.upper()}"
        pill_width = max(230, len(status_text) * 12)
        self._draw_panel(frame, (width - pill_width - 42, 34), (width - 42, 82), stage_color, alpha=0.34)
        self._draw_text(
            frame,
            status_text,
            (width - pill_width - 20, 67),
            accent_color,
            scale=0.62,
            thickness=2,
        )

        self._draw_panel(frame, (32, 108), (280, 206), (37, 41, 54), alpha=0.56)
        self._draw_text(frame, "STUDIETIJD", (46, 138), (154, 164, 184), scale=0.52, thickness=1)
        self._draw_text(frame, study_time, (46, 182), (255, 255, 255), scale=0.95, thickness=2)

        if self.calibration_center is None:
            calibration_progress = self.current_calibration_index / len(self.calibration_steps)
            self._draw_text(frame, "Kalibratie", (42, 244), (208, 214, 228), scale=0.6, thickness=2)
            self._draw_progress_bar(frame, (42, 258), min(320, width - 84), 18, calibration_progress, stage_color)

        if self.stage == "Afgeleid":
            remaining = max(0.0, self.distracted_grace_seconds - distracted_seconds)
            countdown_progress = 1.0 - (remaining / self.distracted_grace_seconds)
            self._draw_progress_bar(
                frame,
                (42, 320),
                min(320, width - 84),
                18,
                countdown_progress,
                (60, 140, 255),
                background_color=(58, 44, 44),
            )

        # Sneltoetsen blijven actief.


if __name__ == "__main__":
    # Start de app.
    camera = Camera()
    camera.detect_face()
