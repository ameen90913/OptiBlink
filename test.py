import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            static_image_mode=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        # Standard EAR landmarks
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Blink detection parameters
        self.blink_counter = 0
        self.short_blinks = 0
        self.long_blinks = 0
        self.short_blink_threshold = 0.35
        self.blink_state = "Open"
        self.blink_start_time = None
        self.is_blinking = False

        # EAR smoothing
        self.ear_history = deque(maxlen=5)

        # Calibration
        self.ear_baseline = 0
        self.calibration_counter = 0
        self.calibration_frames = 30
        self.is_calibrated = False

        # Morse code
        self.current_morse = ""
        self.morse_timeout = 3.0
        self.last_blink_end_time = None

    def calculate_ear(self, eye_points, frame_width, frame_height):
        try:
            points = np.array([(int(p.x * frame_width), int(p.y * frame_height)) for p in eye_points])
            A = np.linalg.norm(points[1] - points[5])
            B = np.linalg.norm(points[2] - points[4])
            C = np.linalg.norm(points[0] - points[3])
            if C == 0:  # Prevent division by zero
                return 0.0, points
            ear = (A + B) / (2.0 * C)
            return ear, points
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0.0, None

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            left_eye = [face_landmarks.landmark[i] for i in self.LEFT_EYE]
            right_eye = [face_landmarks.landmark[i] for i in self.RIGHT_EYE]

            left_ear, left_points = self.calculate_ear(left_eye, frame_width, frame_height)
            right_ear, right_points = self.calculate_ear(right_eye, frame_width, frame_height)

            avg_ear = (left_ear + right_ear) / 2.0
            self.ear_history.append(avg_ear)
            smoothed_ear = np.mean(self.ear_history)

            # Calibration
            if not self.is_calibrated:
                self.ear_baseline += smoothed_ear
                self.calibration_counter += 1
                if self.calibration_counter >= self.calibration_frames:
                    self.ear_baseline /= self.calibration_frames
                    self.is_calibrated = True
                cv2.putText(frame, f"Calibrating... {self.calibration_counter}/{self.calibration_frames}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                return frame

            ear_ratio = smoothed_ear / self.ear_baseline
            is_blink = ear_ratio < self.short_blink_threshold
            current_time = time.time()

            if is_blink and not self.is_blinking:
                self.is_blinking = True
                self.blink_start_time = current_time
                self.blink_state = "Closed"

            elif not is_blink and self.is_blinking:
                self.is_blinking = False
                self.blink_state = "Open"
                blink_duration = current_time - self.blink_start_time

                if blink_duration > 0.05:
                    blink_type = "Short" if blink_duration < 0.5 else "Long"
                    self.current_morse += "." if blink_type == "Short" else "-"
                    self.blink_counter += 1
                    if blink_type == "Short":
                        self.short_blinks += 1
                    else:
                        self.long_blinks += 1
                    self.last_blink_end_time = current_time

            # Clear Morse code if timeout
            if self.last_blink_end_time:
                if time.time() - self.last_blink_end_time > self.morse_timeout:
                    self.current_morse = ""

            # Draw contours
            if left_points is not None:
                cv2.polylines(frame, [left_points], True, (255, 0, 0), 1)
            if right_points is not None:
                cv2.polylines(frame, [right_points], True, (255, 0, 0), 1)

            # Info display
            cv2.putText(frame, f"Blinks: {self.blink_counter}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Short: {self.short_blinks}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Long: {self.long_blinks}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Morse: {self.current_morse}", (10, frame_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Blink indicator
            if self.blink_state == "Closed":
                cv2.circle(frame, (frame_width - 50, 50), 20, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (frame_width - 50, 50), 20, (0, 255, 0), -1)

        return frame

def main():
    cap = cv2.VideoCapture(0)
    eye_tracker = EyeTracker()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            processed_frame = eye_tracker.process_frame(frame)
            cv2.imshow("Eye Blink Morse Code", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()