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
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Morse code to letter mapping
        self.morse_to_letter = {
            '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
            '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
            '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
            '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
            '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
            '--..': 'Z',
            # Adding number mappings
            '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
            '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
            # Emergency SOS
            '.....': 'Emergency SOS'
        }
        
        # Eye landmarks indices
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Blink detection parameters
        self.blink_counter = 0
        self.last_blink_time = time.time()
        
        # Blink duration tracking
        self.blink_start_time = None
        self.is_blinking = False
        self.short_blink_threshold = 0.5
        self.short_blinks = 0
        self.long_blinks = 0
        
        # Moving average for smoothing
        self.ear_history = deque(maxlen=3)
        self.area_history = deque(maxlen=3)
        
        # Calibration parameters
        self.is_calibrated = False
        self.calibration_frames = 30
        self.calibration_counter = 0
        self.ear_baseline = 0
        self.area_baseline = 0
        
        # Morse code parameters
        self.current_morse = ""
        self.last_blink_type = None
        self.morse_timeout = 3.0  # Time in seconds to wait before considering the sequence complete
        self.last_blink_end_time = None
        
        # Debug information
        self.current_ear = 0
        self.current_area = 0
        self.blink_state = "Open"
        self.last_blink_ear = 0
        self.last_blink_area = 0
        
    def calculate_eye_features(self, eye_landmarks, frame_width, frame_height):
        """Calculate multiple features for eye state detection"""
        try:
            # Convert landmarks to numpy array
            points = np.array([(int(landmark.x * frame_width), int(landmark.y * frame_height)) 
                             for landmark in eye_landmarks])
            
            # Calculate eye aspect ratio
            v1 = np.linalg.norm(points[1] - points[5])
            v2 = np.linalg.norm(points[2] - points[4])
            v3 = np.linalg.norm(points[3] - points[7])
            h = np.linalg.norm(points[0] - points[3])
            ear = (v1 + v2 + v3) / (3.0 * h)
            
            # Calculate eye contour area
            area = cv2.contourArea(points)
            
            return ear, area, points
            
        except Exception as e:
            return 0.0, 0.0, None
    
    def detect_blink(self, left_ear, right_ear, left_area, right_area):
        """Detect blink using multiple features"""
        # Average the features from both eyes
        avg_ear = (left_ear + right_ear) / 2.0
        avg_area = (left_area + right_area) / 2.0
        
        # Add to history
        self.ear_history.append(avg_ear)
        self.area_history.append(avg_area)
        
        # Get smoothed values
        smoothed_ear = np.mean(self.ear_history)
        smoothed_area = np.mean(self.area_history)
        
        # Calculate relative changes
        ear_ratio = smoothed_ear / self.ear_baseline if self.ear_baseline > 0 else 1.0
        area_ratio = smoothed_area / self.area_baseline if self.area_baseline > 0 else 1.0
        
        # Update current values
        self.current_ear = smoothed_ear
        self.current_area = smoothed_area
        
        # Detect blink using both features
        is_blink = ear_ratio < 0.7 or area_ratio < 0.7
        
        return is_blink, smoothed_ear, smoothed_area
    
    def update_morse_code(self, blink_type):
        """Update the Morse code sequence based on blink type"""
        current_time = time.time()
        
        # If this is the first blink or enough time has passed since the last blink
        if self.last_blink_end_time is None or (current_time - self.last_blink_end_time) > self.morse_timeout:
            self.current_morse = ""
        
        # Add the new blink to the sequence
        if blink_type == "Short":
            self.current_morse += "."
        else:  # Long blink
            self.current_morse += "-"
        
        self.last_blink_end_time = current_time
        
        # Try to convert Morse code to letter
        if self.current_morse in self.morse_to_letter:
            return self.morse_to_letter[self.current_morse]
        return None
    
    def calibrate(self, ear, area):
        """Calibrate the system with baseline measurements"""
        if self.calibration_counter < self.calibration_frames:
            self.ear_baseline += ear
            self.area_baseline += area
            self.calibration_counter += 1
            return False
        else:
            self.ear_baseline /= self.calibration_frames
            self.area_baseline /= self.calibration_frames
            self.is_calibrated = True
            return True
    
    def process_frame(self, frame):
        """Process a single frame to detect blinks and update Morse code"""
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        
        # Process the frame with MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get eye landmarks
            left_eye = [face_landmarks.landmark[idx] for idx in self.LEFT_EYE]
            right_eye = [face_landmarks.landmark[idx] for idx in self.RIGHT_EYE]
            
            # Calculate features for both eyes
            left_ear, left_area, left_points = self.calculate_eye_features(left_eye, frame_width, frame_height)
            right_ear, right_area, right_points = self.calculate_eye_features(right_eye, frame_width, frame_height)
            
            # Calibration
            if not self.is_calibrated:
                if self.calibrate((left_ear + right_ear)/2, (left_area + right_area)/2):
                    cv2.putText(frame, "Calibration Complete!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Calibrating... {self.calibration_counter}/{self.calibration_frames}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                return frame
            
            # Detect blink
            is_blink, smoothed_ear, smoothed_area = self.detect_blink(left_ear, right_ear, left_area, right_area)
            
            # Blink duration tracking
            current_time = time.time()
            
            if is_blink and not self.is_blinking:
                # Blink started
                self.is_blinking = True
                self.blink_start_time = current_time
                self.last_blink_ear = smoothed_ear
                self.last_blink_area = smoothed_area
                self.blink_state = "Closed"
            elif not is_blink and self.is_blinking:
                # Blink ended
                self.is_blinking = False
                self.blink_state = "Open"
                blink_duration = current_time - self.blink_start_time
                
                # Only count blinks if they're not too short
                if blink_duration > 0.05:
                    # Classify blink duration
                    if blink_duration < self.short_blink_threshold:
                        self.short_blinks += 1
                        blink_type = "Short"
                    else:
                        self.long_blinks += 1
                        blink_type = "Long"
                    
                    self.blink_counter += 1
                    self.last_blink_time = current_time
                    
                    # Update Morse code sequence and get letter if complete
                    letter = self.update_morse_code(blink_type)
            
            # Draw eye contours
            if left_points is not None and right_points is not None:
                cv2.polylines(frame, [left_points], True, (0, 255, 0), 1)
                cv2.polylines(frame, [right_points], True, (0, 255, 0), 1)
            
            # Display information
            cv2.putText(frame, f"Blinks: {self.blink_counter}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Short: {self.short_blinks}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Long: {self.long_blinks}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display current Morse code sequence and letter
            morse_text = f"Morse: {self.current_morse}"
            text_size = cv2.getTextSize(morse_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame_width - text_size[0]) // 2
            cv2.putText(frame, morse_text, (text_x, frame_height - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the translated letter if available
            if self.current_morse in self.morse_to_letter:
                letter = self.morse_to_letter[self.current_morse]
                letter_text = f"Letter: {letter}"
                letter_size = cv2.getTextSize(letter_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                letter_x = (frame_width - letter_size[0]) // 2
                cv2.putText(frame, letter_text, (letter_x, frame_height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Draw blink state indicator
            if self.blink_state == "Closed":
                cv2.circle(frame, (frame_width - 50, 50), 20, (0, 0, 255), -1)  # Red circle for closed
            else:
                cv2.circle(frame, (frame_width - 50, 50), 20, (0, 255, 0), -1)  # Green circle for open
        
        return frame

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    eye_tracker = EyeTracker()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        processed_frame = eye_tracker.process_frame(frame)
        
        # Display the frame
        cv2.imshow("Eye Tracking", processed_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 