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
            # Letters
            '.-': 'A', '-...': 'B', '---.': 'C', '-..': 'D', '.': 'E',
            '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
            '-.-': 'K', '.-..': 'L', '----': 'M', '-.': 'N', '---': 'O',
            '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
            '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
            '--..': 'Z',
            # Numbers
            '-----': '0', '.---': '1', '..---': '2', '...--': '3', '....-': '4',
            '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
            # Special characters
            '.-.-': 'ENTER',  # Enter key
            '..--': 'SPACE',  # Space
            '--': 'BACKSPACE',  # Backspace
            '..--': 'CAPS',  # Caps Lock
            # Emergency SOS (6 dots)
            '......': 'SOS'
        }
        
        # Eye landmarks indices
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Blink detection parameters
        self.blink_counter = 0
        self.short_blinks = 0
        self.long_blinks = 0
        
        # Time-based blink detection
        self.blink_start_time = None
        self.open_start_time = None
        self.is_blinking = False
        self.short_blink_threshold = 0.5 # seconds for short blink
        self.open_threshold = 1.0  # seconds for end of character
        
        # Moving average for smoothing
        self.ear_history = deque(maxlen=5)
        self.area_history = deque(maxlen=3)
        
        # Calibration parameters
        self.is_calibrated = False
        self.calibration_frames = 30
        self.calibration_counter = 0
        self.ear_baseline = 0
        self.area_baseline = 0
        
        # Morse code parameters
        self.morse_char_buffer = ""  # Buffer for current character's Morse code
        self.message_history = []
        self.word_buffer = ""
        self.caps_lock = False
        
        # Debug information
        self.current_ear = 0
        self.current_area = 0
        self.blink_state = "Open"
        self.last_blink_ear = 0
        self.last_blink_area = 0
        self.blink_duration = 0
        self.open_duration = 0
        
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
    
    def process_special_character(self, char):
        """Process special characters and update state accordingly"""
        if char == 'ENTER':
            if self.word_buffer:
                print(f"Message: {self.word_buffer}")  # Print message with prefix
                self.message_history.append(self.word_buffer)
                self.word_buffer = ""
        elif char == 'SPACE':
            if self.word_buffer:
                self.message_history.append(self.word_buffer)
                self.word_buffer = ""
        elif char == 'BACKSPACE':
            if self.word_buffer:
                self.word_buffer = self.word_buffer[:-1]
        elif char == 'CAPS':
            self.caps_lock = not self.caps_lock
        elif char == 'SOS':
            self.message_history.append("SOS")
            self.word_buffer = ""

    def decode_morse_char(self):
        """Decode the current Morse code buffer into a character"""
        if self.morse_char_buffer in self.morse_to_letter:
            char = self.morse_to_letter[self.morse_char_buffer]
            
            # Process special characters
            if char in ['ENTER', 'SPACE', 'BACKSPACE', 'CAPS', 'SOS']:
                self.process_special_character(char)
            else:
                # Add letter to word buffer with caps lock consideration
                if self.caps_lock:
                    self.word_buffer += char.upper()
                else:
                    self.word_buffer += char.lower()
            
            # Clear buffer after successful translation
            self.morse_char_buffer = ""
            return char
        else:
            self.morse_char_buffer = ""
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
            
            # Time-based blink detection
            current_time = time.time()
            
            if is_blink:
                if not self.is_blinking:
                    # Blink started
                    self.blink_start_time = current_time
                    self.open_start_time = None
                    self.is_blinking = True
                    self.blink_state = "Closed"
                self.blink_duration = current_time - self.blink_start_time
                self.open_duration = 0
            else:
                if self.is_blinking:
                    # Blink ended
                    blink_duration = current_time - self.blink_start_time
                    self.is_blinking = False
                    self.blink_state = "Open"
                    self.open_start_time = current_time
                    
                    # Classify blink duration and add to buffer
                    if blink_duration < self.short_blink_threshold:
                        self.short_blinks += 1
                        self.morse_char_buffer += "."
                    else:
                        self.long_blinks += 1
                        self.morse_char_buffer += "-"
                    
                    self.blink_counter += 1
                elif self.open_start_time:
                    # Track open duration
                    self.open_duration = current_time - self.open_start_time
                    self.blink_duration = 0
                    
                    # Check for end of character
                    if self.open_duration >= self.open_threshold and self.morse_char_buffer:
                        self.decode_morse_char()  # Try to decode current buffer
                        self.open_start_time = None
            
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
            
            # Display current Morse buffer and possible letters
            cv2.putText(frame, f"Morse: {self.morse_char_buffer}", (10, frame_height - 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Show possible letters for current buffer
            possible_letters = [char for code, char in self.morse_to_letter.items() 
                              if code.startswith(self.morse_char_buffer)]
            if possible_letters:
                possible_text = f"Possible: {', '.join(possible_letters)}"
                cv2.putText(frame, possible_text, (10, frame_height - 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Display current word and caps lock state
            caps_text = "CAPS ON" if self.caps_lock else "CAPS OFF"
            cv2.putText(frame, f"{caps_text} - Current word: {self.word_buffer}", (10, frame_height - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Display message history
            if self.message_history:
                # Split long messages into multiple lines
                words_per_line = 5
                for i in range(0, len(self.message_history), words_per_line):
                    line = " ".join(self.message_history[i:i + words_per_line])
                    y_pos = frame_height - 60 - (len(self.message_history) // words_per_line) * 30
                    cv2.putText(frame, line, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
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
    
    # Set window size
    window_width = 800  # Set to 800 pixels
    window_height = 600  # Set to 600 pixels
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            processed_frame = eye_tracker.process_frame(frame)
            
            # Resize frame to larger window size
            processed_frame = cv2.resize(processed_frame, (window_width, window_height))
            
            # Display the frame
            cv2.imshow("Eye Blink Morse Code", processed_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 