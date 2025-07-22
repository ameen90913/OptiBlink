import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, defaultdict
import os
import csv
import keyboard

# Load words from CSV or fallback to NLTK
def load_words_from_csv(csv_path, column_name="Word"):
    words = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                word = row.get(column_name)
                if word and word.isalpha() and len(word) > 2:
                    words.append(word.lower())
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}. Falling back to NLTK words.")
        import nltk
        nltk.download('words')
        from nltk.corpus import words as nltk_words
        words = [word.lower() for word in nltk_words.words() if word.isalpha() and len(word) > 2]
    return words

word_list = load_words_from_csv(r"words.csv", column_name="Word")

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class AutoCompleteSystem:
    def __init__(self):
        self.root = TrieNode()
        self.frequency = defaultdict(int)
        for word in word_list:
            self.insert(word)
        self.load_usage_data()

    def insert(self, word):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def _dfs(self, node, prefix, results):
        if node.is_end:
            results.append(prefix)
        for ch, child in node.children.items():
            self._dfs(child, prefix + ch, results)

    def suggest(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        results = []
        self._dfs(node, prefix, results)
        results.sort(key=lambda x: (-self.frequency[x], x))
        return results[:3]

    def record_usage(self, word):
        self.frequency[word] += 1
        with open("usage_data.txt", "a") as f:
            f.write(f"{word},{int(time.time())}\n")

    def load_usage_data(self, filename="usage_data.txt"):
        if not os.path.exists(filename):
            return
        with open(filename, "r") as f:
            for line in f:
                if "," in line:
                    word, _ = line.strip().split(",")
                    self.frequency[word] += 1
                    self.insert(word)

class EyeTracker:
    def __init__(self, auto):
        self.auto = auto
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.morse_to_letter = {
            # Letters
            '.-': 'A', '-...': 'B', '---.': 'C', '-..': 'D', '.': 'E',
            '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
            '-.-': 'K', '.-..': 'L', '----': 'M', '-.': 'N', '---': 'O',
            '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
            '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
            '--..': 'Z',
            # Numbers
            '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
            '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
            # Special characters
            '.-.-': 'ENTER', '.--.-': 'CAPS', '--': 'BACKSPACE', '......': 'SOS',
            '..--': 'SPACE', '.---.': 'SELECT1', '..--.': 'SELECT2',
            '.--..': 'SELECT3'
        }

        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        self.blink_counter = 0
        self.short_blinks = 0
        self.long_blinks = 0

        self.blink_start_time = None
        self.open_start_time = None
        self.is_blinking = False
        
        # Adjusted blink timing thresholds for better recognition
        self.short_blink_threshold = 0.3  # A shorter threshold for a 'dot'
        self.open_threshold = 1.0         # A longer pause to signal end of a character

        self.ear_history = deque(maxlen=5)
        self.area_history = deque(maxlen=3)

        self.is_calibrated = False
        self.calibration_frames = 30
        self.calibration_counter = 0
        self.ear_baseline = 0
        self.area_baseline = 0

        self.morse_char_buffer = ""
        self.message_history = []
        self.word_buffer = ""
        self.caps_lock = False

        self.current_ear = 0
        self.current_area = 0
        self.blink_state = "Open"
        self.last_blink_ear = 0
        self.last_blink_area = 0
        self.blink_duration = 0
        self.open_duration = 0

        self.current_suggestions = []
        self.last_word_printed = ""

    def calculate_eye_features(self, eye_landmarks, frame_width, frame_height):
        try:
            points = np.array([
                (int(landmark.x * frame_width), int(landmark.y * frame_height))
                for landmark in eye_landmarks
            ])
            v1 = np.linalg.norm(points[1] - points[5])
            v2 = np.linalg.norm(points[2] - points[4])
            v3 = np.linalg.norm(points[3] - points[7])
            h = np.linalg.norm(points[0] - points[3])
            ear = (v1 + v2 + v3) / (3.0 * h)
            area = cv2.contourArea(points)
            return ear, area, points
        except Exception:
            return 0.0, 0.0, None

    def detect_blink(self, left_ear, right_ear, left_area, right_area):
        avg_ear = (left_ear + right_ear) / 2.0
        avg_area = (left_area + right_area) / 2.0

        self.ear_history.append(avg_ear)
        self.area_history.append(avg_area)

        smoothed_ear = np.mean(self.ear_history)
        smoothed_area = np.mean(self.area_history)

        ear_ratio = smoothed_ear / self.ear_baseline if self.ear_baseline > 0 else 1.0
        area_ratio = smoothed_area / self.area_baseline if self.area_baseline > 0 else 1.0

        self.current_ear = smoothed_ear
        self.current_area = smoothed_area

        is_blink = (smoothed_ear / self.ear_baseline) < 0.7 or (smoothed_area / self.area_baseline) < 0.7
        return is_blink, smoothed_ear, smoothed_area

    def process_special_character(self, char):
        if char == 'ENTER':
            if self.word_buffer:
                print(f"Message: {self.word_buffer}")
                self.message_history.append(self.word_buffer)
                self.auto.record_usage(self.word_buffer.lower())
                self.last_word_printed = self.word_buffer
                keyboard.write(self.word_buffer)
                keyboard.send('enter')
            self.word_buffer = ""
            self.morse_char_buffer = ""
        elif char == 'SPACE':
            self.word_buffer += " "
            keyboard.send('space')
            self.morse_char_buffer = ""
        elif char == 'BACKSPACE':
            if self.word_buffer:
                self.word_buffer = self.word_buffer[:-1]
                keyboard.send('backspace')
            self.morse_char_buffer = ""
        elif char == 'CAPS':
            self.caps_lock = not self.caps_lock
            self.morse_char_buffer = ""
        elif char == 'SOS':
            self.message_history.append("SOS")
            self.word_buffer = ""
            self.morse_char_buffer = ""
            keyboard.write('SOS')

    def decode_morse_char(self):
        if self.morse_char_buffer in self.morse_to_letter:
            char = self.morse_to_letter[self.morse_char_buffer]
            if char in ['ENTER', 'SPACE', 'BACKSPACE', 'CAPS', 'SOS']:
                self.process_special_character(char)
            elif char.startswith("SELECT"):
                index = int(char[-1]) - 1
                if 0 <= index < len(self.current_suggestions):
                    selected_word = self.current_suggestions[index]
                    last_word = self.word_buffer.strip().split(" ")[-1]
                    # Remove the partial word from both word_buffer and notepad
                    for _ in range(len(last_word)):
                        keyboard.send('backspace')
                    self.word_buffer = self.word_buffer[:-(len(last_word))] + selected_word + " "
                    self.auto.record_usage(selected_word.lower())
                    keyboard.write(selected_word)
            else:
                char_to_add = char.upper() if self.caps_lock else char.lower()
                self.word_buffer += char_to_add
                keyboard.write(char_to_add)
            self.morse_char_buffer = ""
            return char
        else:
            self.morse_char_buffer = ""
            return None

    def calibrate(self, ear, area):
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        results = self.face_mesh.process(rgb_frame)

        line_spacing = 35
        current_y = 30

        if not self.is_calibrated:
            cv2.putText(frame, f"Calibrating... {self.calibration_counter}/{self.calibration_frames}",
                        (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        current_y += 40

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            left_eye = [face_landmarks.landmark[idx] for idx in self.LEFT_EYE]
            right_eye = [face_landmarks.landmark[idx] for idx in self.RIGHT_EYE]
            left_ear, left_area, left_points = self.calculate_eye_features(left_eye, frame_width, frame_height)
            right_ear, right_area, right_points = self.calculate_eye_features(right_eye, frame_width, frame_height)

            if not self.is_calibrated:
                self.calibrate((left_ear + right_ear) / 2, (left_area + right_area) / 2)
            else:
                is_blink, smoothed_ear, smoothed_area = self.detect_blink(left_ear, right_ear, left_area, right_area)
                current_time = time.time()

                if is_blink:
                    if not self.is_blinking:
                        self.blink_start_time = current_time
                        self.open_start_time = None
                        self.is_blinking = True
                        self.blink_state = "Closed"
                    self.blink_duration = current_time - self.blink_start_time
                    self.open_duration = 0
                else:
                    if self.is_blinking:
                        blink_duration = current_time - self.blink_start_time
                        self.is_blinking = False
                        self.blink_state = "Open"
                        self.open_start_time = current_time
                        if blink_duration < self.short_blink_threshold:
                            self.short_blinks += 1
                            self.morse_char_buffer += "."
                        else:
                            self.long_blinks += 1
                            self.morse_char_buffer += "-"
                        self.blink_counter += 1
                    elif self.open_start_time:
                        self.open_duration = current_time - self.open_start_time
                        self.blink_duration = 0
                        if self.open_duration >= self.open_threshold and self.morse_char_buffer:
                            self.decode_morse_char()
                            self.open_start_time = None

            if left_points is not None and right_points is not None:
                cv2.polylines(frame, [left_points], True, (0, 255, 0), 1)
                cv2.polylines(frame, [right_points], True, (0, 255, 0), 1)

        cv2.putText(frame, f"CAPS: {'ON' if self.caps_lock else 'OFF'} - Word: {self.word_buffer}",
                    (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        current_y += line_spacing

        cv2.putText(frame, f"Morse: {self.morse_char_buffer}",
                    (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        current_y += line_spacing

        possible_letters = [char for code, char in self.morse_to_letter.items() if code.startswith(self.morse_char_buffer)]
        if possible_letters:
            possible_text = f"Possible: {', '.join(possible_letters)}"
            cv2.putText(frame, possible_text, (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            current_y += line_spacing

        if self.word_buffer:
            last_word = self.word_buffer.strip().split(" ")[-1].lower()
            self.auto.suggest(last_word)
            self.current_suggestions = self.auto.suggest(last_word)
            start_x = 10
            box_height = 50
            box_width = 200
            padding = 10

            for idx, suggestion in enumerate(self.current_suggestions):
                box_top_left = (start_x + idx * (box_width + padding), current_y)
                box_bottom_right = (box_top_left[0] + box_width, box_top_left[1] + box_height)
                cv2.rectangle(frame, box_top_left, box_bottom_right, (255, 0, 0), 2)

                text_x = box_top_left[0] + 10
                text_y = box_top_left[1] + 25
                cv2.putText(frame, suggestion, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                morse_keys = ['.---.', '..--.', '.--..']
                if idx < len(morse_keys):
                    cv2.putText(frame, morse_keys[idx], (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            current_y += box_height + line_spacing

        if self.blink_state == "Closed":
            cv2.circle(frame, (frame_width - 50, 50), 20, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (frame_width - 50, 50), 20, (0, 255, 0), -1)

        return frame

def main():
    cap = cv2.VideoCapture(0)
    auto = AutoCompleteSystem()
    eye_tracker = EyeTracker(auto)

    window_width = 950
    window_height = 750

    keyboard_image_path = "morse_keyboard.jpg"
    keyboard_img = cv2.imread(keyboard_image_path, cv2.IMREAD_UNCHANGED)

    if keyboard_img is None:
        print(f"Error: Could not load keyboard image from {keyboard_image_path}")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)

            keyboard_height_ratio = 0.40
            keyboard_height = int(window_height * keyboard_height_ratio)
            video_height = window_height - keyboard_height

            original_aspect_ratio = frame.shape[1] / frame.shape[0]
            video_width_normal = int(video_height * original_aspect_ratio)
            
            video_frame_resized = cv2.resize(frame, (video_width_normal, video_height))
            
            video_canvas = np.zeros((video_height, window_width, 3), dtype=np.uint8)
            x_offset = (window_width - video_width_normal) // 2
            video_canvas[0:video_height, x_offset:x_offset+video_width_normal] = video_frame_resized

            processed_video_frame = eye_tracker.process_frame(video_canvas)

            resized_keyboard_img = cv2.resize(keyboard_img, (window_width, keyboard_height))

            full_display_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            full_display_frame[0:video_height, 0:window_width] = processed_video_frame
            full_display_frame[video_height:video_height+keyboard_height, 0:window_width] = resized_keyboard_img

            cv2.imshow("Eye Blink Morse Code", full_display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
