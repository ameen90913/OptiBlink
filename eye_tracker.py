import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, defaultdict
import os
import csv
import keyboard
import ctypes  
import win32gui
import win32con
from gtts import gTTS
import pygame
import threading
import tempfile

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
    return words

# Load CSV words
csv_word_list = load_words_from_csv(r"words.csv", column_name="Word")

# Always load NLTK words for fallback
try:
    import nltk
    nltk.data.find('corpora/words')
except (ImportError, LookupError):
    import nltk
    nltk.download('words')
from nltk.corpus import words as nltk_words
nltk_word_list = [word.lower() for word in nltk_words.words() if word.isalpha() and len(word) > 2]

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class AutoCompleteSystem:
    def __init__(self):
        self.csv_root = TrieNode()
        self.nltk_root = TrieNode()
        self.frequency = defaultdict(int)
        for word in csv_word_list:
            self.insert(word, root=self.csv_root)
        for word in nltk_word_list:
            self.insert(word, root=self.nltk_root)
        self.load_usage_data()

    def insert(self, word, root=None):
        if root is None:
            root = self.csv_root
        node = root
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
        # First, get from CSV trie
        node = self.csv_root
        for ch in prefix:
            if ch not in node.children:
                node = None
                break
            node = node.children[ch]
        results = []
        if node:
            self._dfs(node, prefix, results)
        results = list(set(results))
        results.sort(key=lambda x: (-self.frequency[x], x))
        # If less than 3, supplement with NLTK
        if len(results) < 3:
            nltk_node = self.nltk_root
            for ch in prefix:
                if ch not in nltk_node.children:
                    nltk_node = None
                    break
                nltk_node = nltk_node.children[ch]
            nltk_results = []
            if nltk_node:
                self._dfs(nltk_node, prefix, nltk_results)
            # Exclude duplicates
            extra = [w for w in nltk_results if w not in results]
            extra.sort(key=lambda x: (-self.frequency[x], x))
            results += extra[:3-len(results)]
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
                    self.insert(word, root=self.csv_root)

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
        # Flag to prevent re-sending keys when character is processed
        self.key_sent_for_current_char = False 
        # Track the length of the word_buffer that has already been sent to the keyboard
        self.last_sent_len = 0 
        self.tts_enabled = False  # TTS toggle flag
        self.is_speaking = False  # Track if TTS is currently speaking
        self.speaking_message = ""  # Store the message being spoken
        # Add TTS toggle code to morse_to_letter
        self.morse_to_letter['-.-.-'] = 'TTS_TOGGLE'

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

    def speak(self, text):
        if text and text.strip():
            threading.Thread(target=self._speak_blocking, args=(text,), daemon=True).start()

    def _speak_blocking(self, text):
        try:
            self.is_speaking = True
            self.speaking_message = text
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_filename = temp_file.name
            
            # Generate speech and save to temp file
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(temp_filename)
            
            # Initialize pygame mixer and play the audio file
            pygame.mixer.init()
            pygame.mixer.music.load(temp_filename)
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            
            # Clean up pygame mixer and temporary file
            pygame.mixer.quit()
            try:
                os.unlink(temp_filename)
            except:
                pass  # Ignore cleanup errors
            self.is_speaking = False
            self.speaking_message = ""
        except Exception as e:
            print(f"TTS Error: {e}")
            self.is_speaking = False
            self.speaking_message = ""

    def _delayed_speak(self, text):
        time.sleep(0.5)  # Wait half a second
        self._speak_blocking(text)

    def process_special_character(self, char):
        if char == 'ENTER':
            text_to_send = self.word_buffer[self.last_sent_len:]
            if text_to_send:
                keyboard.write(text_to_send)
            keyboard.send('enter')
            if self.word_buffer:
                print(f"Message: {self.word_buffer}")
                self.message_history.append(self.word_buffer)
                self.auto.record_usage(self.word_buffer.lower())
                self.last_word_printed = self.word_buffer
                if self.tts_enabled:
                    message_to_speak = self.word_buffer  # Store before clearing
                    # Add a short delay before TTS
                    time.sleep(0.3)
                    self.speak(message_to_speak)
            self.word_buffer = ""
            self.morse_char_buffer = ""
            self.last_sent_len = 0
        elif char == 'SPACE':
            self.word_buffer += " "
            keyboard.send('space')
            self.morse_char_buffer = ""
            self.last_sent_len = len(self.word_buffer)
        elif char == 'BACKSPACE':
            if self.word_buffer:
                self.word_buffer = self.word_buffer[:-1]
                keyboard.send('backspace')
            self.morse_char_buffer = ""
            self.last_sent_len = len(self.word_buffer)
        elif char == 'CAPS':
            self.caps_lock = not self.caps_lock
            self.morse_char_buffer = ""
        elif char == 'SOS':
            self.message_history.append("SOS")
            self.word_buffer = ""
            self.morse_char_buffer = ""
            keyboard.write('SOS')
            keyboard.send('enter')
            self.last_sent_len = len(self.word_buffer)
            if self.tts_enabled:
                self.speak("SOS")
        elif char == 'TTS_TOGGLE':
            self.tts_enabled = not self.tts_enabled
            status = "ON" if self.tts_enabled else "OFF"
            print(f"TTS {status}")
            if self.word_buffer:
                self.speak(self.word_buffer)
            else:
                self.speak("No message")
            self.morse_char_buffer = ""
            # Do NOT touch word_buffer here!

    def decode_morse_char(self):
        # Reset the flag at the beginning of decoding a new character
        self.key_sent_for_current_char = False 

        if self.morse_char_buffer in self.morse_to_letter:
            char = self.morse_to_letter[self.morse_char_buffer]
            if char == 'TTS_TOGGLE':
                self.process_special_character(char)
                self.key_sent_for_current_char = True
                self.morse_char_buffer = ""
                return char
            if char in ['ENTER', 'SPACE', 'BACKSPACE', 'CAPS', 'SOS']:
                self.process_special_character(char)
                self.key_sent_for_current_char = True # Mark as key sent
            elif char.startswith("SELECT"):
                index = int(char[-1]) - 1
                if 0 <= index < len(self.current_suggestions):
                    selected_word = self.current_suggestions[index]
                    last_word_in_buffer = self.word_buffer.strip().split(" ")[-1]
                    
                    # Calculate how many backspaces are needed to clear the partial word
                    backspaces_needed = len(last_word_in_buffer)

                    # Remove the partial word from both word_buffer and notepad
                    for _ in range(backspaces_needed):
                        keyboard.send('backspace')
                    
                    # Update word_buffer with the selection
                    # The part of word_buffer *before* the last word needs to be preserved
                    if ' ' in self.word_buffer.strip():
                        # Find the index of the last space
                        last_space_index = self.word_buffer.strip().rfind(' ')
                        # Reconstruct the word_buffer
                        self.word_buffer = self.word_buffer[:last_space_index + 1] + selected_word + " "
                    else:
                        self.word_buffer = selected_word + " "

                    self.auto.record_usage(selected_word.lower())
                    keyboard.write(selected_word + " ") # Write the selected word and a space
                    self.key_sent_for_current_char = True # Mark as key sent
                    self.last_sent_len = len(self.word_buffer) # Update sent length after selection
            else:
                char_to_add = char.upper() if self.caps_lock else char.lower()
                self.word_buffer += char_to_add
                keyboard.write(char_to_add) # Only write the single new character
                self.key_sent_for_current_char = True # Mark as key sent
                self.last_sent_len = len(self.word_buffer) # Update sent length
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
                        # When a blink just finished and a dot/dash is added, reset the key_sent_for_current_char flag
                        self.key_sent_for_current_char = False 
                    elif self.open_start_time:
                        self.open_duration = current_time - self.open_start_time
                        self.blink_duration = 0
                        # Ensure we only decode/send if a character is ready AND it hasn't been sent yet for this buffer
                        if self.open_duration >= self.open_threshold and self.morse_char_buffer and not self.key_sent_for_current_char:
                            self.decode_morse_char()
                            self.open_start_time = None # Reset open_start_time after processing

            if left_points is not None and right_points is not None:
                cv2.polylines(frame, [left_points], True, (0, 255, 0), 1)
                cv2.polylines(frame, [right_points], True, (0, 255, 0), 1)

        cv2.putText(frame, f"CAPS: {'ON' if self.caps_lock else 'OFF'} - TTS: {'ON' if self.tts_enabled else 'OFF'} - Word: {self.word_buffer}",
                    (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        current_y += int(line_spacing * 0.7)

        # Show speaking indicator if TTS is active
        if self.is_speaking:
            cv2.putText(frame, f"Speaking: {self.speaking_message}",
                        (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            current_y += int(line_spacing * 0.7)

        cv2.putText(frame, f"Morse: {self.morse_char_buffer}",
                    (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        current_y += int(line_spacing * 0.7)

        possible_letters = [char for code, char in self.morse_to_letter.items() if code.startswith(self.morse_char_buffer)]
        if possible_letters:
            # Split possible letters into two lines if too long
            possible_text = [f"Possible: {', '.join(possible_letters)}"]
            max_line_length = 40  # max chars per line (adjust as needed)
            if len(possible_text[0]) > max_line_length:
                # Split into two lines
                mid = len(possible_letters) // 2
                line1 = f"Possible: {', '.join(possible_letters[:mid])}"
                line2 = f"{', '.join(possible_letters[mid:])}"
                possible_text = [line1, line2]
            for line in possible_text:
                cv2.putText(frame, line, (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                current_y += int(line_spacing * 0.7)

        # Suggestions logic
        if self.word_buffer:
            # Get the current partial word for suggestions (handle spaces)
            buffer_parts = self.word_buffer.strip().split(" ")
            if buffer_parts and buffer_parts[-1]: # Ensure there's a non-empty last part
                last_word_for_suggestion = buffer_parts[-1].lower()
            else:
                last_word_for_suggestion = "" # No partial word, no suggestions

            if last_word_for_suggestion: # Only suggest if there's something to suggest for
                self.current_suggestions = self.auto.suggest(last_word_for_suggestion)
            else:
                self.current_suggestions = [] # Clear suggestions if no partial word

            start_x = 10
            box_height = 40
            box_width = 150
            padding = 10
            max_x = frame.shape[1] - box_width - padding
            morse_keys = ['.---.', '..--.', '.--..']
            box_x = start_x
            box_y = current_y
            for idx, suggestion in enumerate(self.current_suggestions):
                if box_x > max_x:
                    box_x = start_x
                    box_y += box_height + padding
                box_top_left = (box_x, box_y)
                box_bottom_right = (box_x + box_width, box_y + box_height)
                cv2.rectangle(frame, box_top_left, box_bottom_right, (255, 0, 0), 2)

                text_x = box_x + 10
                text_y = box_y + 20
                cv2.putText(frame, suggestion, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if idx < len(morse_keys):
                    cv2.putText(frame, morse_keys[idx], (text_x, text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
                box_x += box_width + padding
            current_y = box_y + box_height + int(line_spacing * 0.7) # Only increment if suggestions were shown

        if self.blink_state == "Closed":
            cv2.circle(frame, (frame_width - 50, 50), 10, (0, 0, 255), -1)
        else:
            cv2.circle(frame, (frame_width - 50, 50), 10, (0, 255, 0), -1)

        return frame

def main():
    cap = cv2.VideoCapture(0)
    auto = AutoCompleteSystem()
    eye_tracker = EyeTracker(auto)

    window_width = 650
    window_height = 450

    # Move window to top-right corner
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    x_pos = screen_width - window_width
    y_pos = 40  # Leave space for window controls

    # Function to set OpenCV window always on top
    def set_window_always_on_top(window_name):
        hwnd = win32gui.FindWindow(None, window_name)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

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

            keyboard_height_ratio = 0.45
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
            cv2.moveWindow("Eye Blink Morse Code", x_pos, y_pos)
            set_window_always_on_top("Eye Blink Morse Code")

            # Robust window close detection
            try:
                if cv2.getWindowProperty("Eye Blink Morse Code", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            key = cv2.waitKey(1)
            if key == -1:
                # If window is closed, waitKey returns -1
                if cv2.getWindowProperty("Eye Blink Morse Code", cv2.WND_PROP_VISIBLE) < 1:
                    break
            if key & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()