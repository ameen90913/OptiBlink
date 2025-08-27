# Standard library imports
import csv
import ctypes
import json
import os
import tempfile
import threading
import time
import webbrowser
from collections import deque, defaultdict

# CONFIGURATION - Change emergency contact number here
DEFAULT_EMERGENCY_CONTACT = "+91 7892310175"

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Third-party imports
import cv2
import mediapipe as mp
import numpy as np
import pygame
import keyboard
from gtts import gTTS

# Windows-specific imports
try:
    import win32gui
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    print("Warning: win32gui/win32con not available. Some Windows-specific features may not work.")

# Optional imports with fallbacks
try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False
    print("Warning: pyperclip not available. Clipboard functionality disabled.")

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    print("Warning: pyautogui not available. Some automation features may not work.")

# Configuration management
def load_config():
    """Load configuration from config.json file"""
    config_file = "config.json"
    default_config = {
        "emergency_contact": DEFAULT_EMERGENCY_CONTACT
    }
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                # ALWAYS use the centralized constant, ignore what's in the file
                config["emergency_contact"] = DEFAULT_EMERGENCY_CONTACT
                # Update the file to match the centralized constant
                save_config(config)
                return config
        else:
            # Create default config file
            save_config(default_config)
            return default_config
    except Exception as e:
        print(f"Error loading config: {e}. Using defaults.")
        return default_config

def save_config(config):
    """Save configuration to config.json file"""
    try:
        with open("config.json", 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")

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

# Load NLTK words for fallback - OPTIMIZED FOR FAST STARTUP
nltk_word_list = []
print("EyeTracker: Starting word dictionary setup...")

# Try to use CSV first, NLTK only if needed and fast
try:
    # If CSV exists and has words, skip NLTK entirely for faster startup
    if csv_word_list and len(csv_word_list) > 1000:
        print("EyeTracker: CSV words sufficient, skipping NLTK for faster startup")
        nltk_word_list = []
    else:
        print("EyeTracker: Loading NLTK words (may take a moment on first run)...")
        import nltk
        
        # Check if NLTK words are already available
        try:
            nltk.data.find('corpora/words')
            from nltk.corpus import words as nltk_words
            # Load only a subset for faster startup - full dictionary not needed for basic functionality
            all_words = nltk_words.words()
            nltk_word_list = [word.lower() for word in all_words[:5000] if word.isalpha() and len(word) > 2]
            print(f"EyeTracker: NLTK words loaded (subset): {len(nltk_word_list)} words")
        except LookupError:
            print("EyeTracker: NLTK corpus not found. Skipping NLTK for faster startup.")
            print("EyeTracker: (NLTK can be downloaded later if needed)")
            nltk_word_list = []
            
except ImportError:
    print("EyeTracker: NLTK not installed. Using CSV words only.")
    nltk_word_list = []
except Exception as e:
    print(f"EyeTracker: NLTK loading skipped ({e}). Using CSV words only.")
    nltk_word_list = []

print("EyeTracker: Word dictionary setup complete!")

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
        # Only record single words (no spaces, reasonable length)
        word = word.strip().lower()
        if word and ' ' not in word and len(word) <= 20 and word.isalpha():
            self.frequency[word] += 1
            with open("usage_data.txt", "a") as f:
                f.write(f"{word},{int(time.time())}\n")

    def load_usage_data(self, filename="usage_data.txt"):
        if not os.path.exists(filename):
            return
        with open(filename, "r") as f:
            for line in f:
                if "," in line:
                    word, _ = line.strip().split(",", 1)  # Split only on first comma
                    word = word.strip().lower()
                    # Only load single words (no spaces, reasonable length, alphabetic)
                    if word and ' ' not in word and len(word) <= 20 and word.isalpha():
                        self.frequency[word] += 1
                        self.insert(word, root=self.csv_root)

class EyeTracker:
    def __init__(self, auto):
        print("EyeTracker: Starting initialization...")
        self.auto = auto
        print("EyeTracker: AutoComplete assigned")
        
        print("EyeTracker: Setting up MediaPipe...")
        self.mp_face_mesh = mp.solutions.face_mesh
        print("EyeTracker: MediaPipe face_mesh module loaded")
        
        print("EyeTracker: Creating FaceMesh instance...")
        try:
            # Use optimized settings for fastest startup and reasonable performance
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=False,  # Disable for faster startup
                min_detection_confidence=0.5,  # Lower for faster detection
                min_tracking_confidence=0.5    # Lower for faster tracking
            )
            print("EyeTracker: FaceMesh instance created successfully")
        except Exception as e:
            print(f"EyeTracker: Error creating FaceMesh: {e}")
            # Fallback to absolute minimal settings
            try:
                self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1)
                print("EyeTracker: FaceMesh created with minimal settings")
            except Exception as e2:
                print(f"EyeTracker: Fallback failed: {e2} - Using basic initialization")
                self.face_mesh = self.mp_face_mesh.FaceMesh()
                print("EyeTracker: FaceMesh created with default settings")

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
            '.-.-': 'Enter', '.--.-': 'CapsLk', '--': 'Backspace', '......': 'Emergency SOS',
            '..--': 'Space', '.---.': 'SELECT1', '..--.': 'SELECT2',
            '.--..': 'SELECT3', '-.-.-': 'Text to Speech', '..-..': 'Clear'
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
        
        # For keyboard highlighting
        self.last_entered_char = None
        self.highlight_timer = 0
        self.highlight_duration = 1.0  # Show highlight for 1 second

        # For SOS transparency
        self.transparency_timer = 0
        self.transparency_duration = 10.0  # Keep transparent for 10 seconds during SOS
        self.is_transparent = False

        # For sleep/wake detection (5-second eye closure)
        self.is_system_active = True  # System starts active
        self.eyes_closed_start_time = 0
        self.sleep_wake_threshold = 5.0  # 5 seconds of closed eyes
        self.last_system_state = True

        # For SOS cooldown (temporary disable after SOS)
        self.sos_cooldown_timer = 0
        self.sos_cooldown_duration = 8.0  # Disable system for 8 seconds after SOS
        self.was_active_before_sos = True

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
        
        # Load configuration for emergency contact - ALWAYS use centralized constant
        self.config = load_config()
        self.emergency_contact = DEFAULT_EMERGENCY_CONTACT.replace(" ", "")
        print(f"Emergency contact set to: {self.emergency_contact}")
        print("EyeTracker: Initialization completed successfully!")

    def make_emergency_call(self, phone_number):
        """Initiate an emergency call and send WhatsApp message"""
        success = False
        
        # Clean and format the phone number
        clean_number = phone_number.replace(" ", "").replace("-", "")
        print(f"🚨 EMERGENCY: Initiating SOS for {phone_number}")
        print(f"📱 Formatted number: {clean_number}")
        
        try:
            # Method 1: Send WhatsApp message first (more reliable than calling)
            emergency_message = "🚨 EMERGENCY ALERT! I need immediate help. This is an automated SOS message from OptiBlink. Please contact me urgently!"
            
            # Use WhatsApp desktop app protocol (more reliable than web)
            whatsapp_msg_url = f"whatsapp://send?phone={clean_number}&text={emergency_message.replace(' ', '%20').replace('!', '%21')}"
            print("📲 Sending WhatsApp emergency message...")
            webbrowser.open(whatsapp_msg_url)
            time.sleep(4)  # Wait for WhatsApp to open
            
            if PYAUTOGUI_AVAILABLE:
                try:
                    # Wait for WhatsApp to fully load, then send
                    time.sleep(4)  # Additional wait for WhatsApp to fully load
                    print("⌨️ Pressing Enter to send WhatsApp message...")
                    pyautogui.press('enter')  # Send the pre-filled message
                    time.sleep(1)
                    print("✅ WhatsApp emergency message sent!")
                    
                except Exception as wa_error:
                    print(f"⚠️ WhatsApp automation failed: {wa_error}")
            else:
                print("📲 WhatsApp opened - manual send required")
            
            # Wait between WhatsApp message and phone call to avoid confusion
            print("⏳ Waiting 5 seconds between WhatsApp message and phone call...")
            time.sleep(5)
            
            # Method 2: Make emergency phone call - SIMPLE APPROACH
            try:
                print("📞 Making emergency phone call...")
                
                # Try direct Windows dialer first (most reliable)
                import subprocess
                print(f"📱 Attempting system call to {clean_number}...")
                
                # Use tel: protocol - works with Windows default phone handler
                subprocess.run(['start', '', f'tel:{clean_number}'], shell=True, check=False)
                time.sleep(3)  # Wait for dialer to open
                
                if PYAUTOGUI_AVAILABLE:
                    # Simple approach - just press Enter to call
                    print("⌨️ Pressing Enter to initiate call...")
                    pyautogui.press('enter')
                    time.sleep(0.5)
                    pyautogui.press('enter')  # Double press for confirmation
                    print("✅ Emergency call initiated!")
                else:
                    print("📞 System dialer opened - manual call required")
                
            except Exception as phone_error:
                print(f"⚠️ Phone calling failed: {phone_error}")
                print(f"📞 EMERGENCY: Manually call {clean_number} immediately!")
                
                # Automatically attempt to make the emergency call
                print("✅ Phone Link should be loaded - attempting call!")
                
                # Try multiple methods to trigger the call automatically
                if PYAUTOGUI_AVAILABLE:
                    try:
                        print("🔄 Attempting automatic emergency call...")
                        
                        # Method 1: DON'T use Alt+Tab - Phone Link should already be focused
                        # Just try the call immediately since Phone Link opened
                        
                        # Method 2: Try multiple click locations for call button
                        screen_width, screen_height = pyautogui.size()
                        
                        # Try different common call button locations
                        call_locations = [
                            (screen_width // 2, int(screen_height * 0.8)),      # Center-bottom
                            (screen_width // 2, int(screen_height * 0.75)),     # Center-lower
                            (screen_width // 2, int(screen_height * 0.85)),     # Center-very bottom
                            (int(screen_width * 0.7), int(screen_height * 0.8)) # Right-bottom
                        ]
                        
                        for i, (x, y) in enumerate(call_locations):
                            print(f"🖱️ Trying call button location {i+1}: ({x}, {y})")
                            pyautogui.click(x, y)
                            time.sleep(0.8)
                            
                            # After each click, try keyboard shortcuts
                            pyautogui.press('enter')
                            time.sleep(0.3)
                        
                        # Method 3: Try keyboard shortcuts without clicking
                        print("⌨️ Trying direct keyboard shortcuts...")
                        key_combinations = ['enter', 'space', 'return']
                        
                        for key in key_combinations:
                            print(f"⌨️ Pressing {key.upper()}")
                            pyautogui.press(key)
                            time.sleep(0.5)
                        
                        # Method 4: Try common phone app hotkeys
                        phone_hotkeys = [
                            ('ctrl', 'enter'),  # Common call shortcut
                            ('ctrl', 'shift', 'c'),  # Another call shortcut
                            ('f5',),  # Call/dial in some apps
                        ]
                        
                        for hotkey in phone_hotkeys:
                            print(f"⌨️ Trying hotkey: {'+'.join(hotkey)}")
                            pyautogui.hotkey(*hotkey)
                            time.sleep(0.5)
                        
                        print("✅ All emergency call methods attempted!")
                        
                    except Exception as auto_error:
                        print(f"⚠️ Automatic calling failed: {auto_error}")
                        
                        # Final fallback: Try to open default phone dialer
                        try:
                            print("📞 Fallback: Opening system phone dialer...")
                            import subprocess
                            subprocess.run(['start', '', f'tel:{clean_number}'], shell=True, check=False)
                            time.sleep(2)
                            pyautogui.press('enter')
                            print("✅ System dialer attempted!")
                        except Exception as fallback_error:
                            print(f"⚠️ System dialer fallback failed: {fallback_error}")
                            
                else:
                    print("📞 Phone Link opened - automatic calling not available")

                
            except Exception as phone_error:
                print(f"⚠️ Phone Link calling failed: {phone_error}")
                print(f"📞 MANUAL DIAL REQUIRED: {clean_number}")
                
            except Exception as phone_error:
                print(f"⚠️ Phone Link calling failed: {phone_error}")
            
            return True
            
        except Exception as e:
            print(f"❌ Emergency call failed: {e}")
            print(f"🚨 MANUAL ACTION: Call {phone_number} immediately!")
            

            if PYPERCLIP_AVAILABLE:
                try:
                    pyperclip.copy(f"EMERGENCY: CALL {clean_number} - Need immediate help!")
                    print(f"📋 Emergency number copied: {clean_number}")
                except:
                    print(f" Manual dial: {clean_number}")
            else:
                print(f" Manual dial: {clean_number}")
        
            return False

    def set_window_transparency(self, window_name, alpha=0.5):
        """Set window transparency (alpha: 0.0=fully transparent, 1.0=fully opaque)"""
        try:
            if WIN32_AVAILABLE:
                # Find the OpenCV window handle
                hwnd = win32gui.FindWindow(None, window_name)
                if hwnd:
                    # Get current window style
                    ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                    
                    # Set the window as layered (required for transparency)
                    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, 
                                         ex_style | win32con.WS_EX_LAYERED)
                    
                    # Set the transparency level (0-255, where 255 is opaque)
                    alpha_value = int(alpha * 255)
                    win32gui.SetLayeredWindowAttributes(hwnd, 0, alpha_value, 
                                                      win32con.LWA_ALPHA)
                    print(f"Window transparency set to {int(alpha*100)}%")
                    return True
            else:
                print("Transparency requires win32gui (install pywin32)")
        except Exception as e:
            print(f"Could not set transparency: {e}")
        return False
    
    def restore_window_opacity(self, window_name):
        """Restore window to full opacity"""
        return self.set_window_transparency(window_name, 1.0)

    def draw_keyboard(self, width, height):
        """Draw a dynamic keyboard using OpenCV"""
        keyboard_img = np.zeros((height, width, 3), dtype=np.uint8)
        keyboard_img[:] = (40, 40, 40)  # Dark gray background
        
        # Key dimensions for better fit
        # Calculate available space more accurately
        available_width = width - 20  # Leave 10px margin on each side
        
        # Row 0 has the most keys (11), so base calculation on that
        # Account for: TTS(1.4x) + 10 numbers(1x) + margins between keys
        total_key_units = 1.4 + 10.0  # Text to Speech + 10 number keys
        key_spacing = 2  # Space between keys
        total_spacing = key_spacing * 10  # 10 gaps between 11 keys
        
        base_key_width = int((available_width - total_spacing) / total_key_units)
        key_height = (height - 25) // 4 - 3   # Optimized row spacing
        margin_x = 10  # Left margin
        margin_y = 4   # Reduced top margin
        key_gap = 2    # Gap between keys
        
        # Debug info (will print once)
        if not hasattr(self, 'keyboard_debug_printed'):
            print(f"Keyboard: width={width}, available={available_width}, base_key={base_key_width}")
            self.keyboard_debug_printed = True
        
        # Colors - different for active/sleep/SOS cooldown states
        if (self.sos_cooldown_timer > 0 and 
            time.time() - self.sos_cooldown_timer < self.sos_cooldown_duration):
            # SOS cooldown state - yellowish tint
            normal_color = (60, 80, 80)      # Yellowish gray key (cooldown)
            highlight_color = (100, 255, 255) # Yellow highlight (cooldown)
            text_color = (200, 255, 255)     # Yellowish text (cooldown)
            highlight_text_color = (0, 50, 50) # Dark text on highlighted key
        elif self.is_system_active:
            normal_color = (80, 80, 80)      # Gray key (active)
            highlight_color = (0, 255, 0)    # Green highlight (active)
            text_color = (255, 255, 255)     # White text (active)
            highlight_text_color = (0, 0, 0) # Black text on highlighted key
        else:
            normal_color = (40, 40, 40)      # Darker gray key (sleep)
            highlight_color = (60, 60, 60)   # Dark gray highlight (sleep)
            text_color = (100, 100, 100)     # Dimmed text (sleep)
            highlight_text_color = (150, 150, 150) # Dimmed highlight text
        
        # Define each row separately - matching morse_keyboard.jpg reference
        rows = [
            # Row 0: Text to Speech and Numbers
            [('-.-.-', 'Text to Speech'), ('.----', '1'), ('..---', '2'), ('...--', '3'), ('....-', '4'), 
             ('.....', '5'), ('-....', '6'), ('--...', '7'), ('---..', '8'), ('----.', '9'), ('-----', '0')],
            
            # Row 1: QWERTYUIOP + Backspace
            [('--.-', 'q'), ('.--', 'w'), ('.', 'e'), ('.-.', 'r'), ('-', 't'), 
             ('-.--', 'y'), ('..-', 'u'), ('..', 'i'), ('---', 'o'), ('.--.', 'p'), ('--', 'Backspace')],
            
            # Row 2: CapsLk + ASDFGHJKL + Enter
            [('.--.-', 'CapsLk'), ('.-', 'a'), ('...', 's'), ('-..', 'd'), ('..-.', 'f'), 
             ('--.', 'g'), ('....', 'h'), ('.---', 'j'), ('-.-', 'k'), ('.-..', 'l'), ('.-.-', 'Enter')],
            
            # Row 3: Emergency SOS + ZXCVBNM + Clear + Space
            [('......', 'Emergency SOS'), ('--..', 'z'), ('-..-', 'x'), ('-.-.', 'c'), ('...-', 'v'), 
             ('-...', 'b'), ('-.', 'n'), ('----', 'm'), ('..-..', 'Clear'), ('..--', 'Space')]
        ]
        
        for row_idx, row_keys in enumerate(rows):
            current_x = margin_x
            
            for col_idx, (morse_code, display_text) in enumerate(row_keys):
                # More conservative key widths to ensure everything fits
                if display_text in ['Text to Speech', 'Emergency SOS']:
                    key_width = int(base_key_width * 1.4)  # Reduced to 1.4x
                elif display_text == 'Backspace':
                    key_width = int(base_key_width * 1.1)  # Reduced to 1.1x
                elif display_text == 'Space':
                    key_width = int(base_key_width * 1.6)  # Reduced to 1.6x
                elif display_text == 'CapsLk':
                    key_width = int(base_key_width * 1.1)  # Slightly wider
                elif display_text == 'Enter':
                    key_width = int(base_key_width * 1.1)  # Slightly wider
                elif display_text == 'Clear':
                    key_width = int(base_key_width * 1.0)  # Standard width
                else:
                    key_width = base_key_width
                
                # Calculate key position
                key_y = margin_y + row_idx * (key_height + margin_y)
                
                # Ensure we don't exceed the window bounds
                if current_x + key_width > width - margin_x:
                    # If we're running out of space, compress this key
                    key_width = width - current_x - margin_x
                    if key_width < base_key_width * 0.6:  # Don't make it too small
                        key_width = int(base_key_width * 0.6)
                
                # Check if this key should be highlighted
                is_highlighted = (self.last_entered_char and 
                                self.highlight_timer > 0 and 
                                time.time() - self.highlight_timer < self.highlight_duration and
                                ((morse_code in self.morse_to_letter and 
                                  self.morse_to_letter[morse_code] == self.last_entered_char) or
                                 (display_text == self.last_entered_char) or
                                 (display_text == 'Backspace' and self.last_entered_char == 'Back')))
                
                # Draw key background
                key_color = highlight_color if is_highlighted else normal_color
                cv2.rectangle(keyboard_img, (current_x, key_y), (current_x + key_width, key_y + key_height), key_color, -1)
                
                # Draw key border (reduced thickness)
                border_color = (200, 200, 200) if is_highlighted else (120, 120, 120)
                cv2.rectangle(keyboard_img, (current_x, key_y), (current_x + key_width, key_y + key_height), border_color, 1)
                
                # Draw key text with appropriate sizing
                font = cv2.FONT_HERSHEY_SIMPLEX
                if len(display_text) > 8:
                    font_scale = 0.35  # Very small for long text
                elif len(display_text) > 4:
                    font_scale = 0.4   # Small for medium text
                else:
                    font_scale = 0.5   # Normal for short text
                    
                text_col = highlight_text_color if is_highlighted else text_color
                
                # Center text in key
                (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, 1)
                text_x = current_x + (key_width - text_w) // 2
                text_y = key_y + (key_height + text_h) // 2 - 4
                cv2.putText(keyboard_img, display_text, (text_x, text_y), font, font_scale, text_col, 1)
                
                # Draw morse code below (smaller font) - show for all keys that have morse codes
                if morse_code and morse_code != display_text and len(morse_code) <= 6:
                    morse_font_scale = 0.25 if len(display_text) > 6 else 0.3  # Smaller font for long key names
                    (morse_w, morse_h), _ = cv2.getTextSize(morse_code, font, morse_font_scale, 1)
                    morse_x = current_x + (key_width - morse_w) // 2
                    morse_y = key_y + key_height - 3  # Slightly higher for better visibility
                    cv2.putText(keyboard_img, morse_code, (morse_x, morse_y), font, morse_font_scale, text_col, 1)
                
                # Move to next key position with consistent gap
                current_x += key_width + key_gap
        
        # Add overlay if system is not truly active (sleep mode or SOS cooldown)
        if not self.is_system_truly_active():
            overlay = keyboard_img.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(keyboard_img, 0.3, overlay, 0.7, 0, keyboard_img)
            
            # Determine overlay text based on state
            if (self.sos_cooldown_timer > 0 and 
                time.time() - self.sos_cooldown_timer < self.sos_cooldown_duration):
                text = "SOS COOLDOWN"
                text_color = (100, 255, 255)  # Yellow text
            else:
                text = "SLEEP MODE"
                text_color = (255, 255, 255)  # White text
                
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height // 2
            cv2.putText(keyboard_img, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        return keyboard_img

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

    def process_special_character(self, char):
        # Skip all keyboard writing during SOS cooldown period to prevent interference
        if (self.sos_cooldown_timer > 0 and 
            time.time() - self.sos_cooldown_timer < self.sos_cooldown_duration):
            print("SOS cooldown active - skipping keyboard operations")
            return
            
        if char == 'Enter':
            text_to_send = self.word_buffer[self.last_sent_len:]
            if text_to_send:
                keyboard.write(text_to_send)
            keyboard.send('enter')
            if self.word_buffer:
                print(f"Message: {self.word_buffer}")
                self.message_history.append(self.word_buffer)
                
                # Record usage for individual words only, not the entire sentence
                words = self.word_buffer.strip().split()
                for word in words:
                    self.auto.record_usage(word.lower())
                
                self.last_word_printed = self.word_buffer
                if self.tts_enabled:
                    message_to_speak = self.word_buffer  # Store before clearing
                    # Add a short delay before TTS
                    time.sleep(0.3)
                    self.speak(message_to_speak)
            self.word_buffer = ""
            self.morse_char_buffer = ""
            self.last_sent_len = 0
        elif char == 'Space':
            self.word_buffer += " "
            keyboard.send('space')
            self.morse_char_buffer = ""
            self.last_sent_len = len(self.word_buffer)
        elif char == 'Backspace':
            if self.word_buffer:
                self.word_buffer = self.word_buffer[:-1]
                keyboard.send('backspace')
            self.morse_char_buffer = ""
            self.last_sent_len = len(self.word_buffer)
        elif char == 'CapsLk':
            self.caps_lock = not self.caps_lock
            self.morse_char_buffer = ""
        elif char == 'Emergency SOS':
            # Initiate emergency call
            print("\n*** SOS EMERGENCY ***")
            
            # Make window translucent so user can see background (phone app)
            self.set_window_transparency("Eye Blink Morse Code", 0.3)  # 30% opacity
            self.is_transparent = True
            self.transparency_timer = time.time()
            print("Window made translucent to see phone app")
            
            call_success = self.make_emergency_call(self.emergency_contact)
            
            self.message_history.append("SOS - Emergency call initiated")
            # Don't write to keyboard during emergency - it interferes with phone app
            
            if self.tts_enabled:
                self.speak(f"S O S emergency activated. Calling {self.emergency_contact}")
            else:
                # Even if TTS is off, speak emergency message
                self.speak(f"S O S emergency activated. Calling {self.emergency_contact}")
            
            # Start SOS cooldown - temporarily disable system
            self.was_active_before_sos = self.is_system_active
            self.is_system_active = False
            self.sos_cooldown_timer = time.time()
            print("System temporarily disabled to prevent accidental input")
            
            self.word_buffer = ""
            self.morse_char_buffer = ""
            self.last_sent_len = len(self.word_buffer)
        elif char == 'Text to Speech':
            self.tts_enabled = not self.tts_enabled
            status = "ON" if self.tts_enabled else "OFF"
            print(f"TTS {status}")
            if self.word_buffer:
                self.speak(self.word_buffer)
            else:
                self.speak("No message")
            self.morse_char_buffer = ""
        elif char == 'Clear':
            # Clear the word buffer by sending backspaces for each character
            if self.word_buffer:
                for _ in range(len(self.word_buffer)):
                    keyboard.send('backspace')
                self.word_buffer = ""
            self.morse_char_buffer = ""
            self.last_sent_len = 0
            if self.tts_enabled:
                self.speak("Buffer cleared")

    def is_system_truly_active(self):
        """Check if system is active considering both sleep mode and SOS cooldown"""
        # If in SOS cooldown, system is temporarily inactive
        if (self.sos_cooldown_timer > 0 and 
            time.time() - self.sos_cooldown_timer < self.sos_cooldown_duration):
            return False
        return self.is_system_active

    def decode_morse_char(self):
        # Only process morse code if system is truly active
        if not self.is_system_truly_active():
            return None
            
        # Reset the flag at the beginning of decoding a new character
        self.key_sent_for_current_char = False 

        if self.morse_char_buffer in self.morse_to_letter:
            char = self.morse_to_letter[self.morse_char_buffer]
            if char == 'Text to Speech':
                self.process_special_character(char)
                self.key_sent_for_current_char = True
                self.morse_char_buffer = ""
                return char
            if char in ['Enter', 'Space', 'Backspace', 'CapsLk', 'Emergency SOS', 'Clear']:
                self.process_special_character(char)
                self.key_sent_for_current_char = True # Mark as key sent
                # Set highlighting for special characters
                self.last_entered_char = char
                self.highlight_timer = time.time()
            elif char.startswith("SELECT"):
                # Skip keyboard writing during SOS cooldown period
                if (self.sos_cooldown_timer > 0 and 
                    time.time() - self.sos_cooldown_timer < self.sos_cooldown_duration):
                    print("SOS cooldown active - skipping word selection")
                    self.morse_char_buffer = ""  # Clear buffer but don't process
                    return None
                
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
                # Skip keyboard writing during SOS cooldown period
                if (self.sos_cooldown_timer > 0 and 
                    time.time() - self.sos_cooldown_timer < self.sos_cooldown_duration):
                    print("SOS cooldown active - skipping character processing")
                    self.morse_char_buffer = ""  # Clear buffer but don't process
                    return None
                
                char_to_add = char.upper() if self.caps_lock else char.lower()
                self.word_buffer += char_to_add
                keyboard.write(char_to_add) # Only write the single new character
                self.key_sent_for_current_char = True # Mark as key sent
                self.last_sent_len = len(self.word_buffer) # Update sent length
                # Set highlighting for normal characters
                self.last_entered_char = char_to_add.upper()  # Always highlight as uppercase for consistency
                self.highlight_timer = time.time()
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

    def reset_calibration(self):
        """Reset calibration to allow recalibration"""
        self.is_calibrated = False
        self.calibration_counter = 0
        self.ear_baseline = 0
        self.area_baseline = 0
        print("Calibration reset. Please look straight at the camera for recalibration.")

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
                        # Start tracking for sleep/wake detection
                        if self.eyes_closed_start_time == 0:
                            self.eyes_closed_start_time = current_time
                    
                    self.blink_duration = current_time - self.blink_start_time
                    self.open_duration = 0
                    
                    # Check for sleep/wake toggle (5 seconds of closed eyes)
                    if (self.eyes_closed_start_time > 0 and 
                        current_time - self.eyes_closed_start_time >= self.sleep_wake_threshold):
                        # Toggle system state
                        self.is_system_active = not self.is_system_active
                        status = "ACTIVE" if self.is_system_active else "SLEEP"
                        print(f"\n*** SYSTEM {status} ***")
                        
                        if self.tts_enabled:
                            self.speak(f"System {status.lower()}")
                        
                        # Reset timer to prevent immediate re-triggering
                        self.eyes_closed_start_time = 0
                        
                else:
                    # Eyes are open, reset the closed timer
                    self.eyes_closed_start_time = 0
                    
                    if self.is_blinking:
                        blink_duration = current_time - self.blink_start_time
                        self.is_blinking = False
                        self.blink_state = "Open"
                        self.open_start_time = current_time
                        
                        # Only process morse code if system is truly active (not in sleep or SOS cooldown)
                        if self.is_system_truly_active():
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

        # Determine system status display
        if (self.sos_cooldown_timer > 0 and 
            time.time() - self.sos_cooldown_timer < self.sos_cooldown_duration):
            system_status = "SOS COOLDOWN"
            status_color = (0, 255, 255)  # Yellow for cooldown
        elif self.is_system_active:
            system_status = "ACTIVE"
            status_color = (0, 255, 0)  # Green for active
        else:
            system_status = "SLEEP"
            status_color = (0, 0, 255)  # Red for sleep
            
        cv2.putText(frame, f"SYSTEM: {system_status} - CAPS: {'ON' if self.caps_lock else 'OFF'} - TTS: {'ON' if self.tts_enabled else 'OFF'} - Word: {self.word_buffer}",
                    (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
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
    print("Starting OptiBlink Eye Tracker...")
    cap = cv2.VideoCapture(0)
    print("Camera initialized")
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    auto = AutoCompleteSystem()
    print("AutoComplete system loaded")
    
    eye_tracker = EyeTracker(auto)
    print("EyeTracker initialized")

    window_width = 680  # Increased slightly to fit keyboard better
    window_height = 460  # Adjusted height to maintain proportions

    # Move window to top-right corner with some margin
    try:
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        x_pos = screen_width - window_width - 20  # Added 20px margin from edge
    except Exception:
        # Fallback to default position if Windows API is not available
        x_pos = 100
        print("Warning: Could not get screen width. Using default window position.")
    
    y_pos = 40  # Leave space for window controls

    # Function to set OpenCV window always on top
    def set_window_always_on_top(window_name):
        if WIN32_AVAILABLE:
            try:
                hwnd = win32gui.FindWindow(None, window_name)
                if hwnd:
                    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            except Exception as e:
                print(f"Warning: Could not set window always on top: {e}")
        # If win32 is not available, the function does nothing gracefully

    print("Starting main loop...")
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

            # Create dynamic keyboard with highlighting
            keyboard_img = eye_tracker.draw_keyboard(window_width, keyboard_height)

            # Check if transparency timer has expired and restore opacity
            if (eye_tracker.is_transparent and 
                eye_tracker.transparency_timer > 0 and
                time.time() - eye_tracker.transparency_timer > eye_tracker.transparency_duration):
                eye_tracker.restore_window_opacity("OptiBlink")
                eye_tracker.is_transparent = False
                eye_tracker.transparency_timer = 0
                print("Window opacity restored")

            # Check if SOS cooldown has expired and restore system activity
            if (eye_tracker.sos_cooldown_timer > 0 and
                time.time() - eye_tracker.sos_cooldown_timer > eye_tracker.sos_cooldown_duration):
                eye_tracker.is_system_active = eye_tracker.was_active_before_sos
                eye_tracker.sos_cooldown_timer = 0
                status = "ACTIVE" if eye_tracker.is_system_active else "SLEEP"
                print(f"SOS cooldown expired - System restored to {status}")

            full_display_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            full_display_frame[0:video_height, 0:window_width] = processed_video_frame
            full_display_frame[video_height:video_height+keyboard_height, 0:window_width] = keyboard_img

            cv2.imshow("OptiBlink", full_display_frame)
            cv2.moveWindow("OptiBlink", x_pos, y_pos)
            set_window_always_on_top("OptiBlink")

            # Robust window close detection
            try:
                if cv2.getWindowProperty("OptiBlink", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == -1:
                # If window is closed, waitKey returns -1
                if cv2.getWindowProperty("OptiBlink", cv2.WND_PROP_VISIBLE) < 1:
                    break
            
            # Check for actual keyboard input (only when window has focus and key is physically pressed)
            # Use GetAsyncKeyState to check for actual physical key presses
            try:
                import ctypes
                # Check if Q key is currently being pressed (GetAsyncKeyState returns non-zero if pressed)
                if ctypes.windll.user32.GetAsyncKeyState(ord('Q')) & 0x8000:
                    if not hasattr(eye_tracker, '_q_pressed'):
                        eye_tracker._q_pressed = True
                        break
                else:
                    eye_tracker._q_pressed = False
                
                # Check if R key is currently being pressed
                if ctypes.windll.user32.GetAsyncKeyState(ord('R')) & 0x8000:
                    if not hasattr(eye_tracker, '_r_pressed'):
                        eye_tracker._r_pressed = True
                        eye_tracker.reset_calibration()
                else:
                    eye_tracker._r_pressed = False
            except:
                # Fallback to original method if ctypes fails
                if key != 255 and key != -1:
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        eye_tracker.reset_calibration()
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to continue...")
