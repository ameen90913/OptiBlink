# Standard library imports
import csv
import ctypes
import json
import os
import subprocess
import tempfile
import threading
import time
import traceback
import webbrowser
from collections import deque, defaultdict

# CONFIGURATION - Change emergency contact number here
DEFAULT_EMERGENCY_CONTACT = "+91 9632168509"

# WINDOW CONFIGURATION
WINDOW_NAME = "OptiBlink"

# Suppress TensorFlow informational messages EARLY
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Core imports that are always needed
import cv2
import numpy as np
import keyboard

# Defer heavy imports until needed
mp = None  # Will be imported when EyeTracker is created
pygame = None  # Will be imported when TTS is first used
gTTS = None  # Will be imported when TTS is first used

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

try:
    from pywinauto import Application
    PYWINAUTO_AVAILABLE = True
except ImportError:
    PYWINAUTO_AVAILABLE = False
    print("Warning: pywinauto not available. Emergency phone calling may use fallback methods.")

# Configuration management
def load_config():
    """Load configuration from config.json file"""
    config_file = "config.json"
    default_config = {
        "emergency_contact": DEFAULT_EMERGENCY_CONTACT,
        "prefer_whatsapp_web": False  # Set to True to force WhatsApp Web over app
                                     # Useful for users without WhatsApp desktop app
                                     # or those who prefer web interface
    }
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                # ALWAYS use the centralized constant, ignore what's in the file
                config["emergency_contact"] = DEFAULT_EMERGENCY_CONTACT
                # Add prefer_whatsapp_web if not present
                if "prefer_whatsapp_web" not in config:
                    config["prefer_whatsapp_web"] = False
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

# Load words from CSV or fallback to NLTK - OPTIMIZED FOR FAST STARTUP
def load_words_from_csv(csv_path, column_name="Word"):
    words = []
    try:
        # Quick file check first
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}. Will use NLTK if needed.")
            return words
            
        print(f"EyeTracker: Loading words from {csv_path}...")
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # Only load first 3000 words for faster startup
            count = 0
            for row in reader:
                if count >= 3000:  # Limit for faster startup
                    break
                word = row.get(column_name)
                if word and word.isalpha() and len(word) > 2:
                    words.append(word.lower())
                count += 1
        print(f"EyeTracker: Loaded {len(words)} words from CSV for fast startup")
    except Exception as e:
        print(f"CSV loading error: {e}. Will use NLTK fallback if needed.")
    return words

# Load CSV words quickly - no NLTK loading at startup for speed
print("EyeTracker: Fast startup - loading essential words only...")
csv_word_list = load_words_from_csv(r"words.csv", column_name="Word")

# Skip NLTK loading entirely at startup - defer until actually needed
nltk_word_list = []
print("EyeTracker: Word loading complete - NLTK deferred for speed!")

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class AutoCompleteSystem:
    def __init__(self):
        print("AutoComplete: Building word trees...")
        self.csv_root = TrieNode()
        self.nltk_root = TrieNode()
        self.frequency = defaultdict(int)
        self.nltk_loaded = False  # Track if NLTK has been loaded
        
        # Load CSV words first (fast)
        for word in csv_word_list:
            self.insert(word, root=self.csv_root)
        
        # Skip NLTK loading at startup for speed
        print(f"AutoComplete: Loaded {len(csv_word_list)} CSV words quickly!")
        
        self.load_usage_data()
        print("AutoComplete: Ready!")
    
    def _ensure_nltk_loaded(self):
        """Load NLTK words only when actually needed"""
        if self.nltk_loaded or len(csv_word_list) > 1000:
            return  # Skip if already loaded or CSV is sufficient
        
        try:
            print("AutoComplete: Loading NLTK words (first time only)...")
            import nltk
            
            nltk.data.find('corpora/words')
            from nltk.corpus import words as nltk_words
            
            # Load subset for performance
            all_words = nltk_words.words()
            nltk_word_list = [word.lower() for word in all_words[:3000] if word.isalpha() and len(word) > 2]
            
            for word in nltk_word_list:
                self.insert(word, root=self.nltk_root)
            
            self.nltk_loaded = True
            print(f"AutoComplete: NLTK loaded - {len(nltk_word_list)} additional words available!")
            
        except Exception as e:
            print(f"AutoComplete: NLTK loading failed ({e}) - using CSV words only")
            self.nltk_loaded = True  # Prevent retrying

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
        
        # If less than 3 results, try to load NLTK words
        if len(results) < 3:
            self._ensure_nltk_loaded()  # Lazy load NLTK if needed
            
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
        print("EyeTracker: Fast initialization starting...")
        self.auto = auto
        
        # Defer MediaPipe import and initialization for faster startup
        global mp
        if mp is None:
            print("EyeTracker: Importing MediaPipe...")
            import mediapipe as mp_module
            mp = mp_module
        
        print("EyeTracker: Setting up MediaPipe face mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Create FaceMesh with fastest possible settings
        print("EyeTracker: Creating optimized FaceMesh...")
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,  # Disable for speed
            min_detection_confidence=0.3,  # Lower threshold for speed  
            min_tracking_confidence=0.3    # Lower threshold for speed
        )
        
        # Initialize all other attributes quickly
        print("EyeTracker: Setting up core systems...")
        self._initialize_morse_and_detection()
        print("EyeTracker: Initialization complete!")

    def _initialize_morse_and_detection(self):
        """Initialize morse code and eye detection systems"""
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
            '.--..': 'SELECT3', '-.-.-': 'TTS', '..-..': 'Clear'
        }

        # Eye landmark indices
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        # Initialize all state variables
        self.blink_counter = 0
        self.short_blinks = 0
        self.long_blinks = 0
        self.blink_start_time = None
        self.open_start_time = None
        self.is_blinking = False
        
        # Timing thresholds
        self.short_blink_threshold = 0.3
        self.open_threshold = 1.0

        # History tracking
        self.ear_history = deque(maxlen=5)
        self.area_history = deque(maxlen=3)

        # Calibration
        self.is_calibrated = False
        self.calibration_frames = 30
        self.calibration_counter = 0
        self.ear_baseline = 0
        self.area_baseline = 0

        # Message and input
        self.morse_char_buffer = ""
        self.message_history = []
        self.word_buffer = ""
        self.caps_lock = False
        
        # UI and highlighting
        self.last_entered_char = None
        self.highlight_timer = 0
        self.highlight_duration = 1.0

        # Transparency and sleep
        self.transparency_timer = 0
        self.transparency_duration = 10.0
        self.is_transparent = False
        self.is_system_active = True
        self.eyes_closed_start_time = 0
        self.sleep_wake_threshold = 5.0
        self.last_system_state = True

        # SOS cooldown
        self.sos_cooldown_timer = 0
        self.sos_cooldown_duration = 8.0
        self.was_active_before_sos = True

        # Phone call pause - prevent OptiBlink interference during emergency calls
        self.phone_call_active = False
        self.phone_call_start_time = 0
        self.phone_call_pause_duration = 15.0  # Pause OptiBlink for 15 seconds during emergency calls
        self.was_active_before_phone_call = True  # Save state before emergency calls
        
        # Add flag to completely disable keyboard processing during calls
        self.keyboard_processing_enabled = True

        # Current state
        self.current_ear = 0
        self.current_area = 0
        self.blink_state = "Open"
        self.last_blink_ear = 0
        self.last_blink_area = 0
        self.blink_duration = 0
        self.open_duration = 0

        # Suggestions and TTS
        self.current_suggestions = []
        self.last_word_printed = ""
        self.key_sent_for_current_char = False
        self.last_sent_len = 0
        self.tts_enabled = False
        self.is_speaking = False
        self.speaking_message = ""
        
        # Configuration
        self.config = load_config()
        self.emergency_contact = DEFAULT_EMERGENCY_CONTACT.replace(" ", "")

    def make_emergency_call(self, phone_number):
        """Initiate an emergency call and send WhatsApp message"""
        
        # Clean and format the phone number
        clean_number = phone_number.replace(" ", "").replace("-", "")
        print(f"🚨 EMERGENCY: Initiating SOS for {phone_number}")
        print(f"📱 Formatted number: {clean_number}")
        
        try:
            # Method 1: Send WhatsApp message with improved reliability
            emergency_message = "🚨 EMERGENCY ALERT! I need immediate help. This is an automated SOS message from OptiBlink. Please contact me urgently!"
            
            print("📲 Sending WhatsApp emergency message...")
            
            # Check if user prefers WhatsApp Web
            prefer_web = self.config.get("prefer_whatsapp_web", False)
            if prefer_web:
                print("🌐 Configuration set to prefer WhatsApp Web - skipping app")
                whatsapp_opened = False
            else:
                whatsapp_msg_url = f"whatsapp://send?phone={clean_number}&text={emergency_message.replace(' ', '%20').replace('!', '%21')}"
                print(f"🔗 WhatsApp URL: {whatsapp_msg_url}")
                
                # Try WhatsApp app first, then fallback to WhatsApp Web
                try:
                    webbrowser.open(whatsapp_msg_url)
                    time.sleep(8)  # Wait for WhatsApp app to load
                    whatsapp_opened = True
                except Exception as app_error:
                    print(f"⚠️ WhatsApp app failed: {app_error}")
                    whatsapp_opened = False
                
                # Check if WhatsApp app actually opened by looking for window
                if WIN32_AVAILABLE and whatsapp_opened:
                    try:
                        def find_whatsapp():
                            def enum_callback(hwnd, windows):
                                if win32gui.IsWindowVisible(hwnd):
                                    title = win32gui.GetWindowText(hwnd).lower()
                                    if 'whatsapp' in title and len(title) > 0:
                                        windows.append((hwnd, win32gui.GetWindowText(hwnd)))
                                return True
                            
                            windows = []
                            win32gui.EnumWindows(enum_callback, windows)
                            return windows[0] if windows else (0, "")
                        
                        time.sleep(3)  # Give app time to open
                        whatsapp_hwnd, whatsapp_title = find_whatsapp()
                        if whatsapp_hwnd == 0:
                            print("⚠️ WhatsApp app not found, trying WhatsApp Web fallback...")
                            whatsapp_opened = False
                    except:
                        whatsapp_opened = False
            
            # WhatsApp Web fallback if app didn't open
            if not whatsapp_opened:
                print("🌐 Falling back to WhatsApp Web...")
                print("ℹ️  Note: WhatsApp Web requires:")
                print("   • Internet connection")
                print("   • WhatsApp account logged in on web browser")
                print("   • May need QR code scan if not previously logged in")
                # Format number for WhatsApp Web (remove + if present)
                web_number = clean_number.replace('+', '')
                whatsapp_web_url = f"https://web.whatsapp.com/send?phone={web_number}&text={emergency_message.replace(' ', '%20').replace('!', '%21')}"
                print(f"🔗 WhatsApp Web URL: {whatsapp_web_url}")
                webbrowser.open(whatsapp_web_url)
                time.sleep(10)  # Extra time for WhatsApp Web to load
                print("✅ WhatsApp Web opened - may require QR code scan if not logged in")
            
            if PYAUTOGUI_AVAILABLE:
                try:
                    # Focus on WhatsApp and send message immediately
                    print("🔄 Focusing on WhatsApp window...")
                    if WIN32_AVAILABLE:
                        # Find WhatsApp window (app or web) specifically
                        try:
                            def find_whatsapp_any():
                                def enum_callback(hwnd, windows):
                                    if win32gui.IsWindowVisible(hwnd):
                                        title = win32gui.GetWindowText(hwnd).lower()
                                        # Check for both WhatsApp app and WhatsApp Web in browser
                                        if ('whatsapp' in title and len(title) > 0) or \
                                           ('whatsapp web' in title) or \
                                           (title.startswith('whatsapp') and 'browser' in title) or \
                                           ('web.whatsapp.com' in title):
                                            windows.append((hwnd, win32gui.GetWindowText(hwnd)))
                                    return True
                                
                                windows = []
                                win32gui.EnumWindows(enum_callback, windows)
                                return windows[0] if windows else (0, "")
                            
                            whatsapp_hwnd, whatsapp_title = find_whatsapp_any()
                            if whatsapp_hwnd != 0:
                                print(f"📱 Found WhatsApp: {whatsapp_title}")
                                win32gui.SetForegroundWindow(whatsapp_hwnd)
                                time.sleep(3)  # Extra time for web version
                                print("✅ WhatsApp window focused!")
                            else:
                                print("⚠️ WhatsApp window not found, using Alt+Tab")
                                pyautogui.hotkey('alt', 'tab')
                                time.sleep(2)
                        except Exception as wa_focus_error:
                            print(f"⚠️ WhatsApp focus failed, using Alt+Tab: {wa_focus_error}")
                            pyautogui.hotkey('alt', 'tab')
                            time.sleep(2)
                    
                    # Send WhatsApp message immediately
                    print("⌨️ Sending WhatsApp message NOW...")
                    
                    # Check if we're using WhatsApp Web or app
                    is_web_whatsapp = False
                    if WIN32_AVAILABLE:
                        try:
                            current_window = win32gui.GetForegroundWindow()
                            current_title = win32gui.GetWindowText(current_window).lower()
                            if 'web.whatsapp.com' in current_title or 'browser' in current_title:
                                is_web_whatsapp = True
                                print("🌐 Detected WhatsApp Web - using web-specific actions")
                        except:
                            pass
                    
                    if is_web_whatsapp:
                        # WhatsApp Web: May need to wait for page load and handle send button
                        print("🌐 WhatsApp Web: Waiting for page to fully load...")
                        time.sleep(3)
                        
                        # Try Tab to navigate to send button, then Enter
                        print("⌨️ Navigating to send button...")
                        pyautogui.press('tab')
                        time.sleep(0.5)
                        pyautogui.press('enter')
                        time.sleep(0.5)
                        
                        # Alternative: Direct Enter attempts
                        for i in range(5):
                            pyautogui.press('enter')
                            time.sleep(0.5)
                            print(f"⌨️ WhatsApp Web Enter {i+1}/5")
                    else:
                        # WhatsApp App: Standard approach
                        print("📱 WhatsApp App: Using standard send approach")
                        for i in range(3):
                            pyautogui.press('enter')
                            time.sleep(0.3)
                            print(f"⌨️ WhatsApp App Enter {i+1}/3")
                    
                    print("✅ WhatsApp message sent!")
                    print("� Switching to WhatsApp using Alt+Tab...")
                    pyautogui.hotkey('alt', 'tab')
                    time.sleep(2)
                    
                    # Check if we're in WhatsApp by looking at window title
                    if WIN32_AVAILABLE:
                        try:
                            current_window = win32gui.GetForegroundWindow()
                            current_title = win32gui.GetWindowText(current_window).lower()
                            print(f"🔍 Current window: {current_title}")
                            
                            if 'whatsapp' not in current_title:
                                # Try Alt+Tab again
                                print("🔄 Trying Alt+Tab again...")
                                pyautogui.hotkey('alt', 'tab')
                                time.sleep(2)
                                current_window = win32gui.GetForegroundWindow()
                                current_title = win32gui.GetWindowText(current_window).lower()
                                print(f"🔍 After second Alt+Tab: {current_title}")
                        except Exception as title_error:
                            print(f"⚠️ Could not check window title: {title_error}")
                    
                    # Send WhatsApp message with multiple attempts
                    print("⌨️ Sending WhatsApp message...")
                    for i in range(3):  # Try 3 times
                        pyautogui.press('enter')
                        time.sleep(0.5)
                        print(f"⌨️ Enter press {i+1}/3 for WhatsApp")
                    print("✅ WhatsApp message sending attempted!")
                    
                    # Verify if message was sent by checking if we can find input field
                    try:
                        current_window = win32gui.GetForegroundWindow()
                        current_title = win32gui.GetWindowText(current_window)
                        print(f"🔍 Current focused window: {current_title}")
                    except:
                        pass
                    
                except Exception as wa_error:
                    print(f"⚠️ WhatsApp automation failed: {wa_error}")
                    print("🔄 Trying manual Alt+Tab method...")
                    try:
                        pyautogui.hotkey('alt', 'tab')
                        time.sleep(2)
                        pyautogui.press('enter')
                        pyautogui.press('enter')
                        print("🔄 Alt+Tab fallback attempted")
                    except:
                        print("⚠️ Alt+Tab fallback also failed")
            else:
                print("📲 WhatsApp opened - pyautogui not available, manual send required")
                if not whatsapp_opened:
                    print("ℹ️  MANUAL ACTION REQUIRED:")
                    print("   🌐 WhatsApp Web should be open in your browser")
                    print("   📝 Emergency message is pre-filled")
                    print("   ▶️  Click the SEND button to send the message")
                    print("   📱 Or manually send this message:")
                    print(f"       '{emergency_message}'")
            
            # Wait between WhatsApp and phone call
            print("⏳ Waiting 2 seconds before making call...")
            time.sleep(2)
            
            # Method 2: Simple phone call using tel: protocol
            try:
                print("📞 Making emergency phone call...")
                
                # PAUSE OptiBlink keyboard monitoring during phone call
                # Save current system state to restore later
                self.was_active_before_phone_call = self.is_system_active
                self.phone_call_active = True
                self.phone_call_start_time = time.time()
                self.keyboard_processing_enabled = False  # Completely disable keyboard processing
                print("🛑 OptiBlink keyboard monitoring COMPLETELY DISABLED for SOS call")
                
                # Use generic tel: protocol (works with default phone handler)
                print(f"📱 Dialing {clean_number} using system dialer...")
                subprocess.run(['start', '', f'tel:{clean_number}'], shell=True, check=False)
                
                # Wait for dialer to load completely
                print("⏳ Waiting for dialer to load...")
                time.sleep(3)
                
                if PYAUTOGUI_AVAILABLE:
                    # Focus specifically on Phone Link and call immediately
                    print("🔄 Focusing on Phone Link window...")
                    if WIN32_AVAILABLE:
                        try:
                            def find_phone_link():
                                def enum_callback(hwnd, windows):
                                    if win32gui.IsWindowVisible(hwnd):
                                        title = win32gui.GetWindowText(hwnd).lower()
                                        phone_keywords = ['phone link', 'your phone', 'phone', 'dialer', 'call']
                                        if any(keyword in title for keyword in phone_keywords) and len(title) > 0:
                                            windows.append((hwnd, win32gui.GetWindowText(hwnd)))
                                    return True
                                
                                windows = []
                                win32gui.EnumWindows(enum_callback, windows)
                                return windows[0] if windows else (0, "")
                            
                            phone_hwnd, phone_title = find_phone_link()
                            if phone_hwnd != 0:
                                print(f"📱 Found Phone app: {phone_title}")
                                win32gui.SetForegroundWindow(phone_hwnd)
                                time.sleep(2)
                                print("✅ Phone app window focused!")
                            else:
                                print("⚠️ Phone app window not found, using Alt+Tab")
                                pyautogui.hotkey('alt', 'tab')
                                time.sleep(2)
                        except Exception as phone_focus_error:
                            print(f"⚠️ Phone focus failed, using Alt+Tab: {phone_focus_error}")
                            pyautogui.hotkey('alt', 'tab')
                            time.sleep(2)
                    
                    # Call immediately while Phone Link is focused
                    print("⌨️ Starting gentle phone call sequence...")
                    time.sleep(2)  # Initial wait for Phone Link to be ready
                    
                    # Use precise navigation to avoid screen casting button
                    if WIN32_AVAILABLE and phone_hwnd != 0:
                        print("🎯 Using precise navigation to locate call button (avoiding screen cast)...")
                        
                        # Ensure Phone Link is focused
                        win32gui.SetForegroundWindow(phone_hwnd)
                        time.sleep(1)
                        
                        # Simple navigation to call button (since it works on first try)
                        print("🔍 Navigating to call button...")
                        pyautogui.hotkey('ctrl', 'home')  # Start from beginning
                        time.sleep(0.5)
                        
                        # Single tab to call button
                        pyautogui.press('tab')
                        time.sleep(0.3)
                        
                        # Activate call button with Enter key
                        print("📞 Placing SOS call...")
                        pyautogui.press('enter')
                        time.sleep(1)
                        print("✅ SOS call placed successfully!")
                            
                    else:
                        # Fallback to simple pyautogui if Windows API not available
                        print("⌨️ Fallback: Using simple pyautogui method...")
                        for i in range(5):
                            pyautogui.press('enter')
                            time.sleep(1)
                            print(f"⌨️ SOS fallback enter {i+1}/5")
                    
                    print("✅ SOS call sequence completed!")
                    
                    # Final check - ensure Phone Link window remains visible
                    if WIN32_AVAILABLE and phone_hwnd != 0:
                        try:
                            print("🔍 Final SOS window visibility check...")
                            win32gui.ShowWindow(phone_hwnd, 5)  # SW_SHOW - ensure visible
                            win32gui.SetForegroundWindow(phone_hwnd)
                            print("✅ Phone Link window kept visible after SOS")
                        except Exception as final_sos_error:
                            print(f"⚠️ Final SOS visibility check failed: {final_sos_error}")
                    
                else:
                    print("📞 Phone dialer opened - pyautogui not available, manual call required")
                
            except Exception as phone_error:
                print(f"⚠️ Phone dialer failed: {phone_error}")
                print(f"📞 EMERGENCY: Manually call {clean_number} immediately!")
            
            return True
            
        except Exception as e:
            print(f"❌ Emergency call failed: {e}")
            print(f"🚨 MANUAL ACTION: Call {phone_number} immediately!")
            
            # Re-enable keyboard processing on error
            self.phone_call_active = False
            self.keyboard_processing_enabled = True
            # Restore system to the state it was in before the phone call
            if hasattr(self, 'was_active_before_phone_call'):
                self.is_system_active = self.was_active_before_phone_call
                state = "ACTIVE" if self.is_system_active else "SLEEP"
                print(f"✅ System restored to {state} state after call error")
            else:
                # Fallback: assume user was active since they attempted a call
                self.is_system_active = True
                print("✅ System restored to ACTIVE state after call error")
            
            # Copy number to clipboard as last resort
            if PYPERCLIP_AVAILABLE:
                try:
                    pyperclip.copy(f"EMERGENCY: CALL {clean_number}")
                    print(f"📋 Emergency number copied to clipboard: {clean_number}")
                except:
                    print(f"📞 Manual dial required: {clean_number}")
            else:
                print(f"📞 Manual dial required: {clean_number}")
            
            return False

    def make_direct_emergency_call(self, phone_number):
        """Make a direct emergency phone call without WhatsApp message"""
        
        # Clean and format the phone number
        clean_number = phone_number.replace(" ", "").replace("-", "")
        print(f"📞 DIRECT EMERGENCY CALL: {phone_number}")
        print(f"📱 Formatted number: {clean_number}")
        
        try:
            # Direct phone call using tel: protocol
            print("📞 Making direct emergency phone call...")
            
            # PAUSE OptiBlink keyboard monitoring during phone call
            # Save current system state to restore later
            self.was_active_before_phone_call = self.is_system_active
            self.phone_call_active = True
            self.phone_call_start_time = time.time()
            self.keyboard_processing_enabled = False  # Completely disable keyboard processing
            print("🛑 OptiBlink keyboard monitoring COMPLETELY DISABLED for phone call")
            
            # Use generic tel: protocol (works with default phone handler)
            print(f"📱 Dialing {clean_number} using system dialer...")
            subprocess.run(['start', '', f'tel:{clean_number}'], shell=True, check=False)
            
            # Wait for dialer to load completely
            print("⏳ Waiting for dialer to load...")
            time.sleep(3)
            
            if PYAUTOGUI_AVAILABLE:
                # Focus specifically on Phone Link and call immediately
                print("🔄 Focusing on Phone Link window...")
                if WIN32_AVAILABLE:
                    try:
                        # Find Phone Link window specifically
                        def find_phone_link():
                            def enum_callback(hwnd, windows):
                                if win32gui.IsWindowVisible(hwnd):
                                    title = win32gui.GetWindowText(hwnd).lower()
                                    if ('phone' in title and 'link' in title) or 'your phone' in title:
                                        windows.append((hwnd, win32gui.GetWindowText(hwnd)))
                                return True
                            
                            windows = []
                            win32gui.EnumWindows(enum_callback, windows)
                            return windows[0] if windows else (0, "")
                        
                        phone_hwnd, phone_title = find_phone_link()
                        if phone_hwnd != 0:
                            print(f"📱 Found Phone Link: {phone_title}")
                            
                            # CRITICAL: Stop OptiBlink keyboard interference
                            print("🛑 Temporarily stopping OptiBlink keyboard monitoring...")
                            time.sleep(1)  # Brief pause for message to be seen
                            
                            # Focus Phone Link window properly WITHOUT aggressive window manipulation
                            print("🎯 Focusing Phone Link gently...")
                            win32gui.SetForegroundWindow(phone_hwnd)
                            time.sleep(2)  # Wait for focus
                            print("✅ Phone Link focused and ready!")
                            
                            # Use precise navigation to avoid screen casting button
                            # SOLUTION: Use Tab navigation + Space key instead of direct Enter presses
                            # OPTIMIZED: Single Tab + Space works on first try, no loops needed
                            print("🎯 Navigating to call button (avoiding screen cast)...")
                            
                            # Navigate to call button - simple and effective
                            pyautogui.hotkey('ctrl', 'home')  # Go to beginning
                            time.sleep(0.5)
                            
                            # Single tab to call button (since it works on first try)
                            print("� Navigating to call button...")
                            pyautogui.press('tab')
                            time.sleep(0.3)
                            
                            # Activate call button with Enter key (Space only dials, Enter calls)
                            print("📞 Placing call...")
                            pyautogui.press('enter')
                            time.sleep(1)
                            print("✅ Call placed successfully!")
                            
                            print("🔄 OptiBlink keyboard monitoring will resume automatically")
                            
                            print("✅ Emergency call attempts completed!")
                            
                            # Final check - ensure Phone Link window remains visible
                            try:
                                print("🔍 Final window visibility check...")
                                win32gui.ShowWindow(phone_hwnd, 5)  # SW_SHOW - ensure visible
                                win32gui.SetForegroundWindow(phone_hwnd)
                                print("✅ Phone Link window kept visible")
                            except Exception as final_check_error:
                                print(f"⚠️ Final visibility check failed: {final_check_error}")
                            
                            return True
                        else:
                            print("⚠️ Phone Link not found, call may need manual confirmation")
                    except Exception as call_error:
                        print(f"⚠️ Direct call automation failed: {call_error}")
                        print("📞 Phone dialer opened - manual call confirmation may be required")
                        # Re-enable keyboard processing on error
                        self.phone_call_active = False
                        self.keyboard_processing_enabled = True
                        # Restore system to the state it was in before the phone call
                        if hasattr(self, 'was_active_before_phone_call'):
                            self.is_system_active = self.was_active_before_phone_call
                            state = "ACTIVE" if self.is_system_active else "SLEEP"
                            print(f"✅ System restored to {state} state after call error")
                        else:
                            # Fallback: assume user was active since they attempted a call
                            self.is_system_active = True
                            print("✅ System restored to ACTIVE state after call error")
            else:
                print("📞 Phone dialer opened - pyautogui not available, manual confirmation required")
            
            return True
            
        except Exception as e:
            print(f"❌ Emergency call failed: {e}")
            # Re-enable keyboard processing on error
            self.phone_call_active = False
            self.keyboard_processing_enabled = True
            # Restore system to the state it was in before the phone call
            if hasattr(self, 'was_active_before_phone_call'):
                self.is_system_active = self.was_active_before_phone_call
                state = "ACTIVE" if self.is_system_active else "SLEEP"
                print(f"✅ System restored to {state} state after call error")
            else:
                # Fallback: assume user was active since they attempted a call
                self.is_system_active = True
                print("✅ System restored to ACTIVE state after call error")
            
            # Copy number to clipboard as fallback
            if PYPERCLIP_AVAILABLE:
                try:
                    pyperclip.copy(f"EMERGENCY CALL: {clean_number}")
                    print(f"📋 Emergency number copied to clipboard: {clean_number}")
                except:
                    print(f"📞 Manual dial required: {clean_number}")
            else:
                print(f"📞 Manual dial required: {clean_number}")
            
            return False

    def handle_emergency_sos(self):
        """Simplified Emergency SOS handler - ALWAYS works, no restrictions"""
        try:
            print("\n🚨 *** EMERGENCY SOS ACTIVATED ***")
            print("🚨 *** BYPASSING ALL SYSTEM RESTRICTIONS ***")
            
            # Clean phone number
            clean_number = DEFAULT_EMERGENCY_CONTACT.replace(" ", "").replace("-", "")
            print(f"📞 Emergency Contact: {clean_number}")
            
            # Make window fully visible during emergency
            self.set_window_transparency(WINDOW_NAME, 1.0)  # Full opacity
            print("🔍 Window made fully visible for emergency")
            
            # STEP 1: Send WhatsApp message - TRY APP FIRST, then Web fallback
            print("📲 Sending WhatsApp SOS message...")
            emergency_message = "🚨 EMERGENCY! I need immediate help. This is an automated SOS from OptiBlink. Please call me urgently!"
            whatsapp_success = False
            
            try:
                # METHOD 1: Try WhatsApp APP first
                print("📱 Trying WhatsApp APP first...")
                whatsapp_app_url = f"whatsapp://send?phone={clean_number}&text={emergency_message.replace(' ', '%20').replace('!', '%21')}"
                print(f"🔗 WhatsApp App URL: {whatsapp_app_url}")
                
                webbrowser.open(whatsapp_app_url)
                time.sleep(4)  # Wait for app to load
                
                # Check if WhatsApp app actually opened
                app_opened = False
                if WIN32_AVAILABLE:
                    try:
                        def find_whatsapp_app():
                            def enum_callback(hwnd, windows):
                                if win32gui.IsWindowVisible(hwnd):
                                    title = win32gui.GetWindowText(hwnd).lower()
                                    # Look for WhatsApp app specifically (not web)
                                    if 'whatsapp' in title and 'web' not in title and 'browser' not in title:
                                        windows.append((hwnd, win32gui.GetWindowText(hwnd)))
                                return True
                            
                            windows = []
                            win32gui.EnumWindows(enum_callback, windows)
                            return windows[0] if windows else (0, "")
                        
                        whatsapp_hwnd, whatsapp_title = find_whatsapp_app()
                        if whatsapp_hwnd != 0:
                            print(f"✅ WhatsApp APP found: {whatsapp_title}")
                            win32gui.SetForegroundWindow(whatsapp_hwnd)
                            time.sleep(2)
                            app_opened = True
                            whatsapp_success = True
                        else:
                            print("⚠️ WhatsApp APP window not found")
                            
                    except Exception as find_error:
                        print(f"⚠️ Could not detect WhatsApp app: {find_error}")
                
                if app_opened and PYAUTOGUI_AVAILABLE:
                    # Try to send message in WhatsApp app
                    print("📤 Sending message via WhatsApp APP...")
                    time.sleep(1)
                    pyautogui.press('enter')  # Send message
                    time.sleep(0.5)
                    pyautogui.press('enter')  # Extra enter attempt
                    print("✅ WhatsApp APP message sent!")
                    
            except Exception as app_error:
                print(f"⚠️ WhatsApp APP failed: {app_error}")
            
            # METHOD 2: Fallback to WhatsApp Web ONLY if app failed
            if not whatsapp_success:
                try:
                    print("🌐 Falling back to WhatsApp Web...")
                    web_number = clean_number.replace('+', '')
                    whatsapp_web_url = f"https://web.whatsapp.com/send?phone={web_number}&text={emergency_message.replace(' ', '%20').replace('!', '%21')}"
                    print(f"🌐 WhatsApp Web URL: {whatsapp_web_url}")
                    webbrowser.open(whatsapp_web_url)
                    time.sleep(4)  # Wait for web to load
                    
                    # Try to send message automatically
                    if PYAUTOGUI_AVAILABLE:
                        time.sleep(2)  # Extra time for page load
                        pyautogui.press('enter')  # Send message
                        print("✅ WhatsApp Web message sent!")
                    else:
                        print("🌐 WhatsApp Web opened - manual send required")
                        
                except Exception as web_error:
                    print(f"⚠️ WhatsApp Web also failed: {web_error}")
            
            # STEP 2: Make phone call - AVOID CAST SCREEN BUTTON
            print("📞 Making emergency call...")
            try:
                # Use tel: protocol to open phone dialer
                subprocess.run(['start', '', f'tel:{clean_number}'], shell=True, check=False)
                print(f"📱 Dialing {clean_number}...")
                time.sleep(3)  # Wait longer for dialer to fully load
                
                # RELIABLE PHONE CALL using pywinauto (directly targets call button)
                if PYWINAUTO_AVAILABLE:
                    print("🎯 Using RELIABLE pywinauto method to find actual call button...")
                    try:
                        # Connect to Phone Link by process
                        app = Application(backend="uia").connect(path="PhoneExperienceHost.exe")
                        print("✅ Connected to Phone Link process")
                        
                        # Get the main window
                        win = app.top_window()
                        win.set_focus()
                        print("✅ Phone Link window focused")
                        
                        # Find the Call button via AutomationId (most reliable method)
                        call_btn = win.child_window(auto_id="ButtonCall", control_type="Button")
                        
                        if call_btn.exists():
                            call_btn.click_input()
                            print("✅ CALL BUTTON CLICKED SUCCESSFULLY!")
                            print("📞 Emergency call should be connecting...")
                        else:
                            print("⚠️ Call button not found by AutomationId, trying alternatives...")
                            # Fallback to keyboard method
                            if PYAUTOGUI_AVAILABLE:
                                pyautogui.press('enter')
                        
                    except Exception as pywin_error:
                        print(f"⚠️ pywinauto method failed: {pywin_error}")
                        print("🔄 Falling back to keyboard method...")
                        if PYAUTOGUI_AVAILABLE:
                            pyautogui.press('enter')
                        else:
                            print("📞 Manual call confirmation required!")
                            
                elif PYAUTOGUI_AVAILABLE:
                    print("⚠️ pywinauto not available, using basic keyboard method...")
                    pyautogui.press('enter')
                    print("📞 Basic call attempt made")
                    print("🎯 Using SMART navigation to avoid cast screen button...")
                    
                    # Method 1: Use specific keyboard shortcuts for Phone Link
                    print("📞 Trying Ctrl+Enter (Phone Link call shortcut)")
                    pyautogui.hotkey('ctrl', 'enter')
                    time.sleep(1)
                    
                    # Method 2: Use Tab to navigate AWAY from cast screen, then to call button
                    print("⌨️ Using Tab navigation to find call button (avoiding cast screen)")
                    # Reset focus to beginning of dialog
                    pyautogui.hotkey('ctrl', 'home')
                    time.sleep(0.5)
                    
                    # Tab navigation - usually call button is 2nd or 3rd tab stop
                    for i in range(5):
                        pyautogui.press('tab')
                        time.sleep(0.3)
                        # Try Space key (safer than Enter for buttons)
                        pyautogui.press('space')
                        time.sleep(0.5)
                        print(f"📞 Tab attempt {i+1}: Pressed Space at tab position")
                    
                    # Method 3: Direct Alt+C (common call shortcut)
                    print("📞 Trying Alt+C (common call shortcut)")
                    pyautogui.hotkey('alt', 'c')
                    time.sleep(1)
                    
                    # Method 4: Function key F5 (sometimes used for call)
                    print("📞 Trying F5 key")
                    pyautogui.press('f5')
                    time.sleep(1)
                    
                    # Method 5: Look for Phone Link window and use mouse click on call button area
                    if WIN32_AVAILABLE:
                        try:
                            print("🖱️ Trying mouse click approach...")
                            def find_phone_window():
                                def enum_callback(hwnd, windows):
                                    if win32gui.IsWindowVisible(hwnd):
                                        title = win32gui.GetWindowText(hwnd).lower()
                                        if ('phone' in title and 'link' in title) or 'your phone' in title or 'phone' in title:
                                            windows.append((hwnd, win32gui.GetWindowText(hwnd)))
                                    return True
                                
                                windows = []
                                win32gui.EnumWindows(enum_callback, windows)
                                return windows[0] if windows else (0, "")
                            
                            phone_hwnd, phone_title = find_phone_window()
                            if phone_hwnd != 0:
                                print(f"� Found phone window: {phone_title}")
                                # Get window position and size
                                rect = win32gui.GetWindowRect(phone_hwnd)
                                window_x, window_y, window_right, window_bottom = rect
                                window_width = window_right - window_x
                                window_height = window_bottom - window_y
                                
                                # Focus the window
                                win32gui.SetForegroundWindow(phone_hwnd)
                                time.sleep(1)
                                
                                # Click in the bottom right area where call button usually is
                                # Avoid top area where cast screen button is
                                call_x = window_x + int(window_width * 0.75)  # 75% from left
                                call_y = window_y + int(window_height * 0.85)  # 85% from top (bottom area)
                                
                                print(f"🖱️ Clicking call area at ({call_x}, {call_y})")
                                pyautogui.click(call_x, call_y)
                                time.sleep(0.5)
                                
                        except Exception as mouse_error:
                            print(f"⚠️ Mouse click method failed: {mouse_error}")
                    
                    print("✅ EMERGENCY CALL ATTEMPTS COMPLETED!")
                    print("📞 Multiple methods tried to avoid cast screen button")
                else:
                    print("📞 Phone dialer opened - pyautogui not available, MANUAL CALL REQUIRED!")
                    
            except Exception as call_error:
                print(f"❌ Phone call failed: {call_error}")
                print(f"🚨 CRITICAL: MANUALLY DIAL {clean_number} NOW!")
                
            # Always speak SOS message
            print("🔊 Speaking SOS message...")
            self.speak(f"Emergency SOS activated. Calling {clean_number}")
            
            # Record the emergency
            self.message_history.append(f"🚨 EMERGENCY SOS - {clean_number}")
            print("📝 Emergency logged")
            
            # Clear buffers
            self.word_buffer = ""
            self.morse_char_buffer = ""
            self.last_sent_len = 0
            
            print("✅ Emergency SOS procedure completed!")
            return True
            
        except Exception as e:
            print(f"❌ CRITICAL: Emergency SOS failed: {e}")
            print(f"📞 MANUAL ACTION REQUIRED: CALL {DEFAULT_EMERGENCY_CONTACT} IMMEDIATELY!")
            
            # Copy number to clipboard as last resort
            if PYPERCLIP_AVAILABLE:
                try:
                    pyperclip.copy(f"EMERGENCY CALL {clean_number}")
                    print(f"📋 Emergency number copied to clipboard")
                except:
                    pass
                    
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
        
        # Key dimensions for better fit with proper text accommodation
        # Calculate available space more accurately
        available_width = width - 40  # Leave more margin (20px each side)
        
        # More realistic calculation accounting for actual key widths needed
        # Estimate total width needed: TTS (smaller now) + Emergency SOS (large) + other keys
        estimated_total_units = 1.8 + 9.0  # Reduced estimate since TTS is now much shorter
        key_spacing = 3  # Slightly more space between keys
        total_spacing = key_spacing * 10  # 10 gaps between 11 keys in longest row
        
        base_key_width = int((available_width - total_spacing) / estimated_total_units)
        # Ensure minimum key width for readability
        base_key_width = max(base_key_width, 40)  # Reduced minimum since TTS is smaller
        
        key_height = (height - 30) // 4 - 4   # More space for text
        margin_x = 20  # Increased left margin
        margin_y = 5   # Slightly more top margin
        key_gap = 3    # Increased gap between keys
        
        # Debug info (will print once)
        if not hasattr(self, 'keyboard_debug_printed'):
            print(f"Keyboard: width={width}, available={available_width}, base_key={base_key_width}")
            self.keyboard_debug_printed = True
        
        # Enhanced colors for better visibility and accessibility
        if (self.sos_cooldown_timer > 0 and 
            time.time() - self.sos_cooldown_timer < self.sos_cooldown_duration):
            # SOS cooldown state - high contrast yellow/orange theme
            normal_color = (40, 60, 100)     # Dark blue-gray key (cooldown)
            highlight_color = (0, 165, 255)  # Bright orange highlight (cooldown)
            text_color = (255, 255, 255)     # Pure white text (cooldown)
            highlight_text_color = (0, 0, 0) # Black text on highlighted key
        elif self.is_system_active:
            # Active state - high contrast blue/green theme
            normal_color = (60, 60, 60)      # Medium gray key (active)
            highlight_color = (0, 255, 128)  # Bright green highlight (active)
            text_color = (255, 255, 255)     # Pure white text (active)
            highlight_text_color = (0, 0, 0) # Black text on highlighted key
        else:
            # Sleep state - high contrast but dimmed
            normal_color = (30, 30, 30)      # Very dark gray key (sleep)
            highlight_color = (80, 80, 80)   # Medium gray highlight (sleep)
            text_color = (180, 180, 180)     # Light gray text (sleep) - improved from dim
            highlight_text_color = (255, 255, 255) # White highlight text for contrast
        
        # Define each row separately - matching morse_keyboard.jpg reference
        rows = [
            # Row 0: TTS and Numbers
            [('-.-.-', 'TTS'), ('.----', '1'), ('..---', '2'), ('...--', '3'), ('....-', '4'), 
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
                # Calculate key width based on content and ensure text fits
                # Base calculation for text width to ensure proper fit
                font = cv2.FONT_HERSHEY_SIMPLEX
                if len(display_text) > 8:
                    test_font_scale = 0.4
                elif len(display_text) > 4:
                    test_font_scale = 0.45
                else:
                    test_font_scale = 0.55
                
                # Get actual text dimensions for proper sizing
                (text_width, text_height), _ = cv2.getTextSize(display_text, font, test_font_scale, 2)
                min_key_width = text_width + 16  # Add 16px padding (8px each side)
                
                # Set key widths with proper text accommodation
                if display_text == 'TTS':
                    key_width = max(int(base_key_width * 1.1), min_key_width)  # Smaller multiplier for TTS
                elif display_text == 'Emergency SOS':
                    key_width = max(int(base_key_width * 1.8), min_key_width)  # Large multiplier for Emergency SOS
                elif display_text == 'Backspace':
                    key_width = max(int(base_key_width * 1.3), min_key_width)  # Increased for "Backspace"
                elif display_text == 'Space':
                    key_width = max(int(base_key_width * 1.4), min_key_width)  # Space key
                elif display_text == 'CapsLk':
                    key_width = max(int(base_key_width * 1.2), min_key_width)  # CapsLock
                elif display_text == 'Enter':
                    key_width = max(int(base_key_width * 1.2), min_key_width)  # Enter key
                elif display_text == 'Clear':
                    key_width = max(int(base_key_width * 1.1), min_key_width)  # Clear key
                else:
                    key_width = max(base_key_width, min_key_width)  # Single letters/numbers
                
                # Calculate key position
                key_y = margin_y + row_idx * (key_height + margin_y)
                
                # Advanced bounds checking to prevent text overflow
                available_width_remaining = width - current_x - margin_x
                if key_width > available_width_remaining:
                    # Scale down this key but maintain minimum readability
                    key_width = max(available_width_remaining, 35)  # Minimum 35px width
                    # If still too small, reduce font sizes for this key
                    if key_width < min_key_width:
                        # This key will need smaller font - handled in text drawing section
                        key_width = available_width_remaining
                
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
                
                # Enhanced key border for better visibility
                border_thickness = 2  # Thicker border for better contrast
                if is_highlighted:
                    border_color = (255, 255, 255)  # White border for highlighted keys
                else:
                    border_color = (180, 180, 180)  # Brighter gray for better visibility
                cv2.rectangle(keyboard_img, (current_x, key_y), (current_x + key_width, key_y + key_height), border_color, border_thickness)
                
                # Draw key text with improved contrast and sizing for better visibility
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                # Dynamic text thickness - thinner for longer text to improve readability
                if len(display_text) > 8:  # Very long text like "Emergency SOS"
                    text_thickness = 1  # Thin for better readability
                elif len(display_text) > 4:  # Medium text like "Backspace"
                    text_thickness = 1  # Thin for better readability  
                else:  # Short text like single letters
                    text_thickness = 2  # Thick for visibility
                
                # Dynamic font scaling to ensure text fits within key boundaries
                max_attempts = 5
                font_scale = 0.55 if len(display_text) <= 4 else (0.45 if len(display_text) <= 8 else 0.4)
                
                # Ensure text fits within key width with padding
                for attempt in range(max_attempts):
                    (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, text_thickness)
                    if text_w <= key_width - 8:  # 4px padding on each side
                        break
                    font_scale *= 0.9  # Reduce font size by 10%
                    if font_scale < 0.25:  # Minimum readable size
                        font_scale = 0.25
                        break
                
                text_col = highlight_text_color if is_highlighted else text_color
                
                # Center text within key boundaries with bounds checking
                text_x = max(current_x + 4, current_x + (key_width - text_w) // 2)  # Ensure 4px left margin
                text_x = min(text_x, current_x + key_width - text_w - 4)  # Ensure 4px right margin
                text_y = key_y + (key_height + text_h) // 2 - 10  # Adjusted for morse text below
                
                # Enhanced contrast outline system for better visibility
                if not is_highlighted and text_col == (255, 255, 255):  # White text gets black outline
                    outline_color = (0, 0, 0)
                    outline_thickness = 1  # Always use thickness 1 for outlines regardless of text thickness
                    
                    # Enhanced outline for longer text that needs better visibility
                    if len(display_text) > 4:  # For longer text like "Backspace", "Emergency SOS"
                        # Draw a more comprehensive outline for better readability
                        offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
                        for dx, dy in offsets:
                            cv2.putText(keyboard_img, display_text, (text_x + dx, text_y + dy), 
                                       font, font_scale, outline_color, outline_thickness)
                    else:
                        # Standard 4-direction outline for shorter text
                        cv2.putText(keyboard_img, display_text, (text_x - 1, text_y), font, font_scale, outline_color, outline_thickness)
                        cv2.putText(keyboard_img, display_text, (text_x + 1, text_y), font, font_scale, outline_color, outline_thickness)
                        cv2.putText(keyboard_img, display_text, (text_x, text_y - 1), font, font_scale, outline_color, outline_thickness)
                        cv2.putText(keyboard_img, display_text, (text_x, text_y + 1), font, font_scale, outline_color, outline_thickness)
                
                # Draw main text
                cv2.putText(keyboard_img, display_text, (text_x, text_y), font, font_scale, text_col, text_thickness)
                
                # Draw morse code below with improved visibility and proper bounds checking
                if morse_code and morse_code != display_text and len(morse_code) <= 6:
                    # Dynamic font scaling and thickness for morse code
                    morse_font_scale = 0.35 if len(display_text) > 6 else 0.45
                    # Thinner morse code for better readability, especially on busy keys
                    morse_thickness = 1  # Always use thickness 1 for Morse code for clarity
                    
                    # Ensure morse code fits within key width
                    for attempt in range(3):
                        (morse_w, morse_h), _ = cv2.getTextSize(morse_code, font, morse_font_scale, morse_thickness)
                        if morse_w <= key_width - 6:  # 3px padding on each side
                            break
                        morse_font_scale *= 0.85  # Reduce size
                        if morse_font_scale < 0.25:
                            morse_font_scale = 0.25
                            break
                    
                    # High contrast colors for morse code
                    if is_highlighted:
                        morse_color = (255, 255, 0)  # Bright yellow
                    else:
                        morse_color = (255, 255, 255)  # White
                        outline_color = (0, 0, 0)  # Black outline
                    
                    # Center morse code within key boundaries with bounds checking
                    morse_x = max(current_x + 3, current_x + (key_width - morse_w) // 2)  # 3px left margin
                    morse_x = min(morse_x, current_x + key_width - morse_w - 3)  # 3px right margin
                    morse_y = key_y + key_height - 6  # Space from bottom
                    
                    # Draw black outline for better contrast (only for non-highlighted keys)
                    if not is_highlighted:
                        # 4-direction outline for morse code (cleaner than 8-direction)
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            cv2.putText(keyboard_img, morse_code, (morse_x + dx, morse_y + dy), 
                                       font, morse_font_scale, outline_color, 1)  # Always use thickness 1 for outlines
                    
                    # Draw main morse code text
                    cv2.putText(keyboard_img, morse_code, (morse_x, morse_y), 
                               font, morse_font_scale, morse_color, morse_thickness)
                
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
            
            # Import heavy libraries only when actually needed for TTS
            global pygame, gTTS
            if pygame is None:
                print("TTS: Loading pygame...")
                import pygame as pygame_module
                pygame = pygame_module
            if gTTS is None:
                print("TTS: Loading gTTS...")
                from gtts import gTTS as gTTS_module
                gTTS = gTTS_module
            
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
        # EMERGENCY SOS OVERRIDE - ALWAYS allow SOS to work regardless of any restrictions
        if char == 'Emergency SOS':
            # Jump directly to SOS handling - bypass ALL restrictions
            print("\n🚨 *** EMERGENCY SOS OVERRIDE - BYPASSING ALL RESTRICTIONS ***")
            self.handle_emergency_sos()
            return
            
        # Check if keyboard processing is disabled (during phone calls) - but NOT for SOS
        if not self.keyboard_processing_enabled:
            print("Keyboard processing disabled - skipping operation")
            return
            
        # Skip all keyboard writing during SOS cooldown period to prevent interference - but NOT for SOS
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
        elif char == 'TTS':
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
                self.speak("Clear")
                self.speak("Buffer cleared")

    def check_phone_call_status(self):
        """Check if phone call pause period has ended and re-enable keyboard processing"""
        if self.phone_call_active and (time.time() - self.phone_call_start_time > self.phone_call_pause_duration):
            self.phone_call_active = False
            self.keyboard_processing_enabled = True
            
            # Restore system to the state it was in before the phone call
            if hasattr(self, 'was_active_before_phone_call'):
                self.is_system_active = self.was_active_before_phone_call
                state = "ACTIVE" if self.is_system_active else "SLEEP"
                print(f"✅ System restored to {state} state after phone call")
            else:
                # Fallback: assume user was active since they made a call
                self.is_system_active = True
                print("✅ System restored to ACTIVE state after phone call")
            print("✅ Phone call period ended - OptiBlink keyboard processing re-enabled")

    def is_system_truly_active(self):
        """Check if system is active considering both sleep mode, SOS cooldown, and phone calls"""
        # If keyboard processing is disabled (phone call), system is temporarily inactive for morse code
        # but sleep/wake detection should still work
        if not self.keyboard_processing_enabled:
            return False
            
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
            if char == 'TTS':
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
        # Check if phone call is active and pause OptiBlink processing
        if (self.phone_call_active and 
            time.time() - self.phone_call_start_time < self.phone_call_pause_duration):
            # During phone call - minimal processing, no keyboard capture
            cv2.putText(frame, "PHONE CALL ACTIVE - OptiBlink Paused", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Resuming in {int(self.phone_call_pause_duration - (time.time() - self.phone_call_start_time))}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            return frame
        elif self.phone_call_active:
            # Phone call timeout reached - resume normal operation using the proper restoration function
            self.check_phone_call_status()  # This will handle all restoration properly
        
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
                    # NOTE: This works regardless of current system state - allows waking up from sleep
                    if (self.eyes_closed_start_time > 0 and 
                        current_time - self.eyes_closed_start_time >= self.sleep_wake_threshold):
                        # Toggle system state
                        self.is_system_active = not self.is_system_active
                        status = "ACTIVE" if self.is_system_active else "SLEEP"
                        print(f"\n*** SYSTEM {status} ***")
                        print(f"Sleep/Wake toggle activated after {self.sleep_wake_threshold} seconds")
                        
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

        # Show sleep/wake progress if eyes are closed for more than 50% of threshold
        if self.eyes_closed_start_time > 0:
            elapsed = time.time() - self.eyes_closed_start_time
            # Only show progress after 50% of the required time to avoid showing on every blink
            # This prevents the indicator from appearing during normal blinking
            if elapsed > self.sleep_wake_threshold * 0.5:  # Show after 2.5 seconds (50% of 5 seconds)
                progress = min(elapsed / self.sleep_wake_threshold, 1.0) * 100
                action = "WAKE UP" if not self.is_system_active else "SLEEP"
                cv2.putText(frame, f"Hold to {action}: {progress:.1f}% ({self.sleep_wake_threshold - elapsed:.1f}s left)",
                            (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                current_y += int(line_spacing * 0.7)

        # Show speaking indicator if TTS is active
        if self.is_speaking:
            cv2.putText(frame, f"Speaking: {self.speaking_message}",
                        (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            current_y += int(line_spacing * 0.7)

        cv2.putText(frame, f"Morse: {self.morse_char_buffer}",
                    (10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        current_y += int(line_spacing * 0.7)

        # Only show possible letters if there's something in the morse buffer
        if self.morse_char_buffer:
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

    # Initialize camera first for immediate feedback
    print("📹 Initializing camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    print("✅ Camera ready!")
    
    # Initialize systems with progress feedback
    print("🧠 Loading AutoComplete system...")
    auto = AutoCompleteSystem()
    print("✅ AutoComplete ready!")
    
    print("👁️ Initializing EyeTracker...")  
    eye_tracker = EyeTracker(auto)
    print("✅ EyeTracker ready!")
    
    print("🖥️ Setting up window...")
    window_width = 800
    window_height = 550

    # Move window to top-right corner with some margin
    try:
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
        x_pos = screen_width - window_width - 30  # Increased margin for larger window
        print(f"Screen dimensions: {screen_width}x{screen_height}")
        print(f"Window size: {window_width}x{window_height}")
        print(f"Calculated position: x={x_pos}, y=40")
        print(f"This should place window at top-right corner with 30px margin")
    except Exception as e:
        # Fallback to default position if Windows API is not available
        x_pos = 100
        print(f"Warning: Could not get screen dimensions: {e}. Using default window position.")
    
    y_pos = 40  # Leave space for window controls

    # Function to set OpenCV window always on top
    def set_window_always_on_top(window_name):
        if WIN32_AVAILABLE:
            try:
                hwnd = win32gui.FindWindow(None, window_name)
                if hwnd:
                    # Set as topmost window that stays above all other windows
                    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
                    
                    # Additionally, ensure window is visible and not minimized
                    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
                    return True
            except Exception as e:
                print(f"Warning: Could not set window always on top: {e}")
                return False
        return False
        
    # Function to force window to top-right position and topmost
    def force_window_position_and_topmost(window_name, x, y):
        if WIN32_AVAILABLE:
            try:
                hwnd = win32gui.FindWindow(None, window_name)
                if hwnd:
                    # Force position and make topmost in one call
                    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, x, y, 0, 0,
                                        win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
                    return True
            except Exception as e:
                print(f"Warning: Could not force window position: {e}")
                return False
        return False

    print("Starting main loop...")
    
    # Variables for window positioning
    window_positioned = False
    position_attempts = 0
    max_position_attempts = 10
    
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
                eye_tracker.set_window_transparency(WINDOW_NAME, 0.85)  # Restore to permanent transparency
                eye_tracker.is_transparent = False
                eye_tracker.transparency_timer = 0
                print("Window opacity restored to normal transparency")

            # Check if SOS cooldown has expired and restore system activity
            if (eye_tracker.sos_cooldown_timer > 0 and
                time.time() - eye_tracker.sos_cooldown_timer > eye_tracker.sos_cooldown_duration):
                eye_tracker.is_system_active = eye_tracker.was_active_before_sos
                eye_tracker.sos_cooldown_timer = 0
                status = "ACTIVE" if eye_tracker.is_system_active else "SLEEP"
                print(f"SOS cooldown expired - System restored to {status}")
            
            # Check if phone call pause has expired and restore keyboard processing
            eye_tracker.check_phone_call_status()

            full_display_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)
            full_display_frame[0:video_height, 0:window_width] = processed_video_frame
            full_display_frame[video_height:video_height+keyboard_height, 0:window_width] = keyboard_img

            cv2.imshow(WINDOW_NAME, full_display_frame)
            
            # Position window in top-right corner (only try for first few frames)
            if not window_positioned and position_attempts < max_position_attempts:
                position_attempts += 1
                print(f"Positioning attempt {position_attempts}: trying to move to ({x_pos}, {y_pos})")
                
                # First, try OpenCV positioning (works immediately)
                cv2.moveWindow(WINDOW_NAME, x_pos, y_pos)
                
                # Then, force position and topmost with Windows API
                if force_window_position_and_topmost(WINDOW_NAME, x_pos, y_pos):
                    # Verify the position
                    try:
                        hwnd = win32gui.FindWindow(None, WINDOW_NAME)
                        if hwnd:
                            rect = win32gui.GetWindowRect(hwnd)
                            current_x, current_y = rect[0], rect[1]
                            print(f"Window positioned at ({current_x}, {current_y}), target was ({x_pos}, {y_pos})")
                            
                            if abs(current_x - x_pos) < 50 and abs(current_y - y_pos) < 50:
                                window_positioned = True
                                print("Window successfully positioned in top-right corner!")
                    except Exception as verify_error:
                        print(f"Could not verify position: {verify_error}")
                else:
                    print(f"Attempt {position_attempts}: Windows API positioning failed")
                        
                # If we've tried enough times, stop trying
                if position_attempts >= max_position_attempts:
                    window_positioned = True
                    print("Window positioning attempts completed")
            
            # Continuously enforce always-on-top behavior (every 10 frames to avoid performance impact)
            frame_count = getattr(eye_tracker, '_frame_count', 0) + 1
            eye_tracker._frame_count = frame_count
            
            if frame_count % 10 == 0:  # Every 10 frames (roughly every 0.3 seconds)
                success = set_window_always_on_top(WINDOW_NAME)
                if not success and frame_count % 50 == 0:  # Report failures every 50 frames
                    print("Warning: Could not maintain always-on-top status")
            
            # Additional enforcement every 30 frames - reposition if needed
            if frame_count % 30 == 0 and window_positioned:
                force_window_position_and_topmost(WINDOW_NAME, x_pos, y_pos)
            
            # Set permanent transparency (allow users to see behind the window)
            if not hasattr(eye_tracker, '_permanent_transparency_set'):
                eye_tracker.set_window_transparency(WINDOW_NAME, 0.85)  # 85% opacity permanently
                eye_tracker._permanent_transparency_set = True
                print(f"Window positioned at top-right corner: ({x_pos}, {y_pos})")

            # Robust window close detection
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == -1:
                # If window is closed, waitKey returns -1
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            
            # Check for actual keyboard input (only when window has focus and key is physically pressed)
            # Use GetAsyncKeyState to check for actual physical key presses
            try:
                # Check if Q key is currently being pressed (GetAsyncKeyState returns non-zero if pressed)
                q_pressed = (ctypes.windll.user32.GetAsyncKeyState(ord('Q')) & 0x8000) or (ctypes.windll.user32.GetAsyncKeyState(ord('q')) & 0x8000)
                if q_pressed:
                    if not hasattr(eye_tracker, '_q_pressed') or not eye_tracker._q_pressed:
                        eye_tracker._q_pressed = True
                        print("Q key pressed - exiting program")
                        break
                else:
                    eye_tracker._q_pressed = False
                
                # Check if R key is currently being pressed
                r_pressed = (ctypes.windll.user32.GetAsyncKeyState(ord('R')) & 0x8000) or (ctypes.windll.user32.GetAsyncKeyState(ord('r')) & 0x8000)
                if r_pressed:
                    if not hasattr(eye_tracker, '_r_pressed') or not eye_tracker._r_pressed:
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
        traceback.print_exc()
        input("Press Enter to continue...")