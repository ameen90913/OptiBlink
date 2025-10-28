# Standard library imports
import sys
import os

_BOOT_SAVED_FD = None
try:
    _BOOT_SAVED_FD = os.dup(2)  # stderr only
    nul_path = 'NUL' if os.name == 'nt' else '/dev/null'
    nul_fd = os.open(nul_path, os.O_WRONLY)
    os.dup2(nul_fd, 2)  # Redirect stderr only
    os.close(nul_fd)
except Exception:
    pass

# LEVEL 2: Set ALL possible TensorFlow/ABSL/GLOG environment variables
# These need to be set BEFORE any TensorFlow/MediaPipe imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow C++ logging
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'  # TensorFlow verbose logging
os.environ['TF_CPP_MAX_VLOG_LEVEL'] = '0'  # Maximum verbose level
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages

# ABSL (Abseil) logging - Google's C++ logging library
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'  # Minimum log level
os.environ['ABSL_LOGGING_MIN_LOG_LEVEL'] = '3'  # Alternative name

# Google logging (GLOG)
os.environ['GLOG_minloglevel'] = '3'  # Minimum severity level
os.environ['GLOG_logtostderr'] = '0'  # Don't log to stderr
os.environ['GLOG_stderrthreshold'] = '3'  # Stderr threshold
os.environ['GLOG_v'] = '0'  # Verbosity level

# TensorFlow Lite specific
os.environ['TFLITE_LOG_LEVEL'] = '3'  # TFLite log level
os.environ['TFLITE_SUPPRESS_LOG'] = '1'  # Suppress TFLite logs
os.environ['XNNPACK_VERBOSE'] = '0'  # XNNPACK delegate verbosity

# MediaPipe
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Disable GPU warnings

# TensorFlow module specific
os.environ['TF_CPP_VMODULE'] = 'inference_feedback_manager=0,xnnpack_delegate=0,external_delegate_adaptor=0'

# General TensorFlow flags
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Can reduce some warnings
os.environ['AUTOGRAPH_VERBOSITY'] = '0'  # AutoGraph verbosity

if sys.stdout:
    sys.stdout.reconfigure(encoding='utf-8')
# DO NOT reconfigure stderr yet - keep it suppressed!

# LEVEL 4: Additional environment variables for completeness
os.environ['PYTHONWARNINGS'] = 'ignore'  # Ignore all Python warnings
os.environ['PYTHONIOENCODING'] = 'utf-8'  # IO encoding

# Environment variable suppression for protobuf warnings (must be before imports)
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning:google.protobuf.symbol_database'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Additional suppression
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # PyGame
os.environ['KMP_WARNINGS'] = '0'  # Intel MKL warnings
os.environ['OMP_NUM_THREADS'] = '1'  # OpenMP warnings


# LEVEL 5: Comprehensive warning suppression for cleaner startup (must be before other imports)
import warnings
warnings.filterwarnings("ignore")  # Ignore ALL warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# LEVEL 6: Python logging configuration - set EVERYTHING to CRITICAL
import logging
logging.basicConfig(
    level=logging.CRITICAL,
    format='',
    handlers=[logging.NullHandler()]
)

# Disable ALL known loggers
for logger_name in [
    'tensorflow', 'absl', 'mediapipe', 'google.protobuf',
    'google.protobuf.symbol_database', 'root', 'py.warnings'
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.disabled = True
    logger.propagate = False

# Disable root logger completely
logging.disable(logging.CRITICAL)

import csv
import ctypes
import json
import subprocess
import tempfile
import threading
import time
import traceback
import webbrowser
from collections import deque, defaultdict


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller."""
    try:
        base_path = sys._MEIPASS  # For PyInstaller
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def get_config_path():
    """Return a user-writable path for the app config file."""
    base_dir = os.getenv('APPDATA') or os.getenv('LOCALAPPDATA') or os.path.expanduser('~')
    cfg_dir = os.path.join(base_dir, 'OptiBlink')
    try:
        os.makedirs(cfg_dir, exist_ok=True)
    except Exception:
        pass
    return os.path.join(cfg_dir, 'config.json')

# Global config path constant used by config helpers
CONFIG_PATH = get_config_path()


def _encode_message_for_url(text):
    """Encode message for use in URL query parameters with fallback when urllib is unavailable."""
    try:
        if URLLIB_AVAILABLE:
            return urllib.parse.quote(text)
    except Exception:
        pass
    # Fallback encoding for common special characters
    return (
        text.replace(' ', '%20')
            .replace('!', '%21')
            .replace('\n', '%0A')
            .replace(':', '%3A')
            .replace('?', '%3F')
            .replace('&', '%26')
            .replace('=', '%3D')
    )


def _fallback_copy_and_prompt(clean_number, label="EMERGENCY CALL"):
    """Copy an emergency call prompt to clipboard if available and print manual prompt."""
    try:
        if PYPERCLIP_AVAILABLE:
            try:
                pyperclip.copy(f"{label} {clean_number}")
                print(f"📋 Emergency number copied to clipboard: {clean_number}")
            except Exception:
                print(f"📞 Manual dial required: {clean_number}")
        else:
            print(f"📞 Manual dial required: {clean_number}")
    except Exception:
        print(f"📞 Manual dial required: {clean_number}")


# Native (C-level) stderr redirection to swallow absl/TFLite C++ logs
import contextlib
import io

@contextlib.contextmanager
def _suppress_native_stderr():
    """ULTRA-AGGRESSIVE: Suppress C++ level stderr output from TensorFlow/MediaPipe libraries
    
    This function creates a complete blackhole for stderr at the OS file descriptor level.
    Nothing that writes to stderr (including C++ code) can escape this suppression.
    """
    saved_fd = None
    nul_fd = None
    try:
        # Save current stderr file descriptor
        saved_fd = os.dup(2)
        
        # Open the system's null device
        nul_path = 'NUL' if os.name == 'nt' else '/dev/null'
        nul_fd = os.open(nul_path, os.O_WRONLY)
        
        # Replace stderr (fd 2) with null device
        os.dup2(nul_fd, 2)
        os.close(nul_fd)
        nul_fd = None
        
        # Flush any buffered content before yielding
        try:
            sys.stderr.flush()
        except:
            pass
            
        yield
    finally:
        try:
            # Restore original stderr file descriptor
            if saved_fd is not None:
                # Flush before restoring
                try:
                    sys.stderr.flush()
                except:
                    pass
                os.dup2(saved_fd, 2)
                os.close(saved_fd)
        except Exception:
            pass

@contextlib.contextmanager
def _suppress_all_warnings():
    """NUCLEAR OPTION: Comprehensive multi-layer warning suppression
    
    This combines Python-level and logging-level suppression.
    NOTE: Does NOT nest _suppress_native_stderr() to avoid conflicts with boot-time suppression.
    """
    # Save original stderr and logging state
    original_stderr = sys.stderr
    original_levels = {}
    
    # ALL possible loggers that could emit warnings
    loggers_to_silence = [
        'tensorflow', 'absl', 'mediapipe', 'google.protobuf',
        'google.protobuf.symbol_database', 'google.protobuf.text_format',
        'root', 'py.warnings', 'asyncio'
    ]
    
    for logger_name in loggers_to_silence:
        logger = logging.getLogger(logger_name)
        original_levels[logger_name] = logger.level
        logger.setLevel(logging.CRITICAL)
        logger.disabled = True
    
    try:
        # Redirect Python's stderr to a black hole StringIO
        devnull = io.StringIO()
        sys.stderr = devnull
        
        # Suppress at warnings module level
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings('ignore')
            yield
                
    finally:
        # Restore original stderr
        sys.stderr = original_stderr
        
        # Restore original logging levels
        for logger_name, original_level in original_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(original_level)
            logger.disabled = False


# Location services imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. Location sharing will be disabled.")

try:
    import urllib.parse
    URLLIB_AVAILABLE = True
except ImportError:
    URLLIB_AVAILABLE = False
    print("Warning: urllib not available. Some location features may not work.")

# Try to import winsdk for Windows device location (winrt deprecated)
try:
    import asyncio
    from winsdk.windows.devices.geolocation import Geolocator, PositionStatus # type: ignore
    WINRT_AVAILABLE = True
except ImportError:
    WINRT_AVAILABLE = False
    print("Warning: winsdk not available. Device location will use IP geolocation fallback.")
    print("To enable device-based geolocation, install winsdk with: python -m pip install winsdk")

# CONFIGURATION - Change emergency contact number here
DEFAULT_EMERGENCY_CONTACT = "+91 9999999900"

# WINDOW CONFIGURATION
WINDOW_NAME = "OptiBlink"

# Additional environment variables for cleaner output
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Disable GPU warnings if needed

# Suppress protobuf warnings specifically
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Additional TFLite and XNNPACK delegate warning suppression
os.environ['XNNPACK_VERBOSE'] = '0'
os.environ['TFLITE_SUPPRESS_LOG'] = '1'


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
    import win32api
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
    # Additional protobuf warning suppression right before pywinauto import
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
        from pywinauto import Application
    PYWINAUTO_AVAILABLE = True
except ImportError:
    PYWINAUTO_AVAILABLE = False
    # Suppressed: Emergency phone calling may use fallback methods

# Configuration management
def load_config():
    """Load configuration from the user config path, prompting on first run."""
    default_config = {
        "emergency_contact": "",
        "prefer_whatsapp_web": False,
    }
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # Ensure expected keys exist
            updated = False
            for k, v in default_config.items():
                if k not in config:
                    config[k] = v
                    updated = True
            if updated:
                save_config(config)
            return config
        else:
            # Prompt user and create config
            return load_or_create_config()
    except Exception as e:
        print(f"Error loading config: {e}. Using defaults.")
        return default_config

def save_config(config):
    """Save configuration to the user config path"""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")


def get_current_location():
    """Get current location using Windows device location if available, else IP geolocation services"""
    # Try Windows device location first
    if WINRT_AVAILABLE:
        try:
            async def get_winrt_location():
                locator = Geolocator()
                pos = await locator.get_geoposition_async()
                lat = pos.coordinate.point.position.latitude
                lon = pos.coordinate.point.position.longitude
                accuracy = pos.coordinate.accuracy
                return lat, lon, accuracy

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            lat, lon, accuracy = loop.run_until_complete(get_winrt_location())
            print(f"✅ Device location found: {lat}, {lon} (accuracy: {accuracy}m)")
            location_info = {
                'latitude': lat,
                'longitude': lon,
                'city': 'Device',
                'region': '',
                'country': '',
                'google_maps_url': f"https://maps.google.com/?q={lat},{lon}",
                'address': f"Device location (accuracy: {accuracy}m)"
            }
            return location_info
        except Exception as e:
            print(f"⚠️ Device location failed: {e}. Falling back to IP geolocation.")

    # Fallback to IP geolocation
    try:
        if not REQUESTS_AVAILABLE:
            print("⚠️ Requests module not available for location services")
            return None

        services = [
            "http://ip-api.com/json/",
            "https://ipapi.co/json/",
            "https://ipinfo.io/json"
        ]
        for service_url in services:
            try:
                print(f"🌍 Trying location service: {service_url}")
                response = requests.get(service_url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if service_url.startswith("http://ip-api.com"):
                        if data.get('status') == 'success':
                            lat = data.get('lat')
                            lon = data.get('lon')
                            city = data.get('city', 'Unknown')
                            region = data.get('regionName', 'Unknown')
                            country = data.get('country', 'Unknown')
                            if lat and lon:
                                location_info = {
                                    'latitude': lat,
                                    'longitude': lon,
                                    'city': city,
                                    'region': region,
                                    'country': country,
                                    'google_maps_url': f"https://maps.google.com/?q={lat},{lon}",
                                    'address': f"{city}, {region}, {country} (IP-based)"
                                }
                                print(f"✅ Location found: {location_info['address']}")
                                return location_info
                    elif service_url.startswith("https://ipapi.co"):
                        lat = data.get('latitude')
                        lon = data.get('longitude')
                        city = data.get('city', 'Unknown')
                        region = data.get('region', 'Unknown')
                        country = data.get('country_name', 'Unknown')
                        if lat and lon:
                            location_info = {
                                'latitude': lat,
                                'longitude': lon,
                                'city': city,
                                'region': region,
                                'country': country,
                                'google_maps_url': f"https://maps.google.com/?q={lat},{lon}",
                                'address': f"{city}, {region}, {country} (IP-based)"
                            }
                            print(f"✅ Location found: {location_info['address']}")
                            return location_info
                    elif service_url.startswith("https://ipinfo.io"):
                        loc = data.get('loc')
                        city = data.get('city', 'Unknown')
                        region = data.get('region', 'Unknown')
                        country = data.get('country', 'Unknown')
                        if loc and ',' in loc:
                            lat, lon = loc.split(',')
                            lat, lon = float(lat.strip()), float(lon.strip())
                            location_info = {
                                'latitude': lat,
                                'longitude': lon,
                                'city': city,
                                'region': region,
                                'country': country,
                                'google_maps_url': f"https://maps.google.com/?q={lat},{lon}",
                                'address': f"{city}, {region}, {country} (IP-based)"
                            }
                            print(f"✅ Location found: {location_info['address']}")
                            return location_info
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Location service {service_url} failed: {e}")
                continue
            except Exception as e:
                print(f"⚠️ Error parsing location from {service_url}: {e}")
                continue
        print("⚠️ All location services failed")
        return None
    except Exception as e:
        print(f"❌ Location detection failed: {e}")
        return None

# Load words from CSV or fallback to NLTK - OPTIMIZED FOR FAST STARTUP
def load_words_from_csv(csv_path, column_name="Word"):
    words = []
    try:
        # Quick file check first
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}. Will use NLTK if needed.")
            return words
        
        # Simple direct approach - read as CSV with two columns
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip header row
            count = 0
            for row in reader:
                if count >= 3000:  # Limit for faster startup
                    break
                if len(row) >= 2:  # Ensure there's at least a letter and word
                    word = row[1]  # Second column contains the word
                    if word and word.isalpha() and len(word) > 2:
                        words.append(word.lower())
                count += 1
                
        # Add some default words if none were loaded
        if not words:
            default_words = ["the", "and", "for", "you", "that", "have", "with", "this", "from", "they", 
                            "will", "would", "there", "their", "what", "about", "which", "when", "make", 
                            "like", "time", "just", "know", "take", "people", "into", "year", "your", "good"]
            words = default_words
            
        print(f"Loaded {len(words)} words for suggestions")
    except Exception as e:
        print(f"CSV loading error: {e}. Using default words.")
        # Add default words as fallback
        default_words = ["the", "and", "for", "you", "that", "have", "with", "this", "from", "they"]
        words = default_words
    return words

# Load CSV words quickly - no NLTK loading at startup for speed
# print("EyeTracker: Fast startup - loading essential words only...")
csv_word_list = load_words_from_csv(resource_path("words.csv"), column_name="Word")

# Skip NLTK loading entirely at startup - defer until actually needed
nltk_word_list = []
# print("EyeTracker: Word loading complete - NLTK deferred for speed!")


# Define config path (user-writable)
# Note: CONFIG_PATH is created above using get_config_path()

def load_or_create_config():
    """Load user config, or prompt to create it on first run."""
    default_config = {
        "emergency_contact": "",
        "prefer_whatsapp_web": False
    }

    # Case 1: File doesn't exist — prompt user and create
    if not os.path.exists(CONFIG_PATH):
        print("🔧 No configuration found. Let's set it up!")

        # Detect if running in an interactive console
        interactive = False
        try:
            interactive = bool(sys.stdin and sys.stdin.isatty())
        except Exception:
            interactive = False

        phone = ""
        use_whatsapp = False

        if interactive:
            try:
                phone = input("Enter your emergency contact number (e.g. +91 9876543210): ").strip()
                use_whatsapp = input("Prefer WhatsApp Web for alerts? (y/n): ").strip().lower() == "y"
            except Exception:
                phone = ""
                use_whatsapp = False
        else:
            # Try to prompt the user with a simple GUI dialog when no stdin is available
            try:
                import tkinter as _tk
                from tkinter import simpledialog as _sd, messagebox as _mb
                root = _tk.Tk()
                root.withdraw()
                new_phone = _sd.askstring("Emergency Contact", "Enter emergency contact number (with country code, e.g., +919876543210):", parent=root)
                if new_phone and len(new_phone.strip()) >= 10:
                    phone = new_phone.strip()
                use_whatsapp = _mb.askyesno("WhatsApp Preference", "Prefer WhatsApp Web for alerts?", parent=root)
                root.destroy()
            except Exception:
                phone = ""
                use_whatsapp = False

        default_config["emergency_contact"] = phone
        default_config["prefer_whatsapp_web"] = use_whatsapp

        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=4)
            print("✅ Configuration file created successfully!\n")
        except Exception as e:
            print(f"⚠️ Could not write configuration file: {e}")

        return default_config

    # Case 2: File exists — load it
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # Validate fields (in case user deleted one)
    updated = False
    for key in ["emergency_contact", "prefer_whatsapp_web"]:
        if key not in config_data:
            config_data[key] = default_config[key]
            updated = True

    if updated:
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print(f"⚠️ Could not update configuration file: {e}")

    return config_data

def prompt_update_emergency_contact(config: dict) -> dict:
    """Ask the user at startup whether to update the emergency contact.

    - If the user answers 'y', prompt for a new phone number and WhatsApp Web preference.
    - Save updates to the user config path.
    - Always returns the (possibly updated) config dict.
    """
    try:
        # Detect interactive console
        interactive = False
        try:
            interactive = bool(sys.stdin and sys.stdin.isatty())
        except Exception:
            interactive = False

        # Show current value for context
        current = (config or {}).get("emergency_contact", "").strip() or "<not set>"

        if interactive:
            try:
                ans = input(f"Would you like to update the emergency contact? (current: {current}) (y/n): ").strip().lower()
                if ans == 'y':
                    new_phone = input("Enter new emergency contact number (e.g. +91 9876543210): ").strip()
                    if new_phone:
                        config["emergency_contact"] = new_phone
                    # Optional preference update
                    pref = input("Prefer WhatsApp Web for alerts? (y/n, Enter to skip): ").strip().lower()
                    if pref in ('y', 'n'):
                        config["prefer_whatsapp_web"] = (pref == 'y')
                    save_config(config)
                    print("✅ Emergency contact updated.\n")
                else:
                    print("ℹ️ Keeping existing emergency contact.\n")
            except Exception:
                # If stdin fails mid-prompt, just return existing config
                pass
        else:
            # Non-interactive: try GUI dialogs; if unavailable, skip prompting
            try:
                import tkinter as _tk
                from tkinter import simpledialog as _sd, messagebox as _mb
                root = _tk.Tk()
                root.withdraw()
                update = _mb.askyesno("Update Emergency Contact", f"Current: {current}\n\nDo you want to update the emergency contact?", parent=root)
                if update:
                    new_phone = _sd.askstring("Update Emergency Contact", "Enter new emergency contact number (e.g. +91 9876543210):", parent=root)
                    if new_phone:
                        config["emergency_contact"] = new_phone.strip()
                    pref = _mb.askyesno("WhatsApp Preference", "Prefer WhatsApp Web for alerts?", parent=root)
                    config["prefer_whatsapp_web"] = pref
                    save_config(config)
                    print("✅ Emergency contact updated via GUI.\n")
                else:
                    print("ℹ️ Keeping existing emergency contact.\n")
                root.destroy()
            except Exception:
                # No GUI available either - skip prompting silently
                pass
    except Exception:
        # If prompting fails for any reason, just continue with existing config
        pass
    return config

# Configuration will be loaded and managed by the main function
# The prompt_phone_number_update() function handles user interaction


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class AutoCompleteSystem:
    def __init__(self):
        # print("AutoComplete: Building word trees...")
        self.csv_root = TrieNode()
        self.nltk_root = TrieNode()
        self.frequency = defaultdict(int)
        self.nltk_loaded = False  # Track if NLTK has been loaded
        
        # Load CSV words first (fast)
        for word in csv_word_list:
            self.insert(word, root=self.csv_root)
        
        # Skip NLTK loading at startup for speed
        # print(f"AutoComplete: Loaded {len(csv_word_list)} CSV words from database")
        
        self.load_usage_data()
        # print("AutoComplete: Ready!")
    
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
            
        # If still no results, provide fallback suggestions based on first letter
        if not results and prefix:
            first_letter = prefix[0].lower()
            fallback_suggestions = {
                'a': ['and', 'all', 'about'],
                'b': ['but', 'because', 'before'],
                'c': ['can', 'come', 'could'],
                'd': ['do', 'day', 'down'],
                'e': ['even', 'every', 'ever'],
                'f': ['for', 'from', 'find'],
                'g': ['get', 'good', 'give'],
                'h': ['have', 'how', 'here'],
                'i': ['in', 'it', 'if'],
                'j': ['just', 'job', 'join'],
                'k': ['know', 'keep', 'key'],
                'l': ['like', 'look', 'life'],
                'm': ['make', 'more', 'most'],
                'n': ['not', 'new', 'now'],
                'o': ['one', 'other', 'out'],
                'p': ['people', 'part', 'place'],
                'q': ['question', 'quick', 'quite'],
                'r': ['right', 'really', 'run'],
                's': ['say', 'see', 'some'],
                't': ['the', 'that', 'time'],
                'u': ['up', 'use', 'us'],
                'v': ['very', 'view', 'value'],
                'w': ['with', 'would', 'when'],
                'x': ['x-ray', 'xmas', 'xenon'],
                'y': ['you', 'year', 'yes'],
                'z': ['zero', 'zone', 'zoom']
            }
            if first_letter in fallback_suggestions:
                results = [word for word in fallback_suggestions[first_letter] if word.startswith(prefix)]
            
            # If still no results, just return some common words
            if not results:
                results = ['the', 'and', 'for'][:3]
                
        return results[:3]

    def record_usage(self, word):
        # Only record single words (no spaces, reasonable length)
        word = word.strip().lower()
        if word and ' ' not in word and len(word) <= 20 and word.isalpha():
            self.frequency[word] += 1
            try:
                with open(resource_path("usage_data.txt"), "a") as f:
                    f.write(f"{word},{int(time.time())}\n")
            except Exception as e:
                # If we can't write to bundled resources, try current directory
                try:
                    with open("usage_data.txt", "a") as f:
                        f.write(f"{word},{int(time.time())}\n")
                except Exception:
                    pass  # Silently fail if we can't save usage data

    def load_usage_data(self, filename="usage_data.txt"):
        # Try bundled resource first, then current directory
        usage_file_paths = [resource_path(filename), filename]
        
        for file_path in usage_file_paths:
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        for line in f:
                            if "," in line:
                                word, _ = line.strip().split(",", 1)  # Split only on first comma
                                word = word.strip().lower()
                                # Only load single words (no spaces, reasonable length, alphabetic)
                                if word and ' ' not in word and len(word) <= 20 and word.isalpha():
                                    self.frequency[word] += 1
                                    self.insert(word, root=self.csv_root)
                    break  # Successfully loaded, don't try other paths
                except Exception:
                    continue  # Try next path

class EyeTracker:
    def __init__(self, auto):
        # print("EyeTracker: Fast initialization starting...")
        self.auto = auto
        
        # Defer MediaPipe import and initialization for faster startup
        global mp
        if mp is None:
            # print("EyeTracker: Importing MediaPipe...")
            
            # Import MediaPipe with comprehensive warning suppression
            try:
                with _suppress_all_warnings():
                    import mediapipe as mp_module
            except Exception:
                # Fallback without suppression if needed
                import mediapipe as mp_module
            mp = mp_module
        
        # print("EyeTracker: Setting up MediaPipe face mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Create FaceMesh with fastest possible settings
        # print("EyeTracker: Creating optimized FaceMesh...")
        
        # Create FaceMesh with comprehensive warning suppression (both Python and C++ level)
        with _suppress_native_stderr():
            with _suppress_all_warnings():
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=False,  # Disable for speed
                    min_detection_confidence=0.3,  # Lower threshold for speed  
                    min_tracking_confidence=0.3    # Lower threshold for speed
                )
        
        # Initialize all other attributes quickly
        # print("EyeTracker: Setting up core systems...")
        self._initialize_morse_and_detection()
        # print("EyeTracker: Initialization complete!")

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
        # Cache the emergency contact number without spaces for quick use
        self.emergency_contact = self.config.get("emergency_contact", "").replace(" ", "")
        
        # Call icon hover state
        self._icon_hovering = False
        self._icon_bbox_attr = None

    def _start_phone_call(self, clean_number):
        """Centralized phone dialing via tel: and Phone Link automation (pywinauto)."""
        print("📞 Making emergency call...")
        try:
            os.system(f'start tel:{clean_number}')
            print(f"📱 Dialing {clean_number}...")
            time.sleep(5)

            if PYWINAUTO_AVAILABLE:
                print("🎯 Using pywinauto to trigger call button...")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        warnings.filterwarnings("ignore", message=".*SymbolDatabase.GetPrototype.*")
                        app = Application(backend="uia").connect(path="PhoneExperienceHost.exe")

                    win = app.top_window()
                    win.set_focus()
                    call_btn = win.child_window(auto_id="ButtonCall", control_type="Button")
                    if call_btn.exists():
                        call_btn.click_input()
                        print("✅ Call button clicked successfully!")
                        print("📞 Emergency call should be connecting...")
                    else:
                        print("⚠️ Call button not found.")
                        raise Exception("Call button not found")
                except Exception as pywin_error:
                    print(f"⚠️ pywinauto method failed: {pywin_error}")
                    print("📞 Manual call confirmation may be required!")
            else:
                print("📞 Phone dialer opened - pywinauto not available, MANUAL CALL REQUIRED!")

            return True
        except Exception as call_error:
            print(f"❌ Phone call failed: {call_error}")
            print(f"🚨 CRITICAL: MANUALLY DIAL {clean_number} NOW!")
            return False

    def make_emergency_call(self, phone_number):
        """Initiate an emergency call and send WhatsApp message"""
        
        # Clean and format the phone number
        clean_number = phone_number.replace(" ", "").replace("-", "")
        print(f"🚨 EMERGENCY: Initiating SOS for {phone_number}")
        print(f"📱 Formatted number: {clean_number}")
        
        try:
            # Get current location for emergency message
            print("🌍 Getting location for emergency message...")
            location_info = get_current_location()
            
            # Create emergency message with location
            base_message = "🚨 EMERGENCY ALERT! I need immediate help. This is an automated SOS message from OptiBlink. Please contact me urgently!"
            
            if location_info:
                emergency_message = f"{base_message}\n\n📍 My location: {location_info['address']}\n🗺️ Google Maps: {location_info['google_maps_url']}\n📱 Coordinates: {location_info['latitude']}, {location_info['longitude']}"
                print(f"📍 Location added to emergency message: {location_info['address']}")
            else:
                emergency_message = f"{base_message}\n\n📍 Location: Unable to determine current location"
                print("⚠️ Could not get location - sending message without location info")
                        
            print("📲 Sending WhatsApp emergency message...")
            
            # Check if user prefers WhatsApp Web
            prefer_web = self.config.get("prefer_whatsapp_web", False)
            if prefer_web:
                print("🌐 Configuration set to prefer WhatsApp Web - skipping app")
                whatsapp_opened = False
            else:
                encoded_message = _encode_message_for_url(emergency_message)
                
                whatsapp_msg_url = f"whatsapp://send?phone={clean_number}&text={encoded_message}"
                print(f"🔗 WhatsApp URL: {whatsapp_msg_url[:100]}..." if len(whatsapp_msg_url) > 100 else f"🔗 WhatsApp URL: {whatsapp_msg_url}")
                
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
                            windows = []
                            def enum_callback(hwnd, windows_list):
                                if win32gui.IsWindowVisible(hwnd):
                                    title = win32gui.GetWindowText(hwnd).lower()
                                    if 'whatsapp' in title and len(title) > 0:
                                        windows_list.append((hwnd, win32gui.GetWindowText(hwnd)))
                                return True

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
                
                # Properly encode the emergency message for URL
                if URLLIB_AVAILABLE:
                    encoded_message = urllib.parse.quote(emergency_message)
                else:
                    # Fallback encoding for special characters
                    encoded_message = emergency_message.replace(' ', '%20').replace('!', '%21').replace('\n', '%0A').replace(':', '%3A').replace('?', '%3F').replace('&', '%26').replace('=', '%3D')
                
                whatsapp_web_url = f"https://web.whatsapp.com/send?phone={web_number}&text={encoded_message}"
                print(f"🔗 WhatsApp Web URL: {whatsapp_web_url[:100]}..." if len(whatsapp_web_url) > 100 else f"🔗 WhatsApp Web URL: {whatsapp_web_url}")
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
                                windows = []
                                def enum_callback(hwnd, windows_list):
                                    if win32gui.IsWindowVisible(hwnd):
                                        title = win32gui.GetWindowText(hwnd).lower()
                                        # Check for both WhatsApp app and WhatsApp Web in browser
                                        if ('whatsapp' in title and len(title) > 0) or \
                                           ('whatsapp web' in title) or \
                                           (title.startswith('whatsapp') and 'browser' in title) or \
                                           ('web.whatsapp.com' in title):
                                            windows_list.append((hwnd, win32gui.GetWindowText(hwnd)))
                                    return True

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
                # TTS says 'help' after callee attends the call
                print("🔊 TTS: Saying 'help' after call is attended...")
                self.speak("help")
                
                if PYAUTOGUI_AVAILABLE:
                    # Focus specifically on Phone Link and call immediately
                    print("🔄 Focusing on Phone Link window...")
                    if WIN32_AVAILABLE:
                        try:
                            # Find Phone Link window specifically
                            def find_phone_link():
                                windows = []
                                def enum_callback(hwnd, windows_list):
                                    if win32gui.IsWindowVisible(hwnd):
                                        title = win32gui.GetWindowText(hwnd).lower()
                                        if ('phone' in title and 'link' in title) or 'your phone' in title:
                                            windows_list.append((hwnd, win32gui.GetWindowText(hwnd)))
                                    return True

                                win32gui.EnumWindows(enum_callback, windows)
                                return windows[0] if windows else (0, "")

                            phone_hwnd, phone_title = find_phone_link()
                            if phone_hwnd != 0:
                                print(f"📱 Found Phone Link: {phone_title}")
                                win32gui.SetForegroundWindow(phone_hwnd)
                                time.sleep(2)
                                print("✅ Phone Link window focused!")
                            else:
                                print("⚠️ Phone Link window not found, using Alt+Tab")
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
                        # CHANGE 1 TEST
                        print("🔊 Preparing to speak emergency message...")
                        time.sleep(8)  # Adjust if needed
                        
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
            
            _fallback_copy_and_prompt(clean_number, label="EMERGENCY: CALL")
            
            return False

    def make_direct_emergency_call(self, phone_number):
        """Make a direct emergency phone call without WhatsApp message"""
        
        # Clean and format the phone number
        clean_number = phone_number.replace(" ", "").replace("-", "")
        print(f"📞 DIRECT EMERGENCY CALL: {phone_number}")
        print(f"📱 Formatted number: {clean_number}")
        
        try:
            # PAUSE OptiBlink keyboard monitoring during phone call
            # Save current system state to restore later
            self.was_active_before_phone_call = self.is_system_active
            self.phone_call_active = True
            self.phone_call_start_time = time.time()
            self.keyboard_processing_enabled = False  # Completely disable keyboard processing
            print("🛑 OptiBlink keyboard monitoring COMPLETELY DISABLED for phone call")
            # Centralized dialing and Phone Link automation
            call_ok = self._start_phone_call(clean_number)
           
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
            
            _fallback_copy_and_prompt(clean_number, label="EMERGENCY CALL:")
            
            return False

    def handle_emergency_sos(self):
        """Simplified Emergency SOS handler - ALWAYS works, no restrictions"""
        try:
            print("\n🚨 *** EMERGENCY SOS ACTIVATED ***")
            print("🚨 *** BYPASSING ALL SYSTEM RESTRICTIONS ***")
            
            # Clean phone number
            clean_number = self.emergency_contact.replace(" ", "").replace("-", "")
            print(f"📞 Emergency Contact: {clean_number if clean_number else 'NOT SET'}")
            
            # Get current location
            print("🌍 Getting current location...")
            location_info = get_current_location()
            
            # Create emergency message with location
            base_message = "🚨 EMERGENCY! I need immediate help. This is an automated SOS from OptiBlink. Please call me urgently!"
            
            if location_info:
                # Clarify if device or IP-based
                if 'Device location' in location_info['address']:
                    location_type = '📡 Accurate device location'
                else:
                    location_type = '🌐 Approximate IP-based location'
                emergency_message = (
                    f"{base_message}\n\n{location_type}: {location_info['address']}"
                    f"\n🗺️ Google Maps: {location_info['google_maps_url']}"
                    f"\n📱 Coordinates: {location_info['latitude']}, {location_info['longitude']}"
                )
                print(f"📍 Location added to emergency message: {location_info['address']} ({location_type})")
            else:
                emergency_message = f"{base_message}\n\n📍 Location: Unable to determine current location"
                print("⚠️ Could not get location - sending message without location info")
            
            print(f"📝 Emergency message prepared: {len(emergency_message)} characters")
            
            # Make window fully visible during emergency
            self.set_window_transparency(WINDOW_NAME, 1.0)  # Full opacity
            print("🔍 Window made fully visible for emergency")
            
            # STEP 1: Send WhatsApp message - TRY APP FIRST, then Web fallback
            print("📲 Sending WhatsApp SOS message with location...")
            whatsapp_success = False
            
            try:
                # METHOD 1: Try WhatsApp APP first
                print("📱 Trying WhatsApp APP first...")
                
                # Properly encode the emergency message for URL
                if URLLIB_AVAILABLE:
                    encoded_message = urllib.parse.quote(emergency_message)
                else:
                    # Fallback encoding for special characters
                    encoded_message = emergency_message.replace(' ', '%20').replace('!', '%21').replace('\n', '%0A').replace(':', '%3A').replace('?', '%3F').replace('&', '%26').replace('=', '%3D')
                
                whatsapp_app_url = f"whatsapp://send?phone={clean_number}&text={encoded_message}"
                print(f"🔗 WhatsApp App URL: {whatsapp_app_url[:100]}..." if len(whatsapp_app_url) > 100 else f"🔗 WhatsApp App URL: {whatsapp_app_url}")
                
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
                    
                    encoded_message = _encode_message_for_url(emergency_message)
                    
                    whatsapp_web_url = f"https://web.whatsapp.com/send?phone={web_number}&text={encoded_message}"
                    print(f"🌐 WhatsApp Web URL: {whatsapp_web_url[:100]}..." if len(whatsapp_web_url) > 100 else f"🌐 WhatsApp Web URL: {whatsapp_web_url}")
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
            
            # STEP 2: Make phone call via centralized helper
            call_ok = self._start_phone_call(clean_number)
                
            # Always speak SOS message
            print("🔊 Speaking SOS message...")
            if location_info:
                spoken_message = f"Emergency SOS activated. Calling {clean_number}. Location shared: {location_info['city']}, {location_info['region']}. Hello, I am in an emergency. I need your help. Location sent via WhatsApp"
            else:
                spoken_message = f"Emergency SOS activated. Calling {clean_number}. Location could not be determined."
            
            self.speak(spoken_message)
            
            # Record the emergency
            self.message_history.append(f"🚨 EMERGENCY SOS - {clean_number}")
            print("📝 Emergency logged")
            
            # Clear buffers
            self.word_buffer = ""
            self.morse_char_buffer = ""
            self.last_sent_len = 0
            

            print("✅ Emergency SOS procedure completed!")
        
        except Exception as e:
            print(f"❌ CRITICAL: Emergency SOS failed: {e}")
            print(f"📞 MANUAL ACTION REQUIRED: CALL {self.emergency_contact or 'YOUR EMERGENCY CONTACT'} IMMEDIATELY!")
            
            _fallback_copy_and_prompt(clean_number, label="EMERGENCY CALL")
                    
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
                    # Suppress transparency message for cleaner output
                    # print(f"Window transparency set to {int(alpha*100)}%")
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
        """Draw a dynamic keyboard using OpenCV."""
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
            # print(f"Keyboard: width={width}, available={available_width}, base_key={base_key_width}")
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

                # Draw morse code below the main key text, always bold and clear
                morse_font_scale = min(font_scale + 0.5, 0.4)  # Slightly larger but capped
                morse_thickness = 2# Always bold
                morse_text = morse_code
                morse_text_w, morse_text_h = cv2.getTextSize(morse_text, font, morse_font_scale, morse_thickness)[0]
                morse_x = max(current_x + 4, current_x + (key_width - morse_text_w) // 2)
                morse_x = min(morse_x, current_x + key_width - morse_text_w - 4)
                morse_y = text_y + text_h + 12  # Place below main text
                # Draw black outline for contrast if not highlighted
                if not is_highlighted:
                    outline_color = (0, 0, 0)
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        cv2.putText(keyboard_img, morse_text, (morse_x + dx, morse_y + dy), font, morse_font_scale, outline_color, 1)
                # Draw main morse code text (bold)
                morse_color = (0, 0, 0) if is_highlighted else (255, 255, 255)  # Black if highlighted, else white
                cv2.putText(keyboard_img, morse_text, (morse_x, morse_y), font, morse_font_scale, morse_color, morse_thickness)
                
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
    
    def draw_call_icon(self, frame):
        """Draw call icon at the middle right extreme of the video frame. Returns icon_bbox (x, y, w, h) or None."""
        icon_bbox = None
        try:
            # Candidate locations for the icon file (project root or assets folder)
            icon_candidates = [
                resource_path(os.path.join('assets', 'call_icon.png')), 
                resource_path('call_icon.png'),
                os.path.join(os.path.dirname(__file__), 'call_icon.png')
            ]
            icon_path = None
            for c in icon_candidates:
                try:
                    if c and os.path.exists(c):
                        icon_path = c
                        break
                except Exception:
                    continue

            if icon_path:
                icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    frame_h, frame_w = frame.shape[:2]
                    
                    # Icon size - make it reasonably sized
                    max_icon_dim = 50  # 50px icon size
                    ih, iw = icon.shape[:2]
                    scale = min(max_icon_dim / ih, max_icon_dim / iw, 1.0)
                    new_w = max(32, int(iw * scale))
                    new_h = max(32, int(ih * scale))
                    icon_resized = cv2.resize(icon, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # Position: just above the keyboard, right side
                    # Right extreme: 10px from right edge
                    # Bottom: just above the keyboard (10px from bottom of video area)
                    icon_x = frame_w - new_w - 10
                    icon_y = frame_h - new_h - 10  # 10px margin from bottom of video area

                    # Ensure coordinates inside frame
                    if icon_y >= 0 and icon_x >= 0 and icon_x + new_w <= frame_w and icon_y + new_h <= frame_h:
                        # Draw a circular background to make it more prominent and clickable
                        padding = 8
                        circle_center = (icon_x + new_w // 2, icon_y + new_h // 2)
                        circle_radius = (max(new_w, new_h) // 2) + padding
                        
                        # Check if mouse is hovering (if attribute exists)
                        is_hovering = getattr(self, '_icon_hovering', False)
                        
                        # Draw circle background with hover effect
                        if is_hovering:
                            # Bright green background when hovering
                            cv2.circle(frame, circle_center, circle_radius, (255, 255, 255), 2)
                        else:
                            # Semi-transparent white background normally
                            overlay = frame.copy()
                            cv2.circle(overlay, circle_center, circle_radius, (255, 255, 255), -1)
                            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                            cv2.circle(frame, circle_center, circle_radius, (200, 200, 200), 2)
                        
                        # If icon has alpha channel, blend using alpha; otherwise simple paste
                        if icon_resized.shape[2] == 4:
                            alpha = icon_resized[:, :, 3] / 255.0
                            for c in range(3):
                                frame[icon_y:icon_y+new_h, icon_x:icon_x+new_w, c] = (
                                    alpha * icon_resized[:, :, c] + (1 - alpha) * frame[icon_y:icon_y+new_h, icon_x:icon_x+new_w, c]
                                ).astype(np.uint8)
                        else:
                            frame[icon_y:icon_y+new_h, icon_x:icon_x+new_w] = icon_resized[:, :, :3]
                        
                        # Add tooltip text when hovering
                        if is_hovering:
                            tooltip_text = "Update Emergency Contact"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.5
                            thickness = 2
                            (text_width, text_height), _ = cv2.getTextSize(tooltip_text, font, font_scale, thickness)
                            
                            # Position tooltip to the left of icon
                            tooltip_x = icon_x - text_width - 15
                            tooltip_y = icon_y + new_h // 2 + text_height // 2
                            
                            # Draw tooltip background
                            padding = 5
                            cv2.rectangle(frame, 
                                        (tooltip_x - padding, tooltip_y - text_height - padding),
                                        (tooltip_x + text_width + padding, tooltip_y + padding),
                                        (0, 0, 0), -1)
                            cv2.rectangle(frame, 
                                        (tooltip_x - padding, tooltip_y - text_height - padding),
                                        (tooltip_x + text_width + padding, tooltip_y + padding),
                                        (255, 255, 255), 1)
                            
                            # Draw tooltip text
                            cv2.putText(frame, tooltip_text, (tooltip_x, tooltip_y), 
                                      font, font_scale, (255, 255, 255), thickness)
                        
                        icon_bbox = (icon_x, icon_y, new_w, new_h)
        except Exception as e:
            # Don't fail the entire frame drawing if icon overlay fails
            pass
        
        return icon_bbox

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

    def speak(self, text, non_blocking=True):
        """Speak text using TTS. By default runs non-blocking in a daemon thread.

        If non_blocking is False, this will call the blocking implementation.
        """
        try:
            if not non_blocking:
                return self._speak_blocking(text)

            # If already speaking, queue a short wait and then speak new text
            if self.is_speaking:
                # Spawn a short thread to wait for current speech to finish before speaking
                def delayed_speak():
                    # Wait up to 5 seconds for current speech to finish
                    waited = 0
                    while self.is_speaking and waited < 5:
                        time.sleep(0.1)
                        waited += 0.1
                    try:
                        self._speak_blocking(text)
                    except Exception as e:
                        print(f"TTS delayed speak failed: {e}")

                t = threading.Thread(target=delayed_speak, daemon=True)
                t.start()
                return None

            # Normal non-blocking path: run in a daemon thread
            t = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
            t.start()
            return None
        except Exception as e:
            print(f"TTS speak wrapper error: {e}")
            return None

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
        # Suppress all MediaPipe/TensorFlow warnings during processing
        with _suppress_all_warnings():
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
                        
                        # Always announce sleep mode changes (regardless of TTS setting)
                        if self.is_system_active:
                            self._speak_blocking("Sleep mode deactivated")
                        else:
                            self.speak("Sleep mode activated")
                        
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
                cv2.rectangle(frame, box_top_left, box_bottom_right, (255, 255, 255), 2)

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

        # Add call icon to the middle right extreme of the video frame
        icon_bbox = self.draw_call_icon(frame)
        # Store icon bbox for mouse callback (in video frame coordinates)
        self._icon_bbox_attr = icon_bbox
        
        return frame

def prompt_phone_number_update():
    """Prompt user to update their emergency phone number on every application run"""
    try:
        # Load current config
        config = load_config()
        current_phone = config.get('emergency_contact', '')

        print("\n📞 EMERGENCY CONTACT VERIFICATION")
        print("=" * 40)

        # Detect interactive console
        interactive = False
        try:
            interactive = bool(sys.stdin and sys.stdin.isatty())
        except Exception:
            interactive = False

        if interactive:
            # Console-based prompts
            try:
                if current_phone:
                    print(f"Current emergency contact: {current_phone}")
                    print("\nThis number will be called automatically if you use the SOS signal (6 dots: ......)")
                    update_choice = input("\nDo you want to update your emergency contact number? (y/n): ").strip().lower()
                    if update_choice in ('y', 'yes'):
                        new_phone = input("\nEnter new emergency contact number (with country code, e.g., +919876543210): ").strip()
                        if new_phone and len(new_phone) >= 10 and new_phone.startswith('+') and any(c.isdigit() for c in new_phone):
                            config['emergency_contact'] = new_phone
                            save_config(config)
                            print(f"✅ Emergency contact updated to: {new_phone}")
                        else:
                            print("⚠️ Invalid phone number. Keeping current number.")
                    else:
                        print(f"Keeping current emergency contact: {current_phone}")
                else:
                    print("⚠️ No emergency contact found!")
                    new_phone = input("Enter emergency contact number (with country code, e.g., +919876543210): ").strip()
                    if new_phone and len(new_phone) >= 10 and new_phone.startswith('+') and any(c.isdigit() for c in new_phone):
                        config['emergency_contact'] = new_phone
                        save_config(config)
                        print(f"✅ Emergency contact set to: {new_phone}")
                    else:
                        print("⚠️ Invalid phone number. Using default.")
                        config['emergency_contact'] = DEFAULT_EMERGENCY_CONTACT
                        save_config(config)

                whatsapp_choice = input("\nDo you want to send WhatsApp message before calling? (y/n): ").strip().lower()
                config['prefer_whatsapp_web'] = (whatsapp_choice == 'y' or whatsapp_choice == 'yes')
                save_config(config)
                whatsapp_status = "enabled" if config['prefer_whatsapp_web'] else "disabled"
                print(f"WhatsApp Web integration: {whatsapp_status}")
                print("\n" + "=" * 40)
                print("Emergency contact verification complete!\n")
                return config
            except Exception as e:
                print(f"Error during phone number update: {e}")
                return load_config()
        else:
            # Non-interactive: use GUI dialogs if tkinter available, otherwise keep existing config
            try:
                import tkinter as _tk
                from tkinter import simpledialog as _sd, messagebox as _mb
                root = _tk.Tk()
                root.withdraw()
                if current_phone:
                    _mb.showinfo("Emergency Contact", f"Current emergency contact: {current_phone}\n\nThis number will be called automatically if you use the SOS signal (6 dots: ......)", parent=root)
                    update = _mb.askyesno("Update Emergency Contact", "Do you want to update your emergency contact number?", parent=root)
                    if update:
                        new_phone = _sd.askstring("Update Emergency Contact", "Enter new emergency contact number (with country code, e.g., +919876543210):", parent=root)
                        if new_phone and len(new_phone) >= 10 and new_phone.startswith('+') and any(c.isdigit() for c in new_phone):
                            config['emergency_contact'] = new_phone.strip()
                            save_config(config)
                else:
                    _mb.showwarning("Emergency Contact", "No emergency contact found! You may set one now.", parent=root)
                    new_phone = _sd.askstring("Set Emergency Contact", "Enter emergency contact number (with country code, e.g., +919876543210):", parent=root)
                    if new_phone and len(new_phone) >= 10 and new_phone.startswith('+') and any(c.isdigit() for c in new_phone):
                        config['emergency_contact'] = new_phone.strip()
                        save_config(config)

                pref = _mb.askyesno("WhatsApp Preference", "Do you want to send WhatsApp message before calling?", parent=root)
                config['prefer_whatsapp_web'] = pref
                save_config(config)
                whatsapp_status = "enabled" if config['prefer_whatsapp_web'] else "disabled"
                print(f"WhatsApp Web integration: {whatsapp_status}")
                print("\n" + "=" * 40)
                print("Emergency contact verification complete!\n")
                root.destroy()
                return config
            except Exception:
                # No GUI available; just return current config
                return config
    except Exception as e:
        print(f"Error during phone number update: {e}")
        return load_config()  # Return current config if error


def main():
    print("🚀 OptiBlink - Eye Tracking Morse Code Interface")
    print("=" * 50)
    
    # Load config silently - no prompts at startup
    current_config = load_config()
    emergency_contact = current_config.get('emergency_contact', 'Not configured')
    print(f"📞 Emergency contact: {emergency_contact} (Click phone icon to update)")
    
    # Initialize camera immediately
    print("📹 Initializing camera...")
    cap = cv2.VideoCapture(0)

    # Initialize systems with comprehensive stderr suppression
    print("🧠 Loading AI models and systems...")
    
    # CRITICAL: Keep boot-time stderr suppression active
    # TensorFlow Lite XNNPACK delegate loads on FIRST USE, not on import!
    # Solution: Never restore stderr, keep it suppressed permanently during runtime
    
    with _suppress_all_warnings():
        auto = AutoCompleteSystem()
        eye_tracker = EyeTracker(auto)
    
    print("✅ OptiBlink ready!")
    
    # OPTION: Keep stderr permanently suppressed to prevent ANY future warnings
    # We've saved it, but we won't restore it - stderr stays blocked forever
    # This is the nuclear option but guarantees zero warnings
    
    # Optionally restore stderr (commented out to keep suppression active):
    # try:
    #     if '_BOOT_SAVED_FD' in globals() and _BOOT_SAVED_FD is not None:
    #         os.dup2(_BOOT_SAVED_FD, 2)
    #         os.close(_BOOT_SAVED_FD)
    #         if sys.stderr:
    #             sys.stderr.reconfigure(encoding='utf-8')
    # except Exception:
    #     pass
    print("👁️  Look at the camera and blink to start calibration")
    print("📖 Morse code reference available in 'morse_keyboard.jpg'")
    print("🆘 Emergency SOS: Blink pattern '......' (6 dots)")
    print("❌ Press 'Q' or close window to exit")
    print("=" * 50)
    
    # print("🖥️ Setting up window...")
    window_width = 800
    window_height = 550

    # Move window to top-right corner with some margin
    try:
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
        x_pos = screen_width - window_width - 30  # Increased margin for larger window
        # print(f"Screen dimensions: {screen_width}x{screen_height}")
        # print(f"Window size: {window_width}x{window_height}")
        # print(f"Calculated position: x={x_pos}, y=40")
        # print(f"This should place window at top-right corner with 30px margin")
    except Exception as e:
        # Fallback to default position if Windows API is not available
        x_pos = 100
        # print(f"Warning: Could not get screen dimensions: {e}. Using default window position.")
    
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

    # print("Starting main loop...")
    
    # Variables for window positioning
    window_positioned = False
    position_attempts = 0
    max_position_attempts = 10
    

    # --- Mouse callback integration ---
    global icon_bbox_global, keyboard_bbox_global, window_height_global, window_width_global, eye_tracker_global, video_height_global
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

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
            # Get call icon bbox from eye_tracker if available
            icon_bbox = getattr(eye_tracker, '_icon_bbox_attr', None)
            # Save icon and keyboard bbox for mouse callback
            icon_bbox_global = icon_bbox
            keyboard_bbox_global = (0, video_height, window_width, keyboard_height)
            window_height_global = window_height
            window_width_global = window_width
            video_height_global = video_height
            eye_tracker_global = eye_tracker
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
                # Suppress positioning messages for cleaner output
                # print(f"Positioning attempt {position_attempts}: trying to move to ({x_pos}, {y_pos})")
                
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
                            # Suppress positioning verification messages
                            # print(f"Window positioned at ({current_x}, {current_y}), target was ({x_pos}, {y_pos})")
                            
                            if abs(current_x - x_pos) < 50 and abs(current_y - y_pos) < 50:
                                window_positioned = True
                                # Suppress success message for cleaner output
                                # print("Window successfully positioned in top-right corner!")
                        else:
                            # Suppress warning for cleaner output
                            # print("Warning: Could not verify window positioning - handle not found")
                            pass
                    except Exception as verify_error:
                        # Suppress error message for cleaner output
                        # print(f"Could not verify position: {verify_error}")
                        pass
                else:
                    # Suppress failure message for cleaner output
                    # print(f"Attempt {position_attempts}: Windows API positioning failed")
                    pass
                        
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
                # Suppress final positioning message for cleaner output
                # print(f"Window positioned at top-right corner: ({x_pos}, {y_pos})")

            # Robust window close detection - check if window was closed by user clicking X button
            try:
                window_property = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                if window_property < 1:
                    print("\n👋 Window closed by user. OptiBlink shutting down...")
                    break
            except cv2.error:
                print("\n👋 Window closed. OptiBlink shutting down...")
                break

            key = cv2.waitKey(1) & 0xFF
            
            # Additional check for window close after waitKey
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n👋 Window closed. OptiBlink shutting down...")
                    break
            except cv2.error:
                print("\n👋 Window closed. OptiBlink shutting down...")
                break
            
            # Check for actual keyboard input (only when window has focus and key is physically pressed)
            # Use GetAsyncKeyState to check for actual physical key presses
            try:
                # Check if Q key is currently being pressed (GetAsyncKeyState returns non-zero if pressed)
                q_pressed = (ctypes.windll.user32.GetAsyncKeyState(ord('Q')) & 0x8000) or (ctypes.windll.user32.GetAsyncKeyState(ord('q')) & 0x8000)
                if q_pressed:
                    if not hasattr(eye_tracker, '_q_pressed') or not eye_tracker._q_pressed:
                        eye_tracker._q_pressed = True
                        print("\n👋 OptiBlink shutting down...")
                        print("💾 Saving usage data...")
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


# --- Mouse callback and update dialog ---
import threading
import tkinter as tk
from tkinter import simpledialog, messagebox

icon_bbox_global = None
keyboard_bbox_global = None
window_height_global = None
window_width_global = None
eye_tracker_global = None

def update_emergency_contact_dialog():
    """Update emergency contact in both config.json and the running EyeTracker instance"""
    # Run in a separate thread to avoid blocking OpenCV
    def run_dialog():
        root = tk.Tk()
        root.withdraw()
        root.title("Update Emergency Contact")
        
        current = (eye_tracker_global.config or {}).get("emergency_contact", "")
        prompt_text = f"Current: {current}\n\nEnter new emergency contact number:\n(Include country code, e.g., +919876543210)"
        
        new_contact = simpledialog.askstring(
            "Update Emergency Contact", 
            prompt_text, 
            parent=root
        )
        
        if new_contact and new_contact.strip():
            new_contact = new_contact.strip()
            
            # Validate the phone number format
            if len(new_contact) < 10:
                messagebox.showerror(
                    "Invalid Number", 
                    "Phone number is too short. Please enter a valid number with country code.",
                    parent=root
                )
                root.destroy()
                return
            
            # Update in the config dictionary
            eye_tracker_global.config["emergency_contact"] = new_contact
            
            # Save to config.json file
            save_config(eye_tracker_global.config)
            print(f"✅ Config file updated with new emergency contact: {new_contact}")
            
            # Update the live EyeTracker instance's emergency_contact attribute
            eye_tracker_global.emergency_contact = new_contact.replace(" ", "").replace("-", "")
            print(f"✅ EyeTracker instance updated with emergency contact: {eye_tracker_global.emergency_contact}")
            
            # Show success message
            messagebox.showinfo(
                "Success", 
                f"Emergency contact updated successfully!\n\nNew contact: {new_contact}\n\nThis number will be used for SOS calls.",
                parent=root
            )
            
            print("━" * 50)
            print("📞 EMERGENCY CONTACT UPDATED")
            print(f"   New Number: {new_contact}")
            print(f"   Cleaned Format: {eye_tracker_global.emergency_contact}")
            print(f"   Saved to: {CONFIG_PATH}")
            print("━" * 50)
        
        root.destroy()
    
    t = threading.Thread(target=run_dialog, daemon=True)
    t.start()

def mouse_callback(event, x, y, flags, param):
    global eye_tracker_global
    
    # Handle mouse movement for hover effect
    if event == cv2.EVENT_MOUSEMOVE:
        if icon_bbox_global and video_height_global and eye_tracker_global:
            icon_x, icon_y, icon_w, icon_h = icon_bbox_global
            # Check if mouse is over the icon in video area
            if y < video_height_global:  # Mouse is in video area
                # Add some padding for easier clicking
                padding = 10
                if (icon_x - padding <= x < icon_x + icon_w + padding) and \
                   (icon_y - padding <= y < icon_y + icon_h + padding):
                    eye_tracker_global._icon_hovering = True
                    # Change cursor to hand pointer using Windows API
                    if WIN32_AVAILABLE:
                        try:
                            import win32api
                            IDC_HAND = 32649
                            hand_cursor = win32api.LoadCursor(0, IDC_HAND)
                            win32api.SetCursor(hand_cursor)
                        except:
                            pass
                else:
                    eye_tracker_global._icon_hovering = False
                    # Change cursor back to default arrow
                    if WIN32_AVAILABLE:
                        try:
                            import win32api
                            IDC_ARROW = 32512
                            arrow_cursor = win32api.LoadCursor(0, IDC_ARROW)
                            win32api.SetCursor(arrow_cursor)
                        except:
                            pass
            else:
                eye_tracker_global._icon_hovering = False
                # Change cursor back to default arrow
                if WIN32_AVAILABLE:
                    try:
                        import win32api
                        IDC_ARROW = 32512
                        arrow_cursor = win32api.LoadCursor(0, IDC_ARROW)
                        win32api.SetCursor(arrow_cursor)
                    except:
                        pass
    
    # Handle left click
    elif event == cv2.EVENT_LBUTTONDOWN:
        # Check if click is on the call icon in video area
        if icon_bbox_global and video_height_global:
            icon_x, icon_y, icon_w, icon_h = icon_bbox_global
            # Icon is in the video area (top portion of window)
            if y < video_height_global:  # Click is in video area
                # Add some padding for easier clicking
                padding = 10
                if (icon_x - padding <= x < icon_x + icon_w + padding) and \
                   (icon_y - padding <= y < icon_y + icon_h + padding):
                    print("📞 Call icon clicked - Opening emergency contact dialog...")
                    update_emergency_contact_dialog()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        try:
            # Only wait for Enter if running in an interactive console
            if sys.stdin and sys.stdin.isatty():
                input("Press Enter to continue...")
        except Exception:
            pass