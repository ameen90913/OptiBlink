# OptiBlink: Blink-Based Text Input System for Paralyzed Patients

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-orange.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Overview

OptiBlink is an innovative assistive technology solution designed to help paralyzed patients communicate through blink-based text input. Using computer vision and machine learning, the system detects eye blinks and converts them into text input using Morse code patterns, enabling individuals with limited mobility to type and communicate effectively.

## ✨ Features

- **Real-time Eye Tracking**: Advanced computer vision using MediaPipe for precise eye landmark detection
- **Blink Detection**: Sophisticated algorithm to detect and differentiate between intentional and natural blinks
- **Morse Code Translation**: Converts blink patterns into text using Morse code system
- **Emergency SOS System**: Critical safety feature for medical emergencies using SOS Morse pattern
- **Auto-completion**: Intelligent word suggestions using Trie data structure and frequency analysis
- **Text-to-Speech**: Built-in speech synthesis for auditory feedback
- **Customizable Interface**: Adjustable sensitivity and calibration options
- **Usage Analytics**: Tracks frequently used words for improved suggestions
- **Cross-platform Support**: Works on Windows, macOS, and Linux systems

## 🎯 Use Cases

- **ALS Patients**: Enabling communication for individuals with Amyotrophic Lateral Sclerosis
- **Spinal Cord Injuries**: Assisting patients with limited upper body mobility
- **Stroke Recovery**: Supporting communication during rehabilitation
- **General Accessibility**: Providing alternative input methods for various disabilities

## 🛠️ Technology Stack

- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: NumPy for numerical computations
- **Natural Language Processing**: NLTK for word processing
- **Audio Processing**: gTTS, Pygame for text-to-speech
- **System Integration**: PyWin32 for Windows integration
- **Data Structures**: Custom Trie implementation for auto-completion

## 📋 Prerequisites

- Python 3.7 or higher
- Webcam or camera device
- Windows 10/11 (primary support), macOS, or Linux
- Internet connection (for initial NLTK data download)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/OptiBlink.git
cd OptiBlink
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data (Automatic)
The system will automatically download required NLTK data on first run.

## 🎮 Usage

### Basic Operation

1. **Launch the Application**:
   ```bash
   python eye_tracker.py
   ```

2. **Calibration**:
   - Position yourself in front of the camera
   - The system will automatically establish baseline blink detection parameters

3. **Text Input**:
   - Use short blinks for dots (.)
   - Use long blinks for dashes (-)
   - Follow Morse code patterns for letters and numbers
   - Use the auto-completion feature for faster typing

### Morse Code Reference

| Character | Morse Code | Blink Pattern |
|-----------|------------|---------------|
| A | .- | Short, Long |
| B | -... | Long, Short, Short, Short |
| C | -.-. | Long, Short, Long, Short |
| ... | ... | ... |

### Advanced Features

- **Auto-completion**: Type partial words and select from suggestions
- **Voice Feedback**: Enable text-to-speech for auditory confirmation
- **Emergency SOS**: Use SOS Morse pattern to send WhatsApp message and initiate phone call to configured emergency contact
- **Customization**: Adjust blink sensitivity and timing parameters
- **Usage Tracking**: System learns from your typing patterns

### Keyboard Controls

- **Q Key**: Exit the application
- **R Key**: Recalibrate eye detection system

## 📁 Project Structure

```
OptiBlink/
├── eye_tracker.py          # Main application file
├── requirements.txt        # Python dependencies
├── config.json            # Emergency contact configuration
├── words.csv              # Custom word dictionary
├── usage_data.txt         # Usage analytics data
├── morse_keyboard.jpg     # Morse code reference image
└── README.md              # Project documentation
```

## 🔧 Configuration

### Sensitivity Settings
- Adjust blink detection sensitivity in the `EyeTracker` class
- Modify timing parameters for short/long blink detection
- Customize auto-completion suggestion count

### Word Dictionary
- Add custom words to `words.csv` for domain-specific vocabulary
- System automatically loads and prioritizes custom words
- Fallback to NLTK dictionary for comprehensive coverage

## 📊 Performance

- **Latency**: <100ms blink detection
- **Accuracy**: >95% blink recognition rate
- **Compatibility**: Works with most USB webcams
- **Resource Usage**: Minimal CPU and memory footprint

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

**OptiBlink** was developed by:

- **[Ameen Muhammed Jumah](https://www.linkedin.com/in/ameen-muhammed-jumah-45844b271)** - Lead Program Developer
- **[Deepika S](https://www.linkedin.com/in/deepika-s-7494a7258/)** - UI/UX Design & Optimization
- **[Jeslin Philip](http://linkedin.com/in/jeslin-philip-965783301)** - Word Suggestion System Developer
- **[B Thanvi Sheetal](https://www.linkedin.com/in/thanvi-sheetal-779a22265/)** - Keyboard Integration & Testing

### Acknowledgments

- MediaPipe team for the excellent computer vision framework
- OpenCV community for robust image processing capabilities
- NLTK contributors for natural language processing tools
- The assistive technology community for inspiration and feedback

## 📞 Support

For support, questions, or feature requests:

- Create an issue on GitHub
- Documentation: [Wiki](https://github.com/your-username/OptiBlink/wiki)
- Contact creators

## 🔮 Roadmap

- [ ] Integration with popular communication platforms
- [ ] Advanced gesture recognition
- [ ] Cloud-based word learning
- [ ] Multi-language support

## 📈 Statistics

- **Lines of Code**: 1,200+ (main application)
- **Dependencies**: 9 core libraries
- **Features**: 10+ major features including emergency system
- **Supported Languages**: English (expandable)
- **Emergency Response**: WhatsApp + Phone call integration
- **Window Management**: Always-on-top with recalibration support

---

**Made with ❤️ for the assistive technology community**

*OptiBlink - Empowering communication through innovation*
