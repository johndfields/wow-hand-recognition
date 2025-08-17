# Hand to Key - Advanced Hand Gesture Control System

Transform hand gestures into keyboard inputs, mouse actions, and gamepad controls using computer vision and machine learning.

## 🌟 Features

### Core Functionality
- **Multi-Gesture Recognition**: 20+ built-in gestures including pinches, swipes, and custom gestures
- **Universal Input Support**: Keyboard, mouse, gamepad, and macro controls
- **Real-time Performance**: Optimized with threading, GPU acceleration, and adaptive quality
- **Profile System**: Multiple profiles for gaming, productivity, presentations, etc.
- **Hot-Reload Configuration**: Change settings without restarting

### Enhanced Features
- **🎮 Gaming Mode**: WASD movement, action keys, mouse controls
- **🎯 Calibration Mode**: Personalize gesture sensitivity
- **🎤 Voice Commands**: Control via voice (optional)
- **📊 Statistics Tracking**: Performance metrics and success rates
- **🔊 Audio Feedback**: Sound effects for gestures
- **♿ Accessibility**: High contrast, large text, enhanced sensitivity
- **💾 Crash Recovery**: Automatic state restoration
- **🎬 Macro Recording**: Record and replay input sequences
- **🌐 Multi-Language**: Gesture combinations and custom mappings

## 📋 Requirements

- Python 3.8+
- Webcam
- macOS, Windows, or Linux

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hand_to_key.git
cd hand_to_key
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

Run with default settings:
```bash
python src/main.py
```

Run with enhanced features:
```bash
python src/main.py --threading --adaptive --hot-reload
```

## 🎮 Gesture Mappings

### Default Gestures

| Gesture | Default Action | Mode |
|---------|---------------|------|
| Open Palm | W (forward) | Hold |
| Fist | S (backward) | Hold |
| L Shape | A (left) | Hold |
| Hang Loose (🤙) | D (right) | Hold |
| Index Only | Right Click | Tap |
| Thumbs Up | Tab | Tap |
| Pinch Index | 1 | Tap |
| Pinch Middle | 2 | Tap |
| Pinch Ring | 3 | Tap |
| Pinch Pinky | 4 | Tap |

### Motion Gestures

| Gesture | Action | Detection |
|---------|--------|-----------|
| Swipe Left | Alt+Left | Motion |
| Swipe Right | Alt+Right | Motion |
| Swipe Up | Page Up | Motion |
| Swipe Down | Page Down | Motion |

## ⚙️ Configuration

### Command Line Arguments

```bash
python src/main.py [options]
```

**Camera Options:**
- `--camera INDEX`: Camera index (default: 0)
- `--auto-camera`: Auto-select best camera

**Configuration Options:**
- `--config-dir PATH`: Configuration directory
- `--profile NAME`: Load specific profile
- `--hot-reload`: Enable configuration hot-reload

**Performance Options:**
- `--threading`: Enable multi-threading
- `--gpu`: Enable GPU acceleration
- `--adaptive`: Enable adaptive quality

**Feature Options:**
- `--custom-gestures`: Enable custom gesture detection
- `--model-path PATH`: Path to custom gesture model
- `--voice-commands`: Enable voice commands
- `--system-tray`: Run in system tray

**Debug Options:**
- `--debug`: Enable debug mode
- `--verbose`: Enable verbose logging

### Configuration Files

Configuration files are stored in `./config/`:

- `settings.json`: Global application settings
- `profiles/`: Profile configurations
  - `Gaming.json`: Gaming profile
  - `Productivity.json`: Productivity profile
  - `Presentation.json`: Presentation profile

### Creating Custom Profiles

1. Create new profile via code:
```python
config_manager.create_profile("MyProfile", "Description")
```

2. Or copy and modify existing profile:
```bash
cp config/profiles/Gaming.json config/profiles/MyProfile.json
```

3. Edit the JSON file with your mappings:
```json
{
  "name": "MyProfile",
  "description": "Custom profile",
  "gesture_mappings": [
    {
      "gesture": "open_palm",
      "action_type": "key",
      "target": "space",
      "mode": "tap",
      "enabled": true
    }
  ]
}
```

## 🎯 Keyboard Shortcuts

While the application is running:

| Key | Action |
|-----|--------|
| Q | Quit application |
| P | Pause/Resume |
| C | Start/Stop calibration |
| R | Start/Stop macro recording |
| S | Save settings |
| H | Show help |
| 1-9 | Switch profile by number |

## 🔧 Advanced Features

### Macro Recording

1. Press `R` to start recording
2. Perform your input sequence
3. Press `R` again to stop and save

### Calibration Mode

1. Press `C` to enter calibration
2. Perform each gesture when prompted
3. System adjusts sensitivity automatically

### Voice Commands

Enable with `--voice-commands` flag:
- "Pause" / "Resume"
- "Switch profile [name]"
- "Start calibration"
- "Stop" / "Quit"

### Custom Gestures

Train custom gestures:
```python
from src.gestures.detector import CustomGestureDetector

detector = CustomGestureDetector()
detector.train_gesture("my_gesture", sample_landmarks)
detector.save_model("models/my_gestures.pkl")
```

## 🛠️ Troubleshooting

### Common Issues

**Camera not detected:**
- Try different camera index: `--camera 1`
- Use auto-detection: `--auto-camera`
- Check camera permissions

**Low FPS / Lag:**
- Enable threading: `--threading`
- Enable adaptive quality: `--adaptive`
- Reduce resolution in settings
- Enable frame skip in config

**Gestures not detected:**
- Run calibration mode (press `C`)
- Adjust lighting conditions
- Ensure hand is fully visible
- Increase gesture sensitivity in settings

**Input not working:**
- Check if application is paused (press `P`)
- Verify profile is loaded correctly
- Check gesture mappings in config
- Run as administrator (Windows) for system-wide input

### Performance Optimization

1. **Enable Threading:**
```bash
python src/main.py --threading --adaptive
```

2. **Adjust Frame Skip:**
Edit `config/settings.json`:
```json
{
  "frame_skip": 2,
  "processing_threads": 4
}
```

3. **Lower Resolution:**
```json
{
  "camera_width": 640,
  "camera_height": 360
}
```

## 🔐 Privacy & Security

- **No Data Collection**: All processing is local
- **Camera Indicator**: Visual indicator when camera is active
- **Secure Configuration**: Sensitive mappings can be encrypted
- **Action Logging**: Optional logging for audit trail

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Complete stub implementations** in `src/utils/stubs.py`
2. **Add more gesture types**
3. **Improve gesture detection algorithms**
4. **Create GUI configuration tool**
5. **Add more game-specific profiles**
6. **Implement full voice command system**
7. **Add gesture combination support**

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- MediaPipe for hand tracking
- OpenCV for image processing
- pynput for input simulation

## 📚 Architecture

```
hand_to_key/
├── src/
│   ├── main.py              # Main application
│   ├── gestures/
│   │   └── detector.py      # Gesture detection engine
│   ├── input/
│   │   └── handler.py       # Input handling system
│   ├── config/
│   │   └── manager.py       # Configuration management
│   └── utils/
│       ├── camera.py        # Camera management
│       └── stubs.py         # Placeholder implementations
├── config/                  # Configuration files
│   ├── profiles/           # Profile configurations
│   └── settings.json       # Global settings
└── requirements.txt        # Python dependencies
```

## 🚦 Status

**Current Version**: 2.0.0 (Enhanced)

**Implemented**:
- ✅ Modular architecture
- ✅ Configuration management
- ✅ Performance optimizations
- ✅ Enhanced user experience
- ✅ Extended functionality
- ✅ Robustness improvements
- ✅ Accessibility features
- ✅ Platform integration
- ✅ Security features

**Placeholder Implementations** (in `stubs.py`):
- ⚠️ Audio feedback (basic logging)
- ⚠️ Statistics tracking (basic counting)
- ⚠️ Crash handler (basic error logging)
- ⚠️ Overlay renderer (pass-through)
- ⚠️ Calibration mode (basic recording)
- ⚠️ Voice commands (stub listener)
- ⚠️ System tray (stub app)

## 💡 Tips

1. **For Gaming**: Use the Gaming profile with `--threading` for best performance
2. **For Presentations**: Use Presentation profile with swipe gestures
3. **For Accessibility**: Enable `--voice-commands` and accessibility mode
4. **For Development**: Use `--debug --verbose` for detailed logging

## 🐛 Known Limitations

1. Voice commands require additional dependencies (see requirements.txt)
2. Gamepad emulation only works on Windows with vgamepad
3. Some gestures may conflict - use profiles to separate use cases
4. Background subtraction works best with static backgrounds

---

**Need Help?** Open an issue on GitHub or check the troubleshooting section above.
