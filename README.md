# WoW Hand Recognition

Transform hand gestures into keyboard inputs, mouse actions, and gamepad controls for World of Warcraft and other games using computer vision and machine learning.

## Features

- **Gesture Recognition**: Detect 20+ hand gestures including open palm, fist, pinches, and custom gestures
- **Input Mapping**: Map gestures to keyboard, mouse, and gamepad controls
- **Profiles**: Switch between Gaming, Productivity, and Presentation modes
- **Performance**: Multi-threading, GPU acceleration, and adaptive quality
- **Customization**: Create custom gestures and mappings

## Quick Start

### Requirements
- Python 3.8+
- Webcam
- Windows, macOS, or Linux

### Installation

```bash
# Clone the repository
git clone https://github.com/johndfields/wow-hand-recognition.git
cd wow-hand-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Basic usage
python src/main.py

# With performance options
python src/main.py --threading --gpu --adaptive

# Gaming profile
python src/main.py --profile Gaming
```

## Default Gesture Mappings

### Gaming Profile

| Gesture | Action | Mode |
|---------|--------|------|
| Open Palm | W (forward) | Hold |
| Fist | S (backward) | Hold |
| L Shape | A (left) | Hold |
| Hang Loose | D (right) | Hold |
| Index Only | Right Click | Tap |
| Thumbs Up | Tab | Hold |
| Pinch Index | 1 | Tap |
| Pinch Middle | 2 | Tap |
| Pinch Ring | 3 | Tap |
| Pinch Pinky | 4 | Tap |

### Productivity Profile

| Gesture | Action | Mode |
|---------|--------|------|
| Swipe Left | Alt+Left | Tap |
| Swipe Right | Alt+Right | Tap |
| Swipe Up | Page Up | Tap |
| Swipe Down | Page Down | Tap |
| Pinch Index | Ctrl+C (Copy) | Tap |
| Pinch Middle | Ctrl+V (Paste) | Tap |
| Pinch Ring | Ctrl+Z (Undo) | Tap |
| Pinch Pinky | Ctrl+Y (Redo) | Tap |
| Open Palm | Ctrl+S (Save) | Tap |
| Fist | Escape | Tap |

## Command Line Options

```
--camera INDEX        Camera index (default: 0)
--auto-camera         Auto-select best camera
--config-dir PATH     Configuration directory
--profile NAME        Load specific profile
--hot-reload          Enable configuration hot-reload
--threading           Enable multi-threading
--gpu                 Enable GPU acceleration
--adaptive            Enable adaptive quality
--custom-gestures     Enable custom gesture detection
--model-path PATH     Path to custom gesture model
--voice-commands      Enable voice commands
--system-tray         Run in system tray
--debug               Enable debug mode
--verbose             Enable verbose logging
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Q | Quit application |
| P | Pause/Resume |
| C | Start/Stop calibration |
| R | Start/Stop macro recording |
| S | Save settings |
| H | Show help |
| 1-9 | Switch profile by number |
| B | Cycle hand preference (right → left → both) |

## Troubleshooting

- **Camera not detected**: Try `--camera 1` or `--auto-camera`
- **Low FPS/Lag**: Enable `--threading` and `--adaptive`
- **Gestures not detected**: Run calibration (press `C`) or adjust lighting
- **Input not working**: Check if application is paused (press `P`)

## Project Structure

```
wow-hand-recognition/
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
│   ├── profiles/            # Profile configurations
│   └── settings.json        # Global settings
└── requirements.txt         # Python dependencies
```

## Contributing

Contributions are welcome! Areas for improvement:

1. Complete stub implementations in `src/utils/stubs.py`
2. Add more gesture types
3. Improve gesture detection algorithms
4. Create GUI configuration tool
5. Add more game-specific profiles

## License

MIT License - See LICENSE file for details

## Acknowledgments

- MediaPipe for hand tracking
- OpenCV for image processing
- pynput for input simulation

