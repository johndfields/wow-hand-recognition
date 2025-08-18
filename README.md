# Hand Gesture Recognition for Input Control

A computer vision-based application that recognizes hand gestures and translates them into keyboard, mouse, and gamepad inputs. This allows for hands-free control of games, presentations, and other applications.

## Features

- **Real-time Hand Gesture Detection**: Recognizes a wide variety of hand gestures including:
  - Basic gestures (open palm, fist, thumbs up)
  - Finger counting (one to four fingers)
  - Special gestures (L-shape, hang loose/shaka, OK sign, rock on)
  - Pinch gestures (with each finger)
  - Motion gestures (swipes in different directions)

- **Flexible Input Mapping**:
  - Map gestures to keyboard keys, key combinations, and special keys
  - Map gestures to mouse movements and clicks
  - Map gestures to gamepad buttons and analog sticks
  - Create and execute complex macro sequences

- **Cross-Platform Compatibility**:
  - Windows, macOS, and Linux support
  - Special optimizations for Mac-to-Windows VM environments (Parallels)
  - Platform-specific input adapters for consistent behavior

- **Configuration System**:
  - Profile-based configuration with inheritance
  - JSON schema validation
  - Hot-reload capability
  - Migration utilities for different configuration formats

- **Performance Optimizations**:
  - Multi-threaded processing
  - GPU acceleration support
  - Adaptive quality settings
  - Frame skipping options

- **User Interface**:
  - Real-time camera preview with hand landmark visualization
  - Performance statistics display
  - Debug information overlay
  - Customizable UI settings

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- pynput (for input control)
- vgamepad (for gamepad emulation)
- jsonschema (for configuration validation)
- watchdog (for configuration hot-reload)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/hand-gesture-recognition.git
   cd hand-gesture-recognition
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/main.py
   ```

## Usage

### Basic Usage

Run the application with default settings:
```
python src/main.py
```

### Command Line Options

```
python src/main.py --help
```

Common options:
- `--mode [simple|modular|auto]`: Select application mode
- `--config PATH`: Path to configuration file
- `--profile NAME`: Profile name to use (modular mode only)
- `--camera INDEX`: Camera index to use
- `--width WIDTH`: Camera width
- `--height HEIGHT`: Camera height
- `--debug`: Enable debug mode
- `--no-preview`: Disable camera preview

### Configuration

The application supports two configuration formats:

1. **Simple Format** (legacy):
   ```json
   {
     "camera_index": 0,
     "sensitivity": 1.0,
     "cooldown": 0.7,
     "mappings": {
       "open_palm": {"action": "key", "target": "w", "mode": "hold"},
       "fist": {"action": "key", "target": "s", "mode": "hold"}
     }
   }
   ```

2. **Modular Format** (recommended):
   ```json
   {
     "name": "Gaming",
     "description": "Optimized for gaming with WASD movement",
     "gesture_mappings": [
       {
         "gesture": "open_palm",
         "action_type": "key",
         "target": "w",
         "mode": "hold"
       },
       {
         "gesture": "fist",
         "action_type": "key",
         "target": "s",
         "mode": "hold"
       }
     ],
     "settings": {
       "gesture_sensitivity": 1.2
     }
   }
   ```

### Migrating Configurations

To migrate from simple to modular format:
```
python src/main.py --migrate --config simple_config.json
```

To migrate all configurations in a directory:
```
python src/main.py --migrate-dir ./configs
```

## Supported Gestures

| Gesture | Description | Default Mapping |
|---------|-------------|----------------|
| `open_palm` | All fingers extended | W key (hold) |
| `fist` | Closed hand | S key (hold) |
| `thumbs_up` | Thumb extended, others closed | Space (tap) |
| `index_only` | Index finger extended | Mouse right click |
| `victory` | Index and middle fingers extended | A key (hold) |
| `three` | Index, middle, and ring fingers extended | D key (hold) |
| `four_fingers` | All fingers except thumb extended | None |
| `pinch_index` | Thumb and index finger pinched | 1 key |
| `pinch_middle` | Thumb and middle finger pinched | 2 key |
| `pinch_ring` | Thumb and ring finger pinched | 3 key |
| `pinch_pinky` | Thumb and pinky finger pinched | 4 key |
| `l_shape` | Thumb and index forming L shape | None |
| `hang_loose` | Thumb and pinky extended (shaka) | None |
| `ok_sign` | Thumb and index forming circle | None |
| `rock_on` | Index and pinky extended | None |
| `swipe_left` | Hand moving left | Alt+Left |
| `swipe_right` | Hand moving right | Alt+Right |
| `swipe_up` | Hand moving up | Page Up |
| `swipe_down` | Hand moving down | Page Down |

## Project Structure

```
hand-gesture-recognition/
├── src/
│   ├── gestures/
│   │   ├── detector.py       # Gesture detection algorithms
│   │   └── visualizer.py     # Visualization utilities
│   ├── input/
│   │   ├── handler.py        # Input handling system
│   │   └── platform_adapters.py # Platform-specific adapters
│   ├── config/
│   │   ├── manager.py        # Configuration management
│   │   └── migration.py      # Configuration migration utilities
│   ├── camera/
│   │   └── capture.py        # Camera capture module
│   ├── ui/
│   │   └── preview.py        # UI preview window
│   ├── main.py               # Main application entry point
│   └── simple_implementation/ # Legacy simple implementation
├── config/
│   ├── profiles/             # Configuration profiles
│   └── macros/               # Macro definitions
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the hand tracking technology
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [pynput](https://pynput.readthedocs.io/) for input control

