# Hand to Key Launcher Scripts

This directory contains several launcher scripts to run the Hand to Key application with different configurations optimized for various use cases.

## Available Launchers

### 🎮 Gaming Mode - `./run_gaming.sh`
**Optimized for gaming and real-time performance**

```bash
./run_gaming.sh
```

**Features:**
- Threading enabled for maximum performance
- Adaptive quality for dynamic performance adjustment
- Hot reload for live configuration updates
- Automatically loads Gaming profile
- Performance mode environment variables

**Best for:**
- Gaming sessions
- Real-time applications
- High-performance gesture recognition

---

### 📱 Standard Mode - `./run.sh`
**Basic launcher with standard settings**

```bash
./run.sh
```

**Features:**
- Default configuration
- Standard performance settings
- Basic gesture detection

**Best for:**
- General productivity use
- Presentation mode
- Learning the application

---

### 🔧 Development Mode - `./run_dev.sh`
**Full debugging and development features**

```bash
./run_dev.sh
```

**Features:**
- Threading enabled for testing
- Adaptive quality testing
- Hot reload for development
- Verbose logging and debug output
- All visualization options enabled
- Development environment variables

**Best for:**
- Development and debugging
- Testing new features
- Troubleshooting issues
- Performance analysis

---

## Quick Start

1. **Make sure you have `uv` installed:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Choose your launcher based on your use case:**
   - Gaming: `./run_gaming.sh`
   - General use: `./run.sh` 
   - Development: `./run_dev.sh`

3. **Grant camera permissions** when prompted

4. **Use keyboard controls** in the application:
   - `h` - Show help and controls
   - `l` - Toggle hand landmarks
   - `j` - Toggle hand connections
   - `b` - Cycle hand preference (right/left/both)
   - `d` - Toggle debug display mode
   - `q` - Quit application

## Hand Preference Configuration

All launchers support the enhanced hand tracking system with these options:

- **Right Hand Only** (`"right"`): Tracks only your right hand
- **Left Hand Only** (`"left"`): Tracks only your left hand  
- **Both Hands** (`"both"`): Tracks both hands simultaneously

Use the `b` key during runtime to cycle through these options, or edit `config/settings.json`.

## Troubleshooting

If you encounter issues:

1. **Check camera permissions** - macOS requires camera access
2. **Verify uv installation** - Run `uv --version` to check
3. **Use development mode** - Run `./run_dev.sh` for detailed logging
4. **Check the logs** - Look for error messages in the console output

## Performance Tips

### For Gaming:
- Use `./run_gaming.sh` for best performance
- Set `hand_preference` to `"right"` or `"left"` (not `"both"`) for better performance
- Ensure good lighting for reliable hand tracking
- Position camera at chest level for optimal hand detection

### For Development:
- Use `./run_dev.sh` to see all debug information
- Monitor console output for performance metrics
- Use visualization toggles to understand hand tracking behavior

## Environment Variables

The launchers set these environment variables:

### Gaming Mode:
```bash
export HAND_TO_KEY_PROFILE="Gaming"
export HAND_TO_KEY_PERFORMANCE_MODE="1"
```

### Development Mode:
```bash
export HAND_TO_KEY_DEBUG="1"
export HAND_TO_KEY_LOG_LEVEL="DEBUG"
export PYTHONUNBUFFERED="1"
```

## Command Line Arguments

All launchers use these command-line flags:

- `--threading` - Enable multithreaded processing
- `--adaptive` - Enable adaptive quality adjustment
- `--hot-reload` - Enable live configuration reloading
- `--verbose` - Enable verbose logging (dev mode only)

For custom configurations, you can run the application directly:
```bash
uv run src/main.py --help
```

---

**Note:** Make sure your camera is connected and permissions are granted before running any launcher!
