#!/bin/bash

# Improved Hand-to-Key Gaming Launcher
# Runs the hand gesture control with improved pinch detection

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the hand-to-key directory
cd "$SCRIPT_DIR"

echo "🎮 Starting Improved Hand Gesture Recognition Gaming Mode..."
echo "📁 Working directory: $PWD"
echo "⚙️  Config: gaming_config.json"
echo "🖥️  Target: Parallels Windows VM"
echo ""
echo "Gesture Controls:"
echo "  👋 Open Palm    → W (forward)"
echo "  ✊ Fist         → S (backward)"  
echo "  🤏 L Shape      → A (left)"
echo "  🤙 Hang Loose   → D (right)"
echo "  👍 Thumbs Up    → Tab"
echo "  👆 Index Only   → Right Click"
echo "  🤏 Pinch Index  → 1"
echo "  🤏 Pinch Middle → 2" 
echo "  🤏 Pinch Ring   → 3"
echo "  🤏 Pinch Pinky  → 4"
echo ""
echo "Improvements:"
echo "  ✅ Enhanced pinch detection with 3D distance calculation"
echo "  ✅ Improved open palm detection to avoid collisions with pinch"
echo "  ✅ Visual debugging for distances between thumb and fingers"
echo "  ✅ Confidence scoring with pinch penalties"
echo ""
echo "Controls:"
echo "  Press 'p' to pause/resume"
echo "  Press 'q' to quit"
echo ""
echo "Make sure your Parallels Windows VM is focused!"
echo "Starting in 3 seconds..."
sleep 3

# Run the application with the unified implementation using uv
uv run --python .venv/bin/python src/main.py --config gaming_config.json --mode modular --debug
