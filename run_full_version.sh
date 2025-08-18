#!/bin/bash

# Hand-to-Key Full Version with Improved Collision Detection
# Runs the advanced hand gesture control with improved collision detection between pinch and open palm gestures

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the hand-to-key directory
cd "$SCRIPT_DIR"

echo "🖐️ Starting Hand-to-Key FULL VERSION with Improved Collision Detection..."
echo "📁 Working directory: $PWD"
echo ""
echo "✨ Advanced Features Available:"
echo "  📊 Profile System - Multiple profiles for different applications"
echo "  🔄 Hot-Reload Configuration - Change settings without restarting"
echo "  📈 Statistics Tracking - Performance metrics and success rates"
echo "  🎮 Enhanced Gesture Detection - Improved collision handling"
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
echo "  ✅ Enhanced collision detection between pinch and open palm gestures"
echo "  ✅ Gesture priority system where pinch gestures take precedence"
echo "  ✅ Temporal filtering for gesture stability"
echo "  ✅ Fixed platform compatibility issues for macOS"
echo ""
echo "Controls:"
echo "  Press 'p' to pause/resume"
echo "  Press 'q' to quit"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Run the application with uv
uv run --python .venv/bin/python src/main.py
