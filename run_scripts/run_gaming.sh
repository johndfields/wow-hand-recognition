#!/bin/bash

# Hand to Key - Gaming Mode Launcher
# This script launches the Hand to Key application with gaming-optimized settings

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "src/main.py" ]; then
    print_error "main.py not found. Please run this script from the hand_to_key directory."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install it first:"
    print_error "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

print_status "Starting Hand to Key in Gaming Mode..."
print_status "Configuration: Gaming profile with performance optimizations"
print_status "Features enabled:"
echo "  • Threading: Enabled (for better performance)"
echo "  • Adaptive Quality: Enabled (dynamic performance adjustment)"
echo "  • Hot Reload: Enabled (live config updates)"
echo "  • Gaming Profile: Will be loaded automatically"

print_warning "Make sure your camera is connected and permissions are granted!"

# Set gaming-specific environment variables
export HAND_TO_KEY_PROFILE="Gaming"
export HAND_TO_KEY_PERFORMANCE_MODE="1"

print_status "Launching application..."
print_status "Press Ctrl+C to stop the application"
print_status "Press 'h' in the application for help and controls"

echo ""
print_success "Starting Hand to Key with command: uv run src/main.py --threading --adaptive --hot-reload"
echo ""

# Run the application with gaming-optimized settings
uv run src/main.py --threading --adaptive --hot-reload

# If we get here, the application exited normally
print_success "Hand to Key application closed."
