#!/bin/bash

# Hand to Key - Development Launcher
# This script launches the Hand to Key application with development and debugging features

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
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

print_dev() {
    echo -e "${PURPLE}[DEV]${NC} $1"
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

print_dev "Starting Hand to Key in Development Mode..."
print_dev "Configuration: Development settings with full debugging"
print_dev "Features enabled:"
echo "  • Threading: Enabled (for performance testing)"
echo "  • Adaptive Quality: Enabled (dynamic adjustment testing)"
echo "  • Hot Reload: Enabled (live config updates)"
echo "  • Verbose Logging: Enabled (detailed debug output)"
echo "  • System Tray: Disabled (easier debugging)"
echo "  • Full Debug Info: Enabled (all visualization options)"

print_warning "Development mode includes extensive logging - check console output!"
print_warning "Make sure your camera is connected and permissions are granted!"

# Set development-specific environment variables
export HAND_TO_KEY_DEBUG="1"
export HAND_TO_KEY_LOG_LEVEL="DEBUG"
export PYTHONUNBUFFERED="1"  # Ensure immediate output

print_dev "Development environment configured"
print_status "Launching application..."
print_status "Press Ctrl+C to stop the application"
print_status "Development controls:"
echo "  • 'h' - Show help and controls"
echo "  • 'l' - Toggle hand landmarks"
echo "  • 'j' - Toggle hand connections" 
echo "  • 'b' - Cycle hand preference (right/left/both)"
echo "  • 'd' - Toggle debug display mode"

echo ""
print_success "Starting Hand to Key with command: uv run src/main.py --threading --adaptive --hot-reload --verbose"
echo ""

# Run the application with development-optimized settings
uv run src/main.py --threading --adaptive --hot-reload --verbose

# If we get here, the application exited normally
print_success "Hand to Key development session closed."
