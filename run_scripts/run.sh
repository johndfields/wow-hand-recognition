#!/bin/bash

# Hand to Key - Basic Launcher
# This script launches the Hand to Key application with standard settings

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

print_status "Starting Hand to Key..."
print_status "Configuration: Default settings"

print_warning "Make sure your camera is connected and permissions are granted!"

print_status "Launching application..."
print_status "Press Ctrl+C to stop the application"
print_status "Press 'h' in the application for help and controls"

echo ""
print_success "Starting Hand to Key with command: uv run src/main.py"
echo ""

# Run the application with standard settings
uv run src/main.py

# If we get here, the application exited normally
print_success "Hand to Key application closed."
