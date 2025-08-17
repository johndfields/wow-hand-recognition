#!/usr/bin/env python3
"""
Test script for digit key insertion - cross platform test
This script tests whether digit keys (1,2,3) are properly inserted using the new KeyCode approach.
"""

import time
import platform
from pynput.keyboard import Controller, KeyCode

def test_digit_keys():
    """Test digit key insertion using different approaches."""
    controller = Controller()
    system = platform.system()
    
    print(f"Testing digit key insertion on {system}")
    print("Please open a text editor and focus on it, then press Enter to continue...")
    input("Press Enter to start test...")
    
    print("Testing in 3 seconds...")
    time.sleep(3)
    
    # Test using KeyCode.from_char (new approach)
    print("Testing KeyCode.from_char approach for digits 1, 2, 3...")
    
    for digit in ['1', '2', '3']:
        print(f"Typing {digit}...")
        key = KeyCode.from_char(digit)
        controller.press(key)
        controller.release(key)
        time.sleep(0.5)
        
        # Add a space for readability
        controller.press(' ')
        controller.release(' ')
        time.sleep(0.5)
    
    # Add a newline
    controller.press(KeyCode.from_char('\n'))
    controller.release(KeyCode.from_char('\n'))
    time.sleep(1)
    
    # Test using direct string approach (old approach for comparison)
    print("Testing direct string approach for digits 4, 5, 6...")
    
    for digit in ['4', '5', '6']:
        print(f"Typing {digit}...")
        controller.press(digit)
        controller.release(digit)
        time.sleep(0.5)
        
        # Add a space for readability
        controller.press(' ')
        controller.release(' ')
        time.sleep(0.5)
    
    print("Test completed. Check your text editor.")
    print("You should see: '1 2 3 \\n4 5 6 '")
    print(f"Both approaches should work on {system}, but KeyCode.from_char is more reliable on Windows.")

if __name__ == "__main__":
    test_digit_keys()
