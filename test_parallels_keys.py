#!/usr/bin/env python3
"""
Test script for Mac-to-Parallels key transmission
This tests how different pynput approaches work when sending keys from Mac to Windows VM.
"""

import time
import platform
from pynput.keyboard import Controller, KeyCode, Key

def test_parallels_keys():
    """Test different key approaches for Mac-to-Parallels scenario."""
    controller = Controller()
    system = platform.system()
    
    print(f"Running on {system} (pynput will use Darwin backend)")
    print("Make sure Parallels Windows VM is active and a text editor is focused")
    print("Press Enter to continue...")
    input()
    
    print("Starting test in 3 seconds...")
    time.sleep(3)
    
    print("=== Testing Different Approaches ===")
    
    # Approach 1: Direct string (original problematic approach)
    print("1. Testing direct string approach...")
    controller.type("Direct: ")
    for digit in ['1', '2', '3']:
        controller.press(digit)
        controller.release(digit)
        controller.type(" ")
        time.sleep(0.3)
    controller.press(Key.enter)
    controller.release(Key.enter)
    time.sleep(1)
    
    # Approach 2: KeyCode.from_char (current fix)
    print("2. Testing KeyCode.from_char approach...")
    controller.type("KeyCode: ")
    for digit in ['1', '2', '3']:
        key = KeyCode.from_char(digit)
        controller.press(key)
        controller.release(key)
        controller.type(" ")
        time.sleep(0.3)
    controller.press(Key.enter)
    controller.release(Key.enter)
    time.sleep(1)
    
    # Approach 3: Using controller.type() (alternative)
    print("3. Testing controller.type() approach...")
    controller.type("Type method: ")
    controller.type("1 2 3 ")
    controller.press(Key.enter)
    controller.release(Key.enter)
    time.sleep(1)
    
    # Approach 4: KeyCode with explicit virtual key codes (if needed)
    print("4. Testing KeyCode with virtual key approach...")
    controller.type("VK codes: ")
    
    # Try to create KeyCode objects that might work better with VMs
    try:
        # These are the typical virtual key codes for digits on Mac
        digit_vk_map = {
            '1': 0x12,  # kVK_ANSI_1
            '2': 0x13,  # kVK_ANSI_2  
            '3': 0x14   # kVK_ANSI_3
        }
        
        for digit in ['1', '2', '3']:
            if digit in digit_vk_map:
                key = KeyCode.from_vk(digit_vk_map[digit], char=digit)
                controller.press(key)
                controller.release(key)
                controller.type(" ")
            time.sleep(0.3)
    except Exception as e:
        controller.type(f"VK error: {e}")
    
    controller.press(Key.enter)
    controller.release(Key.enter)
    time.sleep(1)
    
    print("\n=== Test Complete ===")
    print("Check your Windows VM text editor. You should see:")
    print("Direct: 1 2 3")
    print("KeyCode: 1 2 3") 
    print("Type method: 1 2 3")
    print("VK codes: 1 2 3")
    print()
    print("If any approach doesn't work, that indicates the issue with Parallels key transmission.")

if __name__ == "__main__":
    test_parallels_keys()
