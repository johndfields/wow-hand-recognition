#!/usr/bin/env python3
"""
Debug script to test the enhanced digit key handling in the simple application
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hand_to_key_simple import InputController
from pynput.keyboard import KeyCode
import platform

def test_digit_key_generation():
    """Test how the InputController creates digit keys"""
    controller = InputController()
    
    print(f"Testing on {platform.system()}")
    print("=" * 50)
    
    for digit in ['1', '2', '3', '4']:
        # Test the _create_digit_key method directly
        key = controller._create_digit_key(digit)
        print(f"Digit '{digit}' -> {key}")
        print(f"  Type: {type(key)}")
        if hasattr(key, 'vk'):
            print(f"  VK: {key.vk}")
        if hasattr(key, 'char'):
            print(f"  Char: {key.char}")
        print()
    
    print("Testing through _parse_key method:")
    print("-" * 30)
    
    for digit in ['1', '2', '3', '4']:
        key = controller._parse_key(digit)
        print(f"parse_key('{digit}') -> {key}")
        print(f"  Type: {type(key)}")
        if hasattr(key, 'vk'):
            print(f"  VK: {key.vk}")
        if hasattr(key, 'char'):
            print(f"  Char: {key.char}")
        print()

if __name__ == "__main__":
    test_digit_key_generation()
