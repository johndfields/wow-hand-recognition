#!/usr/bin/env python3
"""
Parallels-optimized key handler for Mac-to-Windows VM key transmission.
This addresses specific issues that can occur when using pynput on Mac to send keys to Windows VMs.
"""

from typing import Optional, Union
import logging
from pynput.keyboard import Controller, KeyCode, Key

logger = logging.getLogger(__name__)

class ParallelsKeyHandler:
    """Enhanced key handler optimized for Mac-to-Parallels Windows VM communication."""
    
    def __init__(self):
        self.controller = Controller()
        
        # Mac virtual key codes for digits (for better VM compatibility)
        self.mac_digit_vk_codes = {
            '0': 0x1D,  # kVK_ANSI_0
            '1': 0x12,  # kVK_ANSI_1
            '2': 0x13,  # kVK_ANSI_2
            '3': 0x14,  # kVK_ANSI_3
            '4': 0x15,  # kVK_ANSI_4
            '5': 0x17,  # kVK_ANSI_5
            '6': 0x16,  # kVK_ANSI_6
            '7': 0x1A,  # kVK_ANSI_7
            '8': 0x1C,  # kVK_ANSI_8
            '9': 0x19   # kVK_ANSI_9
        }
    
    def parse_key_for_parallels(self, key_str: Union[str, Key]) -> Optional[Union[Key, KeyCode]]:
        """Parse a key string with Parallels VM optimization."""
        if isinstance(key_str, Key):
            return key_str
        
        key_str_original = str(key_str)
        key_str_lower = key_str_original.lower()
        
        # Special handling for digits with Parallels VMs
        if len(key_str_original) == 1 and key_str_original.isdigit():
            return self._create_digit_key(key_str_original)
        
        # Handle special keys
        key_mapping = {
            'space': Key.space, 'enter': Key.enter, 'return': Key.enter,
            'tab': Key.tab, 'esc': Key.esc, 'escape': Key.esc,
            'up': Key.up, 'down': Key.down, 'left': Key.left, 'right': Key.right,
            'backspace': Key.backspace, 'delete': Key.delete,
            'home': Key.home, 'end': Key.end,
            'shift': Key.shift, 'ctrl': Key.ctrl, 'control': Key.ctrl,
            'alt': Key.alt, 'cmd': Key.cmd, 'win': Key.cmd, 'windows': Key.cmd,
        }
        
        if key_str_lower in key_mapping:
            return key_mapping[key_str_lower]
        
        # Handle single characters (letters, symbols)
        if len(key_str_original) == 1:
            return KeyCode.from_char(key_str_original)
        
        # Function keys
        if key_str_lower.startswith('f') and key_str_lower[1:].isdigit():
            try:
                return getattr(Key, key_str_lower)
            except AttributeError:
                pass
        
        logger.warning(f"Unknown key: {key_str_original}")
        return None
    
    def _create_digit_key(self, digit: str) -> KeyCode:
        """Create a digit key optimized for Parallels transmission."""
        # Try multiple approaches for best Parallels compatibility
        
        # Approach 1: Virtual key code + character (most robust for VMs)
        if digit in self.mac_digit_vk_codes:
            try:
                return KeyCode.from_vk(self.mac_digit_vk_codes[digit], char=digit)
            except Exception as e:
                logger.debug(f"VK approach failed for {digit}: {e}")
        
        # Approach 2: Character-based KeyCode (fallback)
        return KeyCode.from_char(digit)
    
    def send_key(self, key_str: str, mode: str = "tap") -> bool:
        """Send a key with Parallels-optimized handling."""
        key = self.parse_key_for_parallels(key_str)
        if not key:
            return False
        
        try:
            if mode == "tap":
                self.controller.press(key)
                self.controller.release(key)
            elif mode == "press":
                self.controller.press(key)
            elif mode == "release":
                self.controller.release(key)
            return True
        except Exception as e:
            logger.error(f"Error sending key {key_str}: {e}")
            return False
    
    def type_text(self, text: str) -> bool:
        """Type text with Parallels optimization."""
        try:
            # For digits, use individual key presses for better reliability
            if text.isdigit() and len(text) == 1:
                return self.send_key(text, "tap")
            else:
                # Use controller.type for other text
                self.controller.type(text)
                return True
        except Exception as e:
            logger.error(f"Error typing text '{text}': {e}")
            return False

def test_parallels_handler():
    """Test the Parallels-optimized handler."""
    handler = ParallelsKeyHandler()
    
    print("Testing Parallels-optimized key handler...")
    print("Make sure Parallels Windows VM is focused and a text editor is open.")
    input("Press Enter to continue...")
    
    import time
    time.sleep(2)
    
    # Test digit keys
    handler.type_text("Digits: ")
    for digit in "123":
        handler.send_key(digit, "tap")
        handler.type_text(" ")
        time.sleep(0.2)
    
    handler.send_key("enter", "tap")
    time.sleep(0.5)
    
    # Test letters
    handler.type_text("Letters: ")
    for letter in "abc":
        handler.send_key(letter, "tap") 
        handler.type_text(" ")
        time.sleep(0.2)
    
    print("Test complete. Check Windows VM for output.")

if __name__ == "__main__":
    test_parallels_handler()
