"""
Platform-specific input adapters for cross-platform compatibility.
This module provides specialized input handling for different operating systems
and virtualization environments like Parallels.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import platform
import logging
from pynput.keyboard import Controller as KeyController, Key, KeyCode
from pynput.mouse import Controller as MouseController, Button as MouseButton

logger = logging.getLogger(__name__)


class PlatformType(Enum):
    """Supported platform types."""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    MAC_PARALLELS_VM = "mac_parallels_vm"
    UNKNOWN = "unknown"


def detect_platform() -> PlatformType:
    """Detect the current platform and environment."""
    system = platform.system().lower()
    
    if system == "darwin":
        # Check if we're running in a context where we need to send keys to Parallels VM
        # This is a heuristic and might need to be adjusted based on actual usage
        try:
            import subprocess
            result = subprocess.run(
                ["ps", "-ef"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            if "Parallels Desktop" in result.stdout and "Windows" in result.stdout:
                return PlatformType.MAC_PARALLELS_VM
        except Exception:
            pass
        return PlatformType.MACOS
    elif system == "windows":
        return PlatformType.WINDOWS
    elif system == "linux":
        return PlatformType.LINUX
    else:
        return PlatformType.UNKNOWN


class PlatformAdapter:
    """Base class for platform-specific input adapters."""
    
    def __init__(self):
        self.platform_type = detect_platform()
        self.keyboard = KeyController()
        self.mouse = MouseController()
    
    def parse_key(self, key_str: Union[str, Key]) -> Optional[Union[Key, KeyCode]]:
        """Parse a key string to a Key object or KeyCode."""
        if isinstance(key_str, Key):
            return key_str
        
        key_str_original = str(key_str)
        key_str_lower = key_str_original.lower()
        
        # Common key mapping for all platforms
        key_mapping = {
            'space': Key.space, 'enter': Key.enter, 'return': Key.enter,
            'tab': Key.tab, 'esc': Key.esc, 'escape': Key.esc,
            'up': Key.up, 'down': Key.down, 'left': Key.left, 'right': Key.right,
            'backspace': Key.backspace, 'delete': Key.delete,
            'home': Key.home, 'end': Key.end, 'page_up': Key.page_up, 'page_down': Key.page_down,
            'shift': Key.shift, 'ctrl': Key.ctrl, 'control': Key.ctrl,
            'alt': Key.alt, 'cmd': Key.cmd, 'win': Key.cmd, 'windows': Key.cmd,
        }
        
        # Check special keys
        if key_str_lower in key_mapping:
            return key_mapping[key_str_lower]
        
        # Single character - use KeyCode for better cross-platform compatibility
        if len(key_str_original) == 1:
            # For digits, use platform-specific handling
            if key_str_original.isdigit():
                return self.create_digit_key(key_str_original)
            else:
                # For letters and other characters, use KeyCode.from_char
                return KeyCode.from_char(key_str_original)
        
        # Function keys
        if key_str_lower.startswith('f') and key_str_lower[1:].isdigit():
            try:
                return getattr(Key, key_str_lower)
            except AttributeError:
                pass
        
        logger.warning(f"Unknown key: {key_str_original}")
        return None
    
    def create_digit_key(self, digit: str) -> KeyCode:
        """Create a digit key (platform-specific implementation)."""
        return KeyCode.from_char(digit)
    
    def parse_combo(self, combo_str: str) -> List[Union[Key, KeyCode]]:
        """Parse a key combination string (e.g., 'ctrl+c')."""
        if not combo_str or '+' not in combo_str:
            return []
        
        parts = combo_str.split('+')
        keys = []
        
        for part in parts:
            key = self.parse_key(part.strip())
            if key:
                keys.append(key)
        
        return keys


class WindowsAdapter(PlatformAdapter):
    """Windows-specific input adapter."""
    
    def __init__(self):
        super().__init__()
        
    def create_digit_key(self, digit: str) -> KeyCode:
        """Create a digit key optimized for Windows."""
        # Windows handles digits well with standard KeyCode
        return KeyCode.from_char(digit)


class MacOSAdapter(PlatformAdapter):
    """macOS-specific input adapter."""
    
    def __init__(self):
        super().__init__()
        
    def create_digit_key(self, digit: str) -> KeyCode:
        """Create a digit key optimized for macOS."""
        # macOS handles digits well with standard KeyCode
        return KeyCode.from_char(digit)


class LinuxAdapter(PlatformAdapter):
    """Linux-specific input adapter."""
    
    def __init__(self):
        super().__init__()
        
    def create_digit_key(self, digit: str) -> KeyCode:
        """Create a digit key optimized for Linux."""
        # Linux handles digits well with standard KeyCode
        return KeyCode.from_char(digit)


class ParallelsVMAdapter(PlatformAdapter):
    """Specialized adapter for Mac-to-Parallels Windows VM key transmission."""
    
    def __init__(self):
        super().__init__()
        
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
    
    def create_digit_key(self, digit: str) -> KeyCode:
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


def get_platform_adapter() -> PlatformAdapter:
    """Factory function to get the appropriate platform adapter."""
    platform_type = detect_platform()
    
    if platform_type == PlatformType.WINDOWS:
        return WindowsAdapter()
    elif platform_type == PlatformType.MACOS:
        return MacOSAdapter()
    elif platform_type == PlatformType.LINUX:
        return LinuxAdapter()
    elif platform_type == PlatformType.MAC_PARALLELS_VM:
        return ParallelsVMAdapter()
    else:
        # Default to a basic adapter
        return PlatformAdapter()

