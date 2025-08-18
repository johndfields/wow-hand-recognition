"""
Input handling module.
"""

from .handler import InputType, InputMode, InputAction, InputSequence, UnifiedInputHandler
from .platform_adapters import PlatformType, detect_platform, get_platform_adapter

