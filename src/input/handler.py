"""
Enhanced input handler module with support for keyboard, mouse, and gamepad inputs.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import threading
import queue
from abc import ABC, abstractmethod
import platform

from pynput.keyboard import Controller as KeyController, Key, Listener as KeyListener
from pynput.mouse import Controller as MouseController, Button as MouseButton, Listener as MouseListener

logger = logging.getLogger(__name__)


class InputType(Enum):
    """Types of input actions."""
    KEY_PRESS = "key_press"
    KEY_HOLD = "key_hold"
    KEY_RELEASE = "key_release"
    MOUSE_CLICK = "mouse_click"
    MOUSE_HOLD = "mouse_hold"
    MOUSE_RELEASE = "mouse_release"
    MOUSE_MOVE = "mouse_move"
    MOUSE_SCROLL = "mouse_scroll"
    MOUSE_DRAG = "mouse_drag"
    GAMEPAD_BUTTON = "gamepad_button"
    GAMEPAD_AXIS = "gamepad_axis"
    MACRO = "macro"
    COMBO = "combo"


class InputMode(Enum):
    """Input trigger modes."""
    TAP = "tap"
    HOLD = "hold"
    TOGGLE = "toggle"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"


@dataclass
class InputAction:
    """Represents an input action to be executed."""
    input_type: InputType
    target: Union[str, Key, MouseButton, Tuple[float, float]]
    mode: InputMode = InputMode.TAP
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class InputSequence:
    """Represents a sequence of input actions (macro)."""
    name: str
    actions: List[InputAction]
    repeat_count: int = 1
    delay_between: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class InputHandlerBase(ABC):
    """Abstract base class for input handlers."""
    
    @abstractmethod
    def execute(self, action: InputAction) -> bool:
        """Execute an input action."""
        pass
    
    @abstractmethod
    def execute_sequence(self, sequence: InputSequence) -> bool:
        """Execute a sequence of actions."""
        pass
    
    @abstractmethod
    def stop_all(self):
        """Stop all ongoing inputs."""
        pass


class KeyboardHandler(InputHandlerBase):
    """Handles keyboard input actions with advanced features."""
    
    def __init__(self):
        self.controller = KeyController()
        self.held_keys: Dict[str, bool] = {}
        self.toggle_states: Dict[str, bool] = {}
        self.combo_keys: Dict[str, List[Key]] = {}
        self.key_mapping = self._create_key_mapping()
        self.lock = threading.Lock()
        
        # Macro recording
        self.is_recording = False
        self.recorded_actions: List[InputAction] = []
        self.record_start_time = 0.0
        
    def execute(self, action: InputAction) -> bool:
        """Execute a keyboard action."""
        try:
            with self.lock:
                if action.input_type == InputType.KEY_PRESS:
                    return self._handle_key_press(action)
                elif action.input_type == InputType.KEY_HOLD:
                    return self._handle_key_hold(action)
                elif action.input_type == InputType.KEY_RELEASE:
                    return self._handle_key_release(action)
                elif action.input_type == InputType.COMBO:
                    return self._handle_combo(action)
                return False
        except Exception as e:
            logger.error(f"Error executing keyboard action: {e}")
            return False
    
    def execute_sequence(self, sequence: InputSequence) -> bool:
        """Execute a sequence of keyboard actions."""
        try:
            for _ in range(sequence.repeat_count):
                for action in sequence.actions:
                    if action.input_type in [InputType.KEY_PRESS, InputType.KEY_HOLD, 
                                            InputType.KEY_RELEASE, InputType.COMBO]:
                        self.execute(action)
                        if sequence.delay_between > 0:
                            time.sleep(sequence.delay_between)
            return True
        except Exception as e:
            logger.error(f"Error executing keyboard sequence: {e}")
            return False
    
    def stop_all(self):
        """Release all held keys."""
        with self.lock:
            for key in list(self.held_keys.keys()):
                if self.held_keys[key]:
                    parsed_key = self._parse_key(key)
                    if parsed_key:
                        self.controller.release(parsed_key)
                        self.held_keys[key] = False
            self.toggle_states.clear()
    
    def _handle_key_press(self, action: InputAction) -> bool:
        """Handle a single key press."""
        key = self._parse_key(action.target)
        if not key:
            return False
        
        if action.mode == InputMode.TAP:
            self.controller.press(key)
            if action.duration > 0:
                time.sleep(action.duration)
            self.controller.release(key)
        elif action.mode == InputMode.DOUBLE_TAP:
            for _ in range(2):
                self.controller.press(key)
                self.controller.release(key)
                time.sleep(0.05)
        elif action.mode == InputMode.LONG_PRESS:
            self.controller.press(key)
            time.sleep(action.duration or 0.5)
            self.controller.release(key)
        elif action.mode == InputMode.TOGGLE:
            key_str = str(action.target)
            if key_str not in self.toggle_states:
                self.toggle_states[key_str] = False
            
            if self.toggle_states[key_str]:
                self.controller.release(key)
                self.toggle_states[key_str] = False
            else:
                self.controller.press(key)
                self.toggle_states[key_str] = True
        
        return True
    
    def _handle_key_hold(self, action: InputAction) -> bool:
        """Handle holding a key down."""
        key = self._parse_key(action.target)
        if not key:
            return False
        
        key_str = str(action.target)
        if key_str not in self.held_keys or not self.held_keys[key_str]:
            self.controller.press(key)
            self.held_keys[key_str] = True
        
        return True
    
    def _handle_key_release(self, action: InputAction) -> bool:
        """Handle releasing a held key."""
        key = self._parse_key(action.target)
        if not key:
            return False
        
        key_str = str(action.target)
        if key_str in self.held_keys and self.held_keys[key_str]:
            self.controller.release(key)
            self.held_keys[key_str] = False
        
        return True
    
    def _handle_combo(self, action: InputAction) -> bool:
        """Handle key combinations (e.g., Ctrl+C)."""
        combo_str = str(action.target)
        keys = self._parse_combo(combo_str)
        
        if not keys:
            return False
        
        # Press modifier keys
        for key in keys[:-1]:
            self.controller.press(key)
        
        # Press and release final key
        self.controller.press(keys[-1])
        self.controller.release(keys[-1])
        
        # Release modifier keys in reverse order
        for key in reversed(keys[:-1]):
            self.controller.release(key)
        
        return True
    
    def _parse_key(self, key_str: Union[str, Key]) -> Optional[Union[Key, 'KeyCode']]:
        """Parse a key string to a Key object or KeyCode."""
        if isinstance(key_str, Key):
            return key_str
        
        key_str_original = str(key_str)
        key_str_lower = key_str_original.lower()
        
        # Check special keys
        if key_str_lower in self.key_mapping:
            return self.key_mapping[key_str_lower]
        
        # Single character - use KeyCode for better cross-platform compatibility
        if len(key_str_original) == 1:
            # For digits, use enhanced handling for VM compatibility (e.g., Parallels)
            if key_str_original.isdigit():
                return self._create_digit_key(key_str_original)
            else:
                # For letters and other characters, use KeyCode.from_char
                from pynput.keyboard import KeyCode
                return KeyCode.from_char(key_str_original)
        
        # Function keys
        if key_str_lower.startswith('f') and key_str_lower[1:].isdigit():
            try:
                return getattr(Key, key_str_lower)
            except AttributeError:
                pass
        
        return None
    
    def _create_digit_key(self, digit: str):
        """Create a digit key with enhanced VM compatibility (e.g., for Parallels)."""
        from pynput.keyboard import KeyCode
        import platform
        
        # On Mac (which might be sending to VMs like Parallels), use virtual key codes for digits
        if platform.system() == 'Darwin':
            mac_digit_vk_codes = {
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
            
            if digit in mac_digit_vk_codes:
                try:
                    # Create KeyCode with both virtual key code and character for best compatibility
                    return KeyCode.from_vk(mac_digit_vk_codes[digit], char=digit)
                except Exception as e:
                    logger.debug(f"VK approach failed for digit {digit}: {e}")
        
        # Fallback to character-based KeyCode
        return KeyCode.from_char(digit)
    
    def _parse_combo(self, combo_str: str) -> List[Key]:
        """Parse a key combination string."""
        parts = [p.strip().lower() for p in combo_str.split('+')]
        keys = []
        
        for part in parts:
            key = self._parse_key(part)
            if key:
                keys.append(key)
        
        return keys
    
    def _create_key_mapping(self) -> Dict[str, Key]:
        """Create mapping of string names to Key objects."""
        mapping = {
            'space': Key.space, 'enter': Key.enter, 'return': Key.enter,
            'tab': Key.tab, 'esc': Key.esc, 'escape': Key.esc,
            'up': Key.up, 'down': Key.down, 'left': Key.left, 'right': Key.right,
            'backspace': Key.backspace, 'delete': Key.delete,
            'home': Key.home, 'end': Key.end, 
            'page_up': Key.page_up, 'pageup': Key.page_up,
            'page_down': Key.page_down, 'pagedown': Key.page_down,
            'shift': Key.shift, 'ctrl': Key.ctrl, 'control': Key.ctrl,
            'alt': Key.alt, 'cmd': Key.cmd, 'win': Key.cmd, 'windows': Key.cmd,
            'caps_lock': Key.caps_lock, 'capslock': Key.caps_lock
        }
        
        # Add menu key if available (not available on macOS)
        try:
            mapping['menu'] = Key.menu
        except AttributeError:
            logger.debug("Key 'menu' not available on this platform, skipping")
        
        # Add platform-specific keys if they exist
        # These keys may not be available on all platforms (especially macOS)
        platform_specific_keys = [
            ('pause', 'pause'),
            ('insert', 'insert'),
            ('print_screen', 'print_screen'),
            ('printscreen', 'print_screen'),
            ('num_lock', 'num_lock'),
            ('numlock', 'num_lock'),
            ('scroll_lock', 'scroll_lock'),
            ('scrolllock', 'scroll_lock')
        ]
        
        for key_name, attr_name in platform_specific_keys:
            try:
                mapping[key_name] = getattr(Key, attr_name)
            except AttributeError:
                logger.debug(f"Key '{attr_name}' not available on this platform, skipping")
                pass
            
        return mapping
    
    def start_recording(self):
        """Start recording keyboard actions for macro creation."""
        self.is_recording = True
        self.recorded_actions = []
        self.record_start_time = time.time()
        
        # Start keyboard listener
        self.keyboard_listener = KeyListener(
            on_press=self._on_key_press_record,
            on_release=self._on_key_release_record
        )
        self.keyboard_listener.start()
        
        logger.info("Started recording keyboard actions")
    
    def stop_recording(self) -> InputSequence:
        """Stop recording and return the recorded sequence."""
        self.is_recording = False
        
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
        
        sequence = InputSequence(
            name=f"recorded_{int(time.time())}",
            actions=self.recorded_actions.copy()
        )
        
        self.recorded_actions = []
        logger.info(f"Stopped recording, captured {len(sequence.actions)} actions")
        
        return sequence
    
    def _on_key_press_record(self, key):
        """Callback for recording key press."""
        if self.is_recording:
            action = InputAction(
                input_type=InputType.KEY_PRESS,
                target=str(key),
                mode=InputMode.TAP,
                timestamp=time.time() - self.record_start_time
            )
            self.recorded_actions.append(action)
    
    def _on_key_release_record(self, key):
        """Callback for recording key release."""
        # Only record if it's a hold action
        pass


class MouseHandler(InputHandlerBase):
    """Handles mouse input actions with advanced features."""
    
    def __init__(self):
        self.controller = MouseController()
        self.held_buttons: Dict[MouseButton, bool] = {}
        self.drag_start: Optional[Tuple[float, float]] = None
        self.lock = threading.Lock()
        
        # Mouse recording
        self.is_recording = False
        self.recorded_actions: List[InputAction] = []
        self.record_start_time = 0.0
    
    def execute(self, action: InputAction) -> bool:
        """Execute a mouse action."""
        try:
            with self.lock:
                if action.input_type == InputType.MOUSE_CLICK:
                    return self._handle_mouse_click(action)
                elif action.input_type == InputType.MOUSE_HOLD:
                    return self._handle_mouse_hold(action)
                elif action.input_type == InputType.MOUSE_RELEASE:
                    return self._handle_mouse_release(action)
                elif action.input_type == InputType.MOUSE_MOVE:
                    return self._handle_mouse_move(action)
                elif action.input_type == InputType.MOUSE_SCROLL:
                    return self._handle_mouse_scroll(action)
                elif action.input_type == InputType.MOUSE_DRAG:
                    return self._handle_mouse_drag(action)
                return False
        except Exception as e:
            logger.error(f"Error executing mouse action: {e}")
            return False
    
    def execute_sequence(self, sequence: InputSequence) -> bool:
        """Execute a sequence of mouse actions."""
        try:
            for _ in range(sequence.repeat_count):
                for action in sequence.actions:
                    if action.input_type in [InputType.MOUSE_CLICK, InputType.MOUSE_HOLD,
                                            InputType.MOUSE_RELEASE, InputType.MOUSE_MOVE,
                                            InputType.MOUSE_SCROLL, InputType.MOUSE_DRAG]:
                        self.execute(action)
                        if sequence.delay_between > 0:
                            time.sleep(sequence.delay_between)
            return True
        except Exception as e:
            logger.error(f"Error executing mouse sequence: {e}")
            return False
    
    def stop_all(self):
        """Release all held mouse buttons."""
        with self.lock:
            for button in list(self.held_buttons.keys()):
                if self.held_buttons[button]:
                    self.controller.release(button)
                    self.held_buttons[button] = False
            self.drag_start = None
    
    def _handle_mouse_click(self, action: InputAction) -> bool:
        """Handle mouse click."""
        button = self._parse_button(action.target)
        if not button:
            return False
        
        if action.mode == InputMode.TAP:
            self.controller.click(button, 1)
        elif action.mode == InputMode.DOUBLE_TAP:
            self.controller.click(button, 2)
        elif action.mode == InputMode.LONG_PRESS:
            self.controller.press(button)
            time.sleep(action.duration or 0.5)
            self.controller.release(button)
        elif action.mode == InputMode.TOGGLE:
            if button not in self.held_buttons:
                self.held_buttons[button] = False
            
            if self.held_buttons[button]:
                self.controller.release(button)
                self.held_buttons[button] = False
            else:
                self.controller.press(button)
                self.held_buttons[button] = True
        
        return True
    
    def _handle_mouse_hold(self, action: InputAction) -> bool:
        """Handle holding mouse button."""
        button = self._parse_button(action.target)
        if not button:
            return False
        
        if button not in self.held_buttons or not self.held_buttons[button]:
            self.controller.press(button)
            self.held_buttons[button] = True
        
        return True
    
    def _handle_mouse_release(self, action: InputAction) -> bool:
        """Handle releasing mouse button."""
        button = self._parse_button(action.target)
        if not button:
            return False
        
        if button in self.held_buttons and self.held_buttons[button]:
            self.controller.release(button)
            self.held_buttons[button] = False
        
        return True
    
    def _handle_mouse_move(self, action: InputAction) -> bool:
        """Handle mouse movement."""
        if isinstance(action.target, tuple) and len(action.target) == 2:
            x, y = action.target
            
            if action.metadata.get('relative', False):
                # Relative movement
                current_x, current_y = self.controller.position
                self.controller.position = (current_x + x, current_y + y)
            else:
                # Absolute movement
                self.controller.position = (x, y)
            
            return True
        return False
    
    def _handle_mouse_scroll(self, action: InputAction) -> bool:
        """Handle mouse scrolling."""
        if isinstance(action.target, tuple) and len(action.target) == 2:
            dx, dy = action.target
            self.controller.scroll(dx, dy)
            return True
        return False
    
    def _handle_mouse_drag(self, action: InputAction) -> bool:
        """Handle mouse dragging."""
        if isinstance(action.target, tuple) and len(action.target) == 2:
            if self.drag_start is None:
                # Start drag
                self.drag_start = self.controller.position
                self.controller.press(MouseButton.left)
            
            # Move to target
            x, y = action.target
            self.controller.position = (x, y)
            
            if action.metadata.get('end_drag', False):
                # End drag
                self.controller.release(MouseButton.left)
                self.drag_start = None
            
            return True
        return False
    
    def _parse_button(self, button_str: Union[str, MouseButton]) -> Optional[MouseButton]:
        """Parse a button string to MouseButton."""
        if isinstance(button_str, MouseButton):
            return button_str
        
        button_str = str(button_str).lower()
        
        button_map = {
            'left': MouseButton.left,
            'left_click': MouseButton.left,
            'right': MouseButton.right,
            'right_click': MouseButton.right,
            'middle': MouseButton.middle,
            'middle_click': MouseButton.middle,
            'button1': MouseButton.left,
            'button2': MouseButton.right,
            'button3': MouseButton.middle
        }
        
        return button_map.get(button_str)
    
    def get_position(self) -> Tuple[float, float]:
        """Get current mouse position."""
        return self.controller.position
    
    def start_recording(self):
        """Start recording mouse actions."""
        self.is_recording = True
        self.recorded_actions = []
        self.record_start_time = time.time()
        
        # Start mouse listener
        self.mouse_listener = MouseListener(
            on_click=self._on_click_record,
            on_move=self._on_move_record,
            on_scroll=self._on_scroll_record
        )
        self.mouse_listener.start()
        
        logger.info("Started recording mouse actions")
    
    def stop_recording(self) -> InputSequence:
        """Stop recording and return the recorded sequence."""
        self.is_recording = False
        
        if hasattr(self, 'mouse_listener'):
            self.mouse_listener.stop()
        
        sequence = InputSequence(
            name=f"mouse_recorded_{int(time.time())}",
            actions=self.recorded_actions.copy()
        )
        
        self.recorded_actions = []
        logger.info(f"Stopped recording, captured {len(sequence.actions)} mouse actions")
        
        return sequence
    
    def _on_click_record(self, x, y, button, pressed):
        """Callback for recording mouse clicks."""
        if self.is_recording:
            action = InputAction(
                input_type=InputType.MOUSE_CLICK if pressed else InputType.MOUSE_RELEASE,
                target=button,
                mode=InputMode.TAP,
                metadata={'position': (x, y)},
                timestamp=time.time() - self.record_start_time
            )
            self.recorded_actions.append(action)
    
    def _on_move_record(self, x, y):
        """Callback for recording mouse movement."""
        if self.is_recording and len(self.recorded_actions) > 0:
            # Only record significant movements
            last_action = self.recorded_actions[-1]
            if last_action.input_type == InputType.MOUSE_MOVE:
                last_pos = last_action.target
                if abs(x - last_pos[0]) < 5 and abs(y - last_pos[1]) < 5:
                    return
            
            action = InputAction(
                input_type=InputType.MOUSE_MOVE,
                target=(x, y),
                timestamp=time.time() - self.record_start_time
            )
            self.recorded_actions.append(action)
    
    def _on_scroll_record(self, x, y, dx, dy):
        """Callback for recording mouse scroll."""
        if self.is_recording:
            action = InputAction(
                input_type=InputType.MOUSE_SCROLL,
                target=(dx, dy),
                metadata={'position': (x, y)},
                timestamp=time.time() - self.record_start_time
            )
            self.recorded_actions.append(action)


class GamepadHandler(InputHandlerBase):
    """Handles gamepad/controller input with virtual gamepad support."""
    
    def __init__(self):
        self.virtual_buttons: Dict[str, bool] = {}
        self.virtual_axes: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.gamepad = None  # Always initialize gamepad attribute
        
        # Try to import gamepad library if available
        self.gamepad_available = False
        try:
            import vgamepad as vg
            self.vg = vg
            self.gamepad_available = True
            logger.info("Virtual gamepad support available")
        except ImportError:
            logger.warning("vgamepad not installed, gamepad emulation disabled")
    
    def initialize_gamepad(self):
        """Initialize virtual gamepad."""
        if self.gamepad_available and not self.gamepad:
            try:
                self.gamepad = self.vg.VX360Gamepad()
                logger.info("Virtual Xbox 360 gamepad initialized")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize virtual gamepad: {e}")
                return False
        return False
    
    def execute(self, action: InputAction) -> bool:
        """Execute a gamepad action."""
        if not self.gamepad_available:
            logger.warning("Gamepad support not available")
            return False
        
        if not self.gamepad:
            self.initialize_gamepad()
        
        try:
            with self.lock:
                if action.input_type == InputType.GAMEPAD_BUTTON:
                    return self._handle_button(action)
                elif action.input_type == InputType.GAMEPAD_AXIS:
                    return self._handle_axis(action)
                return False
        except Exception as e:
            logger.error(f"Error executing gamepad action: {e}")
            return False
    
    def execute_sequence(self, sequence: InputSequence) -> bool:
        """Execute a sequence of gamepad actions."""
        try:
            for _ in range(sequence.repeat_count):
                for action in sequence.actions:
                    if action.input_type in [InputType.GAMEPAD_BUTTON, InputType.GAMEPAD_AXIS]:
                        self.execute(action)
                        if sequence.delay_between > 0:
                            time.sleep(sequence.delay_between)
            return True
        except Exception as e:
            logger.error(f"Error executing gamepad sequence: {e}")
            return False
    
    def stop_all(self):
        """Reset all gamepad inputs."""
        with self.lock:
            if self.gamepad:
                try:
                    self.gamepad.reset()
                    self.gamepad.update()
                except:
                    pass
            
            self.virtual_buttons.clear()
            self.virtual_axes.clear()
    
    def _handle_button(self, action: InputAction) -> bool:
        """Handle gamepad button press."""
        if not self.gamepad:
            return False
        
        button_name = str(action.target).upper()
        
        # Map common button names to vgamepad constants
        button_map = {
            'A': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
            'B': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
            'X': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
            'Y': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
            'LB': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
            'RB': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
            'START': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
            'BACK': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
            'LEFT_THUMB': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
            'RIGHT_THUMB': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
            'DPAD_UP': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
            'DPAD_DOWN': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
            'DPAD_LEFT': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
            'DPAD_RIGHT': self.vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
        }
        
        button = button_map.get(button_name)
        if not button:
            return False
        
        if action.mode == InputMode.TAP:
            self.gamepad.press_button(button)
            self.gamepad.update()
            time.sleep(0.05)
            self.gamepad.release_button(button)
            self.gamepad.update()
        elif action.mode == InputMode.HOLD:
            self.gamepad.press_button(button)
            self.gamepad.update()
            self.virtual_buttons[button_name] = True
        elif action.mode == InputMode.TOGGLE:
            if button_name in self.virtual_buttons and self.virtual_buttons[button_name]:
                self.gamepad.release_button(button)
                self.virtual_buttons[button_name] = False
            else:
                self.gamepad.press_button(button)
                self.virtual_buttons[button_name] = True
            self.gamepad.update()
        
        return True
    
    def _handle_axis(self, action: InputAction) -> bool:
        """Handle gamepad axis movement."""
        if not self.gamepad:
            return False
        
        if isinstance(action.target, tuple) and len(action.target) == 2:
            axis_name, value = action.target
            axis_name = str(axis_name).upper()
            
            # Clamp value between -1 and 1
            value = max(-1.0, min(1.0, float(value)))
            
            if axis_name in ['LEFT_X', 'LX']:
                self.gamepad.left_joystick_float(x_value_float=value, 
                                                 y_value_float=self.virtual_axes.get('LEFT_Y', 0))
                self.virtual_axes['LEFT_X'] = value
            elif axis_name in ['LEFT_Y', 'LY']:
                self.gamepad.left_joystick_float(x_value_float=self.virtual_axes.get('LEFT_X', 0),
                                                 y_value_float=value)
                self.virtual_axes['LEFT_Y'] = value
            elif axis_name in ['RIGHT_X', 'RX']:
                self.gamepad.right_joystick_float(x_value_float=value,
                                                  y_value_float=self.virtual_axes.get('RIGHT_Y', 0))
                self.virtual_axes['RIGHT_X'] = value
            elif axis_name in ['RIGHT_Y', 'RY']:
                self.gamepad.right_joystick_float(x_value_float=self.virtual_axes.get('RIGHT_X', 0),
                                                  y_value_float=value)
                self.virtual_axes['RIGHT_Y'] = value
            elif axis_name in ['LEFT_TRIGGER', 'LT']:
                self.gamepad.left_trigger_float(value)
                self.virtual_axes['LEFT_TRIGGER'] = value
            elif axis_name in ['RIGHT_TRIGGER', 'RT']:
                self.gamepad.right_trigger_float(value)
                self.virtual_axes['RIGHT_TRIGGER'] = value
            else:
                return False
            
            self.gamepad.update()
            return True
        
        return False


class UnifiedInputHandler:
    """Unified handler for all input types with macro support."""
    
    def __init__(self):
        self.keyboard_handler = KeyboardHandler()
        self.mouse_handler = MouseHandler()
        self.gamepad_handler = GamepadHandler()
        
        self.action_queue = queue.Queue()
        self.processing_thread = None
        self.is_running = False
        
        self.macros: Dict[str, InputSequence] = {}
        self.gesture_bindings: Dict[str, Union[InputAction, InputSequence]] = {}
        
        # Statistics
        self.stats = {
            'actions_executed': 0,
            'sequences_executed': 0,
            'errors': 0,
            'avg_execution_time': 0.0
        }
    
    def start(self):
        """Start the input processing thread."""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Unified input handler started")
    
    def stop(self):
        """Stop the input processing thread."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        # Stop all handlers
        self.keyboard_handler.stop_all()
        self.mouse_handler.stop_all()
        self.gamepad_handler.stop_all()
        
        logger.info("Unified input handler stopped")
    
    def bind_gesture(self, gesture_name: str, action: Union[InputAction, InputSequence]):
        """Bind a gesture to an input action or sequence."""
        self.gesture_bindings[gesture_name] = action
        logger.info(f"Bound gesture '{gesture_name}' to action")
    
    def unbind_gesture(self, gesture_name: str):
        """Remove a gesture binding."""
        if gesture_name in self.gesture_bindings:
            del self.gesture_bindings[gesture_name]
            logger.info(f"Unbound gesture '{gesture_name}'")
    
    def execute_gesture(self, gesture_name: str) -> bool:
        """Execute the action bound to a gesture."""
        if gesture_name not in self.gesture_bindings:
            return False
        
        action = self.gesture_bindings[gesture_name]
        
        if isinstance(action, InputAction):
            return self.execute_action(action)
        elif isinstance(action, InputSequence):
            return self.execute_sequence(action)
        
        return False
    
    def execute_action(self, action: InputAction) -> bool:
        """Execute a single input action."""
        start_time = time.time()
        
        try:
            # Route to appropriate handler
            if action.input_type in [InputType.KEY_PRESS, InputType.KEY_HOLD, 
                                    InputType.KEY_RELEASE, InputType.COMBO]:
                result = self.keyboard_handler.execute(action)
            elif action.input_type in [InputType.MOUSE_CLICK, InputType.MOUSE_HOLD,
                                      InputType.MOUSE_RELEASE, InputType.MOUSE_MOVE,
                                      InputType.MOUSE_SCROLL, InputType.MOUSE_DRAG]:
                result = self.mouse_handler.execute(action)
            elif action.input_type in [InputType.GAMEPAD_BUTTON, InputType.GAMEPAD_AXIS]:
                result = self.gamepad_handler.execute(action)
            elif action.input_type == InputType.MACRO:
                macro_name = str(action.target)
                if macro_name in self.macros:
                    result = self.execute_sequence(self.macros[macro_name])
                else:
                    result = False
            else:
                result = False
            
            # Update statistics
            self.stats['actions_executed'] += 1
            execution_time = time.time() - start_time
            self.stats['avg_execution_time'] = (
                0.9 * self.stats['avg_execution_time'] + 0.1 * execution_time
            )
            
            if not result:
                self.stats['errors'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            self.stats['errors'] += 1
            return False
    
    def execute_sequence(self, sequence: InputSequence) -> bool:
        """Execute a sequence of actions."""
        try:
            for action in sequence.actions:
                if not self.execute_action(action):
                    return False
                
                if sequence.delay_between > 0:
                    time.sleep(sequence.delay_between)
            
            self.stats['sequences_executed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error executing sequence: {e}")
            return False
    
    def queue_action(self, action: InputAction):
        """Queue an action for asynchronous execution."""
        if self.is_running:
            try:
                self.action_queue.put_nowait(action)
            except queue.Full:
                logger.warning("Action queue full, dropping action")
    
    def _processing_loop(self):
        """Background processing thread for queued actions."""
        while self.is_running:
            try:
                action = self.action_queue.get(timeout=0.1)
                self.execute_action(action)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
    
    def create_macro(self, name: str, sequence: InputSequence):
        """Create or update a macro."""
        self.macros[name] = sequence
        logger.info(f"Created macro '{name}' with {len(sequence.actions)} actions")
    
    def delete_macro(self, name: str):
        """Delete a macro."""
        if name in self.macros:
            del self.macros[name]
            logger.info(f"Deleted macro '{name}'")
    
    def start_recording(self, input_type: str = 'all') -> bool:
        """Start recording inputs for macro creation."""
        if input_type in ['keyboard', 'all']:
            self.keyboard_handler.start_recording()
        if input_type in ['mouse', 'all']:
            self.mouse_handler.start_recording()
        
        return True
    
    def stop_recording(self, macro_name: str = None) -> InputSequence:
        """Stop recording and optionally save as macro."""
        sequences = []
        
        if hasattr(self.keyboard_handler, 'is_recording') and self.keyboard_handler.is_recording:
            sequences.append(self.keyboard_handler.stop_recording())
        
        if hasattr(self.mouse_handler, 'is_recording') and self.mouse_handler.is_recording:
            sequences.append(self.mouse_handler.stop_recording())
        
        # Combine sequences
        combined_actions = []
        for seq in sequences:
            combined_actions.extend(seq.actions)
        
        # Sort by timestamp
        combined_actions.sort(key=lambda a: a.timestamp)
        
        combined_sequence = InputSequence(
            name=macro_name or f"combined_{int(time.time())}",
            actions=combined_actions
        )
        
        if macro_name:
            self.create_macro(macro_name, combined_sequence)
        
        return combined_sequence
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get input handler statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            'actions_executed': 0,
            'sequences_executed': 0,
            'errors': 0,
            'avg_execution_time': 0.0
        }
