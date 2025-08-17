# Windows Number Key Fix

## Problem
The numbers 1, 2, 3 were not being inserted on Windows systems when the corresponding gestures were detected, while they worked correctly on Mac systems. This issue is particularly relevant for Mac users running Windows in virtual machines like Parallels, VMware, or VirtualBox.

## Root Cause
The issue was in the `_parse_key` method in both `src/input/handler.py` and `hand_to_key_simple.py`. The original implementation was returning raw character strings for single-character keys like '1', '2', '3':

```python
# Original problematic code
if len(key_str) == 1:
    return key_str  # This doesn't work reliably on Windows
```

While this approach works on Mac, Windows has different behavior in pynput when handling raw character strings versus proper KeyCode objects.

## Solution
The fix involves using pynput's `KeyCode.from_char()` method to create proper KeyCode objects for single characters:

```python
# Fixed code
if len(key_str) == 1:
    # Use KeyCode for single characters for better Windows compatibility
    from pynput.keyboard import KeyCode
    return KeyCode.from_char(key_str)
```

## Technical Details

### Why This Works
- pynput's `KeyCode.from_char()` creates a proper KeyCode object that works consistently across platforms
- On Mac, pynput can handle both raw strings and KeyCode objects
- On Windows, the KeyCode object provides better integration with the Windows API (specifically through `VkKeyScan` and the Win32 keyboard handling)

### Files Modified
1. `src/input/handler.py` - Line 228-251 in `_parse_key` method
2. `hand_to_key_simple.py` - Line 250-276 in `_parse_key` method

### Platform Compatibility
- **Mac (Darwin)**: Both old and new approaches work
- **Windows**: New KeyCode approach works reliably, old string approach was problematic
- **Linux**: Should work with new approach (same as Mac behavior)

## Testing
To test the fix:

1. Run `python test_digit_keys.py` on both Windows and Mac
2. Use the hand gesture application and test:
   - `one_finger` gesture → should type '1'
   - `two_fingers` gesture → should type '2' 
   - `three_fingers` gesture → should type '3'

## Gesture Mappings Affected
The following default gesture mappings should now work on Windows:

```json
{
    "one_finger": {"action": "key", "target": "1", "mode": "tap"},
    "two_fingers": {"action": "key", "target": "2", "mode": "tap"},
    "three_fingers": {"action": "key", "target": "3", "mode": "tap"},
    "pinch_index": {"action": "key", "target": "4", "mode": "tap"},
    "pinch_middle": {"action": "key", "target": "5", "mode": "tap"},
    "pinch_ring": {"action": "key", "target": "6", "mode": "tap"},
    "pinch_pinky": {"action": "key", "target": "7", "mode": "tap"}
}
```

## Parallels VM Specific Optimization
For Mac users running Windows in Parallels, the input handler now includes enhanced digit key handling:

```python
def _create_digit_key(self, digit: str):
    # On Mac (which might be sending to VMs like Parallels), use virtual key codes
    if platform.system() == 'Darwin':
        mac_digit_vk_codes = {
            '1': 0x12,  # kVK_ANSI_1
            '2': 0x13,  # kVK_ANSI_2  
            '3': 0x14   # kVK_ANSI_3
            # ... etc
        }
        return KeyCode.from_vk(mac_digit_vk_codes[digit], char=digit)
```

This creates KeyCode objects with both virtual key codes and character information, providing the most robust key transmission from Mac host to Windows VM.

### Testing Parallels Setup
1. Run `python test_parallels_keys.py` to test different key transmission methods
2. Run `python parallels_key_handler.py` to test the optimized handler
3. Verify that all approaches work in your Windows VM text editor

## Additional Notes
- This fix applies to all single-character keys, not just numbers
- Letters (a-z) and other single characters also benefit from this change
- No performance impact - `KeyCode.from_char()` is lightweight
- The fix is backwards compatible and doesn't break existing functionality on any platform
- Enhanced VM support for Mac-to-Windows virtualization scenarios

## Future Considerations
If other platform-specific issues arise, consider creating platform-specific key mapping classes or detection logic in the `_create_key_mapping` method.
