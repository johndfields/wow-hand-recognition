# Hand Preference and Tracking Visualization Configuration

This document explains how to configure hand preference settings and hand tracking visualization options in the Hand to Key application.

## Hand Preference Setting

The `hand_preference` setting controls which hand(s) the system will track for gesture detection:

### Configuration Options

Edit your `config/settings.json` file to set the hand preference:

```json
{
  "hand_preference": "right"
}
```

**Available values:**
- `"right"` - Only track the user's right hand (default)
- `"left"` - Only track the user's left hand
- `"both"` - Track both hands

### Benefits of Hand Preference

1. **Performance**: Reduces computational overhead by ignoring irrelevant hands
2. **Accuracy**: Eliminates false positives from unintended hand movements
3. **User Control**: Allows customization based on dominant hand or use case

## Hand Tracking Visualization

Control the visual feedback of hand tracking with these settings:

### Configuration Options

```json
{
  "show_hand_landmarks": true,
  "show_hand_connections": true,
  "show_debug_info": true
}
```

**Settings explained:**
- `show_hand_landmarks`: Show green dots at hand joint positions
- `show_hand_connections`: Show white lines connecting hand joints
- `show_debug_info`: Must be `true` to enable hand tracking visualization

### Visual Elements

When enabled, you'll see:

- **Green dots**: Hand landmark points (joints)
- **White lines**: Connections between hand joints
- **Red center point**: Palm center
- **Green text**: Gesture name and confidence score
- **Yellow text**: Hand index and preference indicator
- **Blue arrows**: Hand movement velocity vectors (when moving)
- **Stats panel**: FPS, confidence, gesture count, and hand preference

## Runtime Controls

While the application is running, you can:

- Press `d` to toggle debug/visualization mode
- Press `h` to show help with all keyboard shortcuts
- The system automatically shows which hand preference is active

## Configuration Examples

### For Left-Handed Users
```json
{
  "hand_preference": "left",
  "show_hand_landmarks": true,
  "show_hand_connections": true,
  "show_debug_info": true
}
```

### For Gaming (Track Both Hands)
```json
{
  "hand_preference": "both",
  "enable_multi_hand": true,
  "max_hands": 2,
  "show_hand_landmarks": true,
  "show_hand_connections": true
}
```

### Minimal Visualization (Performance Mode)
```json
{
  "hand_preference": "right",
  "show_hand_landmarks": false,
  "show_hand_connections": true,
  "show_debug_info": true
}
```

## Troubleshooting

### Hand Not Being Detected
1. Check that your preferred hand is visible to the camera
2. Ensure good lighting conditions
3. Verify `hand_preference` matches your intended hand
4. Try setting `hand_preference` to `"both"` temporarily for testing

### Performance Issues
1. Set `hand_preference` to `"left"` or `"right"` instead of `"both"`
2. Disable visualization: `"show_debug_info": false`
3. Reduce camera resolution in settings

### Visualization Not Showing
1. Ensure `"show_debug_info": true`
2. Check that `"show_preview": true`
3. Press `d` key to toggle debug mode while running

## Command Line Options

You can also run the application with debug mode enabled:

```bash
python src/main.py --debug --verbose
```

This will show detailed logging information about hand detection and filtering.
