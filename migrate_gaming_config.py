#!/usr/bin/env python3
"""
Migration script to convert gaming_config.json to the modular format.
"""

import json
import os
from pathlib import Path

def migrate_gaming_config():
    """Migrate gaming_config.json to the modular format."""
    # Ensure config directories exist
    config_dir = Path("config")
    profiles_dir = config_dir / "profiles"
    
    config_dir.mkdir(exist_ok=True)
    profiles_dir.mkdir(exist_ok=True)
    
    # Load the original gaming config
    try:
        with open("gaming_config.json", "r") as f:
            original_config = json.load(f)
    except FileNotFoundError:
        print("Error: gaming_config.json not found")
        return False
    except json.JSONDecodeError:
        print("Error: gaming_config.json is not valid JSON")
        return False
    
    # Create the modular profile
    gaming_profile = {
        "name": "Gaming",
        "description": "Gaming profile with WASD movement and action keys",
        "gesture_mappings": [],
        "settings": {
            "camera_index": original_config.get("camera_index", 0),
            "camera_width": original_config.get("camera_width", 960),
            "camera_height": original_config.get("camera_height", 540),
            "min_detection_confidence": original_config.get("min_detection_confidence", 0.6),
            "min_tracking_confidence": original_config.get("min_tracking_confidence", 0.6),
            "stable_frames": original_config.get("stable_frames", 3),
            "gesture_sensitivity": original_config.get("sensitivity", 1.0),
            "show_debug_info": True
        }
    }
    
    # Convert mappings
    for gesture, mapping in original_config.get("mappings", {}).items():
        action_type = mapping.get("action", "key")
        target = mapping.get("target", "")
        mode = mapping.get("mode", "tap")
        
        gaming_profile["gesture_mappings"].append({
            "gesture": gesture,
            "action_type": action_type,
            "target": target,
            "mode": mode,
            "enabled": True
        })
    
    # Save the modular profile
    output_path = profiles_dir / "gaming.json"
    with open(output_path, "w") as f:
        json.dump(gaming_profile, f, indent=2)
    
    print(f"Successfully migrated gaming_config.json to {output_path}")
    return True

if __name__ == "__main__":
    migrate_gaming_config()

