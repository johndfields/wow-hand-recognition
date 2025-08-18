#!/usr/bin/env python3
"""
Migration script to convert between configuration formats.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def migrate_simple_to_modular(simple_config_path, output_dir=None):
    """
    Migrate from simple to modular configuration format.
    
    Args:
        simple_config_path: Path to the simple configuration file
        output_dir: Directory to save the modular configuration files
    
    Returns:
        Tuple of (success, message)
    """
    try:
        # Load the simple configuration
        with open(simple_config_path, 'r') as f:
            simple_config = json.load(f)
        
        # Determine output directory
        if output_dir is None:
            output_dir = Path("config")
        else:
            output_dir = Path(output_dir)
        
        # Create directories if they don't exist
        profiles_dir = output_dir / "profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract settings
        settings = {
            "camera_index": simple_config.get("camera_index", 0),
            "camera_width": simple_config.get("camera_width", 960),
            "camera_height": simple_config.get("camera_height", 540),
            "min_detection_confidence": simple_config.get("min_detection_confidence", 0.6),
            "min_tracking_confidence": simple_config.get("min_tracking_confidence", 0.6),
            "stable_frames": simple_config.get("stable_frames", 3),
            "gesture_sensitivity": simple_config.get("sensitivity", 1.0),
            "show_debug_info": True
        }
        
        # Save settings
        settings_path = output_dir / "settings.json"
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        # Create a profile from the simple mappings
        profile_name = os.path.basename(simple_config_path).split('.')[0]
        if profile_name.lower() == "gaming_config":
            profile_name = "Gaming"
            description = "Gaming profile with WASD movement and action keys"
        else:
            profile_name = profile_name.capitalize()
            description = f"Migrated from {simple_config_path}"
        
        # Create profile structure
        profile = {
            "name": profile_name,
            "description": description,
            "gesture_mappings": [],
            "settings": settings,
            "parent_profile": None,
            "is_active": True
        }
        
        # Convert mappings
        if "mappings" in simple_config:
            for gesture, config in simple_config["mappings"].items():
                # Convert gesture name if needed
                if gesture == "l_shape":
                    gesture = "l_shape"  # Keep as is
                elif gesture == "hang_loose":
                    gesture = "hang_loose"  # Keep as is
                elif gesture == "one_finger":
                    gesture = "index_only"  # Map to modular equivalent
                elif gesture == "two_fingers":
                    gesture = "victory"  # Map to modular equivalent
                elif gesture == "three_fingers":
                    gesture = "three"  # Map to modular equivalent
                elif gesture == "four_fingers":
                    gesture = "four_fingers"  # Keep as is
                
                # Create mapping
                mapping = {
                    "gesture": gesture,
                    "action_type": config.get("action", "key"),
                    "target": config.get("target", ""),
                    "mode": config.get("mode", "tap"),
                    "cooldown": simple_config.get("cooldown", 0.7),
                    "sensitivity": simple_config.get("sensitivity", 1.0),
                    "enabled": True
                }
                profile["gesture_mappings"].append(mapping)
            
            # Save profile
            profile_path = profiles_dir / f"{profile_name.lower()}.json"
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)
            
            return True, f"Successfully migrated configuration to profile '{profile_name}'"
        else:
            return False, "No mappings found in simple configuration"
            
    except Exception as e:
        logger.error(f"Error migrating configuration: {e}")
        return False, f"Error migrating configuration: {e}"

def main():
    """Main entry point for the migration script."""
    if len(sys.argv) < 2:
        print("Usage: python migrate_config.py <simple_config_path> [output_dir]")
        return
    
    simple_config_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    success, message = migrate_simple_to_modular(simple_config_path, output_dir)
    print(message)
    
    if success:
        print(f"Migration successful. You can now use the modular configuration.")
    else:
        print(f"Migration failed. Please check the error message above.")

if __name__ == "__main__":
    main()

