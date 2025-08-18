"""
Configuration migration utilities for converting between different configuration formats.
This module handles migration from the simple implementation's configuration format
to the modular implementation's format.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict

from .manager import ConfigurationManager, GestureMapping, ProfileConfig

logger = logging.getLogger(__name__)


class ConfigMigration:
    """Utilities for migrating between different configuration formats."""
    
    @staticmethod
    def migrate_simple_to_modular(
        simple_config_path: str,
        config_manager: ConfigurationManager
    ) -> Tuple[bool, str]:
        """
        Migrate from the simple implementation's configuration format to the modular format.
        
        Args:
            simple_config_path: Path to the simple implementation's config file
            config_manager: Instance of the modular ConfigurationManager
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Load the simple configuration
            with open(simple_config_path, 'r') as f:
                simple_config = json.load(f)
            
            # Extract settings
            settings = {
                "camera_index": simple_config.get("camera_index", 0),
                "camera_width": simple_config.get("camera_width", 960),
                "camera_height": simple_config.get("camera_height", 540),
                "min_detection_confidence": simple_config.get("min_detection_confidence", 0.6),
                "min_tracking_confidence": simple_config.get("min_tracking_confidence", 0.6),
                "stable_frames": simple_config.get("stable_frames", 3),
                "gesture_sensitivity": simple_config.get("sensitivity", 1.0),
            }
            
            # Update application settings
            config_manager.update_settings(settings)
            
            # Create a new profile from the simple mappings
            profile_name = os.path.basename(simple_config_path).split('.')[0]
            if profile_name == "gaming_config":
                profile_name = "Gaming (Simple)"
                description = "Migrated from simple implementation's gaming configuration"
            else:
                profile_name = f"Migrated {profile_name}"
                description = "Migrated from simple implementation"
            
            # Create profile
            profile = config_manager.create_profile(profile_name, description)
            
            # Convert mappings
            if "mappings" in simple_config:
                mappings = []
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
                    mapping = GestureMapping(
                        gesture=gesture,
                        action_type=config.get("action", "key"),
                        target=config.get("target", ""),
                        mode=config.get("mode", "tap"),
                        cooldown=simple_config.get("cooldown", 0.7),
                        sensitivity=simple_config.get("sensitivity", 1.0),
                        enabled=True
                    )
                    mappings.append(mapping)
                
                # Update profile with mappings
                profile.gesture_mappings = mappings
                config_manager.save_profile(profile)
                
                # Activate the profile
                config_manager.activate_profile(profile_name)
                
                return True, f"Successfully migrated configuration to profile '{profile_name}'"
            else:
                return False, "No mappings found in simple configuration"
                
        except Exception as e:
            logger.error(f"Error migrating configuration: {e}")
            return False, f"Error migrating configuration: {e}"
    
    @staticmethod
    def create_simple_config_from_profile(
        profile_name: str,
        output_path: str,
        config_manager: ConfigurationManager
    ) -> Tuple[bool, str]:
        """
        Create a simple implementation configuration file from a modular profile.
        
        Args:
            profile_name: Name of the profile to convert
            output_path: Path to save the simple configuration
            config_manager: Instance of the modular ConfigurationManager
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Get the profile
            profile = config_manager.get_profile(profile_name)
            if not profile:
                return False, f"Profile '{profile_name}' not found"
            
            # Create simple configuration structure
            simple_config = {
                "camera_index": config_manager.settings.camera_index,
                "camera_width": config_manager.settings.camera_width,
                "camera_height": config_manager.settings.camera_height,
                "min_detection_confidence": config_manager.settings.min_detection_confidence,
                "min_tracking_confidence": config_manager.settings.min_tracking_confidence,
                "stable_frames": config_manager.settings.stable_frames,
                "sensitivity": config_manager.settings.gesture_sensitivity,
                "cooldown": 0.7,  # Default value
                "hand": "right",  # Default value
                "mappings": {}
            }
            
            # Convert mappings
            for mapping in profile.gesture_mappings:
                # Convert gesture name if needed
                gesture_name = mapping.gesture
                if gesture_name == "index_only":
                    gesture_name = "one_finger"  # Map to simple equivalent
                elif gesture_name == "victory":
                    gesture_name = "two_fingers"  # Map to simple equivalent
                elif gesture_name == "three":
                    gesture_name = "three_fingers"  # Map to simple equivalent
                
                # Add mapping
                simple_config["mappings"][gesture_name] = {
                    "action": mapping.action_type,
                    "target": mapping.target,
                    "mode": mapping.mode
                }
            
            # Save the configuration
            with open(output_path, 'w') as f:
                json.dump(simple_config, f, indent=2)
            
            return True, f"Successfully created simple configuration at '{output_path}'"
            
        except Exception as e:
            logger.error(f"Error creating simple configuration: {e}")
            return False, f"Error creating simple configuration: {e}"


def detect_config_type(config_path: str) -> str:
    """
    Detect the type of configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        String indicating the configuration type: "simple", "modular", or "unknown"
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for simple implementation format
        if "mappings" in config and isinstance(config["mappings"], dict):
            return "simple"
        
        # Check for modular implementation format (profile)
        if "gesture_mappings" in config and isinstance(config["gesture_mappings"], list):
            return "modular_profile"
        
        # Check for modular implementation format (settings)
        if "camera_index" in config and "gesture_sensitivity" in config:
            return "modular_settings"
        
        return "unknown"
        
    except Exception:
        return "unknown"


def migrate_all_configs(
    simple_config_dir: str,
    config_manager: ConfigurationManager
) -> Dict[str, str]:
    """
    Migrate all simple implementation configurations in a directory.
    
    Args:
        simple_config_dir: Directory containing simple implementation configs
        config_manager: Instance of the modular ConfigurationManager
        
    Returns:
        Dictionary mapping file paths to migration status messages
    """
    results = {}
    
    # Find all JSON files in the directory
    simple_config_path = Path(simple_config_dir)
    for file_path in simple_config_path.glob("*.json"):
        config_type = detect_config_type(str(file_path))
        
        if config_type == "simple":
            success, message = ConfigMigration.migrate_simple_to_modular(
                str(file_path),
                config_manager
            )
            results[str(file_path)] = f"{'Success' if success else 'Failed'}: {message}"
        else:
            results[str(file_path)] = f"Skipped: Not a simple implementation configuration"
    
    return results

