"""
Configuration management module with schema validation, profiles, and hot-reload.
"""

from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os
import time
import threading
import logging
from pathlib import Path
import hashlib
import yaml
from jsonschema import validate, ValidationError, Draft7Validator
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"


@dataclass
class GestureMapping:
    """Represents a gesture to action mapping."""
    gesture: str
    action_type: str  # key, mouse, gamepad, macro
    target: str
    mode: str = "tap"  # tap, hold, toggle, double_tap
    duration: float = 0.0
    cooldown: float = 0.0
    sensitivity: float = 1.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileConfig:
    """Represents a configuration profile."""
    name: str
    description: str = ""
    gesture_mappings: List[GestureMapping] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    parent_profile: Optional[str] = None
    is_active: bool = False
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)


@dataclass
class ApplicationSettings:
    """Global application settings."""
    # Camera settings
    camera_index: int = 0
    camera_width: int = 960
    camera_height: int = 540
    camera_fps: int = 30
    
    # Detection settings
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6
    stable_frames: int = 3
    gesture_sensitivity: float = 1.0
    enable_multi_hand: bool = False
    max_hands: int = 2
    
    # Performance settings
    enable_gpu: bool = True
    frame_skip: int = 0
    processing_threads: int = 2
    adaptive_quality: bool = True
    
    # UI settings
    show_preview: bool = True
    show_statistics: bool = True
    show_debug_info: bool = False
    ui_scale: float = 1.0
    theme: str = "dark"
    
    # Audio settings
    enable_sound_feedback: bool = True
    sound_volume: float = 0.5
    
    # Accessibility settings
    accessibility_mode: bool = False
    high_contrast: bool = False
    large_text: bool = False
    voice_commands: bool = False
    
    # Security settings
    require_confirmation: bool = False
    camera_indicator: bool = True
    log_actions: bool = True
    
    # Paths
    config_dir: str = "./config"
    profiles_dir: str = "./config/profiles"
    macros_dir: str = "./config/macros"
    models_dir: str = "./config/models"
    logs_dir: str = "./logs"


class ConfigSchema:
    """JSON schema definitions for configuration validation."""
    
    GESTURE_MAPPING_SCHEMA = {
        "type": "object",
        "properties": {
            "gesture": {"type": "string"},
            "action_type": {"type": "string", "enum": ["key", "mouse", "gamepad", "macro"]},
            "target": {"type": "string"},
            "mode": {"type": "string", "enum": ["tap", "hold", "toggle", "double_tap", "long_press"]},
            "duration": {"type": "number", "minimum": 0},
            "cooldown": {"type": "number", "minimum": 0},
            "sensitivity": {"type": "number", "minimum": 0.1, "maximum": 3.0},
            "enabled": {"type": "boolean"},
            "metadata": {"type": "object"}
        },
        "required": ["gesture", "action_type", "target"],
        "additionalProperties": False
    }
    
    PROFILE_SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "gesture_mappings": {
                "type": "array",
                "items": GESTURE_MAPPING_SCHEMA
            },
            "settings": {"type": "object"},
            "parent_profile": {"type": ["string", "null"]},
            "is_active": {"type": "boolean"}
        },
        "required": ["name"],
        "additionalProperties": False
    }
    
    SETTINGS_SCHEMA = {
        "type": "object",
        "properties": {
            "camera_index": {"type": "integer", "minimum": 0},
            "camera_width": {"type": "integer", "minimum": 320},
            "camera_height": {"type": "integer", "minimum": 240},
            "camera_fps": {"type": "integer", "minimum": 1, "maximum": 120},
            "min_detection_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "min_tracking_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "stable_frames": {"type": "integer", "minimum": 1},
            "gesture_sensitivity": {"type": "number", "minimum": 0.1, "maximum": 3.0},
            "enable_multi_hand": {"type": "boolean"},
            "max_hands": {"type": "integer", "minimum": 1, "maximum": 4},
            "enable_gpu": {"type": "boolean"},
            "frame_skip": {"type": "integer", "minimum": 0},
            "processing_threads": {"type": "integer", "minimum": 1},
            "adaptive_quality": {"type": "boolean"},
            "show_preview": {"type": "boolean"},
            "show_statistics": {"type": "boolean"},
            "show_debug_info": {"type": "boolean"},
            "ui_scale": {"type": "number", "minimum": 0.5, "maximum": 3.0},
            "theme": {"type": "string", "enum": ["light", "dark", "auto"]},
            "enable_sound_feedback": {"type": "boolean"},
            "sound_volume": {"type": "number", "minimum": 0, "maximum": 1},
            "accessibility_mode": {"type": "boolean"},
            "high_contrast": {"type": "boolean"},
            "large_text": {"type": "boolean"},
            "voice_commands": {"type": "boolean"},
            "require_confirmation": {"type": "boolean"},
            "camera_indicator": {"type": "boolean"},
            "log_actions": {"type": "boolean"}
        },
        "additionalProperties": False
    }


class ConfigFileHandler(FileSystemEventHandler):
    """Handles file system events for configuration hot-reload."""
    
    def __init__(self, config_manager: 'ConfigurationManager'):
        self.config_manager = config_manager
        self.last_reload = {}
        
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
            
        file_path = event.src_path
        current_time = time.time()
        
        # Debounce rapid modifications
        if file_path in self.last_reload:
            if current_time - self.last_reload[file_path] < 1.0:
                return
        
        self.last_reload[file_path] = current_time
        
        # Determine file type and reload
        if file_path.endswith(('.json', '.yaml', '.yml')):
            logger.info(f"Configuration file modified: {file_path}")
            self.config_manager.reload_config(file_path)


class ConfigurationManager:
    """Main configuration management system."""
    
    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiles_dir = self.config_dir / "profiles"
        self.profiles_dir.mkdir(exist_ok=True)
        
        self.macros_dir = self.config_dir / "macros"
        self.macros_dir.mkdir(exist_ok=True)
        
        self.models_dir = self.config_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Configuration state
        self.settings = ApplicationSettings()
        self.profiles: Dict[str, ProfileConfig] = {}
        self.active_profile: Optional[ProfileConfig] = None
        self.gesture_mappings: Dict[str, GestureMapping] = {}
        
        # Hot-reload
        self.observer = None
        self.reload_callbacks: List[Callable] = []
        self.file_checksums: Dict[str, str] = {}
        
        # Load initial configuration
        self.load_all_configs()
        
    def start_hot_reload(self):
        """Start watching configuration files for changes."""
        if self.observer is None:
            self.observer = Observer()
            event_handler = ConfigFileHandler(self)
            self.observer.schedule(event_handler, str(self.config_dir), recursive=True)
            self.observer.start()
            logger.info("Configuration hot-reload started")
    
    def stop_hot_reload(self):
        """Stop watching configuration files."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Configuration hot-reload stopped")
    
    def add_reload_callback(self, callback: Callable):
        """Add a callback to be called when configuration is reloaded."""
        self.reload_callbacks.append(callback)
    
    def remove_reload_callback(self, callback: Callable):
        """Remove a reload callback."""
        if callback in self.reload_callbacks:
            self.reload_callbacks.remove(callback)
    
    def load_all_configs(self):
        """Load all configuration files."""
        # Load main settings
        settings_file = self.config_dir / "settings.json"
        if settings_file.exists():
            self.load_settings(str(settings_file))
        else:
            self.save_settings()
        
        # Load profiles
        for profile_file in self.profiles_dir.glob("*.json"):
            self.load_profile(str(profile_file))
        
        # Set default profile if none active
        if not self.active_profile and self.profiles:
            self.activate_profile(list(self.profiles.keys())[0])
    
    def reload_config(self, file_path: str):
        """Reload a specific configuration file."""
        file_path = Path(file_path)
        
        # Check if file has actually changed
        if not self._has_file_changed(file_path):
            return
        
        try:
            if file_path.name == "settings.json":
                self.load_settings(str(file_path))
            elif file_path.parent == self.profiles_dir:
                self.load_profile(str(file_path))
            
            # Notify callbacks
            for callback in self.reload_callbacks:
                try:
                    callback(file_path)
                except Exception as e:
                    logger.error(f"Error in reload callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error reloading configuration {file_path}: {e}")
    
    def _has_file_changed(self, file_path: Path) -> bool:
        """Check if file content has changed using checksum."""
        if not file_path.exists():
            return False
        
        with open(file_path, 'rb') as f:
            content = f.read()
            checksum = hashlib.md5(content).hexdigest()
        
        old_checksum = self.file_checksums.get(str(file_path))
        self.file_checksums[str(file_path)] = checksum
        
        return checksum != old_checksum
    
    def load_settings(self, file_path: str):
        """Load application settings from file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate schema
            validate(data, ConfigSchema.SETTINGS_SCHEMA)
            
            # Update settings
            for key, value in data.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
            
            logger.info(f"Loaded settings from {file_path}")
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Invalid settings file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading settings {file_path}: {e}")
    
    def save_settings(self):
        """Save application settings to file."""
        settings_file = self.config_dir / "settings.json"
        
        try:
            settings_dict = asdict(self.settings)
            
            # Remove path fields for JSON serialization
            for key in ['config_dir', 'profiles_dir', 'macros_dir', 'models_dir', 'logs_dir']:
                settings_dict.pop(key, None)
            
            with open(settings_file, 'w') as f:
                json.dump(settings_dict, f, indent=2)
            
            logger.info(f"Saved settings to {settings_file}")
            
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
    
    def load_profile(self, file_path: str) -> Optional[ProfileConfig]:
        """Load a profile from file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate schema
            validate(data, ConfigSchema.PROFILE_SCHEMA)
            
            # Create profile
            profile = ProfileConfig(
                name=data['name'],
                description=data.get('description', ''),
                parent_profile=data.get('parent_profile')
            )
            
            # Load gesture mappings
            for mapping_data in data.get('gesture_mappings', []):
                mapping = GestureMapping(**mapping_data)
                profile.gesture_mappings.append(mapping)
            
            # Load settings
            profile.settings = data.get('settings', {})
            
            # Add to profiles
            self.profiles[profile.name] = profile
            
            logger.info(f"Loaded profile '{profile.name}' from {file_path}")
            return profile
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Invalid profile file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading profile {file_path}: {e}")
        
        return None
    
    def save_profile(self, profile: ProfileConfig):
        """Save a profile to file."""
        profile_file = self.profiles_dir / f"{profile.name}.json"
        
        try:
            # Convert to dict
            profile_dict = {
                'name': profile.name,
                'description': profile.description,
                'gesture_mappings': [asdict(m) for m in profile.gesture_mappings],
                'settings': profile.settings,
                'parent_profile': profile.parent_profile,
                'is_active': profile.is_active
            }
            
            # Validate before saving
            validate(profile_dict, ConfigSchema.PROFILE_SCHEMA)
            
            with open(profile_file, 'w') as f:
                json.dump(profile_dict, f, indent=2)
            
            profile.modified_at = time.time()
            logger.info(f"Saved profile '{profile.name}' to {profile_file}")
            
        except ValidationError as e:
            logger.error(f"Invalid profile data: {e}")
        except Exception as e:
            logger.error(f"Error saving profile '{profile.name}': {e}")
    
    def create_profile(self, name: str, description: str = "", 
                      parent_profile: Optional[str] = None) -> ProfileConfig:
        """Create a new profile."""
        profile = ProfileConfig(
            name=name,
            description=description,
            parent_profile=parent_profile
        )
        
        # Inherit from parent if specified
        if parent_profile and parent_profile in self.profiles:
            parent = self.profiles[parent_profile]
            profile.gesture_mappings = parent.gesture_mappings.copy()
            profile.settings = parent.settings.copy()
        
        self.profiles[name] = profile
        self.save_profile(profile)
        
        return profile
    
    def delete_profile(self, name: str) -> bool:
        """Delete a profile."""
        if name not in self.profiles:
            return False
        
        profile_file = self.profiles_dir / f"{name}.json"
        
        try:
            if profile_file.exists():
                profile_file.unlink()
            
            del self.profiles[name]
            
            # Switch to another profile if this was active
            if self.active_profile and self.active_profile.name == name:
                self.active_profile = None
                if self.profiles:
                    self.activate_profile(list(self.profiles.keys())[0])
            
            logger.info(f"Deleted profile '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting profile '{name}': {e}")
            return False
    
    def activate_profile(self, name: str) -> bool:
        """Activate a profile."""
        if name not in self.profiles:
            logger.error(f"Profile '{name}' not found")
            return False
        
        # Deactivate current profile
        if self.active_profile:
            self.active_profile.is_active = False
            self.save_profile(self.active_profile)
        
        # Activate new profile
        self.active_profile = self.profiles[name]
        self.active_profile.is_active = True
        self.save_profile(self.active_profile)
        
        # Update gesture mappings
        self.gesture_mappings.clear()
        for mapping in self.active_profile.gesture_mappings:
            if mapping.enabled:
                self.gesture_mappings[mapping.gesture] = mapping
        
        logger.info(f"Activated profile '{name}'")
        
        # Notify callbacks
        for callback in self.reload_callbacks:
            try:
                callback(name)
            except Exception as e:
                logger.error(f"Error in profile activation callback: {e}")
        
        return True
    
    def get_gesture_mapping(self, gesture: str) -> Optional[GestureMapping]:
        """Get mapping for a specific gesture."""
        return self.gesture_mappings.get(gesture)
    
    def update_gesture_mapping(self, gesture: str, mapping: GestureMapping):
        """Update or add a gesture mapping in the active profile."""
        if not self.active_profile:
            logger.error("No active profile to update")
            return
        
        # Find and update existing mapping or add new one
        found = False
        for i, m in enumerate(self.active_profile.gesture_mappings):
            if m.gesture == gesture:
                self.active_profile.gesture_mappings[i] = mapping
                found = True
                break
        
        if not found:
            self.active_profile.gesture_mappings.append(mapping)
        
        # Update runtime mappings
        if mapping.enabled:
            self.gesture_mappings[gesture] = mapping
        elif gesture in self.gesture_mappings:
            del self.gesture_mappings[gesture]
        
        # Save profile
        self.save_profile(self.active_profile)
    
    def remove_gesture_mapping(self, gesture: str):
        """Remove a gesture mapping from the active profile."""
        if not self.active_profile:
            return
        
        # Remove from profile
        self.active_profile.gesture_mappings = [
            m for m in self.active_profile.gesture_mappings if m.gesture != gesture
        ]
        
        # Remove from runtime mappings
        if gesture in self.gesture_mappings:
            del self.gesture_mappings[gesture]
        
        # Save profile
        self.save_profile(self.active_profile)
    
    def export_profile(self, name: str, file_path: str) -> bool:
        """Export a profile to a file."""
        if name not in self.profiles:
            return False
        
        profile = self.profiles[name]
        
        try:
            profile_dict = {
                'name': profile.name,
                'description': profile.description,
                'gesture_mappings': [asdict(m) for m in profile.gesture_mappings],
                'settings': profile.settings,
                'parent_profile': profile.parent_profile
            }
            
            with open(file_path, 'w') as f:
                json.dump(profile_dict, f, indent=2)
            
            logger.info(f"Exported profile '{name}' to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting profile '{name}': {e}")
            return False
    
    def import_profile(self, file_path: str, new_name: Optional[str] = None) -> Optional[ProfileConfig]:
        """Import a profile from a file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate schema
            validate(data, ConfigSchema.PROFILE_SCHEMA)
            
            # Use new name if provided
            if new_name:
                data['name'] = new_name
            
            # Check for name conflict
            if data['name'] in self.profiles:
                data['name'] = f"{data['name']}_{int(time.time())}"
            
            # Create profile
            profile = ProfileConfig(
                name=data['name'],
                description=data.get('description', ''),
                parent_profile=data.get('parent_profile')
            )
            
            # Load gesture mappings
            for mapping_data in data.get('gesture_mappings', []):
                mapping = GestureMapping(**mapping_data)
                profile.gesture_mappings.append(mapping)
            
            profile.settings = data.get('settings', {})
            
            # Save and add to profiles
            self.profiles[profile.name] = profile
            self.save_profile(profile)
            
            logger.info(f"Imported profile '{profile.name}' from {file_path}")
            return profile
            
        except Exception as e:
            logger.error(f"Error importing profile from {file_path}: {e}")
            return None
    
    def get_all_profiles(self) -> List[str]:
        """Get list of all profile names."""
        return list(self.profiles.keys())
    
    def get_profile(self, name: str) -> Optional[ProfileConfig]:
        """Get a specific profile."""
        return self.profiles.get(name)
    
    def validate_config_file(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """Validate a configuration file against schema."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Determine schema based on content
            if 'gesture_mappings' in data:
                validate(data, ConfigSchema.PROFILE_SCHEMA)
                return True, "Valid profile configuration"
            else:
                validate(data, ConfigSchema.SETTINGS_SCHEMA)
                return True, "Valid settings configuration"
                
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except ValidationError as e:
            return False, f"Schema validation failed: {e.message}"
        except Exception as e:
            return False, f"Error: {e}"
    
    def create_default_profiles(self):
        """Create default profiles for common use cases."""
        # Gaming profile
        gaming_profile = self.create_profile(
            "Gaming",
            "Optimized for gaming with WASD movement and action keys"
        )
        
        gaming_mappings = [
            GestureMapping("open_palm", "key", "w", "hold"),
            GestureMapping("fist", "key", "s", "hold"),
            GestureMapping("victory", "key", "a", "hold"),
            GestureMapping("three", "key", "d", "hold"),
            GestureMapping("thumbs_up", "key", "space", "tap"),
            GestureMapping("pinch_index", "key", "1", "tap"),
            GestureMapping("pinch_middle", "key", "2", "tap"),
            GestureMapping("pinch_ring", "key", "3", "tap"),
            GestureMapping("pinch_pinky", "key", "4", "tap"),
            GestureMapping("index_only", "mouse", "right_click", "tap")
        ]
        
        gaming_profile.gesture_mappings = gaming_mappings
        self.save_profile(gaming_profile)
        
        # Productivity profile
        productivity_profile = self.create_profile(
            "Productivity",
            "Optimized for productivity with shortcuts and navigation"
        )
        
        productivity_mappings = [
            GestureMapping("swipe_left", "key", "alt+left", "tap"),
            GestureMapping("swipe_right", "key", "alt+right", "tap"),
            GestureMapping("swipe_up", "key", "page_up", "tap"),
            GestureMapping("swipe_down", "key", "page_down", "tap"),
            GestureMapping("pinch_index", "key", "ctrl+c", "tap"),
            GestureMapping("pinch_middle", "key", "ctrl+v", "tap"),
            GestureMapping("pinch_ring", "key", "ctrl+z", "tap"),
            GestureMapping("pinch_pinky", "key", "ctrl+y", "tap"),
            GestureMapping("open_palm", "key", "ctrl+s", "tap"),
            GestureMapping("fist", "key", "escape", "tap")
        ]
        
        productivity_profile.gesture_mappings = productivity_mappings
        self.save_profile(productivity_profile)
        
        # Presentation profile
        presentation_profile = self.create_profile(
            "Presentation",
            "Optimized for presentations with slide navigation"
        )
        
        presentation_mappings = [
            GestureMapping("swipe_left", "key", "left", "tap"),
            GestureMapping("swipe_right", "key", "right", "tap"),
            GestureMapping("open_palm", "key", "f5", "tap"),
            GestureMapping("fist", "key", "escape", "tap"),
            GestureMapping("index_only", "mouse", "left_click", "tap"),
            GestureMapping("victory", "key", "b", "tap"),  # Blank screen
            GestureMapping("thumbs_up", "key", "home", "tap"),
            GestureMapping("pinch_index", "key", "1", "tap"),
            GestureMapping("pinch_middle", "key", "2", "tap")
        ]
        
        presentation_profile.gesture_mappings = presentation_mappings
        self.save_profile(presentation_profile)
        
        logger.info("Created default profiles")
