"""
Configuration management module.
"""

from .manager import ConfigFormat, GestureMapping, ProfileConfig, ApplicationSettings, ConfigurationManager
from .migration import ConfigMigration, detect_config_type, migrate_all_configs

