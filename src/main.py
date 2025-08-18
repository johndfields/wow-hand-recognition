"""
Unified entry point for the hand gesture recognition application.
This module provides a single entry point that supports both the simple and modular
implementation modes.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)


def setup_argument_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Hand Gesture Recognition for Input Control"
    )
    
    # Mode selection
    parser.add_argument(
        '--mode', 
        choices=['simple', 'modular', 'auto'],
        default='auto',
        help='Application mode: simple, modular, or auto-detect (default: auto)'
    )
    
    # Configuration
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--profile', 
        type=str,
        help='Profile name to use (modular mode only)'
    )
    
    # Migration
    parser.add_argument(
        '--migrate', 
        action='store_true',
        help='Migrate configuration from simple to modular format'
    )
    
    parser.add_argument(
        '--migrate-dir', 
        type=str,
        help='Directory containing simple configurations to migrate'
    )
    
    # Camera options
    parser.add_argument(
        '--camera', 
        type=int, 
        default=0,
        help='Camera index to use'
    )
    
    parser.add_argument(
        '--width', 
        type=int, 
        default=960,
        help='Camera width'
    )
    
    parser.add_argument(
        '--height', 
        type=int, 
        default=540,
        help='Camera height'
    )
    
    # Debug options
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--no-preview', 
        action='store_true',
        help='Disable camera preview'
    )
    
    return parser


def detect_mode(config_path=None):
    """
    Auto-detect which mode to use based on configuration or available modules.
    
    Args:
        config_path: Optional path to a configuration file
        
    Returns:
        String indicating the mode: "simple" or "modular"
    """
    # If config path is provided, check its format
    if config_path and os.path.exists(config_path):
        try:
            from config.migration import detect_config_type
            config_type = detect_config_type(config_path)
            
            if config_type == "simple":
                return "simple"
            elif config_type in ["modular_profile", "modular_settings"]:
                return "modular"
        except Exception as e:
            logger.warning(f"Error detecting config type: {e}")
    
    # Check for presence of key modules
    try:
        # Try to import a key module from the modular implementation
        from gestures.detector import StandardGestureDetector
        from config.manager import ConfigurationManager
        return "modular"
    except ImportError:
        # If that fails, check for the simple implementation
        try:
            import simple_implementation
            return "simple"
        except ImportError:
            # Default to modular if both are available
            return "modular"


def run_simple_mode(args):
    """
    Run the application in simple mode.
    
    Args:
        args: Command line arguments
    """
    from simple_implementation.main import main as simple_main
    
    # Prepare arguments for simple implementation
    simple_args = {
        "config": args.config,
        "camera": args.camera,
        "width": args.width,
        "height": args.height,
        "debug": args.debug,
        "no_preview": args.no_preview
    }
    
    # Run simple implementation
    simple_main(**simple_args)


def run_modular_mode(args):
    """
    Run the application in modular mode.
    
    Args:
        args: Command line arguments
    """
    from gestures.detector import StandardGestureDetector, GestureProcessor
    from input.handler import UnifiedInputHandler
    from config.manager import ConfigurationManager
    from camera.capture import CameraCapture
    from ui.preview import PreviewWindow
    
    # Initialize configuration
    config_manager = ConfigurationManager()
    
    # Handle migration if requested
    if args.migrate and args.config:
        from config.migration import ConfigMigration
        success, message = ConfigMigration.migrate_simple_to_modular(
            args.config,
            config_manager
        )
        logger.info(message)
        if not success:
            return
    
    # Handle bulk migration if requested
    if args.migrate_dir:
        from config.migration import migrate_all_configs
        results = migrate_all_configs(args.migrate_dir, config_manager)
        for file_path, status in results.items():
            logger.info(f"{file_path}: {status}")
    
    # Load configuration if provided
    if args.config:
        try:
            # Try to load as a profile first
            if os.path.exists(args.config):
                logger.info(f"Loading configuration from {args.config}")
                from config.migration import ConfigMigration
                success, message = ConfigMigration.migrate_simple_to_modular(
                    args.config,
                    config_manager
                )
                logger.info(message)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    # Activate profile if specified
    if args.profile:
        config_manager.activate_profile(args.profile)
    elif not config_manager.active_profile:
        # Activate default profile if none is active
        profiles = config_manager.get_all_profiles()
        if profiles:
            config_manager.activate_profile(profiles[0])
        else:
            # Create and activate a default profile
            config_manager.create_default_profiles()
            config_manager.activate_profile("Gaming")
    
    # Override settings from command line arguments
    settings_override = {}
    if args.camera is not None:
        settings_override["camera_index"] = args.camera
    if args.width is not None:
        settings_override["camera_width"] = args.width
    if args.height is not None:
        settings_override["camera_height"] = args.height
    if args.debug:
        settings_override["show_debug_info"] = True
    if args.no_preview:
        settings_override["show_preview"] = False
    
    if settings_override:
        config_manager.update_settings(settings_override)
    
    # Initialize components
    settings = config_manager.settings
    
    # Initialize camera
    camera = CameraCapture(
        camera_index=settings.camera_index,
        width=settings.camera_width,
        height=settings.camera_height,
        fps=settings.camera_fps
    )
    
    # Initialize gesture detector
    detector = StandardGestureDetector(
        sensitivity=settings.gesture_sensitivity,
        enable_motion_gestures=True,
        enable_custom_gestures=False
    )
    
    # Initialize gesture processor
    processor = GestureProcessor(
        detector=detector,
        enable_multi_hand=settings.enable_multi_hand,
        max_hands=settings.max_hands,
        enable_gpu=settings.enable_gpu,
        confidence_threshold=settings.min_detection_confidence
    )
    
    # Initialize input handler
    input_handler = UnifiedInputHandler()
    
    # Load gesture mappings
    for gesture, mapping in config_manager.gesture_mappings.items():
        input_handler.bind_gesture(
            gesture,
            {
                "key": lambda m: input_handler.keyboard_handler.execute({
                    "input_type": "key_press",
                    "target": m.target,
                    "mode": m.mode,
                    "duration": m.duration
                }),
                "mouse": lambda m: input_handler.mouse_handler.execute({
                    "input_type": "mouse_click" if "click" in m.target else "mouse_move",
                    "target": m.target,
                    "mode": m.mode,
                    "duration": m.duration
                }),
                "gamepad": lambda m: input_handler.gamepad_handler.execute({
                    "input_type": "gamepad_button" if not any(x in m.target for x in ["axis", "stick"]) else "gamepad_axis",
                    "target": m.target,
                    "mode": m.mode,
                    "duration": m.duration
                }),
                "macro": lambda m: input_handler.execute_sequence(
                    input_handler.macros.get(m.target)
                )
            }[mapping.action_type](mapping)
        )
    
    # Initialize UI if preview is enabled
    preview = None
    if settings.show_preview:
        preview = PreviewWindow(
            width=settings.camera_width,
            height=settings.camera_height,
            show_stats=settings.show_statistics,
            show_debug=settings.show_debug_info
        )
    
    # Start components
    camera.start()
    input_handler.start()
    config_manager.start_hot_reload()
    
    try:
        # Main loop
        last_detection_time = 0
        cooldown = 0.7  # Default cooldown between gestures
        
        while True:
            # Get frame from camera
            success, frame = camera.read()
            if not success:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
                continue
            
            # Process frame with gesture processor
            detections = processor.process_frame(frame)
            
            # Handle detected gestures
            if detections:
                # Execute mapped actions if cooldown has passed
                current_time = time.time()
                if current_time - last_detection_time > cooldown:
                    for detection in detections:
                        gesture_name = detection.gesture_type.value
                        if input_handler.execute_gesture(gesture_name):
                            last_detection_time = current_time
                            logger.info(f"Executed gesture: {gesture_name} with confidence {detection.confidence:.2f}")
                            break
            
            # Update preview if enabled
            if preview:
                # Draw landmarks and debug info on frame
                if settings.show_debug_info:
                    frame = processor.draw_debug_info(frame, detections)
                
                # Add additional debug info
                if settings.show_debug_info:
                    # Add FPS, active profile, etc.
                    fps = camera.get_fps()
                    profile_name = config_manager.active_profile.name if config_manager.active_profile else "None"
                    
                    debug_text = [
                        f"FPS: {fps:.1f}",
                        f"Profile: {profile_name}",
                        f"Gestures: {len(config_manager.gesture_mappings)}",
                        f"Last: {time.time() - last_detection_time:.1f}s ago"
                    ]
                    
                    y = 30
                    for text in debug_text:
                        preview.add_text(frame, text, (10, y))
                        y += 30
                
                # Update preview window
                preview.update(frame)
                
                # Check for exit
                if preview.should_exit():
                    break
            
            # Sleep to control frame rate
            if settings.frame_skip > 0:
                time.sleep(settings.frame_skip / 1000.0)
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        # Clean up
        camera.stop()
        input_handler.stop()
        config_manager.stop_hot_reload()
        if preview:
            preview.close()
        
        logger.info("Application stopped")


def main():
    """Main entry point for the application."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine which mode to use
    mode = args.mode
    if mode == 'auto':
        mode = detect_mode(args.config)
    
    logger.info(f"Starting application in {mode} mode")
    
    # Run in the selected mode
    if mode == 'simple':
        run_simple_mode(args)
    else:
        run_modular_mode(args)


if __name__ == "__main__":
    main()
