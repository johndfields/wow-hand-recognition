#!/usr/bin/env python3
"""
Simple test script to verify MediaPipe hand tracking is working.
This will show hand landmarks and connections without the full application.
"""

import cv2
import mediapipe as mp
import numpy as np

def main():
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    
    # Initialize Hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("MediaPipe Hand Tracking Test")
    print("- Hold your hand up to the camera")
    print("- You should see green dots (landmarks) and white lines (connections)")
    print("- Press 'q' to quit")
    print("- Press 'l' to toggle landmarks only")
    print("- Press 'c' to toggle connections only")
    print("- Press 'b' to toggle both")
    
    show_landmarks = True
    show_connections = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks and connections
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand label (Left/Right)
                hand_label = "Unknown"
                if results.multi_handedness:
                    if hand_idx < len(results.multi_handedness):
                        hand_label = results.multi_handedness[hand_idx].classification[0].label
                
                # Custom drawing styles
                landmark_style = mp_drawing.DrawingSpec(
                    color=(0, 255, 0),  # Green landmarks
                    thickness=2,
                    circle_radius=4
                )
                
                connection_style = mp_drawing.DrawingSpec(
                    color=(255, 255, 255),  # White connections
                    thickness=2
                )
                
                # Draw based on toggle settings
                if show_landmarks and show_connections:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_style,
                        connection_style
                    )
                elif show_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks,
                        None,  # No connections
                        landmark_style,
                        None
                    )
                elif show_connections:
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        None,  # No landmark points
                        connection_style
                    )
                
                # Draw hand label
                palm_x = int(hand_landmarks.landmark[9].x * width)
                palm_y = int(hand_landmarks.landmark[9].y * height)
                
                cv2.putText(frame, f"{hand_label} Hand", 
                           (palm_x - 50, palm_y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Draw a center point
                cv2.circle(frame, (palm_x, palm_y), 8, (0, 0, 255), -1)
        
        # Draw instructions
        instructions = [
            f"Landmarks: {'ON' if show_landmarks else 'OFF'} (press 'l')",
            f"Connections: {'ON' if show_connections else 'OFF'} (press 'c')",
            f"Both: {'ON' if show_landmarks and show_connections else 'OFF'} (press 'b')",
            "Press 'q' to quit"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.rectangle(frame, (5, y_offset - 20), (400, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(frame, instruction, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30
        
        # Show the frame
        cv2.imshow('MediaPipe Hand Tracking Test', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            show_landmarks = not show_landmarks
            print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord('c'):
            show_connections = not show_connections
            print(f"Connections: {'ON' if show_connections else 'OFF'}")
        elif key == ord('b'):
            show_landmarks = not show_landmarks
            show_connections = not show_connections
            print(f"Both: {'ON' if show_landmarks and show_connections else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed")

if __name__ == "__main__":
    main()
