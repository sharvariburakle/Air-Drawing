import cv2
import numpy as np
import mediapipe as mp
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Mediapipe setup 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#  Initialize webcam 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

#  Drawing parameters 
drawing_enabled = False
brush_color = (201, 252, 189) 
brush_thickness = 5
prev_x, prev_y = None, None

# UI parameters 
menu_height = 80
color_palette = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "mint green": (201, 252, 189), 
    "white": (255, 255, 255),
}
current_color_name = "mint green" 

#  Functions for gesture recognition and UI 
def get_distance(point1, point2):
    """Calculates Euclidean distance between two points."""
    return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def draw_menu_overlay(menu_area, current_color_name, thickness, drawing_status):
    """Draws the UI menu on the frame with an aesthetic touch."""
    # Background rectangle
    cv2.rectangle(menu_area, (0, 0), (menu_area.shape[1], menu_area.shape[0]), (25, 25, 25), -1)
    
    # Title
    cv2.putText(menu_area, "AIR DRAWING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    
    # Status and thickness indicators
    status_text = "Drawing: ON" if drawing_status else "Drawing: OFF"
    status_color = (0, 255, 0) if drawing_status else (0, 0, 255)
    cv2.putText(menu_area, status_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)

    thickness_text = f"Thickness: {thickness}"
    cv2.putText(menu_area, thickness_text, (200, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    color_text = f"Color: {current_color_name.upper()}"
    cv2.putText(menu_area, color_text, (400, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Color palette squares with a subtle glow for the selected color
    palette_width = 40
    palette_spacing = 10
    total_palette_width = len(color_palette) * (palette_width + palette_spacing)
    start_x_offset = menu_area.shape[1] - total_palette_width - 10
    
    for i, (name, color) in enumerate(color_palette.items()):
        start_x = start_x_offset + i * (palette_width + palette_spacing)
        end_x = start_x + palette_width
        
        cv2.rectangle(menu_area, (start_x, 15), (end_x, 55), color, -1)
        if name == current_color_name:
            cv2.rectangle(menu_area, (start_x - 2, 13), (end_x + 2, 57), (255, 255, 255), 2)


#  Main loop
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    canvas = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Initialize canvas if it's the first frame
        if canvas is None:
            canvas = np.zeros_like(frame)

        # Create menu area for overlay
        menu_area = np.zeros((menu_height, w, 3), dtype=np.uint8)
        draw_menu_overlay(menu_area, current_color_name, brush_thickness, drawing_enabled)
        frame[0:menu_height, 0:w] = menu_area

        # Process with Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get index and thumb tips
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Convert to pixel coordinates
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

                # Gesture recognition
                pinch_distance = get_distance(index_tip, thumb_tip)
                
                # Check for drawing gesture (pinch)
                if pinch_distance < 0.08:
                    drawing_enabled = True
                    thickness_dist = get_distance(pinky_tip, thumb_tip)
                    brush_thickness = int(thickness_dist * 150) + 5
                    brush_thickness = max(5, min(30, brush_thickness))
                    # Draw a cursor circle
                    cv2.circle(frame, (index_x, index_y), brush_thickness // 2, (255, 255, 255), 2)
                    cv2.circle(frame, (index_x, index_y), brush_thickness // 2, brush_color, -1)
                else:
                    drawing_enabled = False

                # Peace sign gesture (clear canvas)
                peace_distance = get_distance(index_tip, middle_tip)
                if peace_distance < 0.05 and (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y):
                    canvas = np.zeros_like(frame)
                    prev_x, prev_y = None, None

                # Open palm for color selection
                if all(landmark.y < hand_landmarks.landmark[mp_hands.HandLandmark(i+1)*4].y for i, landmark in enumerate(
                    [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]])):

                    if index_y < menu_height:
                        start_x_offset = w - (len(color_palette) * 50) - 10 
                        for i, (name, color) in enumerate(color_palette.items()):
                            start_x = start_x_offset + i * 50
                            end_x = start_x + 40
                            if start_x <= index_x <= end_x:
                                brush_color = color
                                current_color_name = name
                                break
                    prev_x, prev_y = None, None

                #  Drawing logic 
                if drawing_enabled and index_y > menu_height:
                    if prev_x is not None and prev_y is not None:
                        if prev_y > menu_height:
                            cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), brush_color, brush_thickness)
                    prev_x, prev_y = index_x, index_y
                else:
                    prev_x, prev_y = None, None
        else:
            prev_x, prev_y = None, None

        # Merge frame and canvas
        combined_frame = cv2.addWeighted(frame, 1, canvas, 1, 0)
        
        cv2.imshow('Enhanced Air Drawing', combined_frame)

        # Key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            drawing_enabled = not drawing_enabled
        elif key == ord('c'):
            canvas = np.zeros_like(frame)
            prev_x, prev_y = None, None

cap.release()
cv2.destroyAllWindows()