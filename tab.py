import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks

task_view_active = False

# Function to recognize gestures
def recognize_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    index_finger_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_finger_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_base = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

    def distance(point1, point2):
        return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    # Check for palm gesture (all fingers extended)
    if (distance(thumb_tip, thumb_base) > 0.1 and
        distance(index_finger_tip, index_finger_base) > 0.1 and
        distance(middle_finger_tip, middle_finger_base) > 0.1 and
        distance(ring_finger_tip, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]) > 0.1 and
        distance(pinky_tip, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]) > 0.1):
        return "Show Task View"

    # Check for pinch gesture (index finger and thumb close together)
    if distance(index_finger_tip, thumb_tip) < 0.05:
        if index_finger_tip.x < thumb_tip.x:
            return "Move Left"
        elif index_finger_tip.x > thumb_tip.x:
            return "Move Right"
        elif index_finger_tip.y < thumb_tip.y:
            return "Move Up"
        elif index_finger_tip.y > thumb_tip.y:
            return "Move Down"

    # Check for scissors gesture (index finger and middle finger touching)
    if distance(index_finger_tip, middle_finger_tip) < 0.05:
        return "Select"

    # Check for fist gesture (all fingers close to the palm)
    if (distance(thumb_tip, thumb_ip) < 0.05 and
          distance(index_finger_tip, index_finger_base) < 0.05 and
          distance(middle_finger_tip, middle_finger_base) < 0.05 and
          distance(ring_finger_tip, hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]) < 0.05 and
          distance(pinky_tip, hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]) < 0.05):
        return "Close Task View"
    
    return "None"

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally for a selfie-view

    # Crop the frame to show only the hand region
    h, w, _ = frame.shape
    new_w = 200  # New width for the smaller output window
    frame = frame[:, w//2-new_w:w//2+new_w]  # Adjust the values to change the size of the output window

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the RGB frame to detect hands
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Recognize gestures and perform operations
            gesture = recognize_gesture(hand_landmarks)
            print(f"Recognized Gesture: {gesture}")  # Debugging print statement
            
            if gesture == "Show Task View" and not task_view_active:
                pyautogui.hotkey('win', 'tab')  # Show Task View
                task_view_active = True
                print("Task View activated")  # Debugging print statement
            elif task_view_active:
                if gesture == "Move Left":
                    pyautogui.press('left')  # Move cursor to the left tab
                    print("Moving left")  # Debugging print statement
                    time.sleep(0.2)  # Short delay for smoother navigation
                elif gesture == "Move Right":
                    pyautogui.press('right')  # Move cursor to the right tab
                    print("Moving right")  # Debugging print statement
                    time.sleep(0.2)  # Short delay for smoother navigation
                elif gesture == "Move Up":
                    pyautogui.press('up')  # Move cursor to the upper tab
                    print("Moving up")  # Debugging print statement
                    time.sleep(0.2)  # Short delay for smoother navigation
                elif gesture == "Move Down":
                    pyautogui.press('down')  # Move cursor to the lower tab
                    print("Moving down")  # Debugging print statement
                    time.sleep(0.2)  # Short delay for smoother navigation
                elif gesture == "Select":
                    pyautogui.press('enter')  # Select the tab
                    task_view_active = False
                    print("Selecting tab")  # Debugging print statement
                elif gesture == "Close Task View":
                    pyautogui.hotkey('esc')  # Close Task View
                    task_view_active = False
                    print("Task View closed")  # Debugging print statement

            # Draw landmarks on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow('Hand-Controlled Task View Navigator', frame)

    # Press 'q' to exit the video capture
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
video_capture.release()
cv2.destroyAllWindows()
