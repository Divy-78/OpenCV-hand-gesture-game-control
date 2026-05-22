import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# drawing utilitys
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Cooldown time
last_action_time = time.time()

while True:

    success, frame = cap.read()

    if not success:
        break

    # Flip frame
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    result = hands.process(rgb)

    # Frame dimensions
    h, w, c = frame.shape

    # -----------------------------
    # Draw Control Zones
    # -----------------------------

    # Horizontal lines
    cv2.line(frame, (0, 150), (w, 150), (0, 255, 0), 2)
    cv2.line(frame, (0, 350), (w, 350), (0, 255, 0), 2)

    # Vertical lines
    cv2.line(frame, (200, 0), (200, h), (0, 255, 0), 2)
    cv2.line(frame, (450, 0), (450, h), (0, 255, 0), 2)

    # Labels
    cv2.putText(frame, "UP", (280, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.putText(frame, "DOWN", (240, 430),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.putText(frame, "LEFT", (50, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.putText(frame, "RIGHT", (500, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    # -----------------------------
    # Hand Detection
    # -----------------------------

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip
            palm = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]

            # Convert coordinates
            x = int(palm.x * w)
            y = int(palm.y * h)

            # Draw finger point
            cv2.circle(frame, (x, y), 15, (255, 0, 255), cv2.FILLED)

            # -----------------------------
            # Gesture Controls
            # -----------------------------

            current_time = time.time()

            # Add cooldown of 0.5 sec
            if current_time - last_action_time > 0.7:

                # UP
                if y < 150:
                    pyautogui.press('up')
                    print("UP")

                # DOWN
                elif y > 350:
                    pyautogui.press('down')
                    print("DOWN")

                # LEFT
                elif x < 200:
                    pyautogui.press('left')
                    print("LEFT")

                # RIGHT
                elif x > 450:
                    pyautogui.press('right')
                    print("RIGHT")

                # Update cooldown timer
                last_action_time = current_time

    # Show webcam
    cv2.namedWindow("Hand Gesture Subway Surfers", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Gesture Subway Surfers", 320, 240)
    cv2.imshow("Hand Gesture Subway Surfers", frame)

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()