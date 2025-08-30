import cv2
import mediapipe as mp
import numpy as np
import joblib

# --- Load ML Model ---
# Make sure you have trained a model and saved it as 'hand_gesture_model.pkl'
model = joblib.load("hand_gesture_model.pkl")

# --- Initialize MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- ML-based gesture classifier ---
def classify_gesture_ml(landmarks):
    # Flatten 21x3 -> 63 features
    features = np.array(landmarks).flatten().reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# --- Video Capture ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            landmarks = []
            xs, ys = [], []

            # Convert normalized landmarks to pixel coordinates
            for lm in handLms.landmark:
                cx, cy = lm.x, lm.y
                landmarks.append([cx, cy, lm.z])
                xs.append(int(cx * w))
                ys.append(int(cy * h))

            # Draw bounding rectangle around hand
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (255, 0, 0), 2)

            # Classify gesture with ML
            gesture = classify_gesture_ml(landmarks)

            # Put label below rectangle
            cv2.putText(frame, gesture, (x_min, y_max + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Recognition (ML)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
