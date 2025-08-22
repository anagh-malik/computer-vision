import cv2
import mediapipe as mp
from ultralytics import YOLO

# 1. Initialize models
yolo_model = YOLO("yolov8n.pt")
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.7)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 2. Define object classes and their colors
# Common stationary and everyday items
common_objects = ['book', 'scissors', 'bottle', 'laptop', 'cell phone', 'pen', 'keyboard', 'mouse', 'chair', 'cup', 'spoon', 'fork', 'knife', 'bowl', 'tv', 'backpack', 'clock']

# Dictionary to map class names to specific colors (in BGR format)
color_map = {
    'hand': (255, 0, 0),       # Blue
    'person': (0, 255, 255),    # Yellow
    'face': (255, 255, 0),      # Cyan
    'default': (0, 255, 0)      # Green for all other objects
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 3. YOLO Object Detection
    results = yolo_model(frame, verbose=False)[0]
    for box in results.boxes:
        # Correctly access tensor data and convert to a list
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0].tolist())
        conf = float(box.conf[0].tolist())
        class_name = yolo_model.names.get(cls, str(cls))

        # Use specific colors for prioritized items
        if class_name in color_map:
            color = color_map.get(class_name)
        elif class_name in common_objects:
            color = color_map.get('default')
        else:
            color = (100, 100, 100)  # Gray for other items

        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 4. MediaPipe Hand Detection
    hand_results = hands.process(rgb)
    if hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=color_map.get('hand'), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))

    # 5. MediaPipe Face Detection
    face_results = face_mesh.process(rgb)
    if face_results.multi_face_landmarks:
        for faceLms in face_results.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, faceLms, mp_face.FACEMESH_TESSELATION,
                                   mp_draw.DrawingSpec(color=color_map.get('face'), thickness=1, circle_radius=1),
                                   mp_draw.DrawingSpec(color=color_map.get('face'), thickness=1, circle_radius=1))

    # 6. Show the frame
    cv2.imshow("Advanced Object Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()