import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import os
from ultralytics import YOLO

# Ask user for video file path
video_path = input(
    "Credits: \nPowered by YOLOv8 - Ultralytics\nMade by @thammammas (Discord)\n\n"
    "https://github.com/mepro123/TDetect\n\nEnter full path to the video file: "
).strip()

if not os.path.isfile(video_path):
    print(f"Error: File does not exist: {video_path}")
    exit()

# Load nano model for speed
model = YOLO("yolov8n.pt")  # yolov8n = nano model

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps) if fps > 0 else 30

frame_skip = 3
frame_count = 0
detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        # Resize frame for model inference - YOLOv8 prefers square images, but supports rectangular
        input_width = 640
        input_height = int(frame.shape[0] * input_width / frame.shape[1])
        small_frame = cv2.resize(frame, (input_width, input_height))

        results = model(small_frame)[0]  # inference results

        scale_x = frame.shape[1] / input_width
        scale_y = frame.shape[0] / input_height

        detections = []
        for result in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = result
            if int(cls) == 0:  # person class
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                detections.append((x1, y1, x2, y2, conf))

    for (x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f'Person {conf:.2f}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow('AI Rig Detection', frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
