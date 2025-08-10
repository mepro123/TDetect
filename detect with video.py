import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import torch
import os

# Ask user for video file path
video_path = input("Credits: \nPowered by YOLOv5 - Ultralytics\nMade by @thammammas (Discord)\n\nhttps://github.com/mepro123/TDetect\n\nEnter full path to the video file: ").strip()

if not os.path.isfile(video_path):
    print(f"Error: File does not exist: {video_path}")
    exit()

# Load nano model for speed
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

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
        small_frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        results = model(small_frame)

        scale_x = frame.shape[1] / small_frame.shape[1]
        scale_y = frame.shape[0] / small_frame.shape[0]

        detections = []
        for *box, conf, cls in results.xyxy[0]:
            if int(cls) == 0:
                x1, y1, x2, y2 = box
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                detections.append((x1, y1, x2, y2, conf))

    for (x1, y1, x2, y2, conf) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('AI Rig Detection', frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

    frame_count += 10
cap.release()
cv2.destroyAllWindows()
