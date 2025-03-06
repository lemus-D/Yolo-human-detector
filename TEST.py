import cv2
import torch
import requests
import datetime

# Telegram Bot API Configuration
TELEGRAM_BOT_TOKEN = "7115856487:AAFYnE4YejYVs9vUg2F3Ea-efL5dD_vWhis"  # Replace with your bot token
TELEGRAM_CHAT_ID = "5921390367"  # Replace with your chat ID

# Frame skipping parameter (process every N-th frame)
FRAME_SKIP = 5  # Adjust to balance performance vs. detection speed
frame_counter = 0  # Counter to track frames

# Confidence threshold (only send alerts for high-confidence detections)
CONFIDENCE_THRESHOLD = 0.75  # Adjust for accuracy

def send_telegram_alert(frame):
    """Send a Telegram message with an image attachment."""
    message = f"🚨 Alert: A person was detected at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}!"
    
    # Save the detected frame as an image
    image_path = "detected_person.jpg"
    cv2.imwrite(image_path, frame)

    # Send text message
    url_message = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url_message, data={"chat_id": TELEGRAM_CHAT_ID, "text": message})

    # Send the image
    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(image_path, "rb") as photo:
        requests.post(url_photo, data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": photo})

    print("✅ Telegram alert with image sent successfully!")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Increase frame counter
    frame_counter += 1

    # Only run YOLO detection every N-th frame
    if frame_counter % FRAME_SKIP == 0:
        results = model(frame)

        # Loop through detections and check for a person
        person_detected = False
        for *xyxy, conf, cls in results.xyxy[0]:  
            if model.names[int(cls)] == "person" and conf > CONFIDENCE_THRESHOLD:  # High-confidence detections only
                person_detected = True
                label = f"Person {conf:.2f}"
                x1, y1, x2, y2 = map(int, xyxy)

                # Draw bounding box & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Send Telegram alert with the frame
        if person_detected:
            send_telegram_alert(frame)

    # Show the video feed with detections
    cv2.imshow('YOLOv5 Live Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
