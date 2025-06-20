import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
from twilio.rest import Client
import time
from datetime import datetime

# ========== Twilio Setup ==========
account_sid = 'id'
auth_token = 'token'
from_number = '‪from'     
to_number = '‪to‬'    

client = Client(account_sid, auth_token)

# ========== Alert Configuration ==========
camera_code = "Camera 09"
gps_location = "Adhiyamaan College of Engineering, NH44, Krishnagiri, Hosur - 635109, Tamil Nadu, India {lat:12.7173595, lon:77.8703219}"

def send_alert_sms(score, max_retries=3, delay_seconds=5):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_body = (
        f" Accident Detected!\n"
        f" Date & Time: {current_time}\n"
        f" Camera: {camera_code}\n"
        f" Confidence: {score:.2f}\n"
        f" Location: {gps_location}"
    )

    for attempt in range(1, max_retries + 1):
        try:
            message = client.messages.create(
                body=message_body,
                from_=from_number,
                to=to_number
            )
            print(f" SMS sent successfully (Attempt {attempt}): {message.sid}")
            break
        except Exception as e:
            print(f" SMS sending failed (Attempt {attempt}): {e}")
            if attempt < max_retries:
                print(f" Retrying in {delay_seconds} seconds...")
                time.sleep(delay_seconds)
            else:
                print(" All retries failed. SMS not sent.")

# ========== Model Setup ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

num_classes = 2  # background + accident
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model_path = "D:/saisai/fasterrcnn_accident.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ========== Video Setup ==========
video_path = "D:/saisai/1.mp4"
# video_path = "D:/saisai/2.mp4"
# video_path = "D:/saisai/3.mov"
# video_path = "D:/saisai/4.mov"
cap = cv2.VideoCapture(video_path)

# Cooldown to avoid spamming SMS
last_alert_time = 0
cooldown_seconds = 180  # 3 minutes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (416, 416))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(rgb_frame).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.autocast("cuda" if device.type == "cuda" else "cpu"):
            predictions = model(image_tensor)

    boxes = predictions[0]['boxes'].detach().cpu().numpy()
    scores = predictions[0]['scores'].detach().cpu().numpy()

    accident_detected = False
    max_score = 0

    for i, score in enumerate(scores):
        if score > 0.9:
            accident_detected = True
            max_score = max(max_score, score)
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(resized_frame, f"Accident {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if accident_detected:
        cv2.putText(resized_frame, "Accident Detected!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Check cooldown before sending SMS
        if time.time() - last_alert_time > cooldown_seconds:
            send_alert_sms(max_score)
            last_alert_time = time.time()

    cv2.imshow("Video Accident Detection", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
