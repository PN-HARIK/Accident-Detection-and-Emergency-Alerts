import cv2
import torch
import torchvision
from torchvision.transforms import functional as F

# Load the trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2  # Background + Accident
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load("C:/saisai/fasterrcnn_accident.pth", map_location=torch.device('cpu')))
model.eval()

# Open video file
video_path = "C:/saisai/vid.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    image_tensor = F.to_tensor(frame).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract detected boxes and scores
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()

    accident_detected = False

    for i, score in enumerate(scores):
        if score > 0.6:  # Confidence threshold
            accident_detected = True
            x1, y1, x2, y2 = map(int, boxes[i])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for accident
            cv2.putText(frame, f"Accident {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)

    if accident_detected:
        cv2.putText(frame, "Accident Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the video stream
    cv2.imshow("Accident Detection", frame)

    # Press 'Q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
