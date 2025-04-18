import torch
import os
import json
import cv2
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), "best_accident.pth")  # Save best model
            print("🔹 Model improved. Saving new best model!")
        else:
            self.counter += 1
            print(f"🔸 No improvement for {self.counter} epochs.")
        
        return self.counter >= self.patience  # Stop training if counter >= patience

# Dataset class
class AccidentDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)

        self.image_ids = list({img["id"] for img in self.coco_data["images"]})
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations = {ann["image_id"]: [] for ann in self.coco_data["annotations"]}
        
        for ann in self.coco_data["annotations"]:
            self.annotations[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]  # Correct indexing
        image_info = self.images[image_id]

        img_path = os.path.join(self.image_dir, image_info["file_name"])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get bounding boxes
        boxes, labels = [], []
        for ann in self.annotations.get(image_info["id"], []):
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:  # Ensure valid bounding boxes
                boxes.append([x, y, x + w, y + h])
                labels.append(1)  # Label '1' means "Accident"

        # If no boxes are found, return empty boxes and labels
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target

# Initialize the model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the classifier to match your dataset
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # background + accident

# Set device to CPU explicitly
device = torch.device("cpu")
model.to(device)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Lower learning rate for stable training

# Initialize Early Stopping
early_stopping = EarlyStopping(patience=2, min_delta=0.001)

# Training loop with validation loss tracking
def train_one_epoch(model, dataloader, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        # Backpropagation
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}] completed, Average Loss: {avg_loss:.4f}")

    return avg_loss

# Prepare dataset and dataloader
train_dataset = AccidentDataset(
    image_dir="C:/saisai/Dataset/train", 
    annotation_file="C:/saisai/Dataset/train/_annotations.coco.json", 
    transform=ToTensor()
)

# Reduce batch size for CPU training
batch_size = 16  

# Prepare DataLoader with CPU-friendly batch size
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x))
)

# Training the model with Early Stopping
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    val_loss = train_one_epoch(model, train_dataloader, optimizer, epoch)

    # Check for early stopping
    if early_stopping(val_loss, model):
        print("⏹ Early stopping triggered! Stopping training.")
        break

# Save the final model
torch.save(model.state_dict(), "new_accident.pth")
print("✅ Model training completed and saved.")
