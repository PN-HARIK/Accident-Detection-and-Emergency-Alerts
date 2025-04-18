import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from dataset import AccidentDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

# Load Pretrained Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the final classifier (only 2 classes: background + accident)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # 2 classes: background + accident

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# DataLoader
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = AccidentDataset("C:/saisai/Dataset/train", "C:/saisai/Dataset/train/_annotations.coco.json", transform=transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Training loop
def train_one_epoch(model, dataloader):
    model.train()
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")

# Training for 5 epochs
for epoch in range(5):  
    train_one_epoch(model, train_dataloader)
