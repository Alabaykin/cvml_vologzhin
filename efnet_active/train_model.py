import torch
import cv2
import torchvision
import numpy as np
from torchvision import transforms
from PIL import Image
import time
from collections import deque   
from pathlib import Path


save_path = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_a(model_path):
    weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    model = torchvision.models.alexnet(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(features, 1)
    if model_path is not None and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model.to(device)

def build_e(model_path):
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False
    features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(features, 1)
    if model_path is not None and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model.to(device)

model_file_a = save_path / "model_a.pth"
model_file_e = save_path / "model_e.pth"

model_a = build_a(model_file_a)
model_e = build_e(model_file_e)


criterion = torch.nn.BCEWithLogitsLoss()
optimizer_a = torch.optim.Adam(filter(lambda p: p.requires_grad, model_a.parameters()), lr=0.001)
optimizer_e = torch.optim.Adam(filter(lambda p: p.requires_grad, model_e.parameters()), lr=0.001)



transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def train(buffer):
    if len(buffer) < 10:
        return None, None
    
    model_a.train()
    model_e.train()
    
    images, labels = buffer.get_batch()
    images = images.to(device) 
    labels = labels.to(device) 
    
    # AlexNet
    optimizer_a.zero_grad()
    predictions_a = model_a(images).squeeze(-1)
    loss_a = criterion(predictions_a, labels)
    loss_a.backward()
    optimizer_a.step()
    
    # EfficientNet
    optimizer_e.zero_grad()
    predictions_e = model_e(images).squeeze(-1)
    loss_e = criterion(predictions_e, labels)
    loss_e.backward()
    optimizer_e.step()
    
    return loss_a.item(), loss_e.item()

def predict(frame, model):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = model(tensor).squeeze(-1)
        prob = torch.sigmoid(predicted).item()
    label = "Person" if prob > 0.5 else "No Person"
    return label, prob

class Buffer:
    def __init__(self, maxsize=16):
        self.frames = deque(maxlen=maxsize)
        self.labels = deque(maxlen=maxsize)

    def append(self, frame, label):
        self.frames.append(frame)
        self.labels.append(label)

    def __len__(self):
        return len(self.frames)
    
    def get_batch(self):
        images = torch.stack(list(self.frames))
        labels = torch.tensor(list(self.labels), dtype=torch.float32)
        return images, labels

    

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera",cv2.WINDOW_GUI_NORMAL)
buffer = Buffer()
count_labeled = 0

while True:
    _, frame = cap.read()

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if key == ord("q"):
        break
    elif key == ord("1"):
        tensor = transform(image)
        buffer.append(tensor, 1.0)
        count_labeled += 1
    elif key == ord("2"):
        tensor = transform(image)
        buffer.append(tensor, 0.0)
        count_labeled += 1
    elif key == ord("p"): 
        t= time.time()
        label_a, conf_a = predict(frame, model_a)
        label_e, conf_e = predict(frame, model_e)
        print(f"Elapsed: {time.time() - t}s | AlexNet: {label_a} ({conf_a}), EfficialNet: {label_e} ({conf_e})")
    elif key == ord("s"):
        torch.save(model_a.state_dict(), model_file_a) 
        torch.save(model_e.state_dict(), model_file_e)

    print(f"Buffer Size: {len(buffer),count_labeled}")
    if count_labeled >= buffer.frames.maxlen:
        loss_a, loss_e = train(buffer)
        loss = train(buffer)
        if loss_a is not None and loss_e is not None:
            print(f"Loss AlexNet = {loss_a}, Loss EfficialNet = {loss_e}")
            log_text = f"Loss AlexNet = {loss_a}, Loss EffNet = {loss_e}"
        count_labeled = 0