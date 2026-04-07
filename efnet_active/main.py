import torch
import cv2
import torchvision
from torchvision import transforms
from pathlib import Path
import time
save_path = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_a(model_path):
    model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
    features = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(features, 1)
    if model_path is not None and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model.to(device).eval()

def build_e(model_path):
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(features, 1)
    if model_path is not None and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model.to(device).eval()


model_alexnet = build_a(save_path / "model_a.pth")
model_efnet = build_e(save_path / "model_e.pth")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(frame, model):
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = model(tensor).squeeze(-1)
        prob = torch.sigmoid(predicted).item()
    label = "Person" if prob > 0.5 else "No Person"
    return label, prob

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    elif key == ord("p"):
        t = time.time()
        label_a, conf_a = predict(frame, model_alexnet)
        label_e, conf_e = predict(frame, model_efnet)
        print(f"AlexNet: {label_a}; Confidence: {conf_a}")
        print(f"EfficientNet: {label_e}; Confidence: {conf_e}")
        print("+" * 60)

cap.release()
cv2.destroyAllWindows()