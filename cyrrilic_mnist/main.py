import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from train_model import DeepCyrillicNet, FastDataset, extract_classes, fix_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")

root = Path(__file__).parent
zip_path = root / "cyrillic.zip"
model_path = root / "model.pth"
all_paths, class2idx = extract_classes(zip_path)
idx2class = {v: k for k, v in class2idx.items()}
num_cls = len(class2idx)

labels = [class2idx[fix_text(p.split('/')[1])] for p in all_paths]
_, test_paths = train_test_split(
    all_paths, test_size=0.2, random_state=42, stratify=labels
)

test_trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_ds = FastDataset(zip_path, test_paths, class2idx, test_trans)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=True)

model = DeepCyrillicNet(num_classes=num_cls).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("Модель загружена")

images, labels = next(iter(test_loader))
images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

plt.figure(figsize=(12, 12))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    img = images[i].cpu().squeeze() * 0.5 + 0.5
    plt.imshow(img, cmap='gray')
    true_char = idx2class[labels[i].item()]
    pred_char = idx2class[preds[i].item()]
    plt.title(f"{true_char} -> {pred_char}", fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()