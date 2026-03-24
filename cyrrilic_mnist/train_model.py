import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path
import zipfile
from PIL import Image
import io
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")

torch.manual_seed(42)

ROOT = Path(__file__).parent
ZIP_PATH = ROOT / "cyrillic.zip"
MODEL_SAVE = ROOT / "model.pth"
PLOT_SAVE = ROOT / "train.png"

BATCH = 64
EPOCHS = 15
LR = 0.001

def extract_classes(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as z:
        all_imgs = [f for f in z.namelist() if f.endswith('.png') and '/' in f]
        classes = sorted(set(f.split('/')[1] for f in all_imgs))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return all_imgs, class_to_idx

class FastDataset(Dataset):
    def __init__(self, zip_path, image_paths, class_dict, transform=None):
        self.transform = transform
        self.data = [] 
        with zipfile.ZipFile(zip_path, 'r') as z:
            for path in image_paths:
                img_bytes = z.read(path)
                img_pil = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
                img = np.array(img_pil)[:, :, 3]
                img = np.expand_dims(img, axis=-1) 
                label = class_dict[path.split('/')[1]]
                self.data.append((img, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class DeepCyrillicNet(nn.Module):
    def __init__(self, num_classes=34):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.relu4 = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    all_paths, class2idx = extract_classes(ZIP_PATH)
    num_cls = len(class2idx)
    print("Найдены классы:", list(class2idx.keys()))
    print(f"Всего изображений: {len(all_paths)}")

    labels = [class2idx[p.split('/')[1]] for p in all_paths]
    train_paths, test_paths = train_test_split(
        all_paths, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Обучающих: {len(train_paths)}, тестовых: {len(test_paths)}")

    train_trans = transforms.Compose([
        transforms.ToPILImage(),        
        transforms.Resize((28, 28)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_ds = FastDataset(ZIP_PATH, train_paths, class2idx, train_trans)
    test_ds  = FastDataset(ZIP_PATH, test_paths,  class2idx, test_trans)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    model = DeepCyrillicNet(num_classes=num_cls).to(device)
    print("Параметров:", sum(p.numel() for p in model.parameters()))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    loss_hist = []
    acc_hist = []

    if not MODEL_SAVE.exists():
        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100.0 * correct / total
            loss_hist.append(epoch_loss)
            acc_hist.append(epoch_acc)
            scheduler.step(epoch_acc)

            print(f"Эпоха {epoch+1:2d}/{EPOCHS} | Loss: {epoch_loss:.4f} | "
                  f"Acc: {epoch_acc:.2f}%")

        torch.save(model.state_dict(), MODEL_SAVE)
        print(f"Модель сохранена в {MODEL_SAVE}")

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(loss_hist)
        plt.title("Loss")
        plt.subplot(1, 2, 2)
        plt.plot(acc_hist)
        plt.title("Acc")
        plt.tight_layout()
        plt.savefig(PLOT_SAVE)
        plt.show()
        print(f"График сохранён в {PLOT_SAVE}")
    else:
        model.load_state_dict(torch.load(MODEL_SAVE, map_location=device))
        print("Модель загружена из файла")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    print(f"Тестовая точность: {100.0 * correct / total:.2f}%")