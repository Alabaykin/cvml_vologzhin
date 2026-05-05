import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transform
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import string

class ImageDataset(Dataset):
    def __init__(self, n=5000, size=128, mode=1):
        super().__init__()
        self.n = n
        self.size = size
        self.mode = mode
        self.transform = transform.Compose([
            transform.ToTensor(),
        ])
        self.chars = string.ascii_uppercase + string.digits
        # Load a font that supports different sizes if possible, otherwise default
        try:
            self.font = ImageFont.truetype("arial.ttf", 20)
        except:
            self.font = ImageFont.load_default()

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # Create a white background image
        image = Image.new("L", (self.size, self.size), color=255)
        draw = ImageDraw.Draw(image)
        
        # Scenario logic
        if self.mode == 1:
            # 1. Текст фиксированный, случайным образом меняется позиция
            text = "FIXED"
            x = random.randint(0, self.size - 60)
            y = random.randint(0, self.size - 20)
        elif self.mode == 2:
            # 2. Текст случайный, но всегда одной длины, позиция фиксированная
            text = "".join(random.choices(self.chars, k=5))
            x, y = self.size // 2 - 30, self.size // 2 - 10
        elif self.mode == 3:
            # 3. Текст случайный, случайной длины, позиция фиксированная
            length = random.randint(1, 8)
            text = "".join(random.choices(self.chars, k=length))
            x, y = self.size // 2 - 30, self.size // 2 - 10
        elif self.mode == 4:
            # 4. Текст случайный, случайной длины, позиция случайная
            length = random.randint(1, 8)
            text = "".join(random.choices(self.chars, k=length))
            x = random.randint(0, max(0, self.size - (length * 12)))
            y = random.randint(0, self.size - 20)
        else:
            text = "ERR"
            x, y = 0, 0
        
        draw.text((x, y), text, fill=0, font=self.font)
        tensor = self.transform(image)
        return tensor, tensor

class Encoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.bottleneck = nn.Linear(256 * 8 * 8, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()
        self.bottleneck = nn.Linear(latent, 256 * 8 * 8)  

        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1), # 128x128
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 8, 8)
        x = self.features(x)
        return x

def train_model(mode, epochs=10, device='cpu'):
    print(f"\n--- Training Scenario {mode} ---")
    dataset = ImageDataset(n=2000, size=128, mode=mode)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            latent = encoder(imgs)
            output = decoder(latent)
            loss = criterion(output, imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    return losses, encoder, decoder

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_losses = {}
    scenarios = {
        1: "Fixed text, Random position",
        2: "Random text (fixed len), Fixed position",
        3: "Random text (random len), Fixed position",
        4: "Random text (random len), Random position"
    }

    for mode in scenarios:
        losses, enc, dec = train_model(mode, epochs=10, device=device)
        all_losses[mode] = losses
        # Save models for the last scenario if needed, or all
        torch.save(enc.state_dict(), f"encoder_mode_{mode}.pth")
        torch.save(dec.state_dict(), f"decoder_mode_{mode}.pth")

    # Plot results
    plt.figure(figsize=(10, 6))
    for mode, losses in all_losses.items():
        plt.plot(losses, label=scenarios[mode])
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Encoder-Decoder Performance across Variability Scenarios')
    plt.legend()
    plt.grid(True)
    plt.savefig("variability_analysis.png")
    # plt.show() # Commented out for non-interactive execution

    print("\nConclusion Analysis:")
    for mode in all_losses:
        print(f"Scenario {mode} final loss: {all_losses[mode][-1]:.6f}")
    
    # Comparison logic
    # We compare mode 1 (position variability) and mode 2 (content variability)
    if all_losses[1][-1] > all_losses[2][-1]:
        print("\nConclusion: Random position has a STRONGER impact on training than random text content.")
    else:
        print("\nConclusion: Random text content has a STRONGER impact on training than random position.")