import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. The Corruption Process
def corrupt(x, amount):
    """Corrupt the input images x by adding noise."""
    noise = torch.randn_like(x)
    amount = amount.view(-1, 1, 1, 1) # reshape for broadcasting
    return x * (1 - amount) + noise * amount

# 2. The Model: A Simple UNet
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        ])
        self.up_layers = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        ])
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = F.relu(l(x))
            if i < len(self.down_layers) - 1:
                h.append(x)
                x = self.pool(x)
        
        for i, l in enumerate(self.up_layers):
            if i > 0:
                x = self.upsample(x)
                # Skip connection logic would go here if we had more layers
                # For this toy model, we'll keep it simple
            x = F.relu(l(x))
        return x

# 3. Data Loading (MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(mnist, batch_size=128, shuffle=True)

# 4. Training Loop
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print(f"Starting Toy Diffusion Training on {device}...")

epochs = 3
for epoch in range(epochs):
    losses = []
    for x, _ in tqdm(train_loader):
        x = x.to(device)
        noise_amount = torch.rand(x.shape[0]).to(device)
        noisy_x = corrupt(x, noise_amount)
        
        optimizer.zero_grad()
        prediction = model(noisy_x)
        loss = criterion(prediction, x) # Trying to recover original from noise
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    print(f"Epoch {epoch+1} | Loss: {np.mean(losses):.4f}")

# 5. Sampling Logic
def sample(model, steps=10):
    """Simple sampling: start with pure noise and denoise it."""
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, 28, 28).to(device)
        for i in range(steps):
            x = model(x)
    return x

print("Generating sample...")
generated = sample(model)

# Save result (mocking a visual output)
if not os.path.exists('outputs'):
    os.makedirs('outputs')
torch.save(generated, 'outputs/toy_sample.pt')
print("Sample saved to outputs/toy_sample.pt")
