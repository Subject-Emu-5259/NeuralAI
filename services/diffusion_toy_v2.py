import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import UNet2DModel, DDPMScheduler
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Config
image_size = 28
train_batch_size = 128
num_epochs = 1
learning_rate = 1e-4

# 2. Model: UNet2DModel from diffusers (small version for toy)
model = UNet2DModel(
    sample_size=image_size,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(32, 64, 64),
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
).to(device)

# 3. Scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=100)

# 4. Data Loading (MNIST)
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=preprocess)
train_dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

# 5. Training
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Starting NeuralAI Diffusion Toy V2 Training on {device}...")

for epoch in range(num_epochs):
    losses = []
    for step, (images, _) in enumerate(tqdm(train_dataloader)):
        images = images.to(device)
        noise = torch.randn(images.shape).to(device)
        bs = images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bs,), device=device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

        # Predict the noise residual
        noise_pred = model(noisy_images, timesteps).sample

        loss = F.mse_loss(noise_pred, noise)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())

    print(f"Epoch {epoch} | Loss: {np.mean(losses):.6f}")

# 6. Sampling Logic
def generate_samples(model, scheduler, num_samples=1):
    model.eval()
    # Start from random noise
    sample = torch.randn(num_samples, 1, image_size, image_size).to(device)
    
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            residual = model(sample, t).sample
        
        # Compute previous image: x_t -> x_t-1
        sample = scheduler.step(residual, t, sample).prev_sample
    
    return sample

print("Generating NeuralAI Diffusion sample...")
generated = generate_samples(model, noise_scheduler)

# Save result
output_dir = "/home/workspace/Projects/NeuralAI/storage/images"
os.makedirs(output_dir, exist_ok=True)

# Convert to PIL and save
gen_img = (generated[0] / 2 + 0.5).clamp(0, 1).cpu().numpy().squeeze()
gen_img = (gen_img * 255).astype(np.uint8)
img = Image.fromarray(gen_img)
img.save(os.path.join(output_dir, "toy_v2_sample.png"))

# Save model checkpoint
checkpoint_dir = "/home/workspace/Projects/NeuralAI/checkpoints/diffusion_toy"
os.makedirs(checkpoint_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "unet_toy.pt"))

print(f"Sample saved to {output_dir}/toy_v2_sample.png")
print(f"Model saved to {checkpoint_dir}/unet_toy.pt")
