import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from combiner import VAE,vae_loss_function
import matplotlib.pyplot as plt

# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),       # Convert images to tensors
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Paths to datasets
train_path = 'train/'
test_path = 'test/'
val_path = 'val/'

# Create datasets for train, test, and validation
train_dataset = datasets.ImageFolder(root=train_path, transform=transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=transform)
val_dataset = datasets.ImageFolder(root=val_path, transform=transform)

batch_size = 32  # Adjust based on memory constraints
num_samples = int(len(train_dataset) * 0.1)
indices = np.random.choice(len(train_dataset), num_samples, replace=False)
train_subset = Subset(train_dataset, indices)
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

# DataLoaders for loading data in batches
#train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Initialize model, optimizer, and device
device = torch.device('cpu')  # Use 'cuda' if a GPU is available
vae = VAE(64).to(device)
optimizer = optim.Adam(vae.parameters(), lr=0.001)

epochs = 5  # Start with fewer epochs
batch_size = 16  # Adjust based on CPU constraints

for epoch in range(epochs):
    vae.train()
    train_loss = 0
    for images, _ in train_loader:
        images = images.to(device)
        reconstructed, mu, logvar = vae(images)
        loss = vae_loss_function(reconstructed, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader)}")

    # Save checkpoint
    torch.save(vae.state_dict(), f'vae_epoch_{epoch + 1}.pth')
    print(f"Checkpoint saved for epoch {epoch + 1}")

vae.eval()
val_loss = 0

with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(device)
        reconstructed, mu, logvar = vae(images)
        loss = vae_loss_function(reconstructed, images, mu, logvar)
        val_loss += loss.item()

print(f"Validation Loss: {val_loss / len(val_loader)}")

vae.eval()
latent_samples = torch.randn(16, 64).to(device)  # Generate 16 random latent vectors
generated_images = vae.decoder(latent_samples)

# Convert tensors to images
generated_images = generated_images.cpu().detach().numpy().transpose(0, 2, 3, 1)

# Plot generated images
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow((generated_images[i] * 255).astype("uint8"))
    ax.axis('off')
plt.show()

def interpolate(latent1, latent2, steps=10):
    interpolated = []
    for alpha in np.linspace(0, 1, steps):
        z = latent1 * (1 - alpha) + latent2 * alpha
        interpolated.append(z)
    return torch.stack(interpolated)

vae.eval()
latent1 = torch.randn(1, 64).to(device)
latent2 = torch.randn(1, 64).to(device)

# Get interpolated latent vectors
interpolated_latents = interpolate(latent1, latent2, steps=10).to(device)
interpolated_images = vae.decoder(interpolated_latents).cpu().detach().numpy()

# Plot interpolated images
fig, axes = plt.subplots(1, 10, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow((interpolated_images[i].transpose(1, 2, 0) * 255).astype("uint8"))
    ax.axis('off')
plt.show()

vae.eval()
latent_vector = torch.randn(1, 64).to(device)

# Modify specific latent dimensions
latent_vector[:, 0] += 2  # Example: Increase value in the 1st latent dimension
latent_vector[:, 1] -= 1  # Example: Decrease value in the 2nd latent dimension

# Decode the modified latent vector
modified_image = vae.decoder(latent_vector).cpu().detach().numpy().squeeze().transpose(1, 2, 0)

# Plot the image
plt.imshow((modified_image * 255).astype("uint8"))
plt.axis('off')
plt.show()

torch.save(vae.state_dict(), 'vae_final.pth')

