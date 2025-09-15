# Import libraries for model, data handling, and visualization
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set device (CPU due to GPU limits)
device = torch.device('cpu')
print(f'Using device: {device}')



# Set paths to dataset
base_path = '/content/drive/MyDrive/archive/dataset'

train_img_dir = os.path.join(base_path, 'train/images')
train_mask_dir = os.path.join(base_path, 'train/masks')
val_img_dir = os.path.join(base_path, 'validation/images')
val_mask_dir = os.path.join(base_path, 'validation/masks')
test_img_dir = os.path.join(base_path, 'test/images')
test_mask_dir = os.path.join(base_path, 'test/masks')

# Verify file counts
print('Train images:', len(os.listdir(train_img_dir)) if os.path.exists(train_img_dir) else 'Not found')
print('Train masks:', len(os.listdir(train_mask_dir)) if os.path.exists(train_mask_dir) else 'Not found')
print('Val images:', len(os.listdir(val_img_dir)) if os.path.exists(val_img_dir) else 'Not found')
print('Val masks:', len(os.listdir(val_mask_dir)) if os.path.exists(val_mask_dir) else 'Not found')
print('Test images:', len(os.listdir(test_img_dir)) if os.path.exists(test_img_dir) else 'Not found')
print('Test masks:', len(os.listdir(test_mask_dir)) if os.path.exists(test_mask_dir) else 'Not found')

# Check filename mapping
train_img_files = sorted([f for f in os.listdir(train_img_dir) if f.endswith('.png')])
train_mask_files = sorted([f for f in os.listdir(train_mask_dir) if f.endswith('.png')])
print('Sample train images:', train_img_files[:5])
print('Sample train masks:', train_mask_files[:5])

# Define dataset class for reloading
class LandslideDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = [f for f in sorted(os.listdir(img_dir)) if f.endswith('.png')]
        mask_files = set(os.listdir(mask_dir))
        self.img_files = [f for f in self.img_files if f.replace('image_', 'mask_') in mask_files]
        print(f"Found {len(self.img_files)} matching pairs in {img_dir}")

        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        mask_file = img_file.replace('image_', 'mask_')
        mask_path = os.path.join(self.mask_dir, mask_file)

        try:
            img = np.array(Image.open(img_path))
            if img.shape[-1] == 4:
                img = img[:, :, :3]  # Drop alpha
            mask = np.array(Image.open(mask_path))
            if mask.shape[-1] == 4:
                mask = mask[:, :, 0]  # Channel 0
            if img.dtype != np.uint8 or mask.dtype != np.uint8:
                print(f'Invalid dtypes: {img_path}={img.dtype}, {mask_path}={mask.dtype}')
                return None, None
            img = A.Resize(128, 128)(image=img)['image']
            mask = A.Resize(128, 128)(image=mask)['image']
            mask = (mask > 0).astype(np.uint8)
            if img.shape != (128, 128, 3):
                print(f'Invalid image shape {img_path}: {img.shape}')
                return None, None
            if mask.shape != (128, 128):
                print(f'Invalid mask shape {mask_path}: {mask.shape}')
                return None, None
            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask = transformed['mask'].unsqueeze(0)
            return img, mask
        except Exception as e:
            print(f'Error loading {img_path} or {mask_path}: {e}')
            return None, None

# Compute mean/std (use previous values or recompute)
# Using your last known values to avoid recomputing
mean = np.array([0.21877019, 0.21720693, 0.20902685])
std = np.array([0.1038029, 0.05841177, 0.03478948])
print(f'Using mean: {mean}, std: {std}')

# Define transforms
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=90, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=mean.tolist(), std=std.tolist()),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=mean.tolist(), std=std.tolist()),
    ToTensorV2()
])

# Create datasets
train_ds = LandslideDataset(train_img_dir, train_mask_dir, transform=train_transform)
val_ds = LandslideDataset(val_img_dir, val_mask_dir, transform=val_transform)
test_ds = LandslideDataset(test_img_dir, test_mask_dir, transform=val_transform)

# Create DataLoaders
def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

print(f'Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}')

# Visualize 3 training samples, clip normalized values
plt.figure(figsize=(12, 8))
count = 0
for i in range(len(train_ds)):
    img, mask = train_ds[i]
    if img is None:
        continue
    img_np = np.transpose(img[:3].numpy(), (1, 2, 0))
    img_np = np.clip(img_np, 0, 1)
    plt.subplot(3, 2, count*2+1)
    plt.title(f'Train Image {count+1} (RGB)')
    plt.imshow(img_np)
    plt.subplot(3, 2, count*2+2)
    plt.title(f'Train Mask {count+1}')
    plt.imshow(mask.squeeze(), cmap='gray')
    count += 1
    if count == 3:
        break
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/landslide_train_samples.png')
plt.show()

# U-Net with ResNet18, 3-channel input
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights="imagenet",
    in_channels=3,  # RGB
    classes=1,
    activation='sigmoid'
).to(device)

pos_weight = torch.tensor([10.0]).to(device)
criterion = lambda pred, target: 0.5 * DiceLoss(mode='binary')(pred, target) + \
                                0.5 * nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred, target)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# Train for 10 epochs with early stopping
num_epochs = 10
best_val_loss = float('inf')
patience = 5
epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_batches = 0
    for imgs, masks in train_loader:
        if imgs.numel() == 0:
            continue
        imgs, masks = imgs.to(device), masks.to(device).float()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

    train_loss = train_loss / train_batches if train_batches > 0 else float('nan')

    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            if imgs.numel() == 0:
                continue
            imgs, masks = imgs.to(device), masks.to(device).float()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            if not torch.isnan(loss):
                val_loss += loss.item()
                val_batches += 1

    val_loss = val_loss / val_batches if val_batches > 0 else float('nan')

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    if not torch.isnan(torch.tensor(val_loss)) and val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '/content/drive/MyDrive/landslide_unet_best.pth')
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print('Early stopping triggered')
            break
        
        # Load saved model
model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,  # No pretrained weights, loading saved state
    in_channels=3,  # RGB
    classes=1,
    activation='sigmoid'
).to(device)

# Load weights from training
model.load_state_dict(torch.load('/content/drive/MyDrive/landslide_unet_best.pth'))
model.eval()
print('Model loaded successfully')

# Predict on 3 training samples
model.eval()
plt.figure(figsize=(12, 8))
count = 0
for i in range(len(train_ds)):
    img, mask = train_ds[i]
    if img is None:
        continue
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img) > 0.5
    pred = pred[0].cpu().squeeze().numpy()

    img_np = np.transpose(train_ds[i][0][:3].numpy(), (1, 2, 0))
    img_np = np.clip(img_np, 0, 1)
    plt.subplot(3, 3, count*3+1)
    plt.title(f'Train Image {count+1} (RGB)')
    plt.imshow(img_np)
    plt.subplot(3, 3, count*3+2)
    plt.title(f'Train True Mask {count+1}')
    plt.imshow(train_ds[i][1].squeeze(), cmap='gray')
    plt.subplot(3, 3, count*3+3)
    plt.title(f'Train Predicted Mask {count+1}')
    plt.imshow(pred, cmap='gray')
    count += 1
    if count == 3:
        break
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/landslide_train_predictions.png')
plt.show()
# Visualize 3 validation samples
plt.figure(figsize=(12, 8))
count = 0
for i in range(len(val_ds)):
    img, mask = val_ds[i]
    if img is None:
        continue
    img_np = np.transpose(img[:3].numpy(), (1, 2, 0))
    img_np = np.clip(img_np, 0, 1)
    plt.subplot(3, 2, count*2+1)
    plt.title(f'Val Image {count+1} (RGB)')
    plt.imshow(img_np)
    plt.subplot(3, 2, count*2+2)
    plt.title(f'Val Mask {count+1}')
    plt.imshow(mask.squeeze(), cmap='gray')
    count += 1
    if count == 3:
        break
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/landslide_val_samples.png')
plt.show()

# Predict on 3 validation samples
model.eval()
plt.figure(figsize=(12, 8))
count = 0
for i in range(len(val_ds)):
    img, mask = val_ds[i]
    if img is None:
        continue
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img) > 0.5
    pred = pred[0].cpu().squeeze().numpy()

    img_np = np.transpose(val_ds[i][0][:3].numpy(), (1, 2, 0))
    img_np = np.clip(img_np, 0, 1)
    plt.subplot(3, 3, count*3+1)
    plt.title(f'Val Image {count+1} (RGB)')
    plt.imshow(img_np)
    plt.subplot(3, 3, count*3+2)
    plt.title(f'Val True Mask {count+1}')
    plt.imshow(val_ds[i][1].squeeze(), cmap='gray')
    plt.subplot(3, 3, count*3+3)
    plt.title(f'Val Predicted Mask {count+1}')
    plt.imshow(pred, cmap='gray')
    count += 1
    if count == 3:
        break
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/landslide_val_predictions.png')
plt.show()

# Evaluate IoU on test set
model.load_state_dict(torch.load('/content/drive/MyDrive/landslide_unet_best.pth'))
model.eval()

iou_scores = []
with torch.no_grad():
    for imgs, masks in test_loader:
        if imgs.numel() == 0:
            continue
        imgs = imgs.to(device)
        preds = model(imgs) > 0.5
        preds = preds.cpu().numpy().flatten()
        true = masks.numpy().flatten()
        iou = jaccard_score(true, preds, average='binary', zero_division=0)
        iou_scores.append(iou)

mean_iou = np.mean(iou_scores) if iou_scores else 0.0
print(f'Mean IoU on Test Set: {mean_iou:.4f}')

# Visualize 3 test samples with predictions
model.eval()
plt.figure(figsize=(12, 8))
count = 0
for i in range(len(test_ds)):
    img, mask = test_ds[i]
    if img is None:
        continue
    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img) > 0.5
    pred = pred[0].cpu().squeeze().numpy()

    img_np = np.transpose(test_ds[i][0][:3].numpy(), (1, 2, 0))
    img_np = np.clip(img_np, 0, 1)
    plt.subplot(3, 3, count*3+1)
    plt.title(f'Test Image {count+1} (RGB)')
    plt.imshow(img_np)
    plt.subplot(3, 3, count*3+2)
    plt.title(f'Test True Mask {count+1}')
    plt.imshow(test_ds[i][1].squeeze(), cmap='gray')
    plt.subplot(3, 3, count*3+3)
    plt.title(f'Test Predicted Mask {count+1}')
    plt.imshow(pred, cmap='gray')
    count += 1
    if count == 3:
        break
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/landslide_test_predictions.png')
plt.show()

# Save model and documentation
torch.save(model, '/content/drive/MyDrive/landslide_unet_full_model.pth')
print('Model saved!')

# Requirements
requirements = """
segmentation-models-pytorch
albumentations
torch
pillow
numpy
scikit-learn
matplotlib
"""
with open('/content/drive/MyDrive/requirements.txt', 'w') as f:
    f.write(requirements)

# Integration instructions
instructions = """
Landslide Detection Model Integration Instructions:
1. Install dependencies: pip install -r requirements.txt
2. Load model: model = torch.load('landslide_unet_full_model.pth')
3. Set model to eval: model.eval()
4. Input: 3-channel RGB image (3x128x128 tensor, normalized with mean={mean}, std={std})
5. Output: Binary mask (1x128x128, 0/1 values)
6. Preprocess input: Use Albumentations Normalize with same mean/std
7. Example inference:
   import torch
   from PIL import Image
   import numpy as np
   import albumentations as A
   model = torch.load('landslide_unet_full_model.pth').to(device)
   model.eval()
   img = np.array(Image.open('input.png'))[:, :, :3]  # 128x128x3
   transform = A.Compose([A.Resize(128, 128), A.Normalize(mean={mean}, std={std}), A.pytorch.ToTensorV2()])
   img = transform(image=img)['image'].unsqueeze(0).to(device)
   with torch.no_grad():
       pred = model(img) > 0.5
   pred = pred[0].cpu().squeeze().numpy()  # 128x128 mask
""".format(mean=mean.tolist(), std=std.tolist())
with open('/content/drive/MyDrive/integration_instructions.txt', 'w') as f:
    f.write(instructions)

# README for GitHub
readme = """
# Landslide Detection Module
U-Net model for landslide detection using 3-channel RGB (128x128) satellite images. Achieved IoU of {iou:.4f} on test set.

## Files
- `landslide_unet_full_model.pth`: Trained model
- `landslide_unet_best.pth`: Best weights
- `requirements.txt`: Dependencies
- `integration_instructions.txt`: Website integration guide
- `landslide_train_samples.png`: Training data
- `landslide_train_predictions.png`: Training predictions
- `landslide_val_samples.png`: Validation data
- `landslide_val_predictions.png`: Validation predictions
- `landslide_test_predictions.png`: Test predictions

## Usage
See `integration_instructions.txt` for website integration.

## Results
- Dataset: 1,385 train, 396 val, 199 test
- Model: U-Net (ResNet18, 3-channel)
- IoU: {iou:.4f}
""".format(iou=mean_iou)
with open('/content/drive/MyDrive/README.md', 'w') as f:
    f.write(readme)

print('Saved requirements.txt, integration_instructions.txt, README.md')

# Import required libraries
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import drive, files
import segmentation_models_pytorch as smp

# Mount Google Drive to access the saved model
drive.mount('/content/drive')

# Set device (CPU as per your setup)
device = torch.device('cpu')
print(f'Using device: {device}')

# Load the trained model
model_path = '/content/drive/MyDrive/landslide_unet_full_model.pth'
model = torch.load(model_path, map_location=device, weights_only=False) # Added weights_only=False
model.eval()
print('Model loaded successfully')

# Define mean and std from your dataset
mean = [0.21877019, 0.21720693, 0.20902685]
std = [0.1038029, 0.05841177, 0.03478948]

# Define preprocessing transform
transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

# Upload an image
print('Please upload a satellite image (PNG, RGB)')
uploaded = files.upload()
if not uploaded:
    raise ValueError('No file uploaded')

# Get the uploaded image
img_name = list(uploaded.keys())[0]
img = np.array(Image.open(img_name))
if img.shape[-1] == 4:
    img = img[:, :, :3]  # Drop alpha channel if present
if img.shape[-1] != 3:
    raise ValueError(f'Image must be RGB, got shape {img.shape}')

# Preprocess the image
transformed = transform(image=img)
img_tensor = transformed['image'].unsqueeze(0).to(device)

# Predict landslide mask
with torch.no_grad():
    pred = model(img_tensor) > 0.5
pred = pred[0].cpu().squeeze().numpy()

# Visualize input image and predicted mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Input Image (RGB)')
plt.imshow(img)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Predicted Landslide Mask')
plt.imshow(pred, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('/content/drive/MyDrive/landslide_prediction_test.png')
plt.show()

print('Prediction saved as /content/drive/MyDrive/landslide_prediction_test.png')