import os
import time
import torch
import numpy as np
from PIL import Image
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import UploadImageForm
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cpu")
model_path = os.path.join(settings.BASE_DIR, 'landslide_unet_full_model.pth')

# Load model once when server starts
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# Albumentations preprocessing
mean = [0.21877019, 0.21720693, 0.20902685]
std = [0.1038029, 0.05841177, 0.03478948]
transform = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

def predict_landslide(image_path):
    img = np.array(Image.open(image_path))[:, :, :3]
    transformed = transform(image=img)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img_tensor) > 0.5
    return pred[0].cpu().squeeze().numpy(), img

def upload_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data['image']
            
            # Create necessary directories if they don't exist
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            result_dir = os.path.join(settings.MEDIA_ROOT, 'results')
            os.makedirs(upload_dir, exist_ok=True)
            os.makedirs(result_dir, exist_ok=True)
            
            # Save uploaded image with a unique name to avoid conflicts
            unique_name = f"{int(time.time())}_{image_file.name}"
            upload_path = os.path.join(upload_dir, unique_name)
            
            with open(upload_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)
            
            # Predict mask
            mask, orig = predict_landslide(upload_path)

            # Save predicted mask image
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            mask_filename = f'mask_{unique_name}'
            mask_path = os.path.join(result_dir, mask_filename)
            mask_image.save(mask_path)
            
            # Calculate some metrics from the prediction
            landslide_area = np.sum(mask)
            total_area = mask.size
            landslide_percentage = (landslide_area / total_area) * 100
            
            # Construct correct URLs for the templates
            original_image_url = f"{settings.MEDIA_URL}uploads/{unique_name}"
            predicted_mask_url = f"{settings.MEDIA_URL}results/{mask_filename}"

            # Render result template with image URLs and prediction data
            return render(request, 'land_slide_prediction/result.html', {
                'original_image': original_image_url,
                'predicted_mask': predicted_mask_url,
                'landslide_percentage': f"{landslide_percentage:.2f}%",
                'affected_area': f"{landslide_area} pixels",
                'total_area': f"{total_area} pixels",
                'risk_level': 'High' if landslide_percentage > 5.0 else 'Medium' if landslide_percentage > 1.0 else 'Low'
            })
    else:
        form = UploadImageForm()

    return render(request, 'land_slide_prediction/predict.html', {'form': form})