# includes/augmented_dataset.py

from PIL import Image
import numpy as np
import torch
from torchvision.transforms import v2 as transforms


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset that applies data augmentation transforms using transforms v2.

    Following the recommended approach from torchvision documentation:
    - Use ToImage() to convert PIL to tensor
    - Use ToDtype(torch.float32, scale=True) to convert to float and scale to [0, 1]

    Args:
        data (np.ndarray): Input images with shape (N, H, W, C)
        labels (np.ndarray): Labels with shape (N,)
        transform (callable, optional): Transform to apply to images
    """
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

        # Base transform: convert to tensor (following v2 guidelines)
        self.to_tensor = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image and label
        image = self.data[idx]
        label = self.labels[idx]

        # Accept either float images in [0,1] or uint8 images in [0,255]
        if np.issubdtype(image.dtype, np.floating):
            img_uint8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)

        # Convert HWC uint8 numpy -> PIL Image (ToImage() then ToDtype will handle conversion to tensor)
        image_pil = Image.fromarray(img_uint8)

        # Convert to tensor using v2 recommended approach
        image_tensor = self.to_tensor(image_pil)

        # Apply additional transforms if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, torch.tensor(int(label), dtype=torch.long)
