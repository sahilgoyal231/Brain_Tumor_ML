import torch
from torch.utils.data import Dataset
import cv2
import os

class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        # 🔥 Debug print
        print(f"Loading images from: {self.root_dir}")

        # 📁 Auto-load images from 'yes' and 'no' folders
        for label in ['yes', 'no']:
            folder_path = os.path.join(root_dir, label)
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(folder_path, img_name))

        print(f"Total images loaded: {len(self.image_paths)}")  # 🔥 Count check

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # ⚠️ Handle read failure
        if image is None:
            raise ValueError(f"Failed to read image at {img_path}")

        label = 1 if 'yes' in img_path.lower() else 0

        # ➕ Add channel dimension for grayscale (1, H, W)
        image = image[:, :, None]

        if self.transform:
            image = self.transform(image)

        return image, label
