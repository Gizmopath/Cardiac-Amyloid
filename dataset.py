import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Elastic and Grid Distortion Augmentations
def elastic_transform(image, alpha, sigma):
    transform = A.ElasticTransform(alpha=alpha, sigma=sigma, alpha_affine=0)
    augmented = transform(image=np.array(image))
    return Image.fromarray(augmented["image"])

def grid_distortion(image, num_steps, distort_limit):
    transform = A.GridDistortion(num_steps=num_steps, distort_limit=distort_limit)
    augmented = transform(image=np.array(image))
    return Image.fromarray(augmented["image"])

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, positive_images, negative_images, transform=None):
        self.file_paths = positive_images + negative_images
        self.labels = [1] * len(positive_images) + [0] * len(negative_images)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Augmentation pipeline
def get_augmentation_pipeline():
    return T.Compose([
        T.RandomApply([T.Lambda(lambda img: img.rotate(random.choice([90, 180, 270])))], p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.6, saturation=0.6, hue=0.1)], p=0.7),
        T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.2, 3.0))], p=0.3),
        T.RandomApply([T.Lambda(lambda img: elastic_transform(img, alpha=34, sigma=4))], p=0.3),
        T.RandomApply([T.Lambda(lambda img: grid_distortion(img, num_steps=5, distort_limit=0.3))], p=0.3),
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

def get_val_transform():
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

