import random
import numpy as np
from PIL import Image
from torchvision import transforms
import imgaug.augmenters as iaa

# Prevent errors
if not hasattr(np, "bool"):
    np.bool = np.bool_
if not hasattr(np, "complex"):
    np.complex = np.complex_
    

def transform_pipeline1():
    return transforms.Compose([
        # Trigger color jitter independently
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            )
        ], p=0.8),
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=(3, 3), 
                sigma=(0.1, 2.0))
        ], p=0.3),
        # Trigger cutout independently
        transforms.RandomApply([
            RandomCutout(num_holes=2, max_size=0.3)
        ], p=0.5),

        # Trigger Gaussian noise independently
        transforms.RandomApply([
            AddGaussianNoise(std=0.03)
        ], p=0.6)
    ])

def transform_pipeline1_gentle():
    """Gentle version of TPP1, consistent but with compressed params and reduced probability"""
    return transforms.Compose([
        # Color jitter - halved probability, weakened params
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.15,  # Reduced from 0.3 to 0.15
                contrast=0.15,    # Reduced from 0.3 to 0.15
                saturation=0.1,   # Reduced from 0.2 to 0.1
                hue=0.05          # Reduced from 0.1 to 0.05
            )
        ], p=0.1),  # Reduced from 0.8 to 0.4
        
        # Gaussian blur - halved probability, weakened blur
        transforms.RandomApply([
            transforms.GaussianBlur(
                kernel_size=(3, 3), 
                sigma=(0.1, 1.0)  # Reduced from (0.1, 2.0) to (0.1, 1.0)
            )
        ], p=0.1),  # Reduced from 0.3 to 0.15
        
        # Cutout - halved probability, weakened occlusion
        transforms.RandomApply([
            RandomCutout(num_holes=1, max_size=0.15)  # Holes reduced from 2 to 1, size from 0.3 to 0.15
        ], p=0.1),  # Reduced from 0.5 to 0.25
        
        # Gaussian noise - halved probability, weakened noise
        transforms.RandomApply([
            AddGaussianNoise(std=0.015)  # Reduced from 0.03 to 0.015
        ], p=0.1)  # Reduced from 0.6 to 0.3
    ])

def transform_pipeline2():
    return transforms.Compose([
        transforms.RandomApply([
            transforms.RandomAffine(degrees=5, translate=(0.07, 0.07), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.7)
        ], p=0.5)
    ])

def transform_pipeline3():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])        
    ])

def transform_pipeline_weather(augname: str = "Mixed"):
    """Weather augmentation modes: Mixed, single Rain,Snow,Cloud,Brightness,Noop"""
    # Define various weather augmentation methods, params adjustable
    aug_list = [
        ("Rain", iaa.Rain(drop_size=(0.01, 0.02), speed=(0.05, 0.10))),
        ("Snow", iaa.Snowflakes(flake_size=(0.3, 0.6), speed=(0.01, 0.03))),
        ("Cloud", iaa.Clouds()),
        ("Brightness", iaa.imgcorruptlike.Brightness(severity=2)),
        ("Noop", iaa.Noop()),
    ]

    if augname != "Mixed":
        # Specify one weather augmentation
        weather_oneof = iaa.OneOf([aug for name, aug in aug_list if name == augname])
    else:
        # Mix multiple weather augmentations
        weather_oneof = iaa.OneOf(
            [
                aug
                for name, aug in aug_list
                if name in ["Rain", "Snow", "Cloud", "Brightness", "Noop"]
            ]
        )

    # Weather augmentation image transform
    def _apply_weather(img: Image.Image) -> Image.Image:
        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        out = weather_oneof(image=arr)
        return Image.fromarray(out)

    return transforms.Compose([transforms.Lambda(_apply_weather)])


class RandomCutout(object):
    def __init__(self, num_holes=1, max_size=0.3):
        self.num_holes = num_holes
        self.max_size = max_size  # Relative size

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f"RandomCutout only accepts PIL.Image, received {type(img)}")

        img_np = np.array(img).copy()  # Convert to numpy, copy to avoid modifying original
        h, w, _ = img_np.shape

        for _ in range(self.num_holes):
            hole_w = int(random.uniform(0.1, self.max_size) * w)
            hole_h = int(random.uniform(0.1, self.max_size) * h)
            x = random.randint(0, max(0, w - hole_w))
            y = random.randint(0, max(0, h - hole_h))
            img_np[y:y+hole_h, x:x+hole_w, :] = 0  # Set to black

        return Image.fromarray(img_np)

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=10.0, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError(f"AddGaussianNoise only accepts PIL.Image, received {type(img)}")

        if random.random() >= self.p:
            return img

        img_np = np.array(img).astype(np.float32)
        noise = np.random.normal(self.mean, self.std, img_np.shape)
        img_np = img_np + noise
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        return Image.fromarray(img_np)