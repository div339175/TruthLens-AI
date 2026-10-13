"""Image preprocessing pipeline for TruthLens AI."""

from pathlib import Path
from typing import Union, BinaryIO
from PIL import Image
import torch
from torchvision import transforms


class ImagePreprocessor:
    """Standard image transforms and loaders for ResNet/ViT models."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        # Raw transform for visualization / CAM overlay
        self.raw_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])

    def load_image(self, source: Union[str, Path, BinaryIO, Image.Image]) -> Image.Image:
        """Load an image source into an RGB PIL Image."""
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        img = Image.open(source)
        return img.convert("RGB")

    def preprocess(self, source: Union[str, Path, BinaryIO, Image.Image]) -> torch.Tensor:
        """Preprocess an image into a normalized tensor [3, H, W]."""
        pil_img = self.load_image(source)
        return self.transform(pil_img)

    def preprocess_batch(self, sources: list[Union[str, Path, BinaryIO, Image.Image]]) -> torch.Tensor:
        """Preprocess a list of images into a batch tensor [B, 3, H, W]."""
        tensors = [self.preprocess(s) for s in sources]
        return torch.stack(tensors, dim=0)
