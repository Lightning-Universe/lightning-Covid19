import torch
import torch.nn as nn

import kornia as K
import numpy as np


class DataAugmentator(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""
    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self._max_val: float = 1024.

        self.transforms = nn.Sequential(
            K.color.Normalize(0., self._max_val),
            K.augmentation.RandomHorizontalFlip(p=0.5)
        )

        self.jitter = K.augmentation.ColorJitter(0.5, 0.5, 0.5, 0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.transforms(x).unsqueeze(1)
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out


class ToTensor(nn.Module):
    """Module to cast numpy image to torch.Tensor."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> torch.Tensor:
        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 3 and x.shape[0] == 1, x.shape
        return torch.tensor(x).unsqueeze(0)  # 1xCxHxW


class Squeeze(nn.Module):
    """Module to squeeze tensor dimensions."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 4, x.shape
        return torch.squeeze(x, dim=0)  # CxHxW


class XRayCenterCrop(nn.Module):
    """Module to perform center crop based on half image size."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 4, x.shape
        crop_size: int = min(x.shape[-2:]) // 2
        return K.center_crop(x, (crop_size, crop_size))


class XRayResizer(nn.Module):
    """Module to image resize spatial resolution."""
    def __init__(self, size: int) -> None:
        super().__init__()
        self._size: int = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 4, x.shape
        return K.resize(x, self._size)
