
import copy
from PIL import Image, ImageFilter
import numpy as np
import random
import torch
import math
import argparse
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import Sequence, Dict, Any, Union, Tuple, List, Optional

from utils import logger
from . import register_transformations, BaseTransformation
from .utils import jaccard_numpy, setup_size

INTERPOLATION_MODE_MAP = {
    "nearest": T.InterpolationMode.NEAREST,
    "bilinear": T.InterpolationMode.BILINEAR,
    "bicubic": T.InterpolationMode.BICUBIC,
    "cubic": T.InterpolationMode.BICUBIC,
    "box": T.InterpolationMode.BOX,
    "hamming": T.InterpolationMode.HAMMING,
    "lanczos": T.InterpolationMode.LANCZOS,
}


def _interpolation_modes_from_str(name: str) -> T.InterpolationMode:
    return INTERPOLATION_MODE_MAP[name]


def _crop_fn(data: Dict, top: int, left: int, height: int, width: int) -> Dict:
    img = data["image"]
    data["image"] = F.crop(img, top=top, left=left, height=height, width=width)

    if "mask" in data:
        mask = data.pop("mask")
        data["mask"] = F.crop(mask, op=top, left=left, height=height, width=width)

    if "box_coordinates" in data:
        boxes = data.pop("box_coordinates")

        area_before_cropping = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1]
        )

        boxes[..., 0::2] = np.clip(boxes[..., 0::2] - left, a_min=0, a_max=left + width)
        boxes[..., 1::2] = np.clip(boxes[..., 1::2] - top, a_min=0, a_max=top + height)
        area_after_cropping = (boxes[..., 2] - boxes[..., 0]) * (
            boxes[..., 3] - boxes[..., 1]
        )
        area_ratio = area_after_cropping / (area_before_cropping + 1)

        # keep the boxes whose area is atleast 20% of the area before cropping
        keep = area_ratio >= 0.2

        box_labels = data.pop("box_labels")

        data["box_coordinates"] = boxes[keep]
        data["box_labels"] = box_labels[keep]

    if "instance_mask" in data:
        assert "instance_coords" in data

        instance_masks = data.pop("instance_mask")
        data["instance_mask"] = F.crop(
            instance_masks, top=top, left=left, height=height, width=width
        )

        instance_coords = data.pop("instance_coords")
        instance_coords[..., 0::2] = np.clip(
            instance_coords[..., 0::2] - left, a_min=0, a_max=left + width
        )
        instance_coords[..., 1::2] = np.clip(
            instance_coords[..., 1::2] - top, a_min=0, a_max=top + height
        )
        data["instance_coords"] = instance_coords

    return data
