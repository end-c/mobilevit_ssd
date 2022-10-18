
import copy
import torch
from torch import Tensor
from torch.utils import data
import cv2
from PIL import Image
from typing import Optional, Union, Dict
import argparse
import psutil  # psutil 方便获取系统硬件和性能信息的库。
import time
import numpy as np
from torchvision.io import (read_image, read_file, decode_jpeg, ImageReadMode, decode_image)
import io

from utils import logger


class BaseImageDataset(data.Dataset):
    def __init__(self, opts,
                 is_training: Optional[bool] = True,
                 is_evaluation: Optional[bool] = False,
                 *args,
                 **kwargs):
        root = (getattr(opts, "dataset.root_train", None)
                if is_training
                else getattr(opts, "dataset.root_val", None))
        self.root = root
        self.is_training = is_training
        self.is_evaluation = is_evaluation
        self.sampler_name = getattr(opts, "sampler.name", None)
        self.opts = opts

        image_device_cuda = getattr(self.opts, "dataset.decode_data_on_gpu", False)
        device = getattr(self.opts, "dev.device", torch.device("cpu"))
        use_cuda = False
        if image_device_cuda and (
            (isinstance(device, str) and device.find("cuda") > -1)
            or (isinstance(device, torch.device) and device.type.find("cuda") > -1)
        ):  # cuda could be cuda:0
            use_cuda = True

        if use_cuda and getattr(opts, "dataset.pin_memory", False):
            logger.error(
                "For loading images on GPU, --dataset.pin-memory should be disabled."
            )

        self.device = device if use_cuda else torch.device("cpu")

        self.cached_data = (
            dict()
            if getattr(opts, "dataset.cache_images_on_ram", False) and is_training
            else None
        )
        if self.cached_data is not None:
            if not getattr(opts, "dataset.persistent_workers", False):
                logger.error(
                    "For caching, --dataset.persistent-workers should be enabled."
                )

        self.cache_limit = getattr(opts, "dataset.cache_limit", 80.0)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def _training_transforms(self, *args, **kwargs):
        raise NotImplementedError

    def _validation_transforms(self, *args, **kwargs):
        raise NotImplementedError

    def _evaluation_transforms(self, *args, **kwargs):
        raise NotImplementedError

    def read_image_pil(self, path: str, *args, **kwargs):
        def convert_to_rgb(inp_data: Union[str, io.BytesIO]):
            try:
                rgb_img = Image.open(inp_data).convert("RGB")
            except:
                rgb_img = None
            return rgb_img

        if self.cached_data is not None:

            used_memory = float(psutil.virtual_memory().percent)  # 内存使用率

            if path in self.cached_data:
                img_byte = self.cached_data[path]

            elif (path not in self.cached_data) and (used_memory <= self.cache_limit):

                with open(path, "rb") as bin_file:
                    bin_file_data = bin_file.read()
                    img_byte = io.BytesIO(bin_file_data)
                    self.cached_data[path] = img_byte
            else:
                with open(path, "rb") as bin_file:
                    bin_file_data = bin_file.read()
                    img_byte = io.BytesIO(bin_file_data)
            img = convert_to_rgb(img_byte)
        else:
            img = convert_to_rgb(path)
        return img

    def read_pil_image_torchvision(self, path: str):
        if self.cached_data is not None:
            used_memory = float(psutil.virtual_memory().percent)

            if path in self.cached_data:
                byte_img = self.cached_data[path]
            elif (path not in self.cached_data) and (used_memory <= self.cache_limit):
                byte_img = read_file(path)
                self.cached_data[path] = byte_img
            else:
                byte_img = read_file(path)
        else:
            byte_img = read_file(path)
        img = decode_image(byte_img, mode=ImageReadMode.RGB)
        return img

    def read_img_tensor(self, path: str):
        if self.cached_data is not None:
            used_memory = float(psutil.virtual_memory().percent)

            if path in self.cached_data:
                byte_img = self.cached_data[path]
            elif (path not in self.cached_data) and (used_memory <= self.cache_limit):
                byte_img = read_file(path)
                self.cached_data[path] = byte_img
            else:
                byte_img = read_file(path)
        else:
            byte_img = read_file(path)
        img = decode_jpeg(byte_img, device=self.device, mode=ImageReadMode.RGB)
        return img

    @staticmethod
    def read_mask_pil(path: str):
        try:
            mask = Image.open(path)
            if mask.mode != 'L':
                logger.error("Mask mode should be L. Got :{}".format(mask.mode))
            return mask
        except:
            return None

    @staticmethod
    def read_image_opencv(path: str):
        return cv2.imread(
            path, cv2.IMREAD_COLOR
        )  # Image is read in BGR Format and not RGB format

    @staticmethod
    def read_mask_opencv(path: str):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def to_device(
        self, data: Union[Dict, Tensor], *args, **kwargs
    ) -> Union[Dict, Tensor]:
        return move_to_device(x=data, device=self.device, *args, **kwargs)


def move_to_device(
        x: Union[Dict, Tensor], device: Optional[str] = "cpu",
        *args, **kwargs) -> Union[Dict, Tensor]:
    if isinstance(x, Dict):
        for k, v in x.items():
            if isinstance(v, Dict):
                x[k] = move_to_device(v, device=device)
            elif isinstance(v, Tensor):
                x[k] = v.to(device=device, non_blocking=True)

    elif isinstance(x, Tensor):
        x = x.to(device=device)
    else:
        logger.error(
            "Inputs of type  Tensor or Dict of Tensors are only supported right now"
        )
    return x