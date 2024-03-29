import cv2
import math
import numpy as np
import rpack
import torchvision.transforms.functional as F
from torch import Tensor
from functools import cache
from PIL import Image
from histolab.filters.compositions import FiltersComposition
from histolab.slide import Slide
from torchvision.transforms import CenterCrop
from typing import Optional, Tuple


def build_contours_by_mask(mask):
  mask = mask[..., None].astype(np.uint8) * 255

  kernel_size = 3
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  contours = sorted(map(cv2.boundingRect, contours))

  return contours


def scale_contours(contours, x_max, y_max, contours_scale):
  contours_scaled = []
  for x, y, w, h in contours:
    x, y, w, h = \
      x * contours_scale, \
      y * contours_scale, \
      w * contours_scale, \
      h * contours_scale

    if x > x_max:
      x = x_max
    if y > y_max:
      y = y_max
    if x + w > x_max:
      w = x_max - x
    if y + h > y_max:
      h = y_max - y
    
    contours_scaled.append((x, y, w, h))

  return contours_scaled


def build_shink_size_and_positions(img, scale=None):
  # Get preview image
  if scale is None:
      img_preview = img
  else:
      img_preview = img.resize(
          (img.size[0] // scale, img.size[1] // scale)
      )

  # Get mask
  composition = FiltersComposition(Slide)
  mask = composition.tissue_mask_filters(img_preview)

  # Build contours
  contours = build_contours_by_mask(mask)

  # Here, scaling is rounded up
  # so scaled contour could exceed image size
  contours_scaled = contours
  if scale is not None:
    contours_scaled = scale_contours(
        contours, 
        x_max=img.size[0], 
        y_max=img.size[1], 
        contours_scale=scale
    )

  sizes = [contour[-2:] for contour in contours_scaled]
  positions = rpack.pack(sizes)

  x_max = max(x + w for (x, _), (w, _) in zip(positions, sizes))
  y_max = max(y + h for (_, y), (_, h) in zip(positions, sizes))

  return sizes, positions, contours_scaled, x_max, y_max


def shrink_image(img, sizes, scale=None, fill=(0, 0, 0)):
  sizes, positions, contours_scaled, x_max, y_max = sizes
  img = np.array(img)
  if img.ndim == 2:
    new_img = np.full((y_max, x_max), fill_value=np.mean(fill).astype(img.dtype), dtype=img.dtype)
  else:
    new_img = np.full((y_max, x_max, img.shape[2]), fill_value=np.array(fill)[None, None, :], dtype=img.dtype)

  for (w, h), (x, y), (x_old, y_old, w_old, h_old) in zip(sizes, positions, contours_scaled):
    new_img[y: y + h, x: x + w] = img[y_old: y_old + h_old, x_old: x_old + w_old]

  return Image.fromarray(new_img)


class Shrink:
    def __init__(self, scale: Optional[int] = None, fill: Tuple[int] = (0, 0, 0)):
        self.scale = scale
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
          sizes = build_shink_size_and_positions(img, self.scale)
          img_shrinked = shrink_image(img, sizes, self.scale, self.fill)
        except ValueError as e:
          img_shrinked = img

        return img_shrinked

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale={self.scale})"
    

class CenterCropPct(CenterCrop):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        assert isinstance(self.size, (float, tuple))
        if isinstance(self.size, float):
            crop_size = int(img.size[-1] * self.size), int(img.size[-2] * self.size)
        else:
            crop_size = int(img.size[-1] * self.size[1]), int(img.size[-2] * self.size[0])
        result = F.center_crop(img, crop_size)
        return result


class SimMIMTransform:
    def __init__(self, mask_generator):
        self.mask_generator = mask_generator

    def __call__(self, img):
        mask = self.mask_generator()
        return img, mask


# https://discuss.pytorch.org/t/torchvision-transforms-set-fillcolor-for-centercrop/98098/2
class PadCenterCrop(object):
    def __init__(self, size, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.pad_if_needed = pad_if_needed
        self.padding_mode = padding_mode
        self.fill = fill

    def __call__(self, img):

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        return F.center_crop(img, self.size)


def generate_patch_bboxes(img_size: tuple, patch_size: tuple):
    """Generator for possibly overlapping patch bounding boxes.
    such that there is img_size[i] // patch_size[i] (if img_size[i] 
    is divisible by patch_size[i]) or img_size[i] // patch_size[i] + 1 
    patches in each dimension.
    """
    assert len(img_size) == 2 and len(patch_size) == 2
    assert img_size[0] >= patch_size[0] and img_size[1] >= patch_size[1]
    
    if img_size[0] % patch_size[0] == 0:
        h_stride = patch_size[0]
        h_steps = img_size[0] // patch_size[0]
    else:
        h_stride = math.floor(
            patch_size[0] - 
            (patch_size[0] - img_size[0] % patch_size[0]) / 
            (img_size[0] // patch_size[0])
        )
        h_steps = img_size[0] // patch_size[0] + 1
    
    if img_size[1] % patch_size[1] == 0:
        w_stride = patch_size[1]
        w_steps = img_size[1] // patch_size[1]
    else:
        w_stride = math.floor(
            patch_size[1] - 
            (patch_size[1] - img_size[1] % patch_size[1]) / 
            (img_size[1] // patch_size[1]))
        w_steps = img_size[1] // patch_size[1] + 1
    
    for h_index in range(h_steps):
        for w_index in range(w_steps):
            h_start = h_index * h_stride
            w_start = w_index * w_stride
            h_end = min(h_start + patch_size[0], img_size[0])
            w_end = min(w_start + patch_size[1], img_size[1])
            yield (h_start, h_end, w_start, w_end)


def generate_tensor_patches(img: Tensor, patch_size: tuple, fill: float):
    """Generator for possibly overlapping patch bounding boxes.
    such that there is img_size[i] // patch_size[i] (if img_size[i] 
    is divisible by patch_size[i]) or img_size[i] // patch_size[i] + 1 
    patches in each dimension.
    """
    assert len(img.shape) == 4, "Image must be a 4D tensor"
    assert len(patch_size) == 2, "Patch size must be a 2-tuple"

    B, C, H, W = img.shape

    # If the image is smaller than the patch size, center pad
    if H < patch_size[0]:
        pad_down = (patch_size[0] - img.shape[0]) // 2
        pad_up = patch_size[0] - img.shape[0] - pad_down
        img = F.pad(img, (0, pad_up, 0, pad_down), padding_mode='constant', fill=fill)
    if W < patch_size[1]:
        pad_right = (patch_size[1] - img.shape[1]) // 2
        pad_left = patch_size[1] - img.shape[1] - pad_right
        img = F.pad(img, (pad_left, 0, pad_right, 0), padding_mode='constant', fill=fill)

    # Generate patches
    img_size = img.shape[2:4]
    for bbox in generate_patch_bboxes(img_size, patch_size):
        h_start, h_end, w_start, w_end = bbox
        yield img[:, :, h_start:h_end, w_start:w_end]
