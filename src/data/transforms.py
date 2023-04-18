from typing import Optional
import cv2
import numpy as np
import rpack
import torchvision.transforms.functional as F
from PIL import Image
from histolab.filters.compositions import FiltersComposition
from histolab.slide import Slide
from torchvision.transforms import CenterCrop


def build_contours(mask):
  mask = mask[..., None].astype(np.uint8) * 255

  kernel_size = 3
  kernel = np.ones((kernel_size, kernel_size), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  contours = sorted(map(cv2.boundingRect, contours))

  return contours, mask


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


def shrink_image(img, contours, contours_scale=None):
  # Here, scaling is rounded up
  # so scaled contour could exceed image size
  contours_scaled = contours
  if contours_scale is not None:
    contours_scaled = scale_contours(
        contours, 
        x_max=img.shape[1], 
        y_max=img.shape[0], 
        contours_scale=contours_scale
    )

  sizes = [contour[-2:] for contour in contours_scaled]
  positions = rpack.pack(sizes)

  x_max = max(x + w for (x, _), (w, _) in zip(positions, sizes))
  y_max = max(y + h for (_, y), (_, h) in zip(positions, sizes))
  if img.ndim == 2:
    new_img = np.zeros((y_max, x_max), dtype=img.dtype)
  else:
    new_img = np.zeros((y_max, x_max, img.shape[2]), dtype=img.dtype)

  for (w, h), (x, y), (x_old, y_old, w_old, h_old) in zip(sizes, positions, contours_scaled):
    new_img[y: y + h, x: x + w] = img[y_old: y_old + h_old, x_old: x_old + w_old]

  return new_img


def scale_back(img, orig):
  shape = orig.size[:2] if isinstance(orig, Image.Image) else orig.shape[:2][::-1]
  img = cv2.resize(img, shape, fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
  return img


def shrink_single(img, mask, scale=None):
  contours, mask_contours = build_contours(mask)
  mask_contours_scaled_back = mask_contours
  if scale is not None:
    mask_contours_scaled_back = scale_back(mask_contours, img)

  mask_contours_scaled_back_shrinked = shrink_image(
    mask_contours_scaled_back, contours, contours_scale=scale)
  img_shrinked = shrink_image(img, contours, contours_scale=scale)

  return img_shrinked, mask_contours_scaled_back_shrinked


class Shrink:
    def __init__(self, scale: Optional[int] = None):
        self.scale = scale
    
    def __call__(self, img: Image.Image) -> Image.Image:
        # Get preview image
        if self.scale is None:
            img_preview = img
        else:
            img_preview = img.resize(
               (img.size[0] // self.scale, img.size[1] // self.scale)
            )

        # Get mask
        composition = FiltersComposition(Slide)
        mask = composition.tissue_mask_filters(img_preview)

        # Shrink image
        img_shrinked, _ = shrink_single(np.array(img), mask, scale=self.scale)
        img_shrinked = Image.fromarray(img_shrinked)

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
