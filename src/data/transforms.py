import cv2
import numpy as np
import rpack
import torchvision.transforms.functional as F
from functools import cache
from PIL import Image
from histolab.filters.compositions import FiltersComposition
from histolab.slide import Slide
from torchvision.transforms import CenterCrop
from typing import Optional


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


def shrink_image(img, sizes, scale=None):
  sizes, positions, contours_scaled, x_max, y_max = sizes
  img = np.array(img)
  if img.ndim == 2:
    new_img = np.zeros((y_max, x_max), dtype=img.dtype)
  else:
    new_img = np.zeros((y_max, x_max, img.shape[2]), dtype=img.dtype)

  for (w, h), (x, y), (x_old, y_old, w_old, h_old) in zip(sizes, positions, contours_scaled):
    new_img[y: y + h, x: x + w] = img[y_old: y_old + h_old, x_old: x_old + w_old]

  return Image.fromarray(new_img)


class Shrink:
    shrink_problem_filenames = [
        '1vig9enh.png', '24xxi0a4.png', '3f4c0t1d.png', '61wseljn.png', '6235nfns.png', 
        '6gt1vaty.png', '7nyiq3xw.png', '8f7dpkpv.png', '8j2n6c96.png', '8pkdid6b.png', 
        '9d9fv6qj.png', '9dp3he0s.png', '9r67fy7w.png', 'aw75mfil.png', 'ba9kpgk3.png', 
        'dzuwx6cz.png', 'evlwfw6j.png', 'f54rwukl.png', 'g64ajigb.png', 'hi63pfrb.png', 
        'hxiubwtq.png', 'i5h28l20.png', 'l6s043vm.png', 'ltwbgd9g.png', 'mut49okk.png', 
        'mwr9rmlg.png', 'onxdf2fq.png', 'oxk5y262.png', 'peesbo3q.png', 'pkfqj0ga.png', 
        'qrimite8.png', 'rdslu5i5.png', 'rr4gq7zb.png', 's4cm9bva.png', 'uuggf1s7.png', 
        'vif6zvk6.png', 'z9o9gqia.png'
    ]
    def __init__(self, scale: Optional[int] = None):
        self.scale = scale

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
          sizes = build_shink_size_and_positions(img, self.scale)
          img_shrinked = shrink_image(img, sizes, self.scale)
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
