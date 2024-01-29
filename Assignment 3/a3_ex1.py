import numpy as np
import torch, glob, os
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from typing import Optional


def to_grayscale(pil_image: np.ndarray) -> np.ndarray:             #we convert the rgb images to grayscale
    if pil_image.ndim == 2:                                        #if the image is alrady in grayscale, a copy of the image with additional dimensions is returned
        return pil_image.copy()[None]
    if pil_image.ndim != 3:                                        #if the image has more than 3 dimensions, we raise ValueError
        raise ValueError("The image must be in either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"The image has the shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")

    rgb = pil_image / 255                        #this block converts RGB images to grayscale, using the formula recommended by (IEC)
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]

    grayscale = np.where(                        #this block scales the grayscale image values to the range 0-255 and returns the image as a numpy array
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255

    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]


class RandomImagePixelationDataset(Dataset):
    def __init__(self, image_dir, width_range: tuple[int, int], height_range: tuple[int, int],
                 size_range: tuple[int, int], dtype: Optional[type] = None):
        self.image_dir = image_dir
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype
        self.images = sorted(glob.glob(os.path.join(self.image_dir, "**", "*.jpg"), recursive=True))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        with Image.open(self.images[idx]) as im:
            image = np.array(im, dtype=self.dtype if self.dtype is not None else None)

        image = to_grayscale(image)
        rng = np.random.default_rng(seed=idx)
        width = int(rng.random() * (self.width_range[1] - self.width_range[0]) + self.width_range[0])
        height = int(rng.random() * (self.height_range[1] - self.height_range[0]) + self.height_range[0])

        x = int(rng.uniform(low=0, high=max(image.shape[1] - width, 0)))
        y = int(rng.uniform(low=0, high=max(image.shape[0] - height, 0)))

        size = int(rng.random() * (self.size_range[1] - self.size_range[0]) + self.size_range[0])

        pixelated_image, known_array, target_array = im_proc_f.prepare_image(image, x, y, width, height, size)

        return pixelated_image, known_array, target_array, self.images[idx]
