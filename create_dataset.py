from glob import glob
from os import path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from prep_image import transform_image, to_grayscale, prepare_image


class RandomImagePixelationDataset(Dataset):

    def __init__(
            self,
            image_dir,
            width_range: tuple[int, int],
            height_range: tuple[int, int],
            size_range: tuple[int, int],
            dtype: Optional[type] = None
    ):
        RandomImagePixelationDataset._check_range(width_range, "width")
        RandomImagePixelationDataset._check_range(height_range, "height")
        RandomImagePixelationDataset._check_range(size_range, "size")
        self.image_files = sorted(path.abspath(f) for f in glob(path.join(image_dir, "**", "*.jpg"), recursive=True))
        self.width_range = width_range
        self.height_range = height_range
        self.size_range = size_range
        self.dtype = dtype

    @staticmethod
    def _check_range(r: tuple[int, int], name: str):
        if r[0] < 2:
            raise ValueError(f"minimum {name} must be >= 2")
        if r[0] > r[1]:
            raise ValueError(f"minimum {name} must be <= maximum {name}")

    def __getitem__(self, index):
        with Image.open(self.image_files[index]) as img:
            img = transform_image(img)
            image = np.array(img, dtype=self.dtype)
        image = to_grayscale(image)  # Image shape is now (1, H, W)
        image_width = image.shape[-1]
        image_height = image.shape[-2]

        # Create RNG in each __getitem__ call to ensure reproducibility even in
        # environments with multiple threads and/or processes
        rng = np.random.default_rng(seed=index)

        # Both width and height can be arbitrary, but they must not exceed the
        # actual image width and height
        width = min(rng.integers(low=self.width_range[0], high=self.width_range[1], endpoint=True), image_width)
        height = min(rng.integers(low=self.height_range[0], high=self.height_range[1], endpoint=True), image_height)

        # Ensure that x and y always fit with the randomly chosen width and
        # height (and not throw an error in "prepare_image")
        x = rng.integers(image_width - width, endpoint=True)
        y = rng.integers(image_height - height, endpoint=True)

        # Block size can be arbitrary again
        size = rng.integers(low=self.size_range[0], high=self.size_range[1], endpoint=True)

        pixelated_image, known_array, target_array = prepare_image(image, x, y, width, height, size)

        return pixelated_image, known_array, target_array, self.image_files[index]

    def __len__(self):
        return len(self.image_files)
