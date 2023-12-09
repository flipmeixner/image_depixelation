import torch
from typing import List, Tuple
import numpy as np


def stack_with_padding(batch_as_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    batch_size = len(batch_as_list)
    max_height = max(pixelated_image.shape[1] for pixelated_image, _, _, _ in batch_as_list)
    max_width = max(pixelated_image.shape[-1] for pixelated_image, _, _, _ in batch_as_list)

    stacked_pixelated_images = np.ones((batch_size, 1, max_height, max_width))
    stacked_known_arrays = np.ones((batch_size, 1, max_height, max_width))
    stacked_target_arrays = np.ones((batch_size, 1, max_height, max_width))
    for i, (img, arr, tar, _) in enumerate(batch_as_list):
        stacked_pixelated_images[i, :, :img.shape[1], :img.shape[2]] = img
        stacked_known_arrays[i, :, :arr.shape[1], : arr.shape[2]] = arr
        stacked_target_arrays[i, :, :arr.shape[1], : arr.shape[2]] = tar

    stacked_pixelated_images = torch.Tensor(np.stack(list(image for image in stacked_pixelated_images)))
    stacked_known_arrays = torch.Tensor(np.stack(list(arr for arr in stacked_known_arrays)))
    stacked_target_arrays = torch.Tensor(np.stack(list(tar for tar in stacked_target_arrays)))

    image_files = [image_file for _, _, _, image_file in batch_as_list]

    return stacked_pixelated_images, stacked_known_arrays, stacked_target_arrays, image_files
