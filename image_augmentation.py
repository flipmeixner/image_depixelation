import torch
from torchvision import transforms
from PIL import Image
import random
from glob import glob
from os import path


def random_augmented_image(
        image: Image,
        seed: int,
) -> torch.Tensor:
    candidates = [transforms.RandomRotation(180), transforms.RandomVerticalFlip(),
                  transforms.RandomHorizontalFlip(), transforms.ColorJitter()]
    random.seed(seed)
    x = image
    random_transforms = random.sample(candidates, k=2)
    for transform in random_transforms:
        x = transform(x)
    x = transforms.ToTensor()(x)
    x = torch.nn.Dropout(p=0.01)(x)
    return transforms.ToPILImage()(x)


if __name__ == "__main__":
    image_dir = r"training_raw"
    augmented_image_dir = r"training_raw/augmented"
    image_files = sorted(path.abspath(f) for f in glob(path.join(image_dir, "**", "*.jpg"), recursive=True))
    i = 0
    for img in image_files:
        with Image.open(img) as image:
            augmented_image = random_augmented_image(image, seed=3)
            filename = path.basename(img)
            augmented_image_path = path.join(augmented_image_dir, f"augmented_{i}.jpg")
            augmented_image.save(augmented_image_path)
            i += 1
