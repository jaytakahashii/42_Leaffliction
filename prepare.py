import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
from typing import Callable

from tools.ImageTransformer import ImageTransformer
from tools.ImageAugmentor import ImageAugmentor
from tools.scanner import DirectoryScanner

TARGET_SIZE = (256, 256)
VAL_SPLIT_RATIO = 0.2  # percentage of data for validation
SEED = 42              # for reproducibility


class DatasetHandler:
    def __init__(self, src_dir: str, dst_dir: str):
        self.src_dir: Path = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.train_dir = self.dst_dir / "train"
        self.val_dir = self.dst_dir / "val"

        random.seed(SEED)

    def clean_and_load(self, path: Path) -> Image.Image | None:
        """
        Use Part3's logic: Remove background and crop ROI, return as PIL Image
        Args:
            path (Path): Image file path
        Returns:
            Image.Image | None: Processed PIL Image or None if failed
        """
        try:
            # 1. Remove background and crop ROI
            transformer: ImageTransformer = ImageTransformer(str(path))
            cropped_array: np.ndarray = transformer.get_cropped_roi()

            # 2. Resize to TARGET_SIZE (for Part4)
            # OpenCV (numpy) -> PIL
            img_pil: Image.Image = Image.fromarray(cropped_array)
            img_resized: Image.Image = img_pil.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

            return img_resized
        except Exception as e:
            print(f"Warning: Failed to transform {path.name}: {e}")
            return None

    def random_augment(self, img: Image.Image) -> ImageAugmentor:
        """
        Uses Part2's logic: Generate specified number of augmented images
        Args:
            img (Image.Image): Base image to augment
            count (int): Number of augmented images to generate
        Returns:
            list[Image.Image]: List of augmented images
        """
        augmentor: ImageAugmentor = ImageAugmentor(img)

        # methods available in ImageAugmentor

        methods: list[Callable[..., ImageAugmentor]] = [
            augmentor.rotate,
            augmentor.shear,
            augmentor.flip,
            augmentor.contrast,
            augmentor.blur,
            augmentor.brightness
        ]

        return random.choice(methods)()

    def process(self):
        scanner: DirectoryScanner = DirectoryScanner(str(self.src_dir))
        distribution: dict[str, int] = scanner.get_distribution()  # {'Apple_scab': 500, ...}

        if not distribution:
            print("No images found.")
            return

        # Get max count for balancing
        max_count: int = max(distribution.values())
        target_train_count = int(max_count * (1 - VAL_SPLIT_RATIO))
        print(f"Target count per class (Training): {target_train_count}")

        # Create directories for training and validation sets
        for category in distribution.keys():
            (self.train_dir / category).mkdir(parents=True, exist_ok=True)
            (self.val_dir / category).mkdir(parents=True, exist_ok=True)

        for category, count in distribution.items():
            print(f"Processing {category}...")

            # 1. Collect image paths
            all_files = list(self.src_dir.rglob(f"{category}/*"))
            image_files = [f for f in all_files if f.suffix.lower() in scanner.IMAGE_EXTENSIONS]

            # 2. Shuffle and split into training and validation sets
            random.shuffle(image_files)
            split_idx = int(len(image_files) * (1 - VAL_SPLIT_RATIO))
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]

            # --- Validation Set processing ---
            for i, fpath in enumerate(tqdm(val_files, desc=f"  Validation ({category})")):
                img: Image.Image | None = self.clean_and_load(fpath)
                if img:
                    save_path = self.val_dir / category / f"{fpath.stem}_val{fpath.suffix}"
                    img.save(save_path)

            # --- Training Set processing ---
            current_train_imgs: dict[Path, Image.Image] = {}

            # Save original training images
            for fpath in tqdm(train_files, desc=f"  Training Load ({category})"):
                img = self.clean_and_load(fpath)
                if img:
                    save_path = self.train_dir / category / f"{fpath.stem}{fpath.suffix}"
                    img.save(save_path)
                    current_train_imgs[fpath] = img

            # Check current count and augment if needed
            needed: int = target_train_count - len(current_train_imgs)
            list_dict_items = list(current_train_imgs.items())

            if needed > 0 and current_train_imgs:
                print(f"    -> Augmenting {needed} images for balance.")
                generated: int = 0
                while generated < needed:
                    # Select a random base image
                    base: tuple[Path, Image.Image] = random.choice(list_dict_items)
                    try:
                        aug: ImageAugmentor = self.random_augment(base[1])
                        base_stem = base[0].stem
                        base_suffix = base[0].suffix
                        method_name = "_".join([name.capitalize() for name in aug.history])
                        # to avoid overwriting, include generated index
                        aug.img.save(str(self.train_dir / category /
                                     f"{base_stem}_{method_name}_{generated}{base_suffix}"))
                    except Exception as e:
                        print(f"      Warning: Augmentation failed: {e}")
                        continue

                    generated += 1


def main():
    SRC_DIR = "images"
    DST_DIR = "dataset_prepared"

    if Path(DST_DIR).exists():
        print(f"Cleaning previous build at {DST_DIR}...")
        shutil.rmtree(DST_DIR)

    handler = DatasetHandler(SRC_DIR, DST_DIR)
    handler.process()

    print("\nDataset preparation complete!")
    print(f"Training data: {DST_DIR}/train")
    print(f"Validation data: {DST_DIR}/val")
    print("Don't forget to ZIP the dataset for submission requirements.")

    print("== Count Summary ==")
    DST_DIR = str(Path(DST_DIR) / "train")
    scanner = DirectoryScanner(DST_DIR)
    distribution = scanner.get_distribution()
    for category, count in distribution.items():
        print(f"  {category}: {count} images")


if __name__ == "__main__":
    main()
