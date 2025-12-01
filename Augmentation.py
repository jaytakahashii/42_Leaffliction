import time
import argparse
from pathlib import Path
from tools.ImageAugmentor import ImageAugmentor
from tools.scanner import DirectoryScanner
import numpy as np
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(
        description='This is a script that takes files from the command-line \
            arguments and creates augmented samples from them.',
    )
    parser.add_argument(
        "files",
        type=str,
        nargs="+",
        help="Paths to one or more input image files to be processed"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="If set, the augmented images will be placed in the evaluation directory"
    )
    args = parser.parse_args()
    return args


def show_images(augmented_images: dict) -> None:
    names = list(augmented_images.keys())
    images = [augmented_images[name].img for name in names]

    n = len(images)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()

    for ax, im, name in zip(axes, images, names):
        ax.imshow(np.array(im.convert("RGB")))
        ax.axis('off')
        ax.set_title(name, fontsize=12)

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def main() -> None:
    args = get_args()
    paths = args.files
    save_dir = "augmented_directory" if args.eval else ""
    img_pool = []
    for path in paths:
        f = Path(path)
        if f.suffix.lower() in DirectoryScanner.IMAGE_EXTENSIONS:
            try:
                img = ImageAugmentor(path)
            except ValueError as e:
                print(e)
                continue
            augmented_images = {
                "Original": img,
                "Rotate": img.rotate(),
                "Shear": img.shear(),
                "Flip": img.flip(),
                "Contrast": img.contrast(),
                "Blur": img.blur(),
                "Brightness": img.brightness()
            }
            for k, v in augmented_images.items():
                v.save(save_dir)
            img_pool.append(augmented_images)
        else:
            print(f"Unavailable extension file: {f}")
    for p in img_pool:
        show_images(p)


if __name__ == '__main__':
    main()
