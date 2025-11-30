import argparse
from pathlib import Path
from tools.ImageAugmentor import ImageAugmentor


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}


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


def main() -> None:
    args = get_args()
    paths = args.files
    save_dir = "augmented_directory" if args.eval else ""
    for path in paths:
        f = Path(path)
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                img = ImageAugmentor(path)
            except ValueError as e:
                print(e)
                continue
            augmented_images = [
                img.rotate(),
                img.shear(),
                img.flip(),
                img.contrast(),
                img.blur(),
                img.brightness()
            ]
            for aug_img in augmented_images:
                aug_img.save(save_dir)
                aug_img.img.show()
        else:
            print(f"Unavailable extension file: {f}")


if __name__ == '__main__':
    main()
