import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tools.ImageTransformer import ImageTransformer
from tools.scanner import DirectoryScanner

COLUMN = 4
FIG_WIDTH = 15
FIG_HEIGHT = 5


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image Transformation tool for Leaffliction."
    )
    # Option 1: Direct file processing (Display mode)
    parser.add_argument(
        "files",
        type=str,
        nargs="*",
        help="Path to image file(s) to process directly."
    )
    # Option 2: Batch processing (Save mode)
    parser.add_argument(
        "-src", "--source",
        type=str,
        help="Source directory containing images."
    )
    parser.add_argument(
        "-dst", "--destination",
        type=str,
        help="Destination directory to save transformed images."
    )
    # Individual transformation flags (optional, but good for control)
    parser.add_argument("-b", "--blur", action="store_true", help="Apply Gaussian Blur")
    parser.add_argument("-m", "--mask", action="store_true", help="Apply Mask")
    parser.add_argument("-a", "--analyze", action="store_true", help="Analyze Object (Contours)")
    parser.add_argument("-r", "--roi", action="store_true", help="ROI Objects")
    parser.add_argument("-p", "--pseudo", action="store_true", help="Pseudolandmarks")
    parser.add_argument("-H", "--hist", action="store_true", help="Color Histogram")

    return parser.parse_args()


def process_single_image(path: str, args: argparse.Namespace) -> dict[str, np.ndarray] | None:
    """
    Runs all transformations on a single image and returns a dict of images.
    Args:
        path (str): Path to the image file.
        args (argparse.Namespace): Parsed command-line arguments to control transformations.
    Returns:
        dict: A dictionary with transformation names as keys and image arrays as values.
        None: If the image could not be processed.
    """
    try:
        transformer = ImageTransformer(path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error processing {path}: {e}")
        return None

    # Dictionary of "Title": Image_Array (np.ndarray)
    transformations: dict[str, np.ndarray] = {
        "Original": transformer.get_original(),
    }
    has_flags = any([args.blur, args.mask, args.analyze, args.roi, args.pseudo, args.hist])
    run_all = not has_flags
    if run_all or args.blur:
        transformations["Gaussian Blur"] = transformer.gaussian_blur()
    if run_all or args.mask:
        transformations["Mask"] = transformer.apply_mask()
    if run_all or args.analyze:
        transformations["Analyzed Contours"] = transformer.analyze_object()
    if run_all or args.roi:
        transformations["ROI Objects"] = transformer.roi_objects()
    if run_all or args.pseudo:
        transformations["Pseudolandmarks"] = transformer.pseudo_landmarks()
    if run_all or args.hist:
        transformations["Color Histogram"] = transformer.color_histogram()

    return transformations


def display_transformations(transformations: dict[str, np.ndarray]) -> None:
    """
    Displays the transformations using Matplotlib.
    Args:
        transformations (dict): A dictionary with transformation names as keys and image arrays as values.
    """
    if not transformations:
        return

    n = len(transformations)
    cols = COLUMN
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT * rows))

    for i, (name, img) in enumerate(transformations.items()):
        plt.subplot(rows, cols, i + 1)
        plt.title(name)
        plt.imshow(img)
        plt.axis('off')  # Hide axes for images
        if name == "Color Histogram":
            plt.axis('off')  # Actually, since we converted plot to img, keep off is cleaner

    plt.tight_layout()
    plt.show()


def save_transformations(transformations: dict, original_path: Path, dst_root: str, src_root: str) -> None:
    """
    Saves transformed images to the destination directory maintaining structure.
    Args:
        transformations (dict): A dictionary with transformation names as keys and image arrays as values.
        original_path (Path): The original image file path.
        dst_root (str): The root destination directory to save images.
        src_root (str): The root source directory to maintain structure.
    """

    # Calculate relative path to maintain subdirectory structure
    # e.g. source/Apple/healthy/img.jpg -> Apple/healthy
    rel_path: Path = original_path.parent.relative_to(src_root)
    save_dir: Path = Path(dst_root) / rel_path

    save_dir.mkdir(parents=True, exist_ok=True)

    stem: str = original_path.stem
    suffix: str = original_path.suffix

    for name, img in transformations.items():
        if name == "Original":
            continue  # Don't resave original

        # Clean name for filename (e.g. "Gaussian Blur" -> "Gaussian_Blur")
        suffix_name: str = name.replace(" ", "_")
        file_name: str = f"{stem}_{suffix_name}{suffix}"
        save_path: str = str(save_dir / file_name)

        # Convert RGB back to BGR for OpenCV saving
        img_bgr: np.ndarray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_bgr)

    print(f"Saved transformations for {original_path.name}")


def main():
    args = get_args()

    if (args.source and not args.destination) \
            or (not args.source and args.destination):
        print("Both -src and -dst must be provided for batch processing.")
        sys.exit(1)

    # Mode 1: Batch Processing (-src and -dst provided)
    if args.source and args.destination:
        src_path = Path(args.source)
        if not src_path.exists():
            print("Source directory does not exist.")
            sys.exit(1)

        print(f"Processing directory: {args.source} -> {args.destination}")
        for file_path in src_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in DirectoryScanner.IMAGE_EXTENSIONS:
                trans = process_single_image(str(file_path), args)
                if trans:
                    save_transformations(trans, file_path, args.destination, args.source)

    # Mode 2: Direct File Display (files provided)
    elif args.files:
        for file_path in args.files:
            print(f"Displaying transformations for: {file_path}")
            trans = process_single_image(str(file_path), args)
            if trans:
                display_transformations(trans)

    else:
        print("Please provide files to process or use -src and -dst arguments.")
        print("Use -h for help.")


if __name__ == "__main__":
    main()
