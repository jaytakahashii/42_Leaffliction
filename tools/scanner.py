from pathlib import Path
from collections import Counter


class DirectoryScanner:
    """
    Scans a directory for image files organized in subdirectories,
    and counts the number of images in each subdirectory.
    """

    # Define supported image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

    def __init__(self, root_path: str):
        self.root_path: Path = Path(root_path)
        if not self.root_path.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_path}")
        if not self.root_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {self.root_path}")

    def get_distribution(self) -> dict:
        """
        Scans the directory and returns a dictionary with the count of images per class (subdirectory).
        Returns:
            dict: A dictionary where keys are class names (subdirectory names) and values are image counts.
        """
        distribution: Counter = Counter()

        # recursion through all files in the directory
        for file_path in self.root_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                # Use the parent directory name as the class label
                # Example: ./Apple/apple_healthy/image.jpg -> 'apple_healthy'
                label = file_path.parent.name

                # only count images not in the root directory
                if file_path.parent != self.root_path:
                    distribution[label] += 1

        return dict(distribution)

    def print_summary(self, data: dict) -> None:
        """
        Prints a summary of the image distribution.
        Args:
            data (dict): A dictionary with class names as keys and image counts as values.
        Returns:
            None
        """
        print(f"Found {sum(data.values())} images in {len(data)} classes:")
        for label, count in data.items():
            print(f" - {label}: {count}")
