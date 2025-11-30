import sys
import argparse
from tools.scanner import DirectoryScanner


def main() -> None:
    # 1. Create parser
    parser = argparse.ArgumentParser(description="Analyze image dataset distribution.")
    parser.add_argument("directory", help="Path to the dataset directory")
    args = parser.parse_args()

    # 2. Get distribution data
    try:
        scanner: DirectoryScanner = DirectoryScanner(args.directory)
        data: dict = scanner.get_distribution()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not data:
        print("No images found in the specified directory.")
        sys.exit(0)

    # 3. Print summary
    scanner.print_summary(data)


if __name__ == "__main__":
    main()
