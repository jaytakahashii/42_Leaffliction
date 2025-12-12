import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from tools.ImageTransformer import ImageTransformer
from tools.SimpleCNN import SimpleCNN

# ハイパーパラメータ設定
BATCH_SIZE = 32
IMG_SIZE = (256, 256)
MODEL_SAVE_NAME = "leaf_model.pth"
NUM_CLASSES = 8


def predict(
    model: nn.Module, device: torch.device, path: str
) -> torch.Tensor:
    """
    Evaluates the model on the validation dataset.
    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run the evaluation on.
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()

    with torch.no_grad():
        transformer = ImageTransformer(path)
        cropped = transformer.get_cropped_roi()
        resize = Image.fromarray(cropped).resize(IMG_SIZE, Image.Resampling.LANCZOS)
        target = transforms.functional.to_tensor(resize).unsqueeze(0).to(device)
        outputs = model(target)
        _, predicted = torch.max(outputs.data, 1)

    return predicted


def visualize(path: str, disease: str) -> None:
    transform = ImageTransformer(path)
    cropped = transform.get_cropped_roi()
    resize = Image.fromarray(cropped).resize(IMG_SIZE, Image.Resampling.LANCZOS)
    origin = transform.img_rgb
    transformed = resize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 左: 元画像
    axes[0].imshow(origin)
    axes[0].axis('off')
    axes[0].set_title("Original")

    # 右: 加工後画像
    axes[1].imshow(transformed)
    axes[1].axis('off')
    axes[1].set_title("Transformed")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predict leaf disease classification.")
    parser.add_argument("file", help="image file path")
    args = parser.parse_args()

    # cuda: NVIDIA GPU, mps: Apple Silicon GPU, cpu: CPU
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)

    # 1. Load data
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_NAME, map_location=device))
        class_index = predict(model, device, args.file)
        visualize(args.file, "")

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
