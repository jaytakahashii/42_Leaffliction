import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from tools.ImageTransformer import ImageTransformer
from tools.SimpleCNN import SimpleCNN

# Hyperparameters
BATCH_SIZE = 32
IMG_SIZE = (256, 256)
MODEL_SAVE_NAME = "leaf_model.pth"


def predict(
    model: nn.Module, device: torch.device, path: str
) -> torch.Tensor:
    """
    Predict the class index of the input image using the trained model.
    Args:
        model (nn.Module): The trained model for prediction.
        device (torch.device): The device to run the model on.
        path (str): The file path of the input image.
    Returns:
        torch.Tensor: The predicted class index.
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

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, height_ratios=[10, 1], hspace=0.05)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax_label = fig.add_subplot(gs[1, :])

    ax0.imshow(origin)
    ax0.axis('off')
    ax0.set_title('Original')

    ax1.imshow(transformed)
    ax1.axis('off')
    ax1.set_title('Transformed')

    ax_label.axis('off')
    ax_label.text(
        0.5, 0.5, f"Class predicted : {disease}",
        ha='center', va='center', fontsize=14, weight='bold'
    )

    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.2)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Predict leaf disease classification.")
    parser.add_argument("file", help="image file path")
    parser.add_argument("--model", type=str, default=MODEL_SAVE_NAME, help="model file name")
    args = parser.parse_args()

    # cuda: NVIDIA GPU, mps: Apple Silicon GPU, cpu: CPU
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    try:
        checkpoint = torch.load(args.model, map_location=device)
        model = SimpleCNN(num_classes=len(checkpoint["class_names"])).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        class_index = predict(model, device, args.file)
        visualize(args.file, checkpoint["class_names"][int(class_index)])

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
