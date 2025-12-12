import argparse
import os
import sys
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = (256, 256)
MODEL_SAVE_NAME = "leaf_model.pth"
ZIP_NAME = "submission.zip"


class SimpleCNN(nn.Module):
    """
    Model of a simple Convolutional Neural Network (CNN)
    """

    def __init__(self, num_classes: int, input_size: tuple[int, int] = IMG_SIZE):
        # super(SimpleCNN, self).__init__()
        super().__init__()
        # Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256 -> 128

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128 -> 64

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64 -> 32

            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32 -> 16
        )

        # To calculate the flatten size dynamically
        # Use a dummy input tensor to infer the size after conv layers
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        dummy_output = self.features(dummy_input)

        # Calculate the size of the output (128 * 16 * 16)
        self.flatten_size = dummy_output.view(1, -1).size(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_data_loaders(data_dir: str) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Creates training and validation data loaders.
    Args:
        data_dir (str): Path to the prepared dataset directory.
    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dir: Path = Path(data_dir) / "train"
    val_dir: Path = Path(data_dir) / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Train/Val directories not found in {data_dir}. Run prepare.py first.")

    # Class: Load datasets from directories
    train_dataset: datasets.ImageFolder = datasets.ImageFolder(root=str(train_dir), transform=transform)
    val_dataset: datasets.ImageFolder = datasets.ImageFolder(root=str(val_dir), transform=transform)

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, train_dataset.classes


def train(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device
) -> tuple[float, float]:
    """
    Trains the model for one epoch.
    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the training on.
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy


def validate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
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
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy


def create_submission_zip(source_dir: str, model_path: str, output_zip: str) -> None:
    """
    Creates a zip file containing the dataset and the trained model.
    Args:
        source_dir (str): Path to the prepared dataset directory.
        model_path (str): Path to the trained model file.
        output_zip (str): Path to the output zip file.
    """
    print(f"Creating submission zip: {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # モデルを追加
        zipf.write(model_path, arcname=os.path.basename(model_path))

        # データセットディレクトリを追加
        src_path = Path(source_dir)
        for file in tqdm(list(src_path.rglob('*')), desc="Zipping Dataset"):
            if file.is_file():
                # zip内のパスを dataset_prepared/... から始めるように調整
                zipf.write(file, arcname=str(file))
    print("Zip created successfully.")


def main():
    parser = argparse.ArgumentParser(description="Train a CNN for leaf disease classification.")
    parser.add_argument("directory", help="Path to the prepared dataset directory (e.g., dataset_prepared)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    args = parser.parse_args()

    # cuda: NVIDIA GPU, mps: Apple Silicon GPU, cpu: CPU
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load data
    try:
        train_loader: DataLoader
        val_loader: DataLoader
        class_names: list[str]
        train_loader, val_loader, class_names = get_data_loaders(args.directory)
        print(f"Classes found: {class_names}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # 2. Define model, loss function, optimizer
    model: SimpleCNN = SimpleCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Loop over epochs
    best_acc = 0.0

    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_NAME)
            if best_acc >= 95.0:
                print("Target accuracy reached, stopping training early.")
                break

    print(f"\nTraining finished. Best Validation Accuracy: {best_acc:.2f}%")
    if best_acc < 90:
        print("Warning: Validation accuracy is below 90%. Consider training longer or adjusting parameters.")
    else:
        print("Success: Target accuracy reached!")

    # 4. Create submission zip
    if Path(MODEL_SAVE_NAME).exists():
        create_submission_zip(args.directory, MODEL_SAVE_NAME, ZIP_NAME)
    else:
        print("Model file not found, skipping zip creation.")


if __name__ == "__main__":
    main()
