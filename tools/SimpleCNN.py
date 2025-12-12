import torch
import torch.nn as nn

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
