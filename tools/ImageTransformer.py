import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, cast


class ImageTransformer:
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # OpenCV loads in BGR format
        self.img_bgr = cv2.imread(str(path))
        if self.img_bgr is None:
            raise ValueError(f"Could not load image: {path}")

        # Keep an RGB version for Matplotlib display
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)

        # Cache for the mask to avoid recalculating it for every step
        self._mask: np.ndarray | None = None

    def get_original(self):
        """Returns the original image in RGB."""
        return self.img_rgb

    def gaussian_blur(self, ksize=(5, 5)):
        """Applies Gaussian Blur to reduce noise."""
        blurred = cv2.GaussianBlur(self.img_rgb, ksize, 0)
        return blurred

    def _create_mask(self):
        """Internal method to create a binary mask of the leaf."""
        if self._mask is not None:
            return self._mask

        # Convert to HSV color space (better for color filtering)
        if self.img_bgr is None:
            raise ValueError("Image data is invalid or not loaded properly.")
        hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)

        # Define range for green colors (healthy leaf)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Define range for brown/yellow colors (diseased leaf parts)
        lower_brown = np.array([10, 40, 40])
        upper_brown = np.array([25, 255, 255])
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

        # Combine masks
        self._mask = cv2.bitwise_or(mask_green, mask_brown)

        # Clean up mask (Morphological operations) to remove small noise
        kernel = np.ones((5, 5), np.uint8)
        self._mask = cv2.morphologyEx(self._mask, cv2.MORPH_CLOSE, kernel)
        self._mask = cv2.morphologyEx(self._mask, cv2.MORPH_OPEN, kernel)

        return self._mask

    def apply_mask(self):
        """Applies the binary mask to the original image."""
        mask = self._create_mask()
        # Bitwise-AND mask and original image
        result = cv2.bitwise_and(self.img_rgb, self.img_rgb, mask=mask)
        return result

    def analyze_object(self):
        """Finds and draws contours of the leaf."""
        mask = self._create_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw on a copy
        result = self.img_rgb.copy()
        # Draw all contours in green, thickness 2
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        return result

    def roi_objects(self):
        """Draws the Region of Interest (Bounding Box) around the leaf."""
        mask = self._create_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = self.img_rgb.copy()

        if contours:
            # Find the largest contour (assuming it's the leaf)
            c = max(contours, key=cv2.contourArea)

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(c)

            # Draw rectangle (Blue)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 3)

        return result

    def pseudo_landmarks(self):
        """Finds geometric landmarks (centroid, top, bottom, etc.)."""
        mask = self._create_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        result = self.img_rgb.copy()

        if contours:
            c = max(contours, key=cv2.contourArea)

            # 1. Centroid (using moments)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(result, (cX, cY), 10, (255, 0, 0), -1)  # Red dot center

            # 2. Extreme Points
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            # Draw points
            for point in [extLeft, extRight, extTop, extBot]:
                cv2.circle(result, point, 8, (255, 0, 255), -1)  # Magenta dots

        return result

    def color_histogram(self):
        """Generates a color histogram plot image."""
        # Split channels
        colors = ('r', 'g', 'b')
        channel_ids = (0, 1, 2)  # Note: self.img_rgb is RGB

        # Create a figure using matplotlib
        fig = plt.figure(figsize=(4, 3), dpi=100)
        plt.title("Color Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")

        mask = self._create_mask()

        for channel_id, color in zip(channel_ids, colors):
            hist = cv2.calcHist([self.img_rgb], [channel_id], mask, [256], [0, 256])
            plt.plot(hist, color=color)
            plt.xlim([0, 256])

        plt.tight_layout()

        # Convert Matplotlib figure to NumPy array (Image)
        fig.canvas.draw()

        # Matplotlib 3.8+ 対応: buffer_rgba() を使用して RGBA バッファを取得
        canvas_obj = cast(Any, fig.canvas)
        data = np.frombuffer(canvas_obj.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()

        # Reshape (高さ, 幅, 4チャンネル)
        img_rgba = data.reshape((h, w, 4))

        # RGBA -> RGB に変換 (アルファチャンネルを削除)
        img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)

        plt.close(fig)
        return img_rgb
