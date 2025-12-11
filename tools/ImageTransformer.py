import io
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageTransformer:
    LOWER_GREEN = np.array([25, 40, 40])
    UPPER_GREEN = np.array([90, 255, 255])
    LOWER_BROWN = np.array([10, 40, 40])
    UPPER_BROWN = np.array([25, 255, 255])
    GREEN_RGB = (0, 255, 0)
    BLUE_RGB = (0, 0, 255)
    RED_RGB = (255, 0, 0)
    MAGENTA_RGB = (255, 0, 255)
    CIRCLE_RADIUS = 4
    MARGIN = 10
    FONT_SIZE_LARGE = 24
    FONT_SIZE_MEDIUM = 20
    FONT_SIZE_SMALL = 18
    PLOT_LINE_WIDTH = 2.5

    def __init__(self, path: str):
        self.path: Path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # OpenCV loads in BGR format
        self.img_bgr: np.ndarray | None = cv2.imread(str(path))
        if self.img_bgr is None:
            raise ValueError(f"Could not load image: {path}")

        # Keep an RGB version for Matplotlib display
        self.img_rgb: np.ndarray = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)

        # Cache for the mask to avoid recalculating it for every step
        self._mask: np.ndarray | None = None

    def get_original(self) -> np.ndarray:
        """Returns the original image in RGB."""
        return self.img_rgb

    def gaussian_blur(self, ksize: tuple[int, int] = (5, 5)) -> np.ndarray:
        """
        Applies Gaussian Blur to reduce noise.
        Args:
            ksize (tuple): Kernel size for Gaussian Blur.
            Must be odd and positive.
        Returns:
            np.ndarray: Blurred image.
        Raises:
            ValueError: If ksize is not odd and positive.
        """
        # check ksize validity
        if ksize[0] % 2 == 0 or ksize[1] % 2 == 0 or (ksize[0] * ksize[1]) <= 0:
            raise ValueError("Kernel size must be odd and positive integers.")
        blurred: np.ndarray = cv2.GaussianBlur(self.img_rgb, ksize, 0)
        return blurred

    def _create_mask(self) -> np.ndarray:
        """
        Internal method to create a binary mask of the leaf.
        Returns:
            np.ndarray: Binary mask where leaf areas are white (255) and background is black (0).
        Raises:
            ValueError: If image data is invalid.
        """
        if self._mask is not None:
            return self._mask

        # Convert to HSV
        if self.img_bgr is None:
            raise ValueError("Image data is invalid or not loaded properly.")
        hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)

        # Create masks for green and brown colors
        mask_green = cv2.inRange(hsv, self.LOWER_GREEN, self.UPPER_GREEN)
        mask_brown = cv2.inRange(hsv, self.LOWER_BROWN, self.UPPER_BROWN)
        self._mask = cv2.bitwise_or(mask_green, mask_brown)

        # Clean up mask (Morphological operations) to remove small noise
        kernel = np.ones((5, 5), np.uint8)
        self._mask = cv2.morphologyEx(self._mask, cv2.MORPH_CLOSE, kernel)
        self._mask = cv2.morphologyEx(self._mask, cv2.MORPH_OPEN, kernel)

        return self._mask

    def apply_mask(self) -> np.ndarray:
        """
        Applies the binary mask to the original image.
        Returns:
            np.ndarray: Image with the mask applied.
        """
        mask = self._create_mask()
        result = cv2.bitwise_and(self.img_rgb, self.img_rgb, mask=mask)
        return result

    def analyze_object(self) -> np.ndarray:
        """
        Finds and draws contours of the leaf.
        Returns:
            np.ndarray: Image with contours drawn.
        """
        mask = self._create_mask()
        # RETR_EXTERNAL: retrieve only the extreme outer contours
        # CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw on a copy
        result = self.img_rgb.copy()
        # Draw all contours in green, thickness 2
        # cv2.drawContours(image, contours, contourIdx, color, thickness)
        cv2.drawContours(result, contours, -1, self.GREEN_RGB, 2)
        return result

    def roi_objects(self) -> np.ndarray:
        """
        Draws the Region of Interest (Bounding Box) around the leaf.
        Returns:
            np.ndarray: Image with bounding box drawn.
        """
        mask = self._create_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = self.img_rgb.copy()

        if contours:
            # Find the largest contour (assuming it's the leaf)
            c = max(contours, key=cv2.contourArea)

            # Get bounding rectangle
            # x, y: top-left corner; w, h: width and height
            x, y, w, h = cv2.boundingRect(c)

            # Draw rectangle (Blue)
            cv2.rectangle(result, (x, y), (x + w, y + h), self.BLUE_RGB, 3)

        return result

    def pseudo_landmarks(self) -> np.ndarray:
        """
        Finds geometric landmarks (centroid, top, bottom, etc.).
        Returns:
            np.ndarray: Image with pseudolandmarks drawn.
        """
        mask = self._create_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = self.img_rgb.copy()

        if contours:
            c = max(contours, key=cv2.contourArea)

            # 1. Centroid (using moments)
            M = cv2.moments(c)
            if M["m00"] != 0:
                # moments row: x = m10/m00, y = m01/m00
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # cv2.circle(image, center, radius, color, thickness)
                # -1 thickness fills the circle
                cv2.circle(result, (cX, cY), self.CIRCLE_RADIUS, self.RED_RGB, -1)

            # 2. Extreme Points
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])

            # Draw points
            for point in [extLeft, extRight, extTop, extBot]:
                cv2.circle(result, point, self.CIRCLE_RADIUS, self.MAGENTA_RGB, -1)

        return result

    def color_histogram(self) -> np.ndarray:
        """
        Generates a color histogram plot image.
        Returns:
            np.ndarray: Image of the color histogram plot.
        """
        # Split channels
        colors = ('r', 'g', 'b')
        channel_ids = (0, 1, 2)  # Note: self.img_rgb is RGB
        channel_names = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}

        fig = plt.figure(figsize=(12, 8), dpi=200)
        plt.xlabel("Pixel intensity", fontsize=self.FONT_SIZE_LARGE)
        plt.ylabel("Proportion of pixels(%)", fontsize=self.FONT_SIZE_LARGE)
        plt.xticks(fontsize=self.FONT_SIZE_MEDIUM)
        plt.yticks(fontsize=self.FONT_SIZE_MEDIUM)

        mask = self._create_mask()

        # calculate total number of pixels in the mask
        # --- RGB ---
        mask_pixel_count = cv2.countNonZero(mask)

        for channel_id, color in zip(channel_ids, colors):
            hist = cv2.calcHist([self.img_rgb], [channel_id], mask, [256], [0, 256])

            # modify normalization to avoid division by zero
            if mask_pixel_count > 0:
                hist = (hist / mask_pixel_count) * 100

            # Add labels for the legend
            plt.plot(hist, color=color, linewidth=self.PLOT_LINE_WIDTH, label=channel_names[color])
            plt.xlim([0, 256])

        # --- HSV ---
        img_bgr: np.ndarray | None = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR)
        assert img_bgr is not None, "Failed to decode image from buffer."
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        hsv_colors = ('purple', 'cyan', 'orange')
        hsv_ids = (0, 1, 2)
        hsv_names = {0: 'hue', 1: 'saturation', 2: 'value'}

        for channel_id, color in zip(hsv_ids, hsv_colors):
            bins = 180 if channel_id == 0 else 256
            max_val = 180 if channel_id == 0 else 256

            hist = cv2.calcHist([hsv], [channel_id], mask, [bins], [0, max_val])
            if mask_pixel_count > 0:
                hist = (hist / mask_pixel_count) * 100
            plt.plot(hist, color=color, linewidth=self.PLOT_LINE_WIDTH, label=hsv_names[channel_id])
            plt.xlim([0, max_val])

        # --- Lab ---
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        lab_colors = ('gray', 'yellow', 'magenta')
        lab_ids = (0, 1, 2)
        lab_names = {0: 'lightness', 1: 'blue-yellow', 2: 'green-magenta'}

        for channel_id, color in zip(lab_ids, lab_colors):
            hist = cv2.calcHist([lab], [channel_id], mask, [256], [0, 256])
            if mask_pixel_count > 0:
                hist = (hist / mask_pixel_count) * 100
            plt.plot(hist, color=color, linewidth=self.PLOT_LINE_WIDTH, label=lab_names[channel_id])
            plt.xlim([0, 256])

        handles, labels = plt.gca().get_legend_handles_labels()
        order: list[int] = [2, 7, 1, 8, 3, 6, 0, 4, 5]
        plt.legend(
            [handles[i] for i in order],
            [labels[i] for i in order],
            loc='upper right',
            fontsize=self.FONT_SIZE_MEDIUM,
            title="color Channel",
            title_fontsize=self.FONT_SIZE_LARGE)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200)
        buf.seek(0)

        data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Failed to decode image from buffer.")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        buf.close()
        plt.close(fig)

        return img_rgb

    def get_cropped_roi(self) -> np.ndarray:
        """
        Returns the image cropped to the Region of Interest (Leaf area).
        If no contour is found, returns the original image.
        MARGIN is added around the bounding box. That's because tight cropping may cut off parts of the leaf.
        Returns:
            np.ndarray: Cropped image of the leaf area.
        """
        mask = self._create_mask()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            img_h, img_w = self.img_rgb.shape[:2]
            x_start = max(0, x - self.MARGIN)
            y_start = max(0, y - self.MARGIN)
            x_end = min(img_w, x + w + self.MARGIN)
            y_end = min(img_h, y + h + self.MARGIN)

            return self.img_rgb[y_start:y_end, x_start:x_end]

        return self.img_rgb
