from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import copy
import functools


class ImageAugmentor:
    def __init__(self, path: str):
        try:
            img: Image.Image = Image.open(path)
            img.verify()
            self.img: Image.Image = Image.open(path)
        except (OSError, IOError) as e:
            raise ValueError(f"Can't open image {path}: {e}")
        self.history: list[str] = []
        self.file = Path(path)

    @staticmethod
    def _augment_method(func):

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            new = self.copy()
            func(new, *args, **kwargs)
            new.history.append((func.__name__))
            return new
        return wrapper

    def copy(self):
        new = copy.copy(self)
        new.img = self.img.copy()
        new.history = self.history.copy()
        return new

    @_augment_method
    def rotate(self, degree=-25):
        self.img = self.img.rotate(degree)

    @_augment_method
    def shear(self, a=1, b=0.1, c=0, d=0.1, e=1, f=0):
        self.img = self.img.transform(self.img.size, Image.Transform.AFFINE, (a, b, c, d, e, f))

    @_augment_method
    def flip(self, mode=Image.Transpose.FLIP_LEFT_RIGHT):
        self.img = self.img.transpose(mode)

    @_augment_method
    def contrast(self, factor=3):
        self.img = ImageEnhance.Contrast(self.img).enhance(factor)

    @_augment_method
    def blur(self, radius=2):
        self.img = self.img.filter(ImageFilter.GaussianBlur(radius))

    @_augment_method
    def brightness(self, factor=1.7):
        self.img = ImageEnhance.Brightness(self.img).enhance(factor)

    def save(self, dir: str = "") -> None:
        stem = self.file.stem
        suffix = self.file.suffix
        aug = "_".join([name.capitalize() for name in self.history])
        new_name = f"{stem}_{aug}{suffix}" if aug else f"{stem}{suffix}"
        save_dir = Path(dir) if dir else self.file.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / new_name
        print(f"Augmentation Picture was Saved at {save_path}")
        self.img.save(save_path)
