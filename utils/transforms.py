from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

class ResizePad:
    """
    Resize image to fit within target size while maintaining aspect ratio,
    then pad to reach exact target dimensions.
    """
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        self.size = size  # (H, W)
        self.interp = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        target_h, target_w = self.size
        w, h = img.size  # PIL is (W, H)

        # Calculate scale to fit within target size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = F.resize(img, (new_h, new_w), interpolation=self.interp, antialias=True)

        # Calculate padding to reach target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        img = F.pad(img, [left, top, right, bottom])
        return img
