import io
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_TARGET_SIZE = (224, 224)
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
MIN_SIDE = 64
ALLOWED_FORMATS = {"PNG", "JPEG"}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class InvalidImageError(ValueError):
    """Raised when an upload is not a safe, usable X-ray image."""


def load_safe_image(file_bytes: bytes) -> Image.Image:
    """Check an uploaded file and return it as a PIL image, or raise InvalidImageError."""
    if not file_bytes:
        raise InvalidImageError("The file is empty.")
    if len(file_bytes) > MAX_FILE_BYTES:
        raise InvalidImageError("The file is larger than 5 MB.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(file_bytes)) as probe:
                probe.verify()
            image = Image.open(io.BytesIO(file_bytes))
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise InvalidImageError("The image is too large.") from error
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise InvalidImageError("The file is not a valid image.") from error

    if image.format not in ALLOWED_FORMATS:
        raise InvalidImageError("Only PNG and JPEG images are accepted.")
    width, height = image.size
    if width * height > MAX_PIXELS:
        raise InvalidImageError("The image is too large.")
    if min(width, height) < MIN_SIDE:
        raise InvalidImageError("The image is too small to analyse.")

    try:
        image.load()
    except (OSError, SyntaxError) as error:
        raise InvalidImageError("The file is not a valid image.") from error
    return image


def to_model_input(image: Image.Image) -> np.ndarray:
    """Grayscale, resize to 224x224, copy to 3 channels, scale to 0-1, add batch dimension."""
    grayscale = image.convert("L").resize(IMAGE_TARGET_SIZE)
    array = np.asarray(grayscale, dtype=np.float32) / 255.0
    three_channel = np.repeat(array[..., np.newaxis], 3, axis=-1)
    return three_channel[np.newaxis, ...]
