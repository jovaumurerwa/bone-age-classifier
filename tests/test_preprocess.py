import io

import numpy as np
import pytest
from PIL import Image

from bone_age.preprocess import InvalidImageError, MAX_FILE_BYTES, load_safe_image, to_model_input


def make_image_bytes(size=(300, 400), fmt="PNG"):
    buffer = io.BytesIO()
    Image.new("L", size, color=128).save(buffer, format=fmt)
    return buffer.getvalue()


def test_valid_png_becomes_correct_model_input():
    model_input = to_model_input(load_safe_image(make_image_bytes()))
    assert model_input.shape == (1, 224, 224, 3)
    assert model_input.dtype == np.float32
    assert 0.0 <= model_input.min() <= model_input.max() <= 1.0


def test_valid_jpeg_is_accepted():
    assert load_safe_image(make_image_bytes(fmt="JPEG")).format == "JPEG"


def test_empty_file_is_rejected():
    with pytest.raises(InvalidImageError):
        load_safe_image(b"")


def test_non_image_is_rejected():
    with pytest.raises(InvalidImageError):
        load_safe_image(b"<script>alert('x')</script>")


def test_file_over_size_limit_is_rejected():
    with pytest.raises(InvalidImageError):
        load_safe_image(b"0" * (MAX_FILE_BYTES + 1))


def test_other_image_formats_are_rejected():
    with pytest.raises(InvalidImageError):
        load_safe_image(make_image_bytes(fmt="GIF"))


def test_tiny_image_is_rejected():
    with pytest.raises(InvalidImageError):
        load_safe_image(make_image_bytes(size=(10, 10)))


def test_decompression_bomb_is_rejected():
    with pytest.raises(InvalidImageError):
        load_safe_image(make_image_bytes(size=(5000, 5000)))
