import hashlib
from pathlib import Path

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "bone_age_classifier" / "model.weights.h5"
WEIGHTS_SHA256 = "3972863d59d59fbb7158a0c466a1e5eac7d874d00dc176a46d11fce6b7b97a2d"


class ModelIntegrityError(RuntimeError):
    """Raised when the weights file does not match the expected fingerprint."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weights(path: Path = WEIGHTS_PATH, expected: str = WEIGHTS_SHA256) -> Path:
    """Refuse to use a weights file that was changed or replaced."""
    if not path.is_file():
        raise ModelIntegrityError(f"Weights file not found: {path.name}")
    if sha256_of(path) != expected:
        raise ModelIntegrityError("Weights file fingerprint does not match. Refusing to load it.")
    return path
