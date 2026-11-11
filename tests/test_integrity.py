import pytest

from bone_age.integrity import ModelIntegrityError, WEIGHTS_PATH, sha256_of, verify_weights


def test_shipped_weights_pass_integrity_check():
    assert verify_weights() == WEIGHTS_PATH


def test_tampered_weights_are_refused(tmp_path):
    fake = tmp_path / "model.weights.h5"
    fake.write_bytes(b"tampered")
    with pytest.raises(ModelIntegrityError):
        verify_weights(path=fake)


def test_missing_weights_are_refused(tmp_path):
    with pytest.raises(ModelIntegrityError):
        verify_weights(path=tmp_path / "missing.h5")


def test_hash_is_stable(tmp_path):
    file = tmp_path / "a.bin"
    file.write_bytes(b"abc")
    assert sha256_of(file) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
