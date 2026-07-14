"""Unit tests for S3 storage paths without making network calls."""

from unittest.mock import MagicMock

import pytest

from app.utils import s3 as s3_module


def test_download_single_exam_uses_canonical_client(tmp_path, monkeypatch):
    monkeypatch.setattr(s3_module, "EXAMS_DIR", str(tmp_path))
    storage = s3_module.S3StorageService()
    storage._client = MagicMock()

    assert storage.download_exam("de01.json") is True
    storage._client.download_file.assert_called_once_with(
        storage._bucket,
        "exams/de01.json",
        str(tmp_path / "de01.json"),
    )


@pytest.mark.parametrize("filename", ["../secret.json", "folder/de01.json", "de01.txt"])
def test_download_single_exam_rejects_unsafe_filename(filename):
    with pytest.raises(ValueError):
        s3_module.S3StorageService().download_exam(filename)
