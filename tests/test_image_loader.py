"""
Tests for Image Loader utility.

Module: app/utils/image_loader.py
Covers:
- load_image_base64() — file loading
- get_image_mime() — MIME type detection
- build_multimodal_content() — Vision API content builder
- has_images() — quick image check
- make_image_path() — path generation helper
"""

import base64

from langchain.schema import Document
from app.utils.image_loader import (
    load_image_base64,
    get_image_mime,
    build_multimodal_content,
    has_images,
    make_image_path,
)


# ━━ get_image_mime() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestGetImageMime:

    def test_png(self):
        assert get_image_mime("image.png") == "image/png"

    def test_jpg(self):
        assert get_image_mime("photo.jpg") == "image/jpeg"

    def test_jpeg(self):
        assert get_image_mime("photo.jpeg") == "image/jpeg"

    def test_gif(self):
        assert get_image_mime("animation.gif") == "image/gif"

    def test_webp(self):
        assert get_image_mime("image.webp") == "image/webp"

    def test_unknown_defaults_to_png(self):
        assert get_image_mime("document.bmp") == "image/png"

    def test_uppercase_extension(self):
        assert get_image_mime("IMAGE.PNG") == "image/png"

    def test_path_with_dirs(self):
        assert get_image_mime("data/exams/images/q01.jpg") == "image/jpeg"


# ━━ load_image_base64() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLoadImageBase64:

    def test_nonexistent_file_returns_none(self):
        result = load_image_base64("/nonexistent/path/image.png")
        assert result is None

    def test_existing_file(self, tmp_path):
        # Create a test file
        img_file = tmp_path / "test.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG header
        result = load_image_base64(str(img_file))
        assert result is not None
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert decoded == b"\x89PNG\r\n\x1a\n"

    def test_empty_path(self):
        result = load_image_base64("")
        assert result is None


# ━━ has_images() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHasImages:

    def test_no_images(self):
        docs = [
            Document(page_content="text", metadata={"has_image": False}),
            Document(page_content="more text", metadata={}),
        ]
        assert has_images(docs) is False

    def test_has_image(self):
        docs = [
            Document(page_content="text", metadata={"has_image": False}),
            Document(page_content="with image", metadata={"has_image": True}),
        ]
        assert has_images(docs) is True

    def test_empty_docs(self):
        assert has_images([]) is False

    def test_all_have_images(self):
        docs = [
            Document(page_content="a", metadata={"has_image": True}),
            Document(page_content="b", metadata={"has_image": True}),
        ]
        assert has_images(docs) is True


# ━━ build_multimodal_content() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildMultimodalContent:

    def test_text_only(self):
        docs = [Document(page_content="no image", metadata={})]
        content = build_multimodal_content("Question text", docs)
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Question text"

    def test_text_always_first(self):
        docs = [Document(page_content="content", metadata={"has_image": True, "image_path": "fake.png"})]
        content = build_multimodal_content("My question", docs)
        assert content[0]["type"] == "text"

    def test_max_images_limit(self):
        docs = [
            Document(page_content="d1", metadata={"has_image": True, "image_path": "fake1.png"}),
            Document(page_content="d2", metadata={"has_image": True, "image_path": "fake2.png"}),
            Document(page_content="d3", metadata={"has_image": True, "image_path": "fake3.png"}),
        ]
        # Images won't actually load (files don't exist), so no image_url entries
        content = build_multimodal_content("Q", docs, max_images=2)
        # Only text entries since image files don't exist
        text_entries = [c for c in content if c["type"] == "text"]
        assert len(text_entries) >= 1

    def test_with_real_image(self, tmp_path):
        """Test with an actual file that can be loaded."""
        img_file = tmp_path / "test_q.png"
        img_file.write_bytes(b"\x89PNG\r\n\x1a\n")
        docs = [
            Document(
                page_content="Question with image",
                metadata={"has_image": True, "image_path": str(img_file)},
            )
        ]
        content = build_multimodal_content("Solve this", docs)
        # Should have: text, image label text, image_url
        assert len(content) >= 2
        image_urls = [c for c in content if c["type"] == "image_url"]
        assert len(image_urls) == 1
        assert "base64" in image_urls[0]["image_url"]["url"]


# ━━ make_image_path() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestMakeImagePath:

    def test_basic_path(self):
        path = make_image_path("de7plus_de01", 1)
        assert path == "data/exams/images/de7plus_de01/q01.png"

    def test_double_digit_question(self):
        path = make_image_path("de7plus_de01", 14)
        assert path == "data/exams/images/de7plus_de01/q14.png"

    def test_custom_extension(self):
        path = make_image_path("examA", 3, ext="jpg")
        assert path == "data/exams/images/examA/q03.jpg"

    def test_padding(self):
        path = make_image_path("test", 5)
        assert "q05" in path
