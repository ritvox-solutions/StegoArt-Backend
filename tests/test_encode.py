from ..ml.config import MAX_TEXT_CHARS


def test_encode_text_secret_returns_expected_shape(client, sample_image_bytes):
    response = client.post(
        "/api/encode",
        files={"cover_image": ("cover.png", sample_image_bytes, "image/png")},
        data={"secret_type": "text", "secret_text": "hello world"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["stego_image_base64"]
    assert body["styled_image_base64"] is None
    assert isinstance(body["psnr"], float)
    assert isinstance(body["ssim"], float)
    assert body["styled_decode_supported"] is True


def test_encode_image_secret(client, sample_image_bytes, sample_image_bytes_2):
    response = client.post(
        "/api/encode",
        files={
            "cover_image": ("cover.png", sample_image_bytes, "image/png"),
            "secret_image": ("secret.png", sample_image_bytes_2, "image/png"),
        },
        data={"secret_type": "image"},
    )
    assert response.status_code == 200
    assert response.json()["stego_image_base64"]


def test_encode_with_style_flags_styled_decode_unsupported(client, sample_image_bytes):
    response = client.post(
        "/api/encode",
        files={"cover_image": ("cover.png", sample_image_bytes, "image/png")},
        data={
            "secret_type": "text",
            "secret_text": "styled test",
            "apply_style": "true",
            "style_name": "mosaic",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["styled_image_base64"]
    assert body["styled_decode_supported"] is False


def test_encode_unknown_style_returns_400(client, sample_image_bytes):
    response = client.post(
        "/api/encode",
        files={"cover_image": ("cover.png", sample_image_bytes, "image/png")},
        data={
            "secret_type": "text",
            "secret_text": "hi",
            "apply_style": "true",
            "style_name": "not-a-real-style",
        },
    )
    assert response.status_code == 400


def test_encode_missing_secret_text_returns_422(client, sample_image_bytes):
    response = client.post(
        "/api/encode",
        files={"cover_image": ("cover.png", sample_image_bytes, "image/png")},
        data={"secret_type": "text"},
    )
    assert response.status_code == 422


def test_encode_missing_secret_image_returns_422(client, sample_image_bytes):
    response = client.post(
        "/api/encode",
        files={"cover_image": ("cover.png", sample_image_bytes, "image/png")},
        data={"secret_type": "image"},
    )
    assert response.status_code == 422


def test_encode_text_too_long_returns_422(client, sample_image_bytes):
    response = client.post(
        "/api/encode",
        files={"cover_image": ("cover.png", sample_image_bytes, "image/png")},
        data={"secret_type": "text", "secret_text": "x" * (MAX_TEXT_CHARS + 1)},
    )
    assert response.status_code == 422


def test_encode_invalid_file_type_returns_400(client):
    response = client.post(
        "/api/encode",
        files={"cover_image": ("cover.txt", b"not an image", "text/plain")},
        data={"secret_type": "text", "secret_text": "hi"},
    )
    assert response.status_code == 400


def test_encode_oversized_file_returns_413(client):
    huge = b"\x00" * (10 * 1024 * 1024 + 1)
    response = client.post(
        "/api/encode",
        files={"cover_image": ("cover.png", huge, "image/png")},
        data={"secret_type": "text", "secret_text": "hi"},
    )
    assert response.status_code == 413
