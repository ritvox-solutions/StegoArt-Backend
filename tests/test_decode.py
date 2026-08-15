import base64


def test_decode_after_encode_roundtrip_text(client, sample_image_bytes):
    encode_resp = client.post(
        "/api/encode",
        files={"cover_image": ("cover.png", sample_image_bytes, "image/png")},
        data={"secret_type": "text", "secret_text": "round trip check"},
    )
    assert encode_resp.status_code == 200
    stego_bytes = base64.b64decode(encode_resp.json()["stego_image_base64"])

    decode_resp = client.post(
        "/api/decode",
        files={"stego_image": ("stego.png", stego_bytes, "image/png")},
        data={"secret_type_hint": "text"},
    )
    assert decode_resp.status_code == 200
    body = decode_resp.json()
    assert body["recovered_image_base64"] is None
    assert isinstance(body["recovered_text"], str)
    assert 0.0 <= body["confidence"] <= 1.0


def test_decode_image_secret_shape(client, sample_image_bytes, sample_image_bytes_2):
    encode_resp = client.post(
        "/api/encode",
        files={
            "cover_image": ("cover.png", sample_image_bytes, "image/png"),
            "secret_image": ("secret.png", sample_image_bytes_2, "image/png"),
        },
        data={"secret_type": "image"},
    )
    stego_bytes = base64.b64decode(encode_resp.json()["stego_image_base64"])

    decode_resp = client.post(
        "/api/decode",
        files={"stego_image": ("stego.png", stego_bytes, "image/png")},
        data={"secret_type_hint": "image"},
    )
    assert decode_resp.status_code == 200
    body = decode_resp.json()
    assert body["recovered_text"] is None
    assert body["recovered_image_base64"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_decode_invalid_file_returns_400(client):
    response = client.post(
        "/api/decode",
        files={"stego_image": ("bad.png", b"not an image", "image/png")},
        data={"secret_type_hint": "text"},
    )
    assert response.status_code == 400
