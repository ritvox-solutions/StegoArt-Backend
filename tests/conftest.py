"""Shared pytest fixtures for the backend test suite.

The `client` fixture is session-scoped because model loading (encoder,
decoder, 4 style checkpoints) happens once in the app's lifespan handler —
re-triggering that per test would be slow and pointless, since none of these
tests mutate model state.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def _make_image_bytes(size: int = 64, color=(120, 180, 90)) -> bytes:
    img = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_image_bytes():
    return _make_image_bytes(color=(120, 180, 90))


@pytest.fixture
def sample_image_bytes_2():
    return _make_image_bytes(color=(30, 60, 200))
