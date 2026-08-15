def test_get_styles_returns_the_four_verified_styles(client):
    response = client.get("/api/styles")
    assert response.status_code == 200
    body = response.json()
    names = {s["name"] for s in body["styles"]}
    assert names == {"candy", "mosaic", "rain_princess", "udnie"}
    for s in body["styles"]:
        assert s["thumbnail_url"].startswith("/api/styles/")


def test_style_thumbnail_returns_png(client):
    response = client.get("/api/styles/mosaic/thumbnail")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0


def test_unknown_style_thumbnail_returns_404(client):
    response = client.get("/api/styles/not-a-real-style/thumbnail")
    assert response.status_code == 404
