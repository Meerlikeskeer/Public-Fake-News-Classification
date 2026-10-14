import pytest

from fake_news import cli


class Response:
    encoding = "utf-8"

    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield from self.chunks


def test_live_fetch_decodes_split_utf8_and_closes_response(monkeypatch):
    response = Response([b"caf\xc3", b"\xa9"])
    monkeypatch.setattr(cli.requests, "get", lambda *args, **kwargs: response)
    assert cli.fetch_html("example.com") == "café"
    assert response.closed


def test_live_fetch_limits_response_size_and_closes_response(monkeypatch):
    response = Response([b"123", b"456"])
    monkeypatch.setattr(cli.requests, "get", lambda *args, **kwargs: response)
    monkeypatch.setattr(cli, "MAX_RESPONSE_BYTES", 5)
    with pytest.raises(ValueError, match="response limit"):
        cli.fetch_html("example.com")
    assert response.closed
