"""The wallpaper client's bucket-listing parser and downloader (SB-11).

S3ImgFetcher reads the public bucket's XML listing and downloads every 1k/
still. These tests feed it a literal ListBucketResult and a fake
requests.get, so nothing reaches the network.
"""
from sunback.fetcher.S3ImgFetcher import S3ImgFetcher

BASE = "https://s3.us-east-2.amazonaws.com/the-sun-now/"
LISTING = """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>the-sun-now</Name><Prefix></Prefix><MaxKeys>1000</MaxKeys><IsTruncated>true</IsTruncated>
  <Contents><Key>1k/rhef_171_1k.png</Key><Size>1</Size></Contents>
  <Contents><Key>1k/rhef_rainbow_1k.png</Key><Size>1</Size></Contents>
  <Contents><Key>frames/171/20260928T120000_1k.png</Key><Size>1</Size></Contents>
  <Contents><Key>image_times.txt</Key><Size>1</Size></Contents>
  <Contents><Key>manifest/171.json</Key><Size>1</Size></Contents>
  <Contents><Key>thumb/rhef_171_thumb.png</Key><Size>1</Size></Contents>
  <Contents><Key>v/171/20260928T120000.png</Key><Size>1</Size></Contents>
</ListBucketResult>"""


class FakeResponse:
    def __init__(self, body):
        self.body = body
        self.text = body.decode("utf-8", "replace")

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]


def bare_fetcher(download_dir):
    fetcher = S3ImgFetcher.__new__(S3ImgFetcher)  # skip Processor.__init__ (needs Parameters)
    fetcher.xml_url = BASE
    fetcher.download_dir = str(download_dir)
    return fetcher


def test_listing_parser_keeps_only_1k_stills(tmp_path):
    urls = bare_fetcher(tmp_path)._parse_xml_for_images(LISTING)
    assert urls == [BASE + "1k/rhef_171_1k.png", BASE + "1k/rhef_rainbow_1k.png"]


def test_download_writes_the_bytes_and_skips_thumbs(tmp_path, monkeypatch):
    fetched = []

    def fake_get(url, stream=False, **kwargs):
        fetched.append(url)
        return FakeResponse(b"png:" + url.encode())

    monkeypatch.setattr("sunback.fetcher.S3ImgFetcher.requests.get", fake_get)
    fetcher = bare_fetcher(tmp_path)
    fetcher._download_image(BASE + "1k/rhef_171_1k.png")
    fetcher._download_image(BASE + "thumb/rhef_171_thumb.png")
    assert fetched == [BASE + "1k/rhef_171_1k.png"]
    assert (tmp_path / "rhef_171_1k.png").read_bytes() == b"png:" + (BASE + "1k/rhef_171_1k.png").encode()
    assert not (tmp_path / "rhef_171_thumb.png").exists()
