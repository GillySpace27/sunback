import os
import sys
import requests
import xml.etree.ElementTree as ET
from os.path import join, split
from tqdm import tqdm
from sunback.fetcher.Fetcher import Fetcher

class S3ImgFetcher(Fetcher):
    """
    Fetch images from an S3 bucket XML listing and download them to a specified local directory.
    """
    description = "Get images from an S3 bucket XML listing"
    filt_name = "S3 Image Fetcher"

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.load(params)
        self.xml_url = 'https://s3.us-east-2.amazonaws.com/the-sun-now/'

        # Resolve the download directory
        self.download_dir = os.path.abspath(
            os.path.expanduser(self.params.imgs_top_directory()) if self.params else "./images"
        )

        # Log the resolved directory for debugging
        print(f"Resolved download directory: {self.download_dir}")

    def fetch(self, params=None):
        if params is not None:
            self.params = params
            self.__init__(params=params)

        os.makedirs(self.download_dir, exist_ok=True)
        print(f"Downloading images from '{self.xml_url}' to {self.download_dir}", flush=True)
        sys.stdout.flush()

        xml_content = self._fetch_xml_content()
        img_urls = self._parse_xml_for_images(xml_content)

        if not img_urls:
            print("No images found in the XML content.")
            return

        for img_url in tqdm(img_urls, desc="Downloading images", ncols=100):
            self._download_image(img_url)

        self.load()
        print(f"\nDownloaded {len(img_urls)} images in total.")
        sys.stdout.flush()

    def _fetch_xml_content(self):
        """Fetch the XML content from the specified URL."""
        response = requests.get(self.xml_url)
        response.raise_for_status()
        return response.text

    def _parse_xml_for_images(self, xml_content):
        """Parse XML content using ElementTree to extract image URLs."""
        namespaces = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        root = ET.fromstring(xml_content)
        img_urls = [
            f"{self.xml_url}{content.find('s3:Key', namespaces).text}"
            for content in root.findall('s3:Contents', namespaces)
            if content.find('s3:Key', namespaces).text.startswith('renders/')
        ]
        return img_urls

    def _download_image(self, url):
        """Download an image from the given URL to the specified local directory."""
        filename = os.path.basename(url)
        if "thumb" in filename:
            return
        file_path = join(self.download_dir, filename)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)