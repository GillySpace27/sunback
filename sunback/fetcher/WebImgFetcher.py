import os
import sys
from os.path import join
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
from sunback.fetcher.Fetcher import Fetcher


class WebImgFetcher(Fetcher):
    """
    Fetch images from a webpage and download them to a specified local directory.
    """
    description = "Get images from a specified webpage"
    filt_name = "Web Image Fetcher"

    def __init__(self, params=None, quick=False, rp=None):
        super().__init__(params, quick, rp)
        self.load(params)

        # URL to scrape images from
        self.webpage_url = 'https://gilly.space/sun.html'
        # self.webpage_url = "https://s3.us-east-2.amazonaws.com/the-sun-now/"
        # Expand and resolve download directory
        self.download_dir = os.path.abspath(
            os.path.expanduser(self.params.imgs_top_directory()) if self.params else "./images"
        )

        # Log the resolved directory for debugging
        print(f"Resolved download directory: {self.download_dir}")

    def fetch(self, params=None):
        """
        Download all valid image files from the webpage and store them
        in the download directory (self.params.imgs_top_directory()).
        """
        # If the user called fetch() with new params, update them
        if params is not None:
            self.params = params
            self.__init__(params=params)  # Re-init for new parameters

        # Ensure the download directory exists
        os.makedirs(self.download_dir, exist_ok=True)

        html_content = self._fetch_webpage_content()
        img_urls = self._parse_html_for_images(html_content)

        if not img_urls:
            print("No images found on the webpage.")
            return

        # Use tqdm to show progress
        for img_url in tqdm(img_urls, desc="Downloading images", ncols=100):
            self._download_image(img_url)

        # Optionally load into memory or do any post-download processing
        self.load()

        print(f"\nDownloaded {len(img_urls)} images in total.")
        sys.stdout.flush()

    def _fetch_webpage_content(self):
        """Fetch content of the webpage."""
        response = requests.get(self.webpage_url)
        response.raise_for_status()  # Raise an exception for bad responses
        return response.text

    def _parse_html_for_images(self, html_content):
        """Parse HTML content using BeautifulSoup to extract image URLs."""
        soup = BeautifulSoup(html_content, 'html.parser')
        image_elements = soup.find_all('img')
        img_urls = [img['src'] for img in image_elements if img['src'].startswith('https://s3.us-east-2.amazonaws.com/the-sun-now/renders/')]
        return img_urls

    def _download_image(self, url):
        """Download an image from the given URL to the specified local directory."""
        filename = os.path.basename(url)
        file_path = join(self.download_dir, filename)

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
