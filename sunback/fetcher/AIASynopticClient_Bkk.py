import logging
from sunpy.net import attr, attrs, Fido
from sunpy.net.dataretriever import GenericClient
from parfive import Downloader
import astropy.units as u
from tqdm import tqdm
import os
from sunpy.net.dataretriever.client import QueryResponse

# Setup the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default level; can be changed dynamically

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)  # You can set this to INFO or DEBUG as needed

# Create formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)


class AIASynopticData(attr.Attr):
    """
    Custom attribute to indicate the use of low-resolution synoptic AIA data.
    Default to True if no value is provided.
    """

    def __init__(self):
        super().__init__()


class AIASynopticClient(GenericClient):
    # #  Example Link
    # https://jsoc1.stanford.edu/data/aia/synoptic/2022/06/06/H0500/ AIA20220606_0500_0094.fits

    baseurl = r"https://jsoc1.stanford.edu/data/aia/synoptic/%Y/%m/%d/H%H00/"
    pattern = "{baseurl}AIA%Y%m%d_%H%M_{wavelength}.fits"
    local_path = "temp"
    # os.path.abspath(
    # "../sunback_data/synoptic"
    # )  # Adjust this path as needed
    known_wavelengths = ["0094", "0131", "0171", "0193", "0211", "0304", "1600", "1700"]

    @classmethod
    def register_values(cls):
        return {
            attrs.Instrument: [("AIA", "Atmospheric Imaging Assembly")],
            attrs.Source: [("SDO", "Solar Dynamics Observatory")],
            attrs.Provider: [("JSOC", "Joint Science Operations Center")],
            attrs.Wavelength: [
                (str(94 * u.angstrom), "0094"),
                (str(131 * u.angstrom), "0131"),
                (str(171 * u.angstrom), "0171"),
                (str(193 * u.angstrom), "0193"),
                (str(211 * u.angstrom), "0211"),
                (str(304 * u.angstrom), "0304"),
                (str(1600 * u.angstrom), "1600"),
                (str(1700 * u.angstrom), "1700"),
            ],
            AIASynopticData: [("True", "Low-resolution synoptic AIA data")],
        }

    @classmethod
    def _can_handle_query(cls, *query):
        required = {attrs.Time, AIASynopticData}
        all_attrs = {type(x) for x in query}
        return required.issubset(all_attrs)

    def search(self, *query):
        """
        Override the search method to handle multiple wavelengths or wildcard wavelengths,
        and to apply cadence filtering to reduce the data load.
        """
        from sunpy.net import attrs as a
        import astropy.units as u

        time_range = None
        wavelengths = []
        cadence = None

        # Extract time range, wavelength, and cadence from the query
        for q in query:
            if isinstance(q, a.Time):
                time_range = q
            elif isinstance(q, a.Wavelength):
                wavelengths.append(int(q.min.value))
            elif isinstance(q, a.Sample):
                cadence = q.min * u.s if hasattr(q, "min") else q.value * u.s
                cadence = cadence.to(
                    u.minute
                ).value  # Convert to minutes for easier handling

        if not time_range:
            logger.error("Time range must be specified for the AIASynopticClient.")
            raise ValueError("Time range must be specified for the AIASynopticClient.")

        # If no specific wavelength is specified, use all known wavelengths
        if not wavelengths:
            wavelengths = self.known_wavelengths
        else:
            wavelengths = [
                str(wl).zfill(4) for wl in wavelengths
            ]  # Ensure proper format

        # Apply cadence filtering if specified
        urls = self._generate_urls(time_range, wavelengths, cadence)

        # a = [print(ur) for ur in urls]

        return self._fetch_data_from_urls(urls)

    def _generate_urls(self, time_range, wavelengths, cadence=None):
        """
        Generate URLs for all requested wavelengths and time intervals.
        Applies cadence filtering to reduce the number of generated URLs.
        """
        from datetime import timedelta

        current_time = time_range.start.datetime
        end_time = time_range.end.datetime
        urls = []

        while current_time <= end_time:
            for wavelength in wavelengths:
                # Format the base URL and pattern using strftime to substitute time components
                formatted_baseurl = current_time.strftime(self.baseurl)
                formatted_url = current_time.strftime(self.pattern).format(
                    baseurl=formatted_baseurl, wavelength=wavelength
                )
                urls.append(formatted_url)

            # Increment the time based on the specified cadence (default to every minute if not specified)
            current_time += timedelta(minutes=cadence or 1)

        logger.debug(f"Generated {len(urls)} URLs for download.")
        return urls

    from sunpy.net.dataretriever.client import QueryResponse

    def _fetch_data_from_urls(self, urls, ask=1, max_conn=10, enable_retry=False):
        """
        Fetch data from a list of URLs with optional retry capability and display progress with tqdm.
        Includes a confirmation prompt if the number of files is above a threshold.

        Parameters:
        - urls: List of URLs to download.
        - ask: Threshold for number of files before asking for confirmation.
        - enable_retry: Boolean flag to enable or disable retry logic.
        """
        # Threshold for confirmation prompt
        download_threshold = ask
        total_size_mb = self._estimate_total_size(urls)

        if len(urls) > download_threshold:
            confirm = input(
                f"\n\nYou are about to download {len(urls)} files totaling approximately {total_size_mb} MB.\n"
                f"They will be stored in:\n\t{self.out_path}\n"
                "Do you want to proceed? (y/n): "
            )
            if confirm.lower() != "y":
                logger.info("Download process aborted by user.")
                raise UserAbortError("Download process aborted by user.")

        download_path = self.local_path  # Use the same path for all downloads
        downloader = Downloader(
            max_conn=max_conn, progress=tqdm(total=len(urls), desc="Initial Download")
        )

        for url in urls:
            downloader.enqueue_file(url, path=download_path)  # Save to specified path

        # After downloading
        results = downloader.download()
        failed_downloads = results.errors

        # Initialize data dictionary
        data = {
            "fileid": [],
            "url": [],
            "status": [],
        }

        # Check if results are strings or Result objects
        for file in results:
            if isinstance(file, str):
                # file is a file path (string)
                data["fileid"].append(file)
                data["url"].append(
                    "Unknown"
                )  # You can store the URL elsewhere if needed
                data["status"].append("Success")
            else:
                # file is a Result object
                data["fileid"].append(str(file.filepath))
                data["url"].append(file.url)
                data["status"].append("Success")

        # Handle failed downloads
        if failed_downloads:
            for error in failed_downloads:
                data["fileid"].append(None)
                data["url"].append(error.url)
                data["status"].append("Failed")

        # Construct the QueryResponse object
        query_response = QueryResponse(data)

        return query_response

    def _estimate_total_size(self, urls):
        """
        Estimate the total size of the download (for user confirmation purposes).
        This is a placeholder function; actual implementation might require querying the server for file sizes.
        """
        average_file_size_mb = 1.5  # Placeholder average size in MB per file
        total_size_mb = len(urls) * average_file_size_mb
        return total_size_mb


class UserAbortError(Exception):
    """Exception raised when the user aborts an operation."""

    pass


# Automatically register the client with Fido when the module is imported
if AIASynopticClient not in Fido.registry:
    Fido.registry[AIASynopticClient] = AIASynopticClient._can_handle_query
    logger.info("Synoptic Fido Client Loaded!")
