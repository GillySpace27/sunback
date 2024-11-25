import logging
from sunpy.net import attrs, Fido
from sunpy.net.dataretriever import GenericClient
from parfive import Downloader
import astropy.units as u
from sunpy.net import attr
from sunpy.net.dataretriever.client import QueryResponse
import os

# Setup the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)


class AIASynopticData(attr.Attr):
    """
    Custom attribute to indicate the use of low-resolution synoptic AIA data.
    Defaults to True if no value is provided.
    """

    def __init__(self, value=True):
        self.value = value

    def __repr__(self):
        return f"AIASynopticData({self.value})"

    def __eq__(self, other):
        if isinstance(other, AIASynopticData):
            return self.value == other.value
        return False


class AIASynopticClient(GenericClient):
    baseurl = r"https://jsoc1.stanford.edu/data/aia/synoptic/%Y/%m/%d/H%H00/"
    pattern = "{baseurl}AIA%Y%m%d_%H%M_{wavelength}.fits"
    known_wavelengths = ["0094", "0131", "0171", "0193", "0211", "0304", "1600", "1700"]

    def __init__(self):
        super().__init__()

    @classmethod
    def _can_handle_query(cls, *query):
        required_attrs = {attrs.Time}
        optional_attrs = {
            attrs.Instrument,
            attrs.Wavelength,
            attrs.Sample,
            AIASynopticData,
        }
        all_attrs = {type(x) for x in query}

        if not required_attrs.issubset(all_attrs):
            return False

        # Check for AIASynopticData in query
        has_synoptic_data_attr = any(isinstance(x, AIASynopticData) for x in query)
        return has_synoptic_data_attr

    def search(self, *query):
        """
        Perform the search to generate URLs based on the time range and wavelength.
        """
        time_range = None
        wavelengths = []
        cadence = None

        for q in query:
            if isinstance(q, attrs.Time):
                time_range = q
            elif isinstance(q, attrs.Wavelength):
                wavelengths.append(int(q.min.value))
            elif isinstance(q, attrs.Sample):
                cadence = q.value.to(u.minute).value

        if not time_range:
            logger.error("Time range must be specified for the AIASynopticClient.")
            raise ValueError("Time range must be specified for the AIASynopticClient.")

        # Use all known wavelengths if none are specified
        if not wavelengths:
            wavelengths = self.known_wavelengths
        else:
            wavelengths = [str(wl).zfill(4) for wl in wavelengths]

        urls = self._generate_urls(time_range, wavelengths, cadence)
        return self._prepare_query_response(urls)

    def _prepare_query_response(self, urls):
        """
        Create a QueryResponse object from the generated URLs.
        """
        data = {
            "Start Time": [],
            "End Time": [],
            "Instrument": [],
            "Wavelength": [],
            "url": [],
        }
        for url in urls:
            # Extract information from the URL if possible
            data["Start Time"].append(
                None
            )  # You can parse the time from the URL if needed
            data["End Time"].append(None)
            data["Instrument"].append("AIA")
            data["Wavelength"].append(None)  # Extract from URL or elsewhere if possible
            data["url"].append(url)

        return QueryResponse(data, client=self)

    def _generate_urls(self, time_range, wavelengths, cadence=None):
        """
        Generate URLs for all requested wavelengths and time intervals.
        """
        from datetime import timedelta

        current_time = time_range.start.datetime
        end_time = time_range.end.datetime
        urls = []

        while current_time <= end_time:
            for wavelength in wavelengths:
                formatted_baseurl = current_time.strftime(self.baseurl)
                formatted_url = current_time.strftime(self.pattern).format(
                    baseurl=formatted_baseurl, wavelength=wavelength
                )
                urls.append(formatted_url)

            current_time += timedelta(minutes=cadence or 1)

        logger.debug(f"Generated {len(urls)} URLs for download.")
        return urls

    def fetch(self, query_result, *, path, downloader, **kwargs):
        """
        Fetch data using the specified download path.
        """
        download_path = path or "temp"  # Default to 'temp' if no path is provided
        os.makedirs(download_path, exist_ok=True)

        for record in query_result:
            downloader.enqueue_file(record["url"], path=download_path)

        return downloader.download()


# Automatically register the client with Fido when the module is imported
if AIASynopticClient not in Fido.registry:
    Fido.registry[AIASynopticClient] = AIASynopticClient._can_handle_query
    logger.info("Synoptic Fido Client Loaded!")
