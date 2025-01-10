import logging
from sunpy.net import attrs, Fido
from sunpy.net.dataretriever import GenericClient
from sunpy.net.dataretriever.client import QueryResponse
from sunpy.net import attr
import astropy.units as u
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
    known_wavelengths = [
        "0094",
        "0131",
        "0171",
        "0193",
        "0211",
        "0304",
        "1600",
        "1700",
    ]

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

        has_synoptic_data_attr = any(isinstance(x, AIASynopticData) for x in query)
        return has_synoptic_data_attr

    def search(self, *query):
        time_range = None
        wavelengths = []
        cadence_seconds = None

        for q in query:
            if isinstance(q, attrs.Time):
                time_range = q
            elif isinstance(q, attrs.Wavelength):
                # Access q.min, which is an astropy.units.Quantity
                wavelength_value = q.min.to(u.angstrom).value
                wavelengths.append(int(wavelength_value))
            elif isinstance(q, attrs.Sample):
                # q.value is in seconds (float)
                cadence_seconds = q.value
            # Handle other attributes if necessary

        if not time_range:
            logger.error("Time range must be specified for the AIASynopticClient.")
            raise ValueError("Time range must be specified for the AIASynopticClient.")

        if not wavelengths:
            wavelengths = self.known_wavelengths
        else:
            wavelengths = [str(wl).zfill(4) for wl in wavelengths]

        urls = self._generate_urls(time_range, wavelengths, cadence_seconds)
        return self._prepare_query_response(urls)

    def _prepare_query_response(self, urls):
        from sunpy.net.dataretriever.client import QueryResponseTable
        from datetime import datetime

        data = {
            "Start Time": [],
            "End Time": [],
            "Instrument": [],
            "Wavelength": [],
            "url": [],
        }
        for url in urls:
            filename = os.path.basename(url)
            name_part = filename[3:-5]
            parts = name_part.split("_")
            if len(parts) == 3:
                date_str = parts[0]
                time_str = parts[1]
                wavelength_str = parts[2]
                datetime_str = date_str + time_str
                try:
                    start_time = datetime.strptime(datetime_str, "%Y%m%d%H%M")
                except ValueError:
                    start_time = None
                data["Start Time"].append(start_time)
                data["Wavelength"].append(int(wavelength_str))
            else:
                data["Start Time"].append(None)
                data["Wavelength"].append(None)
            data["End Time"].append(None)
            data["Instrument"].append("AIA")
            data["url"].append(url)

        table = QueryResponseTable(data, client=self)
        return QueryResponse(table)

    def _generate_urls(self, time_range, wavelengths, cadence_seconds=None):
        from datetime import timedelta

        current_time = time_range.start.datetime
        end_time = time_range.end.datetime
        urls = []

        if cadence_seconds is not None:
            cadence_timedelta = timedelta(seconds=cadence_seconds)
        else:
            cadence_timedelta = timedelta(minutes=1)

        while current_time <= end_time:
            for wavelength in wavelengths:
                formatted_baseurl = current_time.strftime(self.baseurl)
                formatted_url = current_time.strftime(self.pattern).format(
                    baseurl=formatted_baseurl, wavelength=str(wavelength).zfill(4)
                )
                urls.append(formatted_url)
            current_time += cadence_timedelta

        logger.debug(f"Generated {len(urls)} URLs for download.")
        return urls

    def fetch(self, query_result, *, path, downloader, **kwargs):
        download_path = path or "temp"
        os.makedirs(download_path, exist_ok=True)

        max_conn = kwargs.get("max_conn", 5)
        downloader.max_conn = max_conn

        for record in query_result:
            downloader.enqueue_file(record["url"], path=download_path)

        return downloader.download()


if AIASynopticClient not in Fido.registry:
    Fido.registry[AIASynopticClient] = AIASynopticClient._can_handle_query
    logger.info("Synoptic Fido Client Loaded!")
