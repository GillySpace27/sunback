from fetch.Fetch import Fetch


class NullFetch(Fetch):
    
    def download_fits_files(self, url):
        return []
