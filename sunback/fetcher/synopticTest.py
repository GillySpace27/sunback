from astropy import units as u
from sunpy.net import Fido, attrs as a
from AIASynopticClient import AIASynopticData

search_result = Fido.search(
    a.Time("2015-06-06", "2015-06-07"),
    AIASynopticData(),
    a.Sample(1 * u.hour),
    a.Wavelength(171 * u.angstrom),
)

result = Fido.fetch(search_result, path=r"sunback_data/bb2")
