"""Venue locations and approximate dimensions for descriptive research exports.

Carrara location: https://sportvenues.q2032.au/venues/people-first-stadium
Summit location: https://www.spacetoco.com/space/ssrp-changerooms-cd
The Summit listing belongs to Mount Barker District Council.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class VenueMeta:
    canonical_name: str
    latitude: float
    longitude: float
    timezone: str
    length_m: float | None
    width_m: float | None
    capacity: float | None


# Dimensions/capacity are approximate and can be overridden downstream if desired.
VENUE_METADATA: dict[str, VenueMeta] = {
    "mcg": VenueMeta(
        "M.C.G.", -37.8199, 144.9834, "Australia/Melbourne", 160.0, 141.0, 100024.0
    ),
    "docklands": VenueMeta(
        "Docklands", -37.8164, 144.9475, "Australia/Melbourne", 159.5, 128.8, 53000.0
    ),
    "adelaideoval": VenueMeta(
        "Adelaide Oval", -34.9154, 138.5967, "Australia/Adelaide", 167.0, 123.0, 53500.0
    ),
    "gabba": VenueMeta(
        "Gabba", -27.4850, 153.0381, "Australia/Brisbane", 156.0, 138.0, 42000.0
    ),
    "perthstadium": VenueMeta(
        "Perth Stadium", -31.9509, 115.8890, "Australia/Perth", 165.0, 130.0, 60000.0
    ),
    "carrara": VenueMeta(
        "Carrara", -28.0064, 153.3669, "Australia/Brisbane", 159.0, 134.0, 27000.0
    ),
    "scg": VenueMeta(
        "S.C.G.", -33.8917, 151.2240, "Australia/Sydney", 155.0, 136.0, 48000.0
    ),
    "subiaco": VenueMeta(
        "Subiaco", -31.9431, 115.8329, "Australia/Perth", 175.0, 122.0, 43000.0
    ),
    "kardiniapark": VenueMeta(
        "Kardinia Park",
        -38.1561,
        144.3548,
        "Australia/Melbourne",
        170.0,
        115.0,
        40000.0,
    ),
    "sydneyshowground": VenueMeta(
        "Sydney Showground",
        -33.8474,
        151.0674,
        "Australia/Sydney",
        164.0,
        128.0,
        24000.0,
    ),
    "yorkpark": VenueMeta(
        "York Park", -41.4262, 147.1344, "Australia/Hobart", 175.0, 145.0, 21000.0
    ),
    "footballpark": VenueMeta(
        "Football Park", -34.8940, 138.5200, "Australia/Adelaide", 167.0, 123.0, 51000.0
    ),
    "belleriveoval": VenueMeta(
        "Bellerive Oval", -42.8752, 147.3706, "Australia/Hobart", 175.0, 135.0, 20000.0
    ),
    "manukaoval": VenueMeta(
        "Manuka Oval", -35.3211, 149.1460, "Australia/Sydney", 170.0, 130.0, 15000.0
    ),
    "marraraoval": VenueMeta(
        "Marrara Oval", -12.4013, 130.8835, "Australia/Darwin", 177.0, 145.0, 14000.0
    ),
    "stadiumaustralia": VenueMeta(
        "Stadium Australia",
        -33.8477,
        151.0631,
        "Australia/Sydney",
        170.0,
        145.0,
        83500.0,
    ),
    "eurekastadium": VenueMeta(
        "Eureka Stadium",
        -37.5516,
        143.8513,
        "Australia/Melbourne",
        160.0,
        130.0,
        11000.0,
    ),
    "cazalysstadium": VenueMeta(
        "Cazaly's Stadium",
        -16.9200,
        145.7460,
        "Australia/Brisbane",
        164.0,
        137.0,
        13000.0,
    ),
    "traegerpark": VenueMeta(
        "Traeger Park", -23.7068, 133.8830, "Australia/Darwin", 175.0, 145.0, 10000.0
    ),
    "norwoodoval": VenueMeta(
        "Norwood Oval", -34.9205, 138.6360, "Australia/Adelaide", 167.0, 123.0, 22000.0
    ),
    "wellington": VenueMeta(
        "Wellington", -41.2725, 174.7853, "Pacific/Auckland", 165.0, 135.0, 34500.0
    ),
    "jiangwanstadium": VenueMeta(
        "Jiangwan Stadium", 31.3027, 121.5045, "Asia/Shanghai", 160.0, 130.0, 25000.0
    ),
    "summitsportspark": VenueMeta(
        "Summit Sports Park",
        -35.0754,
        138.8943,
        "Australia/Adelaide",
        160.0,
        130.0,
        5000.0,
    ),
    "barossaoval": VenueMeta(
        "Barossa Oval", -34.5270, 138.9580, "Australia/Adelaide", 165.0, 130.0, 5000.0
    ),
    "blacktown": VenueMeta(
        "Blacktown", -33.7706, 150.8573, "Australia/Sydney", 160.0, 130.0, 10000.0
    ),
    "riverwaystadium": VenueMeta(
        "Riverway Stadium",
        -19.3022,
        146.7299,
        "Australia/Brisbane",
        165.0,
        135.0,
        10000.0,
    ),
    "handsoval": VenueMeta(
        "Hands Oval", -33.3389, 115.6433, "Australia/Perth", 165.0, 130.0, 8000.0
    ),
}


def normalize_venue_name(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", raw.strip().lower())


VENUE_ALIASES = {
    "melbournecricketground": "mcg",
    "marvel": "docklands",
    "marvelstadium": "docklands",
    "etihadstadium": "docklands",
    "telstradome": "docklands",
    "colonialstadium": "docklands",
    "aamistadium": "footballpark",
    "optus": "perthstadium",
    "optusstadium": "perthstadium",
    "domainstadium": "subiaco",
    "patersonsstadium": "subiaco",
    "subiacooval": "subiaco",
    "thegabba": "gabba",
    "metricon": "carrara",
    "metriconstadium": "carrara",
    "heritagebank": "carrara",
    "heritagebankstadium": "carrara",
    "peoplefirststadium": "carrara",
    "sydneycricketground": "scg",
    "gmhba": "kardiniapark",
    "gmhbastadium": "kardiniapark",
    "simondsstadium": "kardiniapark",
    "skilledstadium": "kardiniapark",
    "shellstadium": "kardiniapark",
    "engie": "sydneyshowground",
    "engiestadium": "sydneyshowground",
    "giantsstadium": "sydneyshowground",
    "spotlessstadium": "sydneyshowground",
    "skodastadium": "sydneyshowground",
    "sydneyshowgroundstadium": "sydneyshowground",
    "utas": "yorkpark",
    "utasstadium": "yorkpark",
    "universityoftasmaniastadium": "yorkpark",
    "aurorastadium": "yorkpark",
    "blundstone": "belleriveoval",
    "blundstonearena": "belleriveoval",
    "ninjastadium": "belleriveoval",
    "manuka": "manukaoval",
    "tiostadium": "marraraoval",
    "darwin": "marraraoval",
    "tiotraegerpark": "traegerpark",
    "tiotraegerparkoval": "traegerpark",
    "anzstadium": "stadiumaustralia",
    "accor": "stadiumaustralia",
    "accorstadium": "stadiumaustralia",
    "telstrastadium": "stadiumaustralia",
    "marsstadium": "eurekastadium",
    "cazalys": "cazalysstadium",
    "westpacstadium": "wellington",
    "skystadium": "wellington",
    "summitsportandrecreationpark": "summitsportspark",
    "summitsportsandrecreationpark": "summitsportspark",
    "barossapark": "barossaoval",
}


def get_venue_meta(venue_name: str) -> VenueMeta | None:
    key = normalize_venue_name(venue_name)
    return VENUE_METADATA.get(VENUE_ALIASES.get(key, key))
