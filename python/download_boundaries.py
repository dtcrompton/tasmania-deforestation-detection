"""
Download PTPZ and Tasmanian Reserve Estate boundaries from the LIST ArcGIS REST API.
Queries by category to stay within server record limits.
Output: data/permits/ptpz.geojson, data/permits/reserve_estate.geojson
"""

import json
import urllib.request
import urllib.parse
from pathlib import Path

OUTPUT_DIR = Path("data/permits")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESERVE_ESTATE_URL = (
    "https://services.thelist.tas.gov.au/arcgis/rest/services/"
    "Public/CadastreAndAdministrative/MapServer/29"
)

PUBLIC_LAND_URL = (
    "https://services.thelist.tas.gov.au/arcgis/rest/services/"
    "Public/OpenDataWFS/MapServer/38"
)

RESERVE_ESTATE_CATEGORIES = [
    "National Park",
    "Nature Reserve",
    "State Reserve",
    "Conservation Area",
    "Regional Reserve",
    "Game Reserve",
    "Nature Recreation Area",
    "Historic Site",
    "Informal Reserve on Permanent Timber Production Zone Land or STT managed land",
]

PUBLIC_LAND_CATEGORIES = [
    "Permanent Timber Production Zone Land",
    "National Park",
    "Nature Reserve",
    "State Reserve",
    "Conservation Area",
    "Regional Reserve",
]


def fetch_category(base_url, category_field, category, page_size=500):
    """Fetch all features for a single category, paginating as needed."""
    all_features = []
    offset = 0

    while True:
        where = f"{category_field}='{category}'"
        params = urllib.parse.urlencode({
            "where": where,
            "outFields": "*",
            "resultOffset": offset,
            "resultRecordCount": page_size,
            "f": "geojson"
        })
        url = f"{base_url}/query?{params}"
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = json.loads(response.read().decode())
        except Exception as e:
            print(f"    Error at offset {offset}: {e}")
            break

        if "error" in data:
            print(f"    API error: {data['error']}")
            break

        features = data.get("features", [])
        all_features.extend(features)

        if len(features) < page_size:
            break
        offset += page_size

    return all_features


def download_filtered(filename, base_url, category_field, categories):
    print(f"\nDownloading {filename}...")
    all_features = []

    for category in categories:
        features = fetch_category(base_url, category_field, category)
        print(f"  {len(features):>6}  {category}")
        all_features.extend(features)

    geojson = {"type": "FeatureCollection", "features": all_features}
    out_path = OUTPUT_DIR / filename
    with open(out_path, "w") as f:
        json.dump(geojson, f)
    print(f"  Total: {len(all_features)} features saved to {out_path}")


if __name__ == "__main__":
    download_filtered(
        "reserve_estate.geojson",
        RESERVE_ESTATE_URL,
        "RES_CLASS",
        RESERVE_ESTATE_CATEGORIES,
    )
    download_filtered(
        "ptpz.geojson",
        PUBLIC_LAND_URL,
        "CATEGORY",
        PUBLIC_LAND_CATEGORIES,
    )
    print("\nDone.")