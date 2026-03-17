"""
Quick inspection of downloaded boundary files.
Reports feature counts and unique category values.
"""

import json
from pathlib import Path
from collections import Counter

files = {
    "reserve_estate.geojson": "RES_CLASS",
    "public_land_classification.geojson": "CATEGORY",
}

for filename, category_field in files.items():
    path = Path("data/permits") / filename
    with open(path) as f:
        data = json.load(f)

    features = data["features"]
    print(f"\n{filename}: {len(features)} features")
    print(f"  Fields: {list(features[0]['properties'].keys())}")

    categories = Counter(
        f["properties"].get(category_field, "NO FIELD")
        for f in features
    )
    for cat, count in categories.most_common():
        print(f"  {count:>6}  {cat}")
"""
for filename, category_field in files.items():
    path = Path("data/permits") / filename
    with open(path) as f:
        data = json.load(f)

    features = data["features"]
    print(f"\n{filename}: {len(features)} features")

    categories = Counter(
        f["properties"].get(category_field, "NO FIELD")
        for f in features
    )
    for cat, count in categories.most_common():
        print(f"  {count:>6}  {cat}")
        """