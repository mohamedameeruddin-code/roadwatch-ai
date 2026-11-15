"""
Inspect pothole counts across countries in archive (2).zip
"""

from collections import Counter
from pathlib import Path
import zipfile

zip_path = Path(r"C:\Users\Asus\Downloads\archive (2).zip")
with zipfile.ZipFile(zip_path, "r") as z:
    pothole_by_country = Counter()
    total_by_country = Counter()
    for n in z.namelist():
        if n.endswith(".txt"):
            parts = n.split("/")
            if len(parts) >= 4:
                stem = parts[-1]
                country = stem.split("_")[0]
                total_by_country[country] += 1
                try:
                    content = z.read(n).decode("utf-8").strip()
                    if any(l.split()[0] == "4" for l in content.splitlines() if l.split()):
                        pothole_by_country[country] += 1
                except Exception:
                    pass

    print("Total images by country:", total_by_country)
    print("Pothole images by country:", pothole_by_country)
