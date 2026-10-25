"""Download the Severson repo notebooks we need into repo/ (git is not installed)."""
import urllib.request
from pathlib import Path

BASE = ("https://raw.githubusercontent.com/rdbraatz/"
        "data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/master/")
FILES = ["BuildPkl_Batch1.ipynb", "BuildPkl_Batch2.ipynb", "BuildPkl_Batch3.ipynb",
         "Load%20Data.ipynb", "requirements.txt"]

out = Path(__file__).resolve().parents[1] / "repo"
out.mkdir(exist_ok=True)
for f in FILES:
    try:
        d = urllib.request.urlopen(BASE + f, timeout=60).read()
        (out / f.replace("%20", " ")).write_bytes(d)
        print("ok", f, len(d))
    except Exception as e:
        print("FAIL", f, e)
