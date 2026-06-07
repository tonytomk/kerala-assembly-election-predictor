import json
import re
import requests
from bs4 import BeautifulSoup

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

# myneta
r = session.get("https://myneta.info/Kerala2026/", timeout=30)
print("myneta status", r.status_code, len(r.text))
soup = BeautifulSoup(r.text, "html.parser")
links = [a.get("href") for a in soup.find_all("a", href=True) if "Kerala2026" in a.get("href", "")]
print("links", links[:10], "count", len(links))
tables = soup.find_all("table")
print("tables", len(tables))
if tables:
    for tr in tables[0].find_all("tr")[:5]:
        print([c.get_text(strip=True) for c in tr.find_all(["td", "th"])])

# try first constituency page if exists
if links:
    rr = session.get(links[0] if links[0].startswith("http") else "https://myneta.info/" + links[0].lstrip("/"), timeout=30)
    print("detail", rr.status_code, len(rr.text))
    ds = BeautifulSoup(rr.text, "html.parser")
    for tr in ds.find_all("tr")[:8]:
        cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
        if cells:
            print(cells)

# BOOM json - inspect one with many candidates maybe
build_id = "Mna-2EUOjRjVbd_57zhlj"
data = session.get(
    f"https://elections.boomlive.in/_next/data/{build_id}/elections/kerala.json",
    timeout=60,
).json()
consts = next(s for s in data["pageProps"]["initialSnapshot"]["states"] if s["code"] == "KL")["constituencies"]
max_results = max(len(c["results"]) for c in consts)
print("max boom results entries", max_results)
for c in consts:
    if len(c["results"]) > 2:
        print("multi", c["constituency_number"], c["name"], c["results"])

# search indian express for embedded constituency json
r = session.get(
    "https://indianexpress.com/article/india/kerala-election-results-2026-winners-constituency-wise-full-list-party-wise-seat-tally-10665881/",
    timeout=30,
)
for pat in [r"constituencyResults\s*=\s*(\[.*?\])", r"window\.__DATA__\s*=\s*(\{.*?\});"]:
    m = re.search(pat, r.text, re.S)
    print("ie pattern", pat[:30], bool(m))
