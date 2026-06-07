import time
import requests
from bs4 import BeautifulSoup

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://results.eci.gov.in/ResultAcGenMay2026/",
    "Connection": "keep-alive",
})

for ac in [1, 2, 35, 80, 140]:
    url = f"https://results.eci.gov.in/ResultAcGenMay2026/candidateswise-S11{ac}.htm"
    r = session.get(url, timeout=30)
    print(ac, r.status_code, len(r.text), "denied" if "Access Denied" in r.text else "ok")
    if "Access Denied" not in r.text:
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text("\n", strip=True)
        lines = [ln for ln in text.splitlines() if ln.strip()][:25]
        for ln in lines:
            print(" ", ln)
    time.sleep(0.5)
