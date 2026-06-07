import json
import requests
from bs4 import BeautifulSoup

build_id = "Mna-2EUOjRjVbd_57zhlj"
url = f"https://elections.boomlive.in/_next/data/{build_id}/elections/kerala.json"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
data = r.json()
states = data["pageProps"]["initialSnapshot"]["states"]
kerala = next(s for s in states if s["code"] == "KL")
consts = kerala["constituencies"]
print("count", len(consts))
for idx in [0, 34, 79, 134]:
    print("\n=== AC", consts[idx]["constituency_number"], consts[idx]["name"], "===")
    print(json.dumps(consts[idx], indent=2))

# check if any field has 3 candidates
sample = consts[0]
for k, v in sample.items():
    if isinstance(v, list):
        print("list field", k, len(v))
