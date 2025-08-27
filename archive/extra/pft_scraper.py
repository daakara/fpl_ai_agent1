import requests
import pandas as pd
from bs4 import BeautifulSoup

def fetch_fixture_difficulty_pft():
    url = "https://www.premierfantasytools.com/fpl-fixture-difficulty/"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, verify=False, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        return pd.DataFrame()

    headers = [th.text.strip() for th in table.find_all("th")]
    rows = []
    for tr in table.find_all("tr")[1:]:
        row = [td.text.strip() for td in tr.find_all("td")]
        if row:
            rows.append(row)

    max_len = max(len(row) for row in rows)
    # Pad rows if inconsistent length
    rows = [row + [""] * (max_len - len(row)) for row in rows]
    # Pad headers if short
    if len(headers) < max_len:
        headers += [f"Extra_{i}" for i in range(max_len - len(headers))]

    df = pd.DataFrame(rows, columns=headers)
    return df

if __name__ == "__main__":
    df = fetch_fixture_difficulty_pft()
    print(df.head())
