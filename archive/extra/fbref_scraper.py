import requests
import pandas as pd
import random
from bs4 import BeautifulSoup, Comment
import re
import time

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15)",
]

def rate_limit(calls_per_minute):
    interval = 60.0 / calls_per_minute
    def decorator(func):
        last_called = [0]
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < interval:
                time.sleep(interval - elapsed)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

@rate_limit(10)
def fetch_team_stats(team_url="https://fbref.com/en/squads/18bb7c10/Arsenal-Stats"):
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    for attempt in range(3):
        try:
            r = requests.get(team_url, headers=headers, timeout=7, verify=False)  # temporarily disabling SSL verify for debugging, remove verify=False later
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # Try CSV Download first
            csv_link = soup.find("a", href=re.compile(".+csv$"))
            if csv_link:
                csv_url = "https://fbref.com" + csv_link["href"]
                df = pd.read_csv(csv_url)
                return df

            # Else parse HTML table inside comments
            comments = soup.find_all(string=lambda text: isinstance(text, Comment))
            for comment in comments:
                if 'table' in comment:
                    comment_soup = BeautifulSoup(comment, "html.parser")
                    table = comment_soup.find("table", id="stats_standard")
                    if table:
                        return pd.read_html(str(table))[0]
        except Exception as e:
            print(f"Error fetching FBref data (attempt {attempt+1}): {e}")
            time.sleep(1 + attempt)
    return pd.DataFrame()
