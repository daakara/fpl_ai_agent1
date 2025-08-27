import requests
import logging
import re
from typing import List, Dict

def fetch_reddit_hot_posts(subreddit: str, limit: int = 5) -> List[Dict]:
    url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
    headers = {'User-agent': "FPLAI-Bot/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10, verify=False)  # SSL disabled temporarily, remove verify=False after fix
        r.raise_for_status()
        return [{
            "title": p["data"]["title"],
            "score": p["data"]["score"],
            "url": p["data"]["url"],
        } for p in r.json()["data"]["children"]]
    except Exception as e:
        logging.warning(f"Reddit fetch failed: {e}")
        return []

def fetch_ffs_poll_results(url: str = "https://www.fantasyfootballscout.co.uk/poll-results/") -> List[Dict]:
    try:
        html = requests.get(url, timeout=10, verify=False).text  # SSL disabled temporarily, remove verify=False after fix
        matches = re.findall(r'<li class="poll-result">(.+?)<span class="poll-votes">\\((\d+) votes\\)', html, re.DOTALL)
        results = [{"option": name.strip(), "votes": int(votes)} for name, votes in matches]
        results.sort(key=lambda x: -x["votes"])
        return results
    except Exception as e:
        logging.warning(f"FFS poll scrape failed: {e}")
        return []
