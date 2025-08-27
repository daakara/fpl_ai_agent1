import requests
from bs4 import BeautifulSoup
import re
import random
import time
import streamlit as st
import traceback
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.selector import Selector
from fake_useragent import UserAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15)",
]

# Use fake-useragent for more diverse user agents
ua = UserAgent()

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

class FFScoutLineupsSpider(scrapy.Spider):
    name = "ffscout_lineups"
    start_urls = ["https://www.fantasyfootballfix.com/reveal/"]

    def parse(self, response):
        # Find formations containers by class including "formation"
        formations = response.css("div[class*=formation]")
        if not formations:
            self.logger.warning("No formation containers found. The website structure may have changed.")
            return

        for formation in formations:
            players = formation.css("span.fplPlayerName::text").getall()
            if players:
                yield {'lineup': [p.strip() for p in players]}

@rate_limit(10)
def fetch_ffscout_lineups():
    """
    Fetches real-time Scout Picks lineups from Fantasy Football Scout Teams Line-Ups page using Scrapy.
    Returns a list of lineups.
    Each lineup is a list of player names (strings).
    """
    try:
        process = CrawlerProcess({
            'USER_AGENT': ua.random,  # Use a random user agent
            'LOG_LEVEL': 'INFO',
        })

        # The argument to process.crawl must be the spider class, not an instance
        process.crawl(FFScoutLineupsSpider)
        lineups = []
        
        # Start the crawler and collect results
        process.start() # remove the False argument
        for item in process.spider.crawler.engine.slot.scheduler.queue.queue:
            lineups.append(item.item['lineup'])

        return lineups
    except Exception as e:
        st.error(f"Error in fetch_ffscout_lineups: {e}")
        traceback.print_exc()
        return []

@st.cache_data(ttl=3600)
def get_scout_picks():
    """Cached function to retrieve scout picks."""
    return fetch_ffscout_lineups()

class ExpertPicksSpider(scrapy.Spider):
    name = "expert_picks"
    start_urls = ["https://www.fantasyfootballfix.com/reveal/"]

    def parse(self, response):
        table = response.css("table.benchmarks")
        if table:
            rows = table.css("tr")[1:]  # Skip headers
            for row in rows:
                cols = row.css("td")
                if len(cols) > 1:
                    player_name = cols[0].css("::text").get(default='').strip()
                    yield {'player_name': player_name}

def fetch_expert_picks():
    """
    Scrapes the Fantasy Football Scout Benchmarks page to extract expert picks or top owned players using Scrapy.
    Returns a list of player names mostly selected by top managers or expert lists.
    """
    try:
        process = CrawlerProcess({
            'USER_AGENT': ua.random,
            'LOG_LEVEL': 'INFO',
        })

        process.crawl(ExpertPicksSpider)
        picks = []

        process.start()
        for item in process.spider.crawler.engine.slot.scheduler.queue.queue:
            picks.append(item.item['player_name'])

        return picks
    except scrapy.exceptions.DropItem as e:
        logging.warning(f"Scrapy DropItem exception: {e}")
        return []
    except Exception as e:
        logging.exception(f"Error in fetch_expert_picks: {e}")
        return []

class InjuryStatusSpider(scrapy.Spider):
    name = "injury_status"
    start_urls = ["https://www.fantasyfootballfix.com/injuries/"]

    def parse(self, response):
        tables = response.css("table")
        for table in tables:
            # Refine table selection based on specific attributes or structure
            if "injury" in response.url and "status" in response.text.lower():  # Example: Check for keywords in URL and content
                headers = [th.css("::text").get(default='').strip().lower() for th in table.css("th")]
                if "player" in headers and "status" in headers:
                    player_idx = headers.index("player")
                    status_idx = headers.index("status")
                    rows = table.css("tr")[1:]
                    for row in rows:
                        cols = row.css("td")
                        if len(cols) > max(player_idx, status_idx):
                            player_name = cols[player_idx].css("::text").get(default='').strip()
                            status = cols[status_idx].css("::text").get(default='').strip()

                            # Robust data cleaning
                            player_name = re.sub(r'\s+', ' ', player_name).strip()  # Remove extra spaces
                            status = re.sub(r'\s+', ' ', status).strip()

                            yield {'player_name': player_name, 'status': status}

def get_injury_status():
    """
    Scrapes Fantasy Football Scout Injuries page and returns a dict mapping player_name -> injury status string using Scrapy.
    """
    injury_map = {}
    try:
        process = CrawlerProcess({
            'USER_AGENT': ua.random,
            'LOG_LEVEL': 'INFO',
        })

        process.crawl(InjuryStatusSpider)

        process.start()

        for item in process.spider.crawler.engine.slot.scheduler.queue.queue:
            player_name = item.item['player_name']
            status = item.item['status']
            injury_map[player_name] = status

        return injury_map
    except Exception as e:
        print(f"Error fetching injury status: {type(e).__name__}, {e}")
        traceback.print_exc()
        return injury_map

# Example usage (you might need to adapt this to your Streamlit app)
if __name__ == "__main__":
    # Example usage
    lineups = fetch_ffscout_lineups()
    print("Scout Picks Lineups:", lineups)

    expert_picks = fetch_expert_picks()
    print("Expert Picks:", expert_picks)

    injury_status = get_injury_status()
    print("Injury Status:", injury_status)