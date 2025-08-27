import streamlit as st
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
from thefuzz import process
import asyncio
import diskcache
import logging
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import ssl
from typing import List, Dict, Tuple, Any, Optional

from fpl_official import get_players_data, get_teams_data, get_fixtures_data_async
from fpl_myteam import load_my_fpl_team

cache = diskcache.Cache("fpl_cache")

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
]

class DataFetcher:
    """
    A class to encapsulate data fetching logic with caching and error handling.
    """
    def __init__(self, cache: diskcache.Cache):
        self.cache = cache

    @cache.memoize(expire=3600)
    def fetch_fpl_focal_xg_xa_improved(self) -> pd.DataFrame:
        """
        Fetches xG/xA data from FPL.page using Selenium with improved error handling.
        """
        url = "https://fpl.page/expected"
        
        chrome_options = uc.ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
        
        driver = None
        try:
            driver = uc.Chrome(options=chrome_options)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            driver.get(url)
            
            wait = WebDriverWait(driver, 45)
            
            table_selectors = [
                "//div[contains(@class, 'overflow-auto') and contains(@class, 'overflow-x-hidden') and contains(@class, 'rounded-[20px]')]//table",
                "//table",
                "//div[contains(@class, 'overflow-auto')]//table"
            ]
            
            table_found = False
            for selector in table_selectors:
                try:
                    wait.until(EC.visibility_of_element_located((By.XPATH, selector)))
                    table_found = True
                    break
                except Exception as e:
                    logging.exception(f"Error waiting for table with selector {selector}: {e}") # Log the exception
                    continue
            
            if not table_found:
                st.warning("Could not locate xG/xA table on the page")
                return pd.DataFrame(columns=["web_name", "xG_next_5", "xA_next_5"])
            
            html = driver.page_source
            
            try:
                tables = pd.read_html(html)
            except ValueError as e:
                st.warning(f"No tables found in HTML: {e}")
                logging.exception(f"Error reading HTML tables: {e}") # Log the exception
                return pd.DataFrame(columns=["web_name", "xG_next_5", "xA_next_5"])
            
            if not tables:
                st.warning(f"No tables found on {url}")
                return pd.DataFrame(columns=["web_name", "xG_next_5", "xA_next_5"])
            
            xg_xa_df = None
            for i, table in enumerate(tables):
                if any(col for col in table.columns if 'name' in str(col).lower()):
                    name_cols = [col for col in table.columns if 'name' in str(col).lower()]
                    xg_cols = [col for col in table.columns if 'xg' in str(col).lower() or 'expected' in str(col).lower()]
                    xa_cols = [col for col in table.columns if 'xa' in str(col).lower() or 'assist' in str(col).lower()]

                    if name_cols and (xg_cols or xa_cols):
                        xg_xa_df = table.copy()

                        rename_map = {}
                        if name_cols:
                            if 'Name' in name_cols:
                               rename_map['Name'] = "web_name"
                            else:
                                rename_map[name_cols[0]] = "web_name"
                        if xg_cols:
                            rename_map[xg_cols[0]] = "xG_next_5"

                        if xa_cols:
                            rename_map[xa_cols[0]] = "xA_next_5"

                        
                        xg_xa_df = xg_xa_df.rename(columns=rename_map)
                        break
            if xg_xa_df is None:
                st.warning("Could not find appropriate xG/xA table structure")
                return pd.DataFrame(columns=["web_name", "xG_next_5", "xA_next_5"])
            
            required_cols = ["web_name", "xG_next_5", "xA_next_5"]
            for col in required_cols:
                if col not in xg_xa_df.columns:
                    if col == "xG_next_5":
                        xg_xa_df[col] = 0
                    elif col == "xA_next_5":
                        xg_xa_df[col] = 0
                    else:
                        st.warning(f"Missing required column: {col}")
                        return pd.DataFrame(columns=["web_name", "xG_next_5", "xA_next_5"])
            
            xg_xa_df["xG_next_5"] = pd.to_numeric(xg_xa_df["xG_next_5"], errors="coerce").fillna(0)
            xg_xa_df["xA_next_5"] = pd.to_numeric(xg_xa_df["xA_next_5"], errors="coerce").fillna(0)
            
            xg_xa_df["web_name"] = xg_xa_df["web_name"].astype(str).str.strip()
            
            xg_xa_df["web_name"] = xg_xa_df["web_name"].str.replace(r'[A-Z]{3,}(?:GK|DEF|MID|FWD|GKP|FW)$', '', regex=True).str.strip()
            
            xg_xa_df = xg_xa_df.drop_duplicates(subset=["web_name"])
            
            result_df = xg_xa_df[["web_name", "xG_next_5", "xA_next_5"]].copy()
            
            return result_df
            
        except Exception as e:
            st.warning(f"Failed to fetch xG/xA data with Selenium: {e}")
            logging.exception(f"Error in fetch_fpl_focal_xg_xa_improved: {e}") # Log the exception
            return pd.DataFrame(columns=["web_name", "xG_next_5", "xA_next_5"])
        finally:
            if driver:
                try:
                    driver.quit()
                except Exception as e:
                    logging.exception(f"Error quitting driver: {e}") # Log the exception

    @cache.memoize(expire=3600)
    def estimate_xg_xa_from_fpl_stats(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """
        Estimates xG and xA values based on FPL statistics.
        """
        df = df_players.copy()

        df["creativity"] = pd.to_numeric(df["creativity"], errors="coerce").fillna(0)

        df["estimated_xG_next_5"] = (
            df["goals_scored"].fillna(0) * 0.3 + 
            df["form"].apply(lambda x: float(x) if pd.notna(x) else 0) * 0.4 + 
            (df["total_points"].fillna(0) / 38) * 0.3
        ).clip(0, 10)

        df["estimated_xA_next_5"] = (
            df["assists"].fillna(0) * 0.4 + 
            df["creativity"] / 100 * 0.3 + 
            df["form"].apply(lambda x: float(x) if pd.notna(x) else 0) * 0.3
        ).clip(0, 8)

        return df[["web_name", "estimated_xG_next_5", "estimated_xA_next_5"]].rename(
            columns={"estimated_xG_next_5": "xG_next_5", "estimated_xA_next_5": "xA_next_5"}
        )

    def setup_selenium_with_ssl_bypass(self):
        """Setup Selenium with SSL certificate bypass"""
        chrome_options = Options()
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--ssl-protocol=any')  # Optional: Try this if the above doesn't work
        driver = webdriver.Chrome(options=chrome_options)
        return driver

    def get_xg_xa_data_with_fallback(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """
        Fetches xG/xA data from FPL.page or estimates it from FPL stats as a fallback.
        """
        try:
            with st.spinner("🔄 Fetching xG/xA data from FPL.page..."):
                xg_xa_df = self.fetch_fpl_focal_xg_xa_improved()
                
                if not xg_xa_df.empty and len(xg_xa_df) > 10:
                    st.success(f"✅ Successfully loaded xG/xA data for {len(xg_xa_df)} players from FPL.page")
                    
                    # Ensure xG and xA are numeric
                    xg_xa_df["xG_next_5"] = pd.to_numeric(xg_xa_df["xG_next_5"], errors="coerce").fillna(0)
                    xg_xa_df["xA_next_5"] = pd.to_numeric(xg_xa_df["xA_next_5"], errors="coerce").fillna(0)
                    
                    return xg_xa_df
                else:
                    st.warning("⚠️ FPL.page data seems incomplete, trying fallback...")
                    
        except Exception as e:
            st.warning(f"⚠️ FPL.page scraping failed: {e}")
        
        try:
            st.info("🔄 Trying alternative xG/xA source...")
            pass
        except:
            pass
        
        with st.spinner("🔄 Using estimated xG/xA values based on FPL stats..."):
            estimated_df = self.estimate_xg_xa_from_fpl_stats(df_players)
            
            # Ensure xG and xA are numeric
            estimated_df["xG_next_5"] = pd.to_numeric(estimated_df["xG_next_5"], errors="coerce").fillna(0)
            estimated_df["xA_next_5"] = pd.to_numeric(estimated_df["xA_next_5"], errors="coerce").fillna(0)
            
            st.info(f"📊 Generated estimated xG/xA for {len(estimated_df)} players")
            return estimated_df

class FPLDataProcessor:
    """
    A class to handle data processing and merging.
    """
    def __init__(self):
        pass

    def fuzzy_merge(self, df_left: pd.DataFrame, df_right: pd.DataFrame, left_on: str, right_on: str, threshold: int = 90, limit: int = 1) -> pd.DataFrame:
        """
        Merges two DataFrames using fuzzy matching on specified columns.
        """
        s = df_right[right_on].apply(lambda x: process.extractOne(x, df_left[left_on], score_cutoff=threshold))
        
        df_right_copy = df_right.copy()
        df_right_copy['fuzzy_merge_key'] = [i[0] if i else None for i in s]

        df_right_to_merge = df_right_copy.drop(columns=[right_on])
        
        merged = pd.merge(df_left, df_right_to_merge, left_on=left_on, right_on='fuzzy_merge_key', how='left')
        
        return merged.drop(columns=['fuzzy_merge_key'])

class SeleniumSetup:
    """
    A class to manage Selenium setup and teardown.
    """
    def setup_selenium_with_ssl_bypass(self):
        """Setup Selenium with SSL certificate bypass"""
        chrome_options = Options()
        chrome_options.add_argument('--ignore-certificate-errors')
        chrome_options.add_argument('--ssl-protocol=any')  # Optional: Try this if the above doesn't work
        driver = webdriver.Chrome(options=chrome_options)
        return driver

class FPLDataLoader:
    """
    A class to load FPL data with error handling and data processing.
    """
    def __init__(self, data_fetcher: DataFetcher, data_processor: FPLDataProcessor):
        self.data_fetcher = data_fetcher
        self.data_processor = data_processor

    async def load_fpl_data(self) -> Tuple[List[Dict], List[Dict], Dict[str, Any], List[Dict], List[Dict], List[Dict], Dict[str, Any], Dict[str, Any]]:
        """
        Loads FPL data from different sources with individual error handling.
        """
        players = None
        teams = None
        my_team_data_tuple = None
        
        try:
            players = await get_players_data()
        except Exception as e:
            logging.exception(f"Error loading players data: {e}")
        
        try:
            teams = await get_teams_data()
        except Exception as e:
            logging.exception(f"Error loading teams data: {e}")
        
        try:
            my_team_data_tuple = await load_my_fpl_team(1437677)
        except Exception as e:
            logging.exception(f"Error loading my team data: {e}")
        
        info, picks, transfers, chips, predictions, live_points = my_team_data_tuple if my_team_data_tuple else ({}, [], [], [], [], [])
        return players, teams, info, picks, transfers, chips, predictions, live_points

    async def load_player_data_with_xg_xa(self) -> Tuple[pd.DataFrame, List[Dict], Dict[str, Any], List[Dict], List[Dict], List[Dict], Dict[str, Any], Dict[str, Any]]:
        """
        Loads player data, team data, and xG/xA data, merging them into a single DataFrame.
        """
        try:
            players, teams, info, picks, transfers, chips, predictions, live_points = await self.load_fpl_data()
            
            if players is None or teams is None or info is None or picks is None or transfers is None or chips is None or predictions is None or live_points is None:
                logging.warning("One or more data sources failed to load. Returning empty DataFrame.")
                df_players = pd.DataFrame(columns=['id'])
                return df_players, [], {}, [], [], [], {}, {}
            
            df_players = pd.DataFrame(players)
            df_teams = pd.DataFrame(teams)
            team_id_to_name = df_teams.set_index("id")["name"].to_dict()
            df_players["team_name"] = df_players["team"].map(team_id_to_name)
            
            def safe_float(val):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0

            df_players["expected_points_next_5"] = df_players["form"].apply(safe_float) * 5

            # Ensure required columns exist with default values
            if "minutes_per_game" not in df_players.columns:
                df_players["minutes_per_game"] = 0.0
            if "consistency" not in df_players.columns:
                df_players["consistency"] = 0.0
            if "clean_sheets_rate" not in df_players.columns:
                df_players["clean_sheets_rate"] = 0.0
            
            xg_xa_df = self.data_fetcher.get_xg_xa_data_with_fallback(df_players)
            
            if not xg_xa_df.empty:
                # Ensure web_name is string type in both DataFrames
                df_players["web_name"] = df_players["web_name"].astype(str)
                xg_xa_df["web_name"] = xg_xa_df["web_name"].astype(str)
                
                # Add logging statements
                logging.info(f"df_players before merge: {df_players.dtypes}")
                logging.info(f"xg_xa_df before merge: {xg_xa_df.dtypes}")
                
                df_players = self.data_processor.fuzzy_merge(
                    df_players,
                    xg_xa_df, 
                    left_on="web_name", 
                    right_on="web_name",
                    threshold=85)
                
                # Add logging statements
                logging.info(f"df_players after merge: {df_players.dtypes}")
                
                df_players["xG_next_5"] = df_players["xG_next_5"].fillna(0)
                df_players["xA_next_5"] = df_players["xA_next_5"].fillna(0)
                
                non_zero_xg = (df_players["xG_next_5"] > 0).sum()
                non_zero_xa = (df_players["xA_next_5"] > 0).sum()
                
                st.info(f"📈 xG/xA Data: {non_zero_xg} players with xG > 0, {non_zero_xa} players with xA > 0")
            else:
                st.warning("⚠️ Could not fetch xG/xA data, using zeros")
                df_players["xG_next_5"] = 0
                df_players["xA_next_5"] = 0
            
            return df_players, teams, info, picks, transfers, chips, predictions, live_points
        except Exception as e:
            st.error(f"Error in load_player_data_with_xg_xa: {e}")
            logging.exception(f"Error in load_player_data_with_xg_xa: {e}")
            
            # Create an empty DataFrame with the 'id' column
            df_players = pd.DataFrame(columns=['id'])
            return df_players, [], {}, [], [], [], {}, {}

def setup_undetected_chromedriver():
    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = uc.Chrome(options=options)
    return driver

# Instantiate the classes
data_fetcher = DataFetcher(cache)
data_processor = FPLDataProcessor()
selenium_setup = SeleniumSetup()
fpl_data_loader = FPLDataLoader(data_fetcher, data_processor)

# Example usage (you might need to adjust this based on your app's structure)
# async def main():
#     df_players, teams, info, picks, transfers, chips, predictions, live_points = await fpl_data_loader.load_player_data_with_xg_xa()
#     print(df_players.head())

# if __name__ == "__main__":
#     asyncio.run(main())
