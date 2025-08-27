import streamlit as st
import httpx
import json
from urllib.parse import urljoin
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import certifi

class FixtureDataSource(ABC):
    """Abstract base class for fixture data sources"""
    
    @abstractmethod
    def fetch_fixtures(self) -> Optional[List[Dict[str, Any]]]:
        pass

class FPLAPIFixtureSource(FixtureDataSource):
    """Concrete implementation for FPL API fixture source"""
    
    BASE_URL = "https://fantasy.premierleague.com/api/"
    FIXTURES_ENDPOINT = "fixtures/"
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_fixtures(_self) -> Optional[List[Dict[str, Any]]]:
        """Fetches fixture data from the FPL API."""
        url = urljoin(_self.BASE_URL, _self.FIXTURES_ENDPOINT)
        try:
            response = httpx.get(url, timeout=10.0, verify=False)  # DO NOT DO THIS IN PRODUCTION
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            st.error(f"HTTP error fetching fixtures: {e}")
            return None
        except httpx.RequestError as e:
            st.error(f"Request error fetching fixtures: {e}")
            return None
        except json.JSONDecodeError as e:
            st.error(f"JSON decode error: {e}")
            return None

class FixtureDifficultyCalculator:
    """Handles fixture difficulty calculations and caching"""
    
    def __init__(self, fixture_source: FixtureDataSource):
        self.fixture_source = fixture_source
        self._fixtures_cache = None
    
    @property
    def fixtures_data(self) -> Optional[List[Dict[str, Any]]]:
        """Lazy loading of fixtures data with caching"""
        if self._fixtures_cache is None:
            self._fixtures_cache = self.fixture_source.fetch_fixtures()
        return self._fixtures_cache
    
    def calc_avg_fdr(self, team_id: int, current_gw: int, num_gw: int = 5) -> float:
        """Calculates the average fixture difficulty for a team."""
        fixtures = self.fixtures_data
        if fixtures is None:
            return 3.0  # Default to medium difficulty if no fixtures

        relevant = [
            f for f in fixtures 
            if f.get("event") and current_gw <= f["event"] < current_gw + num_gw
        ]
        
        diffs = []
        for fixture in relevant:
            if fixture["team_h"] == team_id:
                diffs.append(fixture["team_h_difficulty"])
            elif fixture["team_a"] == team_id:
                diffs.append(fixture["team_a_difficulty"])
                
        return round(sum(diffs) / len(diffs), 1) if diffs else 3.0

    @staticmethod
    def fdr_to_label(fdr: float) -> str:
        """Converts FDR to a label."""
        if fdr <= 2:
            return "Easy"
        elif fdr >= 4:
            return "Hard"
        else:
            return "Medium"