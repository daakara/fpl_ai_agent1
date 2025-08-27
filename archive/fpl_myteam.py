import httpx
from typing import Dict, Any, List, Optional, Union
import logging
import asyncio
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import ssl

REQUEST_TIMEOUT = 30  # Increased timeout
VERIFY_SSL = True  # Enable SSL verification
BASE_URL = "https://fantasy.premierleague.com/api/"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

class FPLMyTeamFetcher:
    """Fetches FPL team data with improved error handling and authentication."""

    def __init__(self, team_id: int):
        self.team_id = team_id
        self.session = None

    async def __aenter__(self):
        # Create session with proper headers
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0"
        }
        
        self.session = httpx.AsyncClient(
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            verify=VERIFY_SSL,
            follow_redirects=True
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
    async def _get(self, endpoint: str) -> Optional[Union[Dict, List]]:
        """Get data from FPL API with improved error handling."""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        url = f"{BASE_URL}{endpoint}"
        
        try:
            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.5)
            
            response = await self.session.get(url)
            
            if response.status_code == 403:
                logging.warning(f"Access forbidden for {url} - this is expected for private team data")
                return None
            elif response.status_code == 404:
                logging.warning(f"Endpoint not found: {url}")
                return None
            elif response.status_code == 429:
                logging.warning("Rate limited. Waiting...")
                await asyncio.sleep(5)
                return None
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code in [403, 404, 429]:
                return None
            logging.error(f"HTTP error fetching {url}: {e}")
            return None
        except httpx.RequestError as e:
            logging.warning(f"Request error fetching {url}: {e}")
            return None
        except Exception as e:
            logging.warning(f"Unexpected error fetching {url}: {e}")
            return None

    async def fetch_team_info(self) -> Optional[Dict]:
        """Get general info about team - tries public endpoint first."""
        # Try the public entry endpoint first
        result = await self._get(f"entry/{self.team_id}/")
        if result:
            return result
        
        # If that fails, return basic team structure
        return {
            "id": self.team_id,
            "name": f"Team {self.team_id}",
            "summary_overall_points": 0,
            "overall_rank": "N/A",
            "current_event": None
        }

    async def fetch_gameweek_picks(self, gameweek: Optional[int]) -> Dict:
        """Fetch team picks - will fail without authentication."""
        if gameweek is None or gameweek == 0:
            return {
                "picks": [],
                "message": "Gameweek data not available.",
            }
        
        # This will likely fail with 403
        result = await self._get(f"entry/{self.team_id}/event/{gameweek}/picks/")
        if result is None:
            return {
                "picks": [],
                "message": "Picks data requires FPL login (not available in this app).",
            }
        
        return result

    async def fetch_transfers(self) -> List[Dict]:
        """Fetch transfers - will fail without authentication."""
        result = await self._get(f"entry/{self.team_id}/transfers/")
        return result if result is not None else []

    async def fetch_chips(self) -> List[Dict]:
        """Fetch chip usage - will fail without authentication."""
        result = await self._get(f"entry/{self.team_id}/chips/")
        return result if result is not None else []

    async def fetch_predictions(self, gameweek: Optional[int] = None) -> Dict:
        """Predictions are not available via public API."""
        return {
            "predicted_picks": [],
            "message": "Predictions not available via public API.",
        }

    async def fetch_live_points(self, gameweek: Optional[int] = None) -> Dict:
        """Live points not available without authentication."""
        if gameweek is None or gameweek == 0:
            return {
                "live_points": 0,
                "message": "Gameweek has not started yet.",
            }
        
        return {
            "live_points": 0,
            "message": "Live points require FPL login (not available in this app).",
        }


async def fetch_current_gameweek() -> Optional[int]:
    """Fetch current gameweek from bootstrap API."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.get(
                f"{BASE_URL}bootstrap-static/",
                headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            data = response.json()
            
            current_gw = data.get("current_event")
            if current_gw is None:
                # Look for current event in events list
                for event in data.get("events", []):
                    if event.get("is_current"):
                        current_gw = event.get("id")
                        break
            
            return current_gw
            
        except Exception as e:
            logging.error(f"Error fetching current gameweek: {e}")
            return None


async def load_my_fpl_team(team_id: int):
    """Load team data with graceful handling of authentication issues."""
    async with FPLMyTeamFetcher(team_id) as fetcher:
        try:
            # Get basic team info (this might work)
            info = await fetcher.fetch_team_info()
            if info is None:
                info = {
                    "id": team_id,
                    "name": f"Team {team_id}",
                    "summary_overall_points": 0,
                    "overall_rank": "N/A",
                    "current_event": None
                }
            
            # Get current gameweek from public API
            current_gw = await fetch_current_gameweek()
            info["current_event"] = current_gw
            
            # Try to get other data (will likely fail with 403)
            picks = await fetcher.fetch_gameweek_picks(current_gw)
            transfers = await fetcher.fetch_transfers()
            chips = await fetcher.fetch_chips()
            predictions = await fetcher.fetch_predictions(current_gw)
            live_points = await fetcher.fetch_live_points(current_gw)
            
            return info, picks, transfers, chips, predictions, live_points
        except Exception as e:
            logging.warning(f"Error loading team data (this is expected): {e}")
            # Return safe default data
            return (
                {"id": team_id, "name": f"Team {team_id}", "summary_overall_points": 0, "overall_rank": "N/A"},
                {"picks": [], "message": "Team data requires FPL authentication."},
                [],
                [],
                {"predicted_picks": [], "message": "Predictions not available."},
                {"live_points": 0, "message": "Live points not accessible."}
            )


# Simple public data loader
async def load_public_team_data(team_id: int):
    """Load only publicly available team data."""
    try:
        # Get current gameweek
        current_gw = await fetch_current_gameweek()
        
        return (
            {
                "id": team_id,
                "name": f"Team {team_id}",
                "summary_overall_points": 0,
                "overall_rank": "N/A",
                "current_event": current_gw
            },
            {"picks": [], "message": "Private team data requires FPL login."},
            [],
            [],
            {"predicted_picks": [], "message": "Predictions not available."},
            {"live_points": 0, "message": "Live points not accessible."}
        )
        
    except Exception as e:
        logging.error(f"Error loading public team data: {e}")
        return (
            {"id": team_id, "name": f"Team {team_id}", "summary_overall_points": 0},
            {"picks": [], "message": "Data not accessible."},
            [], [], 
            {"predicted_picks": []}, 
            {"live_points": 0}
        )


if __name__ == "__main__":
    import json
    
    async def test_team_data():
        team_id = 1437677
        
        print("Testing team data fetch...")
        info, picks, transfers, chips, predictions, live_points = await load_my_fpl_team(team_id)
        
        print("Team Info:", json.dumps(info, indent=2))
        print("Picks:", json.dumps(picks, indent=2))
        print("Transfers:", len(transfers) if transfers else 0)
        print("Chips:", len(chips) if chips else 0)
    
    asyncio.run(test_team_data())
