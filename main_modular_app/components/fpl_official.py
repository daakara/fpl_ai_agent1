import httpx
from typing import List, Dict
from functools import lru_cache
import logging
import asyncio
import json

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

class FPLAPIError(Exception):
    """Base class for FPL API errors."""
    pass

class FPLDataNotFoundError(FPLAPIError):
    """Raised when data is not found in the FPL API response."""
    pass

class FPLRequestError(FPLAPIError):
    """Raised when there is an issue with the FPL API request."""
    pass

@lru_cache(maxsize=None)
def fetch_fpl_bootstrap_data_sync() -> Dict:
    """
    Fetches bootstrap static data from Official FPL API synchronously.
    """
    try:
        response = httpx.get(FPL_BOOTSTRAP_URL, timeout=10, verify=False)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        logging.error(f"Error fetching FPL bootstrap data: {e}")
        return {}

async def fetch_fpl_bootstrap_data_async() -> Dict:
    """
    Fetches bootstrap static data from Official FPL API asynchronously.
    """
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.get(FPL_BOOTSTRAP_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data:
                raise FPLDataNotFoundError("No data found in FPL API response.")
            return response.json()
        except httpx.RequestError as e:
            logging.error(f"Error fetching FPL bootstrap data: {e}")
            raise FPLRequestError(f"Request error: {e}") from e
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")
            raise FPLDataNotFoundError(f"JSON decode error: {e}") from e
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise FPLAPIError(f"An unexpected error occurred: {e}") from e

async def get_players_data() -> List[Dict]:
    """
    Returns a list of player data dicts from the FPL API.
    """
    data = await fetch_fpl_bootstrap_data_async()
    players = data.get("elements", [])
    return players

async def get_teams_data() -> List[Dict]:
    """
    Returns a list of team data dicts from the FPL API.
    """
    data = await fetch_fpl_bootstrap_data_async()
    teams = data.get("teams", [])
    return teams

@lru_cache(maxsize=None)
def get_positions_data() -> List[Dict]:
    """
    Returns a list of position types e.g. Goalkeeper, Defender, Midfielder, Forward.
    """
    data = fetch_fpl_bootstrap_data_sync()
    positions = data.get("element_types", [])
    return positions

@lru_cache(maxsize=None)
def get_fixtures_data_sync():
    """Fetch fixtures data from the official FPL API synchronously."""
    try:
        response = httpx.get(FPL_FIXTURES_URL, timeout=10, verify=False)
        response.raise_for_status()
        return response.json()
    except httpx.RequestError as e:
        logging.error(f"Error fetching fixtures data: {e}")
        return []

async def get_fixtures_data_async():
    """Fetch fixtures data from the official FPL API asynchronously."""
    async with httpx.AsyncClient(verify=False) as client:
        try:
            response = await client.get(FPL_FIXTURES_URL, timeout=10)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logging.error(f"Error fetching fixtures data: {e}")
            return []

async def main():
    # Simple test
    print("Fetching players data...")
    players = get_players_data()
    print(f"Total players fetched: {len(players)}")
    print("Sample player:")
    print(players[0])

    print("\nFetching teams data...")
    teams = get_teams_data()
    print(f"Total teams fetched: {len(teams)}")
    print("Sample team:")
    print(teams[0])

    print("\nFetching positions data...")
    positions = get_positions_data()
    print(f"Total positions fetched: {len(positions)}")
    print("Sample position:")
    print(positions[0])

    print("\nFetching fixtures data...")
    fixtures = await get_fixtures_data_async()
    print(f"Total fixtures fetched: {len(fixtures)}")
    print("Sample fixture:")
    print(fixtures[0])

if __name__ == "__main__":
    asyncio.run(main())