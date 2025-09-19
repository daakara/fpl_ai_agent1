"""
Enhanced FPL API Service with comprehensive team integration, caching, and error handling
"""
import requests
import pandas as pd
import streamlit as st
import logging
import time
import json
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict
from functools import wraps


@dataclass
class TeamData:
    """Enhanced team data structure"""
    id: int
    entry_name: str
    player_first_name: str
    player_last_name: str
    summary_overall_points: int
    summary_overall_rank: int
    summary_event_rank: int
    current_event: int
    value: int  # Team value in tenths
    bank: int   # Bank value in tenths
    total_transfers: int
    
    # Performance metrics
    event_transfers: int = 0
    event_transfers_cost: int = 0
    favourite_team: Optional[int] = None
    started_event: int = 1
    
    # Additional computed fields
    team_value_millions: float = 0.0
    bank_millions: float = 0.0
    total_value: float = 0.0
    
    def __post_init__(self):
        self.team_value_millions = self.value / 10.0
        self.bank_millions = self.bank / 10.0
        self.total_value = self.team_value_millions + self.bank_millions


@dataclass 
class PlayerPick:
    """Player pick with enhanced attributes"""
    element: int
    position: int
    multiplier: int
    is_captain: bool
    is_vice_captain: bool
    
    # Player details (populated after API call)
    web_name: str = ""
    team_name: str = ""
    position_name: str = ""
    total_points: int = 0
    form: float = 0.0
    now_cost: int = 0
    selected_by_percent: float = 0.0
    expected_points: float = 0.0


class EnhancedFPLAPIService:
    """Enhanced FPL API service with comprehensive functionality"""
    
    def __init__(self, cache_enabled: bool = True):
        self.base_url = "https://fantasy.premierleague.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Setup caching
        self.cache_enabled = cache_enabled
        self.cache_dir = Path("fpl_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_db = self.cache_dir / "api_cache.db"
        self._init_cache_db()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests
        
        # Cache for bootstrap data
        self._bootstrap_data = None
        self._bootstrap_timestamp = None
        self._bootstrap_ttl = 300  # 5 minutes
    
    def _init_cache_db(self):
        """Initialize SQLite cache database"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS api_cache (
                        key TEXT PRIMARY KEY,
                        data TEXT,
                        timestamp REAL,
                        ttl INTEGER
                    )
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize cache database: {e}")
    
    def _cache_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """Generate cache key for endpoint and parameters"""
        key_data = f"{endpoint}:{json.dumps(params or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get data from cache if valid"""
        if not self.cache_enabled:
            return None
            
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    "SELECT data, timestamp, ttl FROM api_cache WHERE key = ?",
                    (key,)
                )
                result = cursor.fetchone()
                
                if result:
                    data, timestamp, ttl = result
                    if time.time() - timestamp < ttl:
                        return json.loads(data)
                    else:
                        # Remove expired entry
                        conn.execute("DELETE FROM api_cache WHERE key = ?", (key,))
                        conn.commit()
                        
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")
        
        return None
    
    def _save_to_cache(self, key: str, data: Dict, ttl: int = 300):
        """Save data to cache"""
        if not self.cache_enabled:
            return
            
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO api_cache (key, data, timestamp, ttl) VALUES (?, ?, ?, ?)",
                    (key, json.dumps(data), time.time(), ttl)
                )
                conn.commit()
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")
    
    def _rate_limit(self):
        """Implement rate limiting"""
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, ttl: int = 300) -> Optional[Dict]:
        """Make API request with caching and error handling"""
        # Check cache first
        cache_key = self._cache_key(endpoint, params)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Rate limiting
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            response = self.session.get(url, params=params, timeout=30, verify=False)
            response.raise_for_status()
            
            data = response.json()
            
            # Save to cache
            self._save_to_cache(cache_key, data, ttl)
            
            return data
            
        except requests.RequestException as e:
            self.logger.error(f"API request failed for {endpoint}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for {endpoint}: {e}")
            return None
    
    def get_bootstrap_data(self, force_refresh: bool = False) -> Optional[Dict]:
        """Get bootstrap data with intelligent caching"""
        current_time = time.time()
        
        # Check if we have cached bootstrap data that's still valid
        if (not force_refresh and 
            self._bootstrap_data and 
            self._bootstrap_timestamp and 
            current_time - self._bootstrap_timestamp < self._bootstrap_ttl):
            return self._bootstrap_data
        
        # Fetch fresh data
        data = self._make_request("bootstrap-static/", ttl=300)
        if data:
            self._bootstrap_data = data
            self._bootstrap_timestamp = current_time
        
        return data
    
    def get_current_gameweek(self) -> int:
        """Get current gameweek with fallback logic"""
        try:
            data = self.get_bootstrap_data()
            if not data:
                return 1
            
            events = data.get('events', [])
            
            # Find current gameweek
            for event in events:
                if event.get('is_current', False):
                    return event['id']
            
            # Find next gameweek
            for event in events:
                if event.get('is_next', False):
                    return event['id']
            
            # Find latest finished gameweek + 1
            finished_events = [e for e in events if e.get('finished', False)]
            if finished_events:
                latest = max(finished_events, key=lambda x: x['id'])
                return min(latest['id'] + 1, 38)
            
            return 1
            
        except Exception as e:
            self.logger.error(f"Error getting current gameweek: {e}")
            return 1
    
    def get_team_data(self, team_id: int) -> Optional[TeamData]:
        """Get comprehensive team data"""
        try:
            data = self._make_request(f"entry/{team_id}/", ttl=60)  # Cache for 1 minute
            if not data:
                return None
            
            return TeamData(
                id=team_id,
                entry_name=data.get('name', 'Unknown Team'),
                player_first_name=data.get('player_first_name', ''),
                player_last_name=data.get('player_last_name', ''),
                summary_overall_points=data.get('summary_overall_points', 0),
                summary_overall_rank=data.get('summary_overall_rank', 0),
                summary_event_rank=data.get('summary_event_rank', 0),
                current_event=data.get('current_event', 1),
                value=data.get('value', 1000),
                bank=data.get('bank', 0),
                total_transfers=data.get('total_transfers', 0),
                event_transfers=data.get('event_transfers', 0),
                event_transfers_cost=data.get('event_transfers_cost', 0),
                favourite_team=data.get('favourite_team'),
                started_event=data.get('started_event', 1)
            )
            
        except Exception as e:
            self.logger.error(f"Error getting team data for {team_id}: {e}")
            return None
    
    def get_team_picks(self, team_id: int, gameweek: Optional[int] = None) -> List[PlayerPick]:
        """Get team picks for a specific gameweek"""
        if gameweek is None:
            gameweek = self.get_current_gameweek()
        
        try:
            data = self._make_request(f"entry/{team_id}/event/{gameweek}/picks/", ttl=60)
            if not data or 'picks' not in data:
                return []
            
            picks = []
            for pick_data in data['picks']:
                pick = PlayerPick(
                    element=pick_data['element'],
                    position=pick_data['position'],
                    multiplier=pick_data.get('multiplier', 1),
                    is_captain=pick_data.get('is_captain', False),
                    is_vice_captain=pick_data.get('is_vice_captain', False)
                )
                picks.append(pick)
            
            return picks
            
        except Exception as e:
            self.logger.error(f"Error getting picks for team {team_id}, GW {gameweek}: {e}")
            return []
    
    def get_enhanced_team_analysis(self, team_id: int, gameweek: Optional[int] = None) -> Dict:
        """Get comprehensive team analysis with all data"""
        try:
            # Get basic team data
            team_data = self.get_team_data(team_id)
            if not team_data:
                raise Exception(f"Could not load team data for ID {team_id}")
            
            # Get picks
            picks = self.get_team_picks(team_id, gameweek)
            
            # Get bootstrap data for player details
            bootstrap_data = self.get_bootstrap_data()
            if not bootstrap_data:
                raise Exception("Could not load bootstrap data")
            
            # Create player lookup
            players_lookup = {p['id']: p for p in bootstrap_data['elements']}
            teams_lookup = {t['id']: t for t in bootstrap_data['teams']}
            positions_lookup = {p['id']: p for p in bootstrap_data['element_types']}
            
            # Enhance picks with player details
            enhanced_picks = []
            for pick in picks:
                player_data = players_lookup.get(pick.element, {})
                team_data_player = teams_lookup.get(player_data.get('team'), {})
                position_data = positions_lookup.get(player_data.get('element_type'), {})
                
                pick.web_name = player_data.get('web_name', 'Unknown')
                pick.team_name = team_data_player.get('short_name', 'UNK')
                pick.position_name = position_data.get('singular_name_short', 'UNK')
                pick.total_points = player_data.get('total_points', 0)
                pick.form = float(player_data.get('form', 0))
                pick.now_cost = player_data.get('now_cost', 0)
                pick.selected_by_percent = float(player_data.get('selected_by_percent', 0))
                pick.expected_points = float(player_data.get('ep_next', 0))
                
                enhanced_picks.append(pick)
            
            # Calculate team statistics
            starting_xi = [p for p in enhanced_picks if p.position <= 11]
            bench = [p for p in enhanced_picks if p.position > 11]
            
            total_points = sum(p.total_points for p in enhanced_picks)
            total_cost = sum(p.now_cost for p in enhanced_picks)
            avg_ownership = sum(p.selected_by_percent for p in enhanced_picks) / len(enhanced_picks) if enhanced_picks else 0
            
            captain = next((p for p in enhanced_picks if p.is_captain), None)
            vice_captain = next((p for p in enhanced_picks if p.is_vice_captain), None)
            
            return {
                'team_data': asdict(team_data),
                'picks': enhanced_picks,
                'starting_xi': starting_xi,
                'bench': bench,
                'captain': captain,
                'vice_captain': vice_captain,
                'statistics': {
                    'total_points': total_points,
                    'total_cost_millions': total_cost / 10.0,
                    'average_ownership': round(avg_ownership, 1),
                    'players_count': len(enhanced_picks),
                    'gameweek': gameweek or self.get_current_gameweek()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in enhanced team analysis: {e}")
            return {}
    
    def get_chip_status(self, team_id: int) -> Dict[str, bool]:
        """Get chip usage status (requires checking multiple gameweeks)"""
        chips_used = {
            'wildcard': False,
            'bench_boost': False,
            'triple_captain': False,
            'free_hit': False
        }
        
        try:
            current_gw = self.get_current_gameweek()
            
            # Check last few gameweeks for chip usage
            for gw in range(max(1, current_gw - 10), current_gw + 1):
                try:
                    data = self._make_request(f"entry/{team_id}/event/{gw}/picks/", ttl=3600)
                    if data and 'active_chip' in data and data['active_chip']:
                        chip_name = data['active_chip'].lower()
                        if chip_name in chips_used:
                            chips_used[chip_name] = True
                except:
                    continue
            
            return chips_used
            
        except Exception as e:
            self.logger.error(f"Error getting chip status: {e}")
            return chips_used
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute("DELETE FROM api_cache")
                conn.commit()
            
            self._bootstrap_data = None
            self._bootstrap_timestamp = None
            
            st.success("✅ Cache cleared successfully!")
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            st.error(f"❌ Error clearing cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM api_cache")
                total_entries = cursor.fetchone()[0]
                
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM api_cache WHERE timestamp > ?",
                    (time.time() - 3600,)  # Last hour
                )
                fresh_entries = cursor.fetchone()[0]
                
                return {
                    'total_entries': total_entries,
                    'fresh_entries': fresh_entries,
                    'cache_hit_rate': f"{(fresh_entries/total_entries*100):.1f}%" if total_entries > 0 else "0%"
                }
                
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {'total_entries': 0, 'fresh_entries': 0, 'cache_hit_rate': '0%'}


# Global service instance
fpl_api_service = EnhancedFPLAPIService()


def with_error_handling(func):
    """Decorator for consistent error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper


@with_error_handling
def load_team_data(team_id: int, gameweek: Optional[int] = None) -> Optional[Dict]:
    """Main function to load comprehensive team data"""
    return fpl_api_service.get_enhanced_team_analysis(team_id, gameweek)


@with_error_handling  
def get_current_gameweek() -> int:
    """Get current gameweek"""
    return fpl_api_service.get_current_gameweek()


def display_cache_info():
    """Display cache information in sidebar"""
    with st.sidebar:
        st.subheader("🗄️ API Cache")
        stats = fpl_api_service.get_cache_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cached Entries", stats['total_entries'])
        with col2:
            st.metric("Fresh Entries", stats['fresh_entries'])
        
        st.write(f"**Hit Rate:** {stats['cache_hit_rate']}")
        
        if st.button("🗑️ Clear Cache"):
            fpl_api_service.clear_cache()
            st.rerun()