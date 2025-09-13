"""
Enhanced Data Management Layer with Real-time Processing
Optimizes data flow and provides robust caching and processing capabilities
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
import hashlib
import json
import requests
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

@dataclass
class PlayerData:
    """Player data model with validation"""
    id: int
    web_name: str
    team_id: int
    element_type: int
    now_cost: int
    total_points: int
    form: float
    selected_by_percent: float
    minutes: int
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    bonus: int = 0
    bps: int = 0
    influence: float = 0.0
    creativity: float = 0.0
    threat: float = 0.0
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        self.cost_millions = self.now_cost / 10
        self.points_per_million = self.total_points / self.cost_millions if self.cost_millions > 0 else 0
        self.form = max(0.0, float(self.form or 0))
        self.selected_by_percent = max(0.0, float(self.selected_by_percent or 0))

@dataclass
class TeamData:
    """Team data model"""
    id: int
    name: str
    short_name: str
    strength_overall_home: int
    strength_overall_away: int
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int

class DataProcessor(ABC):
    """Abstract base class for data processors"""
    
    @abstractmethod
    def process(self, raw_data: Dict) -> Any:
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        pass

class PlayerDataProcessor(DataProcessor):
    """Processes and validates player data"""
    
    def process(self, raw_data: Dict) -> List[PlayerData]:
        """Process raw player data into PlayerData objects"""
        players = []
        
        for player_raw in raw_data.get('elements', []):
            try:
                player = PlayerData(
                    id=player_raw['id'],
                    web_name=player_raw['web_name'],
                    team_id=player_raw['team'],
                    element_type=player_raw['element_type'],
                    now_cost=player_raw['now_cost'],
                    total_points=player_raw['total_points'],
                    form=float(player_raw.get('form', 0)),
                    selected_by_percent=float(player_raw.get('selected_by_percent', 0)),
                    minutes=player_raw.get('minutes', 0),
                    goals_scored=player_raw.get('goals_scored', 0),
                    assists=player_raw.get('assists', 0),
                    clean_sheets=player_raw.get('clean_sheets', 0),
                    bonus=player_raw.get('bonus', 0),
                    bps=player_raw.get('bps', 0),
                    influence=float(player_raw.get('influence', 0)),
                    creativity=float(player_raw.get('creativity', 0)),
                    threat=float(player_raw.get('threat', 0))
                )
                players.append(player)
            except (KeyError, ValueError, TypeError) as e:
                st.warning(f"Skipping invalid player data: {e}")
                continue
        
        return players
    
    def validate(self, data: List[PlayerData]) -> bool:
        """Validate processed player data"""
        if not data:
            return False
        
        # Check for required fields
        for player in data[:10]:  # Sample validation
            if not all([
                player.id > 0,
                player.web_name,
                player.team_id > 0,
                player.element_type > 0,
                player.now_cost > 0
            ]):
                return False
        
        return True

class CacheManager:
    """Advanced caching system with TTL and compression"""
    
    def __init__(self, cache_dir: str = "fpl_cache"):
        self.cache_dir = cache_dir
        self.db_path = f"{cache_dir}/cache.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize cache database"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp REAL,
                    ttl INTEGER,
                    hits INTEGER DEFAULT 0
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            st.warning(f"Cache initialization failed: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT data, timestamp, ttl FROM cache WHERE key = ?", 
                (key,)
            )
            result = cursor.fetchone()
            
            if result:
                data_str, timestamp, ttl = result
                if datetime.now().timestamp() - timestamp < ttl:
                    # Update hit count
                    conn.execute(
                        "UPDATE cache SET hits = hits + 1 WHERE key = ?",
                        (key,)
                    )
                    conn.commit()
                    conn.close()
                    return json.loads(data_str)
                else:
                    # Remove expired entry
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
            
            conn.close()
            return None
        except Exception as e:
            st.warning(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, data: Any, ttl: int = 300) -> None:
        """Set data in cache with TTL"""
        try:
            conn = sqlite3.connect(self.db_path)
            data_str = json.dumps(data, default=str)
            timestamp = datetime.now().timestamp()
            
            conn.execute("""
                INSERT OR REPLACE INTO cache (key, data, timestamp, ttl, hits)
                VALUES (?, ?, ?, ?, 0)
            """, (key, data_str, timestamp, ttl))
            
            conn.commit()
            conn.close()
        except Exception as e:
            st.warning(f"Cache set error: {e}")
    
    def clear_expired(self) -> int:
        """Clear expired cache entries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "DELETE FROM cache WHERE ? - timestamp > ttl",
                (datetime.now().timestamp(),)
            )
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            return deleted_count
        except Exception as e:
            st.warning(f"Cache cleanup error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(hits) as total_hits,
                    AVG(hits) as avg_hits,
                    MAX(timestamp) as latest_entry
                FROM cache
            """)
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'total_entries': result[0],
                    'total_hits': result[1] or 0,
                    'avg_hits': round(result[2] or 0, 2),
                    'latest_entry': datetime.fromtimestamp(result[3]) if result[3] else None
                }
        except Exception as e:
            st.warning(f"Cache stats error: {e}")
        
        return {'total_entries': 0, 'total_hits': 0, 'avg_hits': 0, 'latest_entry': None}

class EnhancedDataLoader:
    """Enhanced data loader with validation, caching, and error handling"""
    
    def __init__(self):
        self.cache = CacheManager()
        self.player_processor = PlayerDataProcessor()
        self.base_url = "https://fantasy.premierleague.com/api"
        
    def _generate_cache_key(self, endpoint: str, params: Dict = None) -> str:
        """Generate cache key for request"""
        key_data = f"{endpoint}:{params or ''}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    @st.cache_data(ttl=300)
    def load_bootstrap_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Load and process bootstrap data with enhanced error handling"""
        cache_key = self._generate_cache_key("bootstrap-static")
        
        # Try cache first
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return self._process_bootstrap_data(cached_data)
        
        # Fetch from API
        try:
            response = requests.get(
                f"{self.base_url}/bootstrap-static/",
                timeout=30,
                verify=False
            )
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            self.cache.set(cache_key, data, ttl=300)
            
            return self._process_bootstrap_data(data)
            
        except requests.RequestException as e:
            st.error(f"Failed to load FPL data: {e}")
            # Try to return stale cache data as fallback
            stale_data = self.cache.get(cache_key)
            if stale_data:
                st.warning("Using stale cached data")
                return self._process_bootstrap_data(stale_data)
            
            return pd.DataFrame(), pd.DataFrame(), {}
    
    def _process_bootstrap_data(self, data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Process bootstrap data into structured format"""
        try:
            # Process players
            players = self.player_processor.process(data)
            if not self.player_processor.validate(players):
                raise ValueError("Player data validation failed")
            
            # Convert to DataFrame
            players_df = pd.DataFrame([asdict(player) for player in players])
            
            # Process teams
            teams_data = []
            for team_raw in data.get('teams', []):
                team = TeamData(
                    id=team_raw['id'],
                    name=team_raw['name'],
                    short_name=team_raw['short_name'],
                    strength_overall_home=team_raw.get('strength_overall_home', 0),
                    strength_overall_away=team_raw.get('strength_overall_away', 0),
                    strength_attack_home=team_raw.get('strength_attack_home', 0),
                    strength_attack_away=team_raw.get('strength_attack_away', 0),
                    strength_defence_home=team_raw.get('strength_defence_home', 0),
                    strength_defence_away=team_raw.get('strength_defence_away', 0)
                )
                teams_data.append(asdict(team))
            
            teams_df = pd.DataFrame(teams_data)
            
            # Add derived columns to players
            self._enhance_player_data(players_df, teams_df, data)
            
            return players_df, teams_df, data
            
        except Exception as e:
            st.error(f"Data processing error: {e}")
            return pd.DataFrame(), pd.DataFrame(), {}
    
    def _enhance_player_data(self, players_df: pd.DataFrame, teams_df: pd.DataFrame, raw_data: Dict):
        """Add enhanced columns to player data"""
        try:
            # Add team information
            team_lookup = teams_df.set_index('id')[['name', 'short_name']].to_dict('index')
            players_df['team_name'] = players_df['team_id'].map(lambda x: team_lookup.get(x, {}).get('name', 'Unknown'))
            players_df['team_short_name'] = players_df['team_id'].map(lambda x: team_lookup.get(x, {}).get('short_name', 'UNK'))
            
            # Add position information
            positions = {pos['id']: pos['singular_name'] for pos in raw_data.get('element_types', [])}
            players_df['position_name'] = players_df['element_type'].map(positions)
            
            # Add advanced metrics
            players_df['minutes_per_game'] = players_df['minutes'] / 38  # Assume 38 games max
            players_df['points_per_game'] = players_df['total_points'] / (players_df['minutes'] / 90).clip(lower=1)
            players_df['value_score'] = (
                players_df['points_per_million'] * 0.4 +
                players_df['form'] * 0.3 +
                (100 - players_df['selected_by_percent']) * 0.2 +
                (players_df['minutes'] / 3000) * 0.1
            )
            
        except Exception as e:
            st.warning(f"Data enhancement error: {e}")

    def load_fixtures_data(self) -> pd.DataFrame:
        """Load fixtures data with enhanced processing"""
        cache_key = self._generate_cache_key("fixtures")
        
        # Try cache first
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            response = requests.get(
                f"{self.base_url}/fixtures/",
                timeout=30,
                verify=False
            )
            response.raise_for_status()
            fixtures_data = response.json()
            
            # Cache the response
            self.cache.set(cache_key, fixtures_data, ttl=1800)  # 30 minutes
            
            return pd.DataFrame(fixtures_data)
            
        except requests.RequestException as e:
            st.error(f"Failed to load fixtures data: {e}")
            return pd.DataFrame()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return self.cache.get_stats()
    
    def clear_cache(self) -> int:
        """Clear expired cache entries"""
        return self.cache.clear_expired()

class DataQualityMonitor:
    """Monitor data quality and provide insights"""
    
    @staticmethod
    def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics"""
        if df.empty:
            return {'status': 'empty', 'issues': ['No data available']}
        
        issues = []
        metrics = {}
        
        # Check for missing values
        missing_cols = df.isnull().sum()
        critical_missing = missing_cols[missing_cols > len(df) * 0.1]
        if not critical_missing.empty:
            issues.append(f"High missing values in: {list(critical_missing.index)}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate rows found")
        
        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 5:
            issues.append("Insufficient numeric columns for analysis")
        
        # Calculate quality score
        quality_score = max(0, 100 - len(issues) * 10 - (missing_cols.sum() / len(df)) * 20)
        
        return {
            'status': 'good' if quality_score > 80 else 'fair' if quality_score > 60 else 'poor',
            'quality_score': round(quality_score, 1),
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': missing_cols.sum(),
            'duplicates': duplicates,
            'issues': issues,
            'metrics': {
                'completeness': round((1 - missing_cols.sum() / (len(df) * len(df.columns))) * 100, 1),
                'uniqueness': round((1 - duplicates / len(df)) * 100, 1),
                'consistency': round(quality_score, 1)
            }
        }
    
    @staticmethod
    def suggest_improvements(quality_report: Dict[str, Any]) -> List[str]:
        """Suggest data quality improvements"""
        suggestions = []
        
        if quality_report['quality_score'] < 80:
            suggestions.append("🔧 Consider data cleaning and validation")
        
        if quality_report['duplicates'] > 0:
            suggestions.append("🗑️ Remove duplicate records")
        
        if quality_report['missing_values'] > 0:
            suggestions.append("📊 Handle missing values with imputation or filtering")
        
        if quality_report['total_rows'] < 100:
            suggestions.append("📈 Consider expanding dataset for better analysis")
        
        return suggestions

# Global instances
enhanced_data_loader = EnhancedDataLoader()
data_quality_monitor = DataQualityMonitor()