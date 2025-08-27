import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class DataProcessor(ABC):
    """Abstract base class for data processing"""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass

class PlayerDataProcessor(DataProcessor):
    """Processes player data with validation and type conversion"""
    
    REQUIRED_COLUMNS = [
        "selected_by_percent", "expected_points_next_5", "xG_next_5", 
        "xA_next_5", "form", "now_cost", "team_name"
    ]
    
    # Extended statistics that might be available
    EXTENDED_COLUMNS = {
        'expected_assists_per_90': 'float',
        'expected_goal_involvements_per_90': 'float', 
        'expected_goals_conceded_per_90': 'float',
        'expected_goals': 'float',
        'expected_assists': 'float',
        'expected_goal_involvements': 'float',
        'expected_goals_conceded': 'float',
        'recoveries': 'int',
        'tackles': 'int',
        'clearances_blocks_interceptions': 'int',
        'expected_goals_per_90': 'float'
    }
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and validate player data with extended statistics"""
        df = df.copy()
        
        # Validate required columns exist
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            # Create missing columns with default values
            for col in missing_cols:
                if col in ["xG_next_5", "xA_next_5"]:
                    df[col] = 0.0
                elif col == "expected_points_next_5":
                    df[col] = df["form"].apply(self._safe_float) * 5
                else:
                    df[col] = 0.0
        
        # Process extended columns if they exist
        for col, dtype in self.EXTENDED_COLUMNS.items():
            if col in df.columns:
                if dtype == 'float':
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                elif dtype == 'int':
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        
        # Convert base numeric columns
        numeric_columns = ["selected_by_percent", "expected_points_next_5", "xG_next_5", 
                          "xA_next_5", "form", "now_cost"]
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        # Create derived columns
        df["now_cost_m"] = df["now_cost"] / 10
        df["team_name"] = df["team_name"].fillna("Unknown")
        
        # Calculate additional derived metrics if base data is available
        self._calculate_derived_metrics(df)
        
        return df
    
    def _calculate_derived_metrics(self, df):
        """Calculate derived metrics from available data"""
        # Calculate per 90 metrics if total metrics and minutes are available
        if 'minutes' in df.columns and df['minutes'].sum() > 0:
            # Calculate games played (approximate)
            df['games_played'] = (df['minutes'] / 90).round()
            df['games_played'] = df['games_played'].replace(0, 1)  # Avoid division by zero
            
            # Calculate per 90 metrics if total stats are available but per 90 aren't
            if 'expected_goals' in df.columns and 'expected_goals_per_90' not in df.columns:
                df['expected_goals_per_90'] = (df['expected_goals'] / df['games_played'] * 90 / df['minutes']).fillna(0)
            
            if 'expected_assists' in df.columns and 'expected_assists_per_90' not in df.columns:
                df['expected_assists_per_90'] = (df['expected_assists'] / df['games_played'] * 90 / df['minutes']).fillna(0)
            
            # Calculate goal involvements if components are available
            if 'expected_goals' in df.columns and 'expected_assists' in df.columns:
                if 'expected_goal_involvements' not in df.columns:
                    df['expected_goal_involvements'] = df['expected_goals'] + df['expected_assists']
                if 'expected_goal_involvements_per_90' not in df.columns:
                    df['expected_goal_involvements_per_90'] = (df['expected_goal_involvements'] / df['games_played'] * 90 / df['minutes']).fillna(0)
        
        # Combine defensive actions if individual components are available
        defensive_cols = ['clearances', 'blocks', 'interceptions']
        available_defensive = [col for col in defensive_cols if col in df.columns]
        
        if available_defensive and 'clearances_blocks_interceptions' not in df.columns:
            df['clearances_blocks_interceptions'] = sum(df[col].fillna(0) for col in available_defensive)
    
    @staticmethod
    def _safe_float(val):
        """Safely convert value to float"""
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

class InjuryDataProcessor(DataProcessor):
    """Processes injury data and maps to player data"""
    
    def __init__(self, injury_scraper):
        self.injury_scraper = injury_scraper
    
    def process(self, df_players: pd.DataFrame) -> pd.DataFrame:
        """Add injury status to player data"""
        try:
            injury_status_by_name = self.injury_scraper.get_injury_status()
            df_players["injury_status"] = df_players["web_name"].map(
                injury_status_by_name
            ).fillna("")
        except Exception as e:
            # Graceful fallback if injury scraping fails
            df_players["injury_status"] = ""
            
        return df_players

class TeamMappingProcessor(DataProcessor):
    """Creates team ID to name mappings"""
    
    def process(self, teams_data: List[Dict]) -> Dict[int, str]:
        """Create team ID to name mapping"""
        df_teams = pd.DataFrame(teams_data)
        return df_teams.set_index("id")["name"].to_dict()