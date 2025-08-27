"""
Core Domain Models for FPL Analytics Application
Following SOLID principles with clear separation of concerns
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from datetime import datetime
from enum import Enum
import pandas as pd


class Position(Enum):
    """Player positions enum"""
    GOALKEEPER = "GK"
    DEFENDER = "DEF"
    MIDFIELDER = "MID"
    FORWARD = "FWD"


class AnalysisType(Enum):
    """Analysis type options"""
    ALL_FIXTURES = "All Fixtures"
    HOME_ONLY = "Home Only"
    AWAY_ONLY = "Away Only"
    NEXT_THREE = "Next 3 Fixtures"
    CONGESTION = "Fixture Congestion Periods"


class ReliabilityStatus(Enum):
    """Player reliability status"""
    RELIABLE = "🟢 Reliable"
    PROMISING = "🟢 Promising"
    DECENT = "🟡 Decent"
    MONITOR = "🟡 Monitor"
    ROTATION = "🟠 Rotation"
    BENCH_RISK = "🔴 Bench Risk"
    INJURED = "🔴 Injured"
    CONCERN = "🔴 Concern"
    UNKNOWN = "❓ Unknown"


@dataclass
class Player:
    """Core player model with all FPL data"""
    id: int
    web_name: str
    first_name: str
    second_name: str
    team_id: int
    position_id: int
    
    # Basic stats
    total_points: int = 0
    points_per_game: float = 0.0
    form: float = 0.0
    now_cost: int = 0
    selected_by_percent: float = 0.0
    
    # Performance metrics
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    saves: int = 0
    bonus: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    minutes: int = 0
    
    # Advanced metrics
    expected_points: float = 0.0
    cost_change_start: int = 0
    cost_change_event: int = 0
    
    # Calculated properties
    cost_millions: float = field(init=False)
    points_per_million: float = field(init=False)
    position_name: str = field(init=False)
    team_name: str = field(init=False)
    team_short_name: str = field(init=False)
    
    def __post_init__(self):
        """Calculate derived properties"""
        self.cost_millions = self.now_cost / 10.0
        self.points_per_million = (
            self.total_points / self.cost_millions if self.cost_millions > 0 else 0
        )
    
    def set_team_info(self, team_name: str, team_short_name: str):
        """Set team information"""
        self.team_name = team_name
        self.team_short_name = team_short_name
    
    def set_position_name(self, position_name: str):
        """Set position name"""
        self.position_name = position_name
    
    def get_reliability_status(self, total_games: int = 1) -> ReliabilityStatus:
        """Calculate player reliability status"""
        if hasattr(self, 'status') and getattr(self, 'status') != 'a':
            return ReliabilityStatus.INJURED
        
        games_started = self.minutes / 90 if self.minutes > 0 else 0
        
        if total_games < 3:  # Early season
            if self.form >= 5:
                return ReliabilityStatus.PROMISING
            elif self.form >= 3:
                return ReliabilityStatus.MONITOR
            else:
                return ReliabilityStatus.CONCERN
        
        # Regular season assessment
        if games_started >= 0.8 * total_games and self.form >= 5:
            return ReliabilityStatus.RELIABLE
        elif games_started >= 0.6 * total_games and self.form >= 3:
            return ReliabilityStatus.DECENT
        elif games_started >= 0.4 * total_games:
            return ReliabilityStatus.ROTATION
        else:
            return ReliabilityStatus.BENCH_RISK


@dataclass
class Team:
    """Core team model"""
    id: int
    name: str
    short_name: str
    code: int
    
    # Strength metrics
    strength: int = 0
    strength_overall_home: int = 0
    strength_overall_away: int = 0
    strength_attack_home: int = 0
    strength_attack_away: int = 0
    strength_defence_home: int = 0
    strength_defence_away: int = 0
    
    # Performance metrics
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: int = 0
    position: int = 0


@dataclass
class Fixture:
    """Core fixture model"""
    id: int
    event: int  # Gameweek
    team_h: int  # Home team ID
    team_a: int  # Away team ID
    team_h_score: Optional[int] = None
    team_a_score: Optional[int] = None
    finished: bool = False
    kickoff_time: Optional[str] = None
    
    # Difficulty ratings
    team_h_difficulty: int = 3
    team_a_difficulty: int = 3
    
    # Enhanced properties
    team_h_name: str = field(init=False, default="")
    team_a_name: str = field(init=False, default="")
    team_h_short: str = field(init=False, default="")
    team_a_short: str = field(init=False, default="")
    
    def set_team_names(self, home_name: str, away_name: str, home_short: str, away_short: str):
        """Set team names for display"""
        self.team_h_name = home_name
        self.team_a_name = away_name
        self.team_h_short = home_short
        self.team_a_short = away_short


@dataclass
class FDRData:
    """Fixture Difficulty Rating data"""
    team_short_name: str
    opponent_short_name: str
    gameweek: int
    fixture_number: int
    is_home: bool
    difficulty: int
    
    # FDR calculations
    attack_fdr: float = 0.0
    defense_fdr: float = 0.0
    combined_fdr: float = 0.0
    
    # Form adjustments
    form_adjusted_attack: float = 0.0
    form_adjusted_defense: float = 0.0
    form_adjusted_combined: float = 0.0


@dataclass
class PlayerPick:
    """Player pick from user's team"""
    element: int  # Player ID
    position: int  # Squad position (1-15)
    multiplier: int = 1  # Captain multiplier
    is_captain: bool = False
    is_vice_captain: bool = False
    
    def __post_init__(self):
        """Set captain flags based on multiplier"""
        if self.multiplier == 2:
            self.is_captain = True
        elif self.multiplier == 1 and hasattr(self, '_vice_captain'):
            self.is_vice_captain = True


@dataclass
class UserTeam:
    """User's FPL team data"""
    id: int
    entry_name: str
    player_first_name: str
    player_last_name: str
    
    # Performance metrics
    summary_overall_points: int = 0
    summary_overall_rank: int = 0
    summary_event_rank: int = 0
    current_event: int = 1
    
    # Team value
    value: int = 1000  # In tenths of millions
    bank: int = 0
    
    # Squad
    picks: List[PlayerPick] = field(default_factory=list)
    
    @property
    def team_value(self) -> float:
        """Team value in millions"""
        return self.value / 10.0
    
    @property
    def bank_value(self) -> float:
        """Bank value in millions"""
        return self.bank / 10.0
    
    def get_captain(self) -> Optional[PlayerPick]:
        """Get current captain"""
        return next((pick for pick in self.picks if pick.is_captain), None)
    
    def get_vice_captain(self) -> Optional[PlayerPick]:
        """Get current vice captain"""
        return next((pick for pick in self.picks if pick.is_vice_captain), None)
    
    def get_starting_eleven(self) -> List[PlayerPick]:
        """Get starting eleven players"""
        return [pick for pick in self.picks if pick.position <= 11]
    
    def get_bench(self) -> List[PlayerPick]:
        """Get bench players"""
        return [pick for pick in self.picks if pick.position > 11]


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters"""
    gameweeks_ahead: int = 5
    fdr_threshold: float = 2.5
    form_weight: float = 0.3
    analysis_type: AnalysisType = AnalysisType.ALL_FIXTURES
    use_form_adjustment: bool = True
    show_opponents: bool = True
    show_colors: bool = True
    sort_ascending: bool = True


@dataclass
class AIConfig:
    """Configuration for AI providers"""
    openai_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    current_provider: str = "cohere"
    max_tokens: int = 600
    temperature: float = 0.7
    max_retries: int = 3
