"""
Data Services for FPL Analytics Application
Implements Service pattern with clear separation of concerns
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import logging
from datetime import datetime

from models.domain_models import Player, Team, Fixture, FDRData, UserTeam, PlayerPick, AnalysisType


class IDataLoader(ABC):
    """Interface for data loading services (Interface Segregation Principle)"""
    
    @abstractmethod
    def load_players(self) -> List[Player]:
        """Load player data"""
        pass
    
    @abstractmethod
    def load_teams(self) -> List[Team]:
        """Load team data"""
        pass
    
    @abstractmethod
    def load_fixtures(self) -> List[Fixture]:
        """Load fixture data"""
        pass


class IFDRCalculator(ABC):
    """Interface for FDR calculation services"""
    
    @abstractmethod
    def calculate_fdr(self, fixtures: List[Fixture], teams: List[Team]) -> List[FDRData]:
        """Calculate FDR for fixtures"""
        pass
    
    @abstractmethod
    def apply_form_adjustment(self, fdr_data: List[FDRData], players: List[Player], form_weight: float) -> List[FDRData]:
        """Apply form adjustments to FDR"""
        pass


class FPLAPIDataLoader(IDataLoader):
    """Concrete implementation of data loader using FPL API"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.base_url = "https://fantasy.premierleague.com/api"
        self.session = requests.Session()
        self.session.verify = False  # For SSL issues
        
    def _make_request(self, endpoint: str) -> Dict:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"API request failed for {endpoint}: {e}")
            raise
    
    def load_bootstrap_data(self) -> Dict:
        """Load main bootstrap data"""
        return self._make_request("bootstrap-static/")
    
    def load_players(self) -> List[Player]:
        """Load and parse player data"""
        try:
            data = self.load_bootstrap_data()
            players = []
            
            # Create lookup dictionaries
            teams_data = {team['id']: team for team in data['teams']}
            positions_data = {pos['id']: pos for pos in data['element_types']}
            
            for player_data in data['elements']:
                player = Player(
                    id=player_data['id'],
                    web_name=player_data['web_name'],
                    first_name=player_data['first_name'],
                    second_name=player_data['second_name'],
                    team_id=player_data['team'],
                    position_id=player_data['element_type'],
                    total_points=player_data['total_points'],
                    points_per_game=float(player_data.get('points_per_game', 0)),
                    form=float(player_data.get('form', 0)),
                    now_cost=player_data['now_cost'],
                    selected_by_percent=float(player_data.get('selected_by_percent', 0)),
                    goals_scored=player_data.get('goals_scored', 0),
                    assists=player_data.get('assists', 0),
                    clean_sheets=player_data.get('clean_sheets', 0),
                    saves=player_data.get('saves', 0),
                    bonus=player_data.get('bonus', 0),
                    yellow_cards=player_data.get('yellow_cards', 0),
                    red_cards=player_data.get('red_cards', 0),
                    minutes=player_data.get('minutes', 0),
                    expected_points=float(player_data.get('ep_next', 0))
                )
                
                # Set team information
                team_data = teams_data.get(player.team_id, {})
                player.set_team_info(
                    team_data.get('name', 'Unknown'),
                    team_data.get('short_name', 'UNK')
                )
                
                # Set position information
                position_data = positions_data.get(player.position_id, {})
                player.set_position_name(position_data.get('singular_name', 'Unknown'))
                
                players.append(player)
            
            self.logger.info(f"Loaded {len(players)} players")
            return players
            
        except Exception as e:
            self.logger.error(f"Failed to load players: {e}")
            return []
    
    def load_teams(self) -> List[Team]:
        """Load and parse team data"""
        try:
            data = self.load_bootstrap_data()
            teams = []
            
            for team_data in data['teams']:
                team = Team(
                    id=team_data['id'],
                    name=team_data['name'],
                    short_name=team_data['short_name'],
                    code=team_data['code'],
                    strength=team_data.get('strength', 0),
                    strength_overall_home=team_data.get('strength_overall_home', 0),
                    strength_overall_away=team_data.get('strength_overall_away', 0),
                    strength_attack_home=team_data.get('strength_attack_home', 0),
                    strength_attack_away=team_data.get('strength_attack_away', 0),
                    strength_defence_home=team_data.get('strength_defence_home', 0),
                    strength_defence_away=team_data.get('strength_defence_away', 0),
                    played=team_data.get('played', 0),
                    wins=team_data.get('win', 0),
                    draws=team_data.get('draw', 0),
                    losses=team_data.get('loss', 0),
                    points=team_data.get('points', 0),
                    position=team_data.get('position', 0)
                )
                teams.append(team)
            
            self.logger.info(f"Loaded {len(teams)} teams")
            return teams
            
        except Exception as e:
            self.logger.error(f"Failed to load teams: {e}")
            return []
    
    def load_fixtures(self) -> List[Fixture]:
        """Load and parse fixture data"""
        try:
            fixtures_data = self._make_request("fixtures/")
            teams_data = {team.id: team for team in self.load_teams()}
            
            fixtures = []
            for fixture_data in fixtures_data:
                if fixture_data.get('finished', True):
                    continue  # Skip finished fixtures
                
                fixture = Fixture(
                    id=fixture_data['id'],
                    event=fixture_data.get('event', 1),
                    team_h=fixture_data['team_h'],
                    team_a=fixture_data['team_a'],
                    team_h_score=fixture_data.get('team_h_score'),
                    team_a_score=fixture_data.get('team_a_score'),
                    finished=fixture_data.get('finished', False),
                    kickoff_time=fixture_data.get('kickoff_time'),
                    team_h_difficulty=fixture_data.get('team_h_difficulty', 3),
                    team_a_difficulty=fixture_data.get('team_a_difficulty', 3)
                )
                
                # Set team names
                home_team = teams_data.get(fixture.team_h)
                away_team = teams_data.get(fixture.team_a)
                
                if home_team and away_team:
                    fixture.set_team_names(
                        home_team.name, away_team.name,
                        home_team.short_name, away_team.short_name
                    )
                
                fixtures.append(fixture)
            
            self.logger.info(f"Loaded {len(fixtures)} fixtures")
            return fixtures
            
        except Exception as e:
            self.logger.error(f"Failed to load fixtures: {e}")
            return []
    
    def load_user_team(self, team_id: int, gameweek: Optional[int] = None) -> Optional[UserTeam]:
        """Load user team data"""
        try:
            # Load entry data
            entry_data = self._make_request(f"entry/{team_id}/")
            
            user_team = UserTeam(
                id=team_id,
                entry_name=entry_data.get('name', 'Unknown Team'),
                player_first_name=entry_data.get('player_first_name', ''),
                player_last_name=entry_data.get('player_last_name', ''),
                summary_overall_points=entry_data.get('summary_overall_points', 0),
                summary_overall_rank=entry_data.get('summary_overall_rank', 0),
                summary_event_rank=entry_data.get('summary_event_rank', 0),
                current_event=entry_data.get('current_event', 1),
                value=entry_data.get('value', 1000),
                bank=entry_data.get('bank', 0)
            )
            
            # Load picks for specific gameweek
            if gameweek:
                try:
                    picks_data = self._make_request(f"entry/{team_id}/event/{gameweek}/picks/")
                    picks = []
                    
                    for pick_data in picks_data.get('picks', []):
                        pick = PlayerPick(
                            element=pick_data['element'],
                            position=pick_data['position'],
                            multiplier=pick_data.get('multiplier', 1)
                        )
                        picks.append(pick)
                    
                    user_team.picks = picks
                    
                except Exception as e:
                    self.logger.warning(f"Could not load picks for GW {gameweek}: {e}")
            
            return user_team
            
        except Exception as e:
            self.logger.error(f"Failed to load user team {team_id}: {e}")
            return None


class FDRCalculatorService(IFDRCalculator):
    """Service for calculating Fixture Difficulty Ratings"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_fdr(self, fixtures: List[Fixture], teams: List[Team]) -> List[FDRData]:
        """Calculate FDR for all fixtures"""
        try:
            team_lookup = {team.id: team for team in teams}
            fdr_data = []
            
            # Group fixtures by team and process
            team_fixtures = self._group_fixtures_by_team(fixtures, team_lookup)
            
            for team_short, team_fixture_list in team_fixtures.items():
                for i, (fixture, is_home, opponent_short) in enumerate(team_fixture_list[:5]):  # Next 5 fixtures
                    team = next((t for t in teams if t.short_name == team_short), None)
                    opponent = next((t for t in teams if t.short_name == opponent_short), None)
                    
                    if not team or not opponent:
                        continue
                    
                    # Calculate FDR based on team strengths
                    attack_fdr = self._calculate_attack_fdr(team, opponent, is_home)
                    defense_fdr = self._calculate_defense_fdr(team, opponent, is_home)
                    combined_fdr = (attack_fdr + defense_fdr) / 2
                    
                    fdr_entry = FDRData(
                        team_short_name=team_short,
                        opponent_short_name=opponent_short,
                        gameweek=fixture.event,
                        fixture_number=i + 1,
                        is_home=is_home,
                        difficulty=fixture.team_h_difficulty if is_home else fixture.team_a_difficulty,
                        attack_fdr=round(attack_fdr, 2),
                        defense_fdr=round(defense_fdr, 2),
                        combined_fdr=round(combined_fdr, 2)
                    )
                    
                    fdr_data.append(fdr_entry)
            
            self.logger.info(f"Calculated FDR for {len(fdr_data)} fixture entries")
            return fdr_data
            
        except Exception as e:
            self.logger.error(f"Failed to calculate FDR: {e}")
            return []
    
    def _group_fixtures_by_team(self, fixtures: List[Fixture], team_lookup: Dict[int, Team]) -> Dict[str, List[Tuple]]:
        """Group fixtures by team"""
        team_fixtures = {}
        
        for fixture in fixtures:
            home_team = team_lookup.get(fixture.team_h)
            away_team = team_lookup.get(fixture.team_a)
            
            if not home_team or not away_team:
                continue
            
            # Add for home team
            if home_team.short_name not in team_fixtures:
                team_fixtures[home_team.short_name] = []
            team_fixtures[home_team.short_name].append((fixture, True, away_team.short_name))
            
            # Add for away team
            if away_team.short_name not in team_fixtures:
                team_fixtures[away_team.short_name] = []
            team_fixtures[away_team.short_name].append((fixture, False, home_team.short_name))
        
        # Sort by gameweek for each team
        for team_name in team_fixtures:
            team_fixtures[team_name].sort(key=lambda x: x[0].event)
        
        return team_fixtures
    
    def _calculate_attack_fdr(self, team: Team, opponent: Team, is_home: bool) -> float:
        """Calculate attack FDR"""
        # Get team's attacking strength
        team_attack = team.strength_attack_home if is_home else team.strength_attack_away
        
        # Get opponent's defensive strength
        opponent_defense = opponent.strength_defence_away if is_home else opponent.strength_defence_home
        
        # Calculate relative difficulty (higher opponent defense = harder)
        # Scale to 1-5 range where 1 = easy, 5 = very hard
        if opponent_defense >= 1400:
            return 5.0
        elif opponent_defense >= 1350:
            return 4.0
        elif opponent_defense >= 1300:
            return 3.0
        elif opponent_defense >= 1250:
            return 2.0
        else:
            return 1.0
    
    def _calculate_defense_fdr(self, team: Team, opponent: Team, is_home: bool) -> float:
        """Calculate defense FDR"""
        # Get opponent's attacking strength
        opponent_attack = opponent.strength_attack_away if is_home else opponent.strength_attack_home
        
        # Calculate relative difficulty (higher opponent attack = harder to keep clean sheet)
        # Scale to 1-5 range where 1 = easy, 5 = very hard
        if opponent_attack >= 1400:
            return 5.0
        elif opponent_attack >= 1350:
            return 4.0
        elif opponent_attack >= 1300:
            return 3.0
        elif opponent_attack >= 1250:
            return 2.0
        else:
            return 1.0
    
    def apply_form_adjustment(self, fdr_data: List[FDRData], players: List[Player], form_weight: float) -> List[FDRData]:
        """Apply form adjustments to FDR calculations"""
        try:
            # Calculate team form factors
            team_forms = self._calculate_team_forms(players)
            
            for fdr_entry in fdr_data:
                team_form = team_forms.get(fdr_entry.team_short_name, 0)
                opponent_form = team_forms.get(fdr_entry.opponent_short_name, 0)
                
                # Apply form adjustments
                attack_adjustment = -team_form * form_weight
                defense_adjustment = opponent_form * form_weight
                combined_adjustment = (attack_adjustment + defense_adjustment) / 2
                
                # Calculate form-adjusted FDR (keep within 1-5 range)
                fdr_entry.form_adjusted_attack = max(1, min(5, fdr_entry.attack_fdr + attack_adjustment))
                fdr_entry.form_adjusted_defense = max(1, min(5, fdr_entry.defense_fdr + defense_adjustment))
                fdr_entry.form_adjusted_combined = max(1, min(5, fdr_entry.combined_fdr + combined_adjustment))
            
            return fdr_data
            
        except Exception as e:
            self.logger.error(f"Failed to apply form adjustment: {e}")
            return fdr_data
    
    def _calculate_team_forms(self, players: List[Player]) -> Dict[str, float]:
        """Calculate average form for each team"""
        team_forms = {}
        team_players = {}
        
        # Group players by team
        for player in players:
            if player.team_short_name not in team_players:
                team_players[player.team_short_name] = []
            team_players[player.team_short_name].append(player)
        
        # Calculate average form for each team
        for team_name, team_player_list in team_players.items():
            if team_player_list:
                avg_form = sum(p.form for p in team_player_list) / len(team_player_list)
                # Normalize form (5.0 = average form)
                form_factor = (avg_form - 5.0) / 5.0  # Range: -1 to 1
                team_forms[team_name] = form_factor
        
        return team_forms
    
    def filter_fdr_by_analysis_type(self, fdr_data: List[FDRData], analysis_type: AnalysisType) -> List[FDRData]:
        """Filter FDR data by analysis type"""
        if analysis_type == AnalysisType.ALL_FIXTURES:
            return fdr_data
        elif analysis_type == AnalysisType.HOME_ONLY:
            return [fdr for fdr in fdr_data if fdr.is_home]
        elif analysis_type == AnalysisType.AWAY_ONLY:
            return [fdr for fdr in fdr_data if not fdr.is_home]
        elif analysis_type == AnalysisType.NEXT_THREE:
            return [fdr for fdr in fdr_data if fdr.fixture_number <= 3]
        else:
            return fdr_data
