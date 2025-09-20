"""
Enhanced Fixture Service with Official FPL API Integration
"""
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import streamlit as st
import logging
import urllib3

# Disable SSL warnings when certificate verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class FixtureService:
    """Enhanced Fixture Service using Official FPL API data"""
    
    def __init__(self):
        """Initialize the fixture service with FPL API integration"""
        self.base_url = "https://fantasy.premierleague.com/api"
        self.fixtures_cache = {}
        self.teams_cache = {}
        self.bootstrap_cache = {}
        self.cache_timestamp = None
        self.cache_duration = 3600  # 1 hour cache
        self.logger = logging.getLogger(__name__)
        
        # Configure requests session with SSL verification disabled
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL certificate verification
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Load initial data
        self._load_bootstrap_data()
    
    def _load_bootstrap_data(self):
        """Load bootstrap data from FPL API including teams and current gameweek"""
        try:
            if (self.cache_timestamp and 
                datetime.now() - self.cache_timestamp < timedelta(seconds=self.cache_duration) and
                self.bootstrap_cache):
                return self.bootstrap_cache
            
            self.logger.info("Loading bootstrap data from FPL API...")
            response = self.session.get(f"{self.base_url}/bootstrap-static/", timeout=15)
            response.raise_for_status()
            
            self.bootstrap_cache = response.json()
            self.cache_timestamp = datetime.now()
            
            # Extract teams data
            self.teams_cache = {
                team['id']: {
                    'name': team['name'],
                    'short_name': team['short_name'],
                    'code': team['code'],
                    'strength': team.get('strength', 3),
                    'strength_overall_home': team.get('strength_overall_home', 3),
                    'strength_overall_away': team.get('strength_overall_away', 3),
                    'strength_attack_home': team.get('strength_attack_home', 3),
                    'strength_attack_away': team.get('strength_attack_away', 3),
                    'strength_defence_home': team.get('strength_defence_home', 3),
                    'strength_defence_away': team.get('strength_defence_away', 3)
                }
                for team in self.bootstrap_cache.get('teams', [])
            }
            
            self.logger.info(f"✅ Successfully loaded {len(self.teams_cache)} teams from FPL API")
            return self.bootstrap_cache
            
        except requests.exceptions.SSLError as e:
            self.logger.error(f"SSL Error loading bootstrap data: {e}")
            # Return cached data if available, otherwise empty dict
            return self.bootstrap_cache if self.bootstrap_cache else {}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error loading bootstrap data: {e}")
            return self.bootstrap_cache if self.bootstrap_cache else {}
        except Exception as e:
            self.logger.error(f"Unexpected error loading bootstrap data: {e}")
            return self.bootstrap_cache if self.bootstrap_cache else {}
    
    def _load_fixtures_data(self):
        """Load fixtures data from FPL API"""
        try:
            if (self.fixtures_cache and self.cache_timestamp and 
                datetime.now() - self.cache_timestamp < timedelta(seconds=self.cache_duration)):
                return self.fixtures_cache
            
            self.logger.info("Loading fixtures data from FPL API...")
            response = self.session.get(f"{self.base_url}/fixtures/", timeout=15)
            response.raise_for_status()
            
            fixtures_data = response.json()
            
            # Process fixtures into a more usable format
            self.fixtures_cache = {}
            
            for fixture in fixtures_data:
                # Only include fixtures that haven't been played yet
                if not fixture.get('finished', True):
                    team_h = fixture['team_h']
                    team_a = fixture['team_a']
                    event = fixture.get('event')
                    
                    # Add fixture for home team
                    if team_h not in self.fixtures_cache:
                        self.fixtures_cache[team_h] = []
                    
                    self.fixtures_cache[team_h].append({
                        'event': event,
                        'is_home': True,
                        'opponent': team_a,
                        'difficulty': fixture.get('team_h_difficulty', 3),
                        'kickoff_time': fixture.get('kickoff_time'),
                        'finished': fixture.get('finished', False)
                    })
                    
                    # Add fixture for away team
                    if team_a not in self.fixtures_cache:
                        self.fixtures_cache[team_a] = []
                    
                    self.fixtures_cache[team_a].append({
                        'event': event,
                        'is_home': False,
                        'opponent': team_h,
                        'difficulty': fixture.get('team_a_difficulty', 3),
                        'kickoff_time': fixture.get('kickoff_time'),
                        'finished': fixture.get('finished', False)
                    })
            
            # Sort fixtures by gameweek for each team
            for team_id in self.fixtures_cache:
                self.fixtures_cache[team_id].sort(key=lambda x: x.get('event', 0))
            
            self.logger.info(f"✅ Successfully loaded fixtures for {len(self.fixtures_cache)} teams from FPL API")
            return self.fixtures_cache
            
        except requests.exceptions.SSLError as e:
            self.logger.error(f"SSL Error loading fixtures data: {e}")
            return {}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error loading fixtures data: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error loading fixtures data: {e}")
            return {}
    
    def get_current_gameweek(self) -> int:
        """Get current gameweek from FPL API"""
        try:
            bootstrap_data = self._load_bootstrap_data()
            
            # Find current event
            for event in bootstrap_data.get('events', []):
                if event.get('is_current', False):
                    return event['id']
            
            # If no current event, find the next upcoming event
            for event in bootstrap_data.get('events', []):
                if event.get('is_next', False):
                    return event['id']
            
            # Fallback: find first unfinished event
            for event in bootstrap_data.get('events', []):
                if not event.get('finished', True):
                    return event['id']
            
            # Final fallback
            return 1
            
        except Exception as e:
            self.logger.error(f"Error getting current gameweek: {e}")
            return st.session_state.get('current_gameweek', 1)
    
    def get_team_id_by_short_name(self, short_name: str) -> Optional[int]:
        """Get team ID by short name"""
        for team_id, team_data in self.teams_cache.items():
            if team_data['short_name'] == short_name:
                return team_id
        return None
    
    def get_team_short_name_by_id(self, team_id: int) -> str:
        """Get team short name by ID"""
        team_data = self.teams_cache.get(team_id, {})
        return team_data.get('short_name', 'UNK')
    
    def get_upcoming_fixtures_difficulty(self, team_short_name: str, gameweeks: int = 5) -> Dict:
        """
        Get fixture difficulty for upcoming gameweeks using official FPL data
        
        Args:
            team_short_name: Team's short name (e.g., 'ARS')
            gameweeks: Number of upcoming gameweeks to analyze
            
        Returns:
            Dictionary with fixture difficulty analysis using official FPL data
        """
        try:
            # Load latest data
            self._load_fixtures_data()
            
            # Get team ID
            team_id = self.get_team_id_by_short_name(team_short_name)
            if not team_id:
                self.logger.warning(f"Team not found: {team_short_name}")
                return self._get_fallback_fixtures(team_short_name, gameweeks)
            
            # Get team's fixtures
            team_fixtures = self.fixtures_cache.get(team_id, [])
            
            if not team_fixtures:
                self.logger.warning(f"No fixtures found for team: {team_short_name}")
                return self._get_fallback_fixtures(team_short_name, gameweeks)
            
            # Get current gameweek
            current_gw = self.get_current_gameweek()
            
            # Filter upcoming fixtures (current + next gameweeks)
            upcoming_fixtures = [
                f for f in team_fixtures 
                if f.get('event', 0) >= current_gw and not f.get('finished', False)
            ][:gameweeks]
            
            if not upcoming_fixtures:
                self.logger.warning(f"No upcoming fixtures found for team: {team_short_name}")
                return self._get_fallback_fixtures(team_short_name, gameweeks)
            
            # Process fixtures
            fixtures_analysis = []
            total_difficulty = 0
            
            for fixture in upcoming_fixtures:
                opponent_id = fixture['opponent']
                opponent_short_name = self.get_team_short_name_by_id(opponent_id)
                difficulty = fixture['difficulty']
                
                fixtures_analysis.append({
                    'gameweek': fixture.get('event', current_gw),
                    'opponent': opponent_short_name,
                    'home': fixture['is_home'],
                    'difficulty': difficulty,
                    'difficulty_text': self._get_difficulty_text(difficulty),
                    'kickoff_time': fixture.get('kickoff_time')
                })
                
                total_difficulty += difficulty
            
            avg_difficulty = total_difficulty / len(fixtures_analysis) if fixtures_analysis else 3
            
            return {
                'fixtures': fixtures_analysis,
                'average_difficulty': avg_difficulty,
                'total_difficulty': total_difficulty,
                'rating': self._get_fixture_period_rating(avg_difficulty)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting fixtures for {team_short_name}: {e}")
            return self._get_fallback_fixtures(team_short_name, gameweeks)
    
    def _get_fallback_fixtures(self, team_short_name: str, gameweeks: int) -> Dict:
        """Fallback fixture data when API is unavailable"""
        self.logger.info(f"Using fallback fixture data for {team_short_name}")
        
        # Use simplified estimation based on team strength
        current_gw = self.get_current_gameweek()
        
        # Generate estimated fixtures
        fixtures_analysis = []
        total_difficulty = 0
        
        for i in range(gameweeks):
            # Estimate difficulty based on average (3 is neutral)
            difficulty = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])  # Weighted random
            
            fixtures_analysis.append({
                'gameweek': current_gw + i,
                'opponent': 'TBD',
                'home': i % 2 == 0,  # Alternate home/away
                'difficulty': difficulty,
                'difficulty_text': self._get_difficulty_text(difficulty),
                'kickoff_time': None
            })
            
            total_difficulty += difficulty
        
        avg_difficulty = total_difficulty / len(fixtures_analysis)
        
        return {
            'fixtures': fixtures_analysis,
            'average_difficulty': avg_difficulty,
            'total_difficulty': total_difficulty,
            'rating': self._get_fixture_period_rating(avg_difficulty),
            'is_fallback': True
        }
    
    def get_team_attack_defense_strength(self, team_short_name: str) -> Dict:
        """Get team's attacking and defensive strength from official FPL data"""
        team_id = self.get_team_id_by_short_name(team_short_name)
        
        if not team_id or team_id not in self.teams_cache:
            return {'attack': 3, 'defense': 3, 'overall': 3}
        
        team_data = self.teams_cache[team_id]
        
        return {
            'attack_home': team_data.get('strength_attack_home', 3),
            'attack_away': team_data.get('strength_attack_away', 3),
            'defense_home': team_data.get('strength_defence_home', 3),
            'defense_away': team_data.get('strength_defence_away', 3),
            'overall_home': team_data.get('strength_overall_home', 3),
            'overall_away': team_data.get('strength_overall_away', 3),
            'attack': (team_data.get('strength_attack_home', 3) + team_data.get('strength_attack_away', 3)) / 2,
            'defense': (team_data.get('strength_defence_home', 3) + team_data.get('strength_defence_away', 3)) / 2,
            'overall': team_data.get('strength', 3)
        }
    
    def compare_fixture_run(self, team1: str, team2: str, gameweeks: int = 5) -> Dict:
        """Compare fixture difficulty between two teams using official FPL data"""
        team1_fixtures = self.get_upcoming_fixtures_difficulty(team1, gameweeks)
        team2_fixtures = self.get_upcoming_fixtures_difficulty(team2, gameweeks)
        
        return {
            'team1': {'name': team1, 'data': team1_fixtures},
            'team2': {'name': team2, 'data': team2_fixtures},
            'recommendation': team1 if team1_fixtures['average_difficulty'] < team2_fixtures['average_difficulty'] else team2
        }
    
    def get_fixture_difficulty_score(self, team_short_name: str, is_home: bool, opponent_short_name: str) -> float:
        """
        Calculate fixture difficulty score using official FPL team strength data
        """
        try:
            team_id = self.get_team_id_by_short_name(team_short_name)
            opponent_id = self.get_team_id_by_short_name(opponent_short_name)
            
            if not team_id or not opponent_id:
                return 3  # Default neutral difficulty
            
            team_data = self.teams_cache.get(team_id, {})
            opponent_data = self.teams_cache.get(opponent_id, {})
            
            # Get appropriate strength values based on home/away
            if is_home:
                team_strength = team_data.get('strength_overall_home', 3)
                opponent_strength = opponent_data.get('strength_overall_away', 3)
            else:
                team_strength = team_data.get('strength_overall_away', 3)
                opponent_strength = opponent_data.get('strength_overall_home', 3)
            
            # Calculate relative difficulty (opponent strength vs team strength)
            # Higher opponent strength = higher difficulty for team
            if team_strength == 0:
                difficulty = 3
            else:
                difficulty_ratio = opponent_strength / team_strength
                
                # Convert ratio to 1-5 scale
                if difficulty_ratio <= 0.6:
                    difficulty = 1
                elif difficulty_ratio <= 0.8:
                    difficulty = 2
                elif difficulty_ratio <= 1.2:
                    difficulty = 3
                elif difficulty_ratio <= 1.4:
                    difficulty = 4
                else:
                    difficulty = 5
            
            return difficulty
            
        except Exception as e:
            self.logger.error(f"Error calculating difficulty: {e}")
            return 3
    
    def _get_difficulty_text(self, difficulty: float) -> str:
        """Convert difficulty score to text description"""
        if difficulty <= 1.5:
            return "Very Easy"
        elif difficulty <= 2.5:
            return "Easy"
        elif difficulty <= 3.5:
            return "Average"
        elif difficulty <= 4.5:
            return "Hard"
        else:
            return "Very Hard"
    
    def _get_fixture_period_rating(self, avg_difficulty: float) -> str:
        """Get overall rating for fixture period"""
        if avg_difficulty <= 2:
            return "Excellent"
        elif avg_difficulty <= 2.5:
            return "Good"
        elif avg_difficulty <= 3.5:
            return "Average"
        elif avg_difficulty <= 4:
            return "Difficult"
        else:
            return "Very Difficult"

