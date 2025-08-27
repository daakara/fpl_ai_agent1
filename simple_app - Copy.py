# Create simple_app.py - completely self-contained
import streamlit as st
import pandas as pd
import requests
import traceback
import urllib3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any  # Added Any here
import random
import re
import math
from dataclasses import dataclass
from enum import Enum

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# 1. DATA MODELS AND INTERFACES
# ============================================================================

class Player:
    """Player data model with validation"""
    
    def __init__(self, data: Dict):
        self.id = data.get('id', 0)
        self.web_name = data.get('web_name', '')
        self.first_name = data.get('first_name', '')
        self.second_name = data.get('second_name', '')
        self.team_id = data.get('team', 0)
        self.element_type = data.get('element_type', 0)
        self.cost_millions = data.get('cost_millions', 0.0)
        self.total_points = data.get('total_points', 0)
        self.form = data.get('form', 0.0)
        self.selected_by_percent = data.get('selected_by_percent', 0.0)
        self.position_name = data.get('position_name', '')
        self.team_name = data.get('team_name', '')
        
    def to_dict(self) -> Dict:
        return self.__dict__

class Squad:
    """Squad data model with validation"""
    
    def __init__(self, players: List[Player], formation: str):
        self.players = players
        self.formation = formation
        self.starting_xi = []
        self.bench = []
        self._allocate_players()
    
    def _allocate_players(self):
        """Allocate players to starting XI and bench"""
        formation_requirements = {
            "4-3-3": {"Goalkeeper": 1, "Defender": 4, "Midfielder": 3, "Forward": 3},
            "3-4-3": {"Goalkeeper": 1, "Defender": 3, "Midfielder": 4, "Forward": 3},
        }
        
        requirements = formation_requirements.get(self.formation, formation_requirements["4-3-3"])
        
        for position, needed in requirements.items():
            pos_players = [p for p in self.players if p.position_name == position]
            pos_players.sort(key=lambda x: x.total_points, reverse=True)
            
            self.starting_xi.extend(pos_players[:needed])
            self.bench.extend(pos_players[needed:])
    
    @property
    def total_cost(self) -> float:
        return sum(p.cost_millions for p in self.players)
    
    @property
    def starting_xi_cost(self) -> float:
        return sum(p.cost_millions for p in self.starting_xi)
    
    @property
    def bench_cost(self) -> float:
        return sum(p.cost_millions for p in self.bench)

# ============================================================================
# 2. SINGLE RESPONSIBILITY PRINCIPLE - SPECIALIZED SERVICES
# ============================================================================

class DataLoader:
    """Responsible for loading and fetching FPL data"""
    
    @staticmethod
    def load_fpl_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Load FPL data from API with fallback to mock data"""
        try:
            st.info("🔄 Loading FPL data from API...")
            
            ssl_configs = [
                {"verify": True, "timeout": 30},
                {"verify": False, "timeout": 30},
                {
                    "verify": False, "timeout": 30,
                    "headers": {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                }
            ]
            
            response = None
            for i, config in enumerate(ssl_configs):
                try:
                    st.info(f"Attempting connection method {i+1}...")
                    response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", **config)
                    if response.status_code == 200:
                        st.success(f"✅ Connected successfully using method {i+1}")
                        break
                except Exception as e:
                    st.warning(f"Method {i+1} failed: {str(e)[:100]}...")
                    continue
            
            if response is None or response.status_code != 200:
                st.error("All connection methods failed. Using mock data for testing.")
                return MockDataGenerator.create_mock_data()
            
            data = response.json()
            players_df = pd.DataFrame(data.get('elements', []))
            teams_df = pd.DataFrame(data.get('teams', []))
            
            # Process data
            data_processor = DataProcessor()
            players_df = data_processor.process_players_data(players_df, teams_df)
            
            st.success("✅ Data loaded and processed successfully!")
            return players_df, teams_df, data
            
        except Exception as e:
            st.error(f"Error loading FPL data: {e}")
            st.info("Using mock data for testing purposes")
            return MockDataGenerator.create_mock_data()

class MockDataGenerator:
    """Responsible for generating mock data for testing"""
    
    @staticmethod
    def create_mock_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Create realistic mock FPL data"""
        
        # Teams data
        teams_data = [
            {'id': i+1, 'name': team, 'short_name': short} 
            for i, (team, short) in enumerate([
                ('Arsenal', 'ARS'), ('Aston Villa', 'AVL'), ('Bournemouth', 'BOU'),
                ('Brentford', 'BRE'), ('Brighton', 'BHA'), ('Chelsea', 'CHE'),
                ('Crystal Palace', 'CRY'), ('Everton', 'EVE'), ('Fulham', 'FUL'),
                ('Liverpool', 'LIV'), ('Manchester City', 'MCI'), ('Manchester United', 'MUN'),
                ('Newcastle', 'NEW'), ('Nottingham Forest', 'NFO'), ('Sheffield United', 'SHU'),
                ('Tottenham', 'TOT'), ('West Ham', 'WHU'), ('Wolves', 'WOL'),
                ('Burnley', 'BUR'), ('Luton', 'LUT')
            ])
        ]
        
        # Generate players
        player_names = [
            'Salah', 'Haaland', 'Kane', 'Son', 'De Bruyne', 'Rashford', 'Saka', 'Martinelli',
            'Alexander-Arnold', 'Cancelo', 'Robertson', 'Walker', 'James', 'Chilwell',
            'Alisson', 'Ederson', 'Ramsdale', 'Pickford', 'Pope', 'Sa',
            'Bruno Fernandes', 'Odegaard', 'Rice', 'Casemiro', 'Mount', 'Maddison',
            'Jesus', 'Darwin', 'Isak', 'Wilson', 'Watkins', 'Toney'
        ]
        
        players_data = []
        positions = [1, 2, 3, 4]  # GK, DEF, MID, FWD
        
        for i in range(32):
            team_id = random.randint(1, 20)
            element_type = random.choice(positions)
            
            players_data.append({
                'id': i + 1,
                'web_name': f"{random.choice(player_names)}{i}",
                'first_name': f'Player{i}',
                'second_name': random.choice(player_names),
                'team': team_id,
                'element_type': element_type,
                'now_cost': random.randint(40, 150),
                'total_points': random.randint(0, 200),
                'form': round(random.uniform(0, 10), 1),
                'selected_by_percent': round(random.uniform(0.1, 50.0), 1),
                'points_per_game': round(random.uniform(0, 8), 1),
                'minutes': random.randint(0, 2000),
                'goals_scored': random.randint(0, 20),
                'assists': random.randint(0, 15),
                'clean_sheets': random.randint(0, 15) if element_type in [1, 2] else 0,
                'goals_conceded': random.randint(0, 30) if element_type in [1, 2] else 0,
                'yellow_cards': random.randint(0, 10),
                'red_cards': random.randint(0, 2),
                'saves': random.randint(0, 100) if element_type == 1 else 0,
                'bonus': random.randint(0, 20),
                'bps': random.randint(0, 500),
                'influence': str(random.randint(0, 1000)),
                'creativity': str(random.randint(0, 1000)),
                'threat': str(random.randint(0, 1000)),
                'ict_index': str(random.randint(0, 300))
            })
        
        players_df = pd.DataFrame(players_data)
        teams_df = pd.DataFrame(teams_data)
        
        # Process data
        data_processor = DataProcessor()
        players_df = data_processor.process_players_data(players_df, teams_df)
        
        mock_data = {
            'elements': players_data,
            'teams': teams_data,
            'element_types': [
                {'id': 1, 'plural_name': 'Goalkeepers'},
                {'id': 2, 'plural_name': 'Defenders'},
                {'id': 3, 'plural_name': 'Midfielders'},
                {'id': 4, 'plural_name': 'Forwards'}
            ]
        }
        
        st.warning("⚠️ Using mock data - this is for testing purposes only")
        return players_df, teams_df, mock_data

class DataCleaner:
    """Responsible for cleaning and validating data"""
    
    @staticmethod
    def clean_numeric_column(series: pd.Series) -> pd.Series:
        """Clean a single numeric column with robust error handling"""
        
        def ultra_clean_numeric(x):
            if pd.isna(x) or x is None:
                return 0.0
            
            x_str = str(x).strip().lower()
            
            if x_str in ['', 'none', 'null', 'nan', 'n/a', 'na', '-']:
                return 0.0
            
            if len(x_str) > 20:
                return 0.0
            
            number_match = re.search(r'-?\d*\.?\d+', x_str)
            
            if number_match:
                try:
                    result = float(number_match.group())
                    if abs(result) > 1000000:
                        return 0.0
                    return result
                except (ValueError, OverflowError):
                    return 0.0
            else:
                return 0.0
        
        # Apply cleaning
        cleaned_series = series.apply(ultra_clean_numeric)
        cleaned_series = pd.to_numeric(cleaned_series, errors='coerce')
        cleaned_series = cleaned_series.fillna(0.0)
        
        return cleaned_series.astype('float64')
    
    @classmethod
    def clean_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Clean entire dataframe"""
        if df.empty:
            return df
        
        numeric_columns = [
            'now_cost', 'total_points', 'form', 'selected_by_percent', 
            'points_per_game', 'minutes', 'goals_scored', 'assists',
            'clean_sheets', 'goals_conceded', 'yellow_cards', 'red_cards',
            'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat',
            'ict_index', 'ep_this', 'ep_next', 'value_form', 'value_season'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = cls.clean_numeric_column(df[col])
                    
                    # Apply reasonable bounds
                    if col == 'form':
                        df[col] = df[col].clip(0, 10)
                    elif col == 'selected_by_percent':
                        df[col] = df[col].clip(0, 100)
                    elif col == 'now_cost':
                        df[col] = df[col].clip(30, 200)
                    elif col == 'total_points':
                        df[col] = df[col].clip(0, 500)
                    elif col == 'minutes':
                        df[col] = df[col].clip(0, 3500)
                        
                except Exception as e:
                    st.warning(f"⚠️ Error cleaning {col}: {e}")
                    df[col] = 0.0
        
        return df

class DataProcessor:
    """Responsible for processing and transforming data"""
    
    def process_players_data(self, players_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """Process players data with team mapping and derived metrics"""
        
        # Clean data first
        players_df = DataCleaner.clean_dataframe(players_df)
        
        # Add team names
        if not teams_df.empty:
            team_map = dict(zip(teams_df['id'], teams_df['name']))
            players_df['team_name'] = players_df['team'].map(team_map)
        
        # Add position names
        position_map = {1: 'Goalkeeper', 2: 'Defender', 3: 'Midfielder', 4: 'Forward'}
        if 'element_type' in players_df.columns:
            players_df['position_name'] = players_df['element_type'].map(position_map)
        
        # Convert cost to millions
        if 'now_cost' in players_df.columns:
            players_df['cost_millions'] = players_df['now_cost'] / 10
        
        # Calculate derived metrics
        players_df = self._calculate_metrics(players_df)
        
        return players_df
    
    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional player metrics"""
        try:
            # Value metrics
            if 'total_points' in df.columns and 'cost_millions' in df.columns:
                df['cost_millions'] = df['cost_millions'].replace(0, 0.1)
                df['points_per_million'] = df['total_points'] / df['cost_millions']
                df['value_score'] = df['points_per_million'].rank(pct=True) * 100
            
            # Form metrics
            if 'form' in df.columns:
                df['form_rank'] = df['form'].rank(pct=True) * 100
            
            # Team builder score
            if 'total_points' in df.columns and 'cost_millions' in df.columns:
                df['team_builder_score'] = df['total_points'] / df['cost_millions']
            
            # Differential score
            if 'selected_by_percent' in df.columns and 'total_points' in df.columns:
                df['differential_score'] = (
                    df['total_points'].rank(pct=True) * 100 - 
                    df['selected_by_percent']
                )
            
            # Fill NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
        except Exception as e:
            st.warning(f"Warning: Error calculating metrics: {e}")
        
        return df

# ============================================================================
# OPEN/CLOSED PRINCIPLE - STRATEGY PATTERNS
# ============================================================================

class TeamBuildingStrategy(ABC):
    """Abstract base class for team building strategies"""
    
    @abstractmethod
    def build_team(self, players_df: pd.DataFrame, formation: str, budget: float, **kwargs) -> Optional[List[Dict]]:
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        pass

class PremiumHeavyStrategy(TeamBuildingStrategy):
    """Strategy focusing on premium players"""
    
    def build_team(self, players_df: pd.DataFrame, formation: str, budget: float, **kwargs) -> Optional[List[Dict]]:
        target_premiums = kwargs.get('target_premiums', [])
        max_players_per_club = kwargs.get('max_players_per_club', 3)
        
        # CORRECT formation requirements for 15-player squad
        formation_requirements = {
            "4-3-3": {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3},
            "3-4-3": {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3},
            "3-5-2": {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3},
            "4-4-2": {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3},
            "4-5-1": {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3},
            "5-3-2": {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3},
            "5-4-1": {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3}
        }
        
        requirements = formation_requirements.get(formation, formation_requirements["4-3-3"]).copy()
        
        selected_team = []
        total_cost = 0.0  # Track total cost of entire squad
        selected_ids = set()
        club_counts = {}
        
        # IMPROVED: Force include premium players first with strict budget checking
        if target_premiums:
            st.info(f"🌟 Attempting to include premium players: {', '.join(target_premiums)}")
            
            for player_name in target_premiums:
                # Try exact match first
                exact_match = players_df[players_df['web_name'] == player_name]
                
                # If no exact match, try case-insensitive contains
                if exact_match.empty:
                    contains_match = players_df[
                        players_df['web_name'].str.contains(player_name, case=False, na=False)
                    ]
                    player_matches = contains_match
                else:
                    player_matches = exact_match
                
                if not player_matches.empty:
                    # Get the first match (or best match if multiple)
                    if len(player_matches) > 1:
                        if not exact_match.empty:
                            player = exact_match.iloc[0]
                        else:
                            player = player_matches.nlargest(1, 'total_points').iloc[0]
                    else:
                        player = player_matches.iloc[0]
                    
                    # STRICT BUDGET CHECK: Ensure we can afford this player within total budget
                    if total_cost + player['cost_millions'] <= budget:
                        # Check club constraint
                        team_count = club_counts.get(player['team_name'], 0)
                        if team_count < max_players_per_club:
                            selected_team.append(player.to_dict())
                            selected_ids.add(player['id'])
                            total_cost += player['cost_millions']
                            club_counts[player['team_name']] = team_count + 1
                            
                            # Reduce requirement for this position
                            position = player['position_name']
                            if position in requirements and requirements[position] > 0:
                                requirements[position] -= 1
                        
                            st.success(f"✅ Included premium player: {player['web_name']} (£{player['cost_millions']}m) - Total cost: £{total_cost:.1f}m")
                        else:
                            st.warning(f"⚠️ Cannot include {player['web_name']} - too many players from {player['team_name']}")
                    else:
                        remaining_budget = budget - total_cost
                        st.warning(f"⚠️ Cannot afford {player['web_name']} (£{player['cost_millions']}m) - only £{remaining_budget:.1f}m remaining")
                else:
                    st.warning(f"⚠️ Premium player '{player_name}' not found in dataset")
        
        # Calculate remaining budget for filling positions
        remaining_budget = budget - total_cost
        
        # Fill remaining positions with budget-conscious selection
        for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
            needed = requirements.get(position, 0)
            if needed <= 0:
                continue
            
            # Calculate average budget per remaining player for this position
            total_remaining_players = sum(requirements.values())
            if total_remaining_players > 0:
                avg_budget_per_player = remaining_budget / total_remaining_players
            else:
                avg_budget_per_player = remaining_budget
            
            # Set max budget per player for this position (with some flexibility)
            max_budget_per_position_player = min(avg_budget_per_player * 2, remaining_budget / needed) if needed > 0 else remaining_budget
            
            available_players = players_df[
                (players_df['position_name'] == position) & 
                (~players_df['id'].isin(selected_ids)) &
                (players_df['cost_millions'] <= max_budget_per_position_player)
            ].copy()
            
            if available_players.empty:
                st.warning(f"⚠️ No available {position}s within budget (£{remaining_budget:.1f}m remaining, max £{max_budget_per_position_player:.1f}m per player)")
                continue
            
            # Sort by points per million for value
            available_players = available_players.sort_values('points_per_million', ascending=False)
            
            position_added = 0
            for _, player in available_players.iterrows():
                if position_added >= needed:
                    break
                
                # STRICT BUDGET CHECK: Ensure total squad cost doesn't exceed budget
                if total_cost + player['cost_millions'] > budget:
                    continue
                
                team_count = club_counts.get(player['team_name'], 0)
                
                if team_count < max_players_per_club:
                    selected_team.append(player.to_dict())
                    selected_ids.add(player['id'])
                    total_cost += player['cost_millions']
                    remaining_budget = budget - total_cost
                    club_counts[player['team_name']] = team_count + 1
                    position_added += 1
                    
                    # Update requirements
                    requirements[position] -= 1
                    
                    # Recalculate budget constraints for remaining players
                    total_remaining_players = sum(requirements.values())
                    if total_remaining_players > 0:
                        avg_budget_per_player = remaining_budget / total_remaining_players
                        max_budget_per_position_player = min(avg_budget_per_player * 2, remaining_budget / (needed - position_added)) if (needed - position_added) > 0 else remaining_budget
        
        # Final validation
        st.info(f"📊 Final squad: {len(selected_team)} players, Total cost: £{total_cost:.1f}m / £{budget:.1f}m")
        
        # Debug information if squad building fails
        if len(selected_team) != 15:
            st.error(f"❌ Could only select {len(selected_team)} players out of 15 required")
            st.write("**Position breakdown:**")
            position_count = {}
            for player in selected_team:
                pos = player['position_name']
                position_count[pos] = position_count.get(pos, 0) + 1
            
            for position, count in position_count.items():
                required = formation_requirements[formation].get(position, 0)
                st.write(f"• {position}: {count}/{required}")
            
            st.write(f"**Budget used:** £{total_cost:.1f}m / £{budget:.1f}m")
            return None
        
        # Final budget check
        if total_cost > budget:
            st.error(f"❌ Squad exceeds budget: £{total_cost:.1f}m > £{budget:.1f}m")
            return None
        
        st.success(f"✅ Squad built successfully within budget: £{total_cost:.1f}m / £{budget:.1f}m")
        return selected_team

    def get_strategy_name(self) -> str:
        return "Premium Heavy"

class BalancedStrategy(TeamBuildingStrategy):
    """Balanced team building strategy"""
    
    def build_team(self, players_df: pd.DataFrame, formation: str, budget: float, **kwargs) -> Optional[List[Dict]]:
        max_players_per_club = kwargs.get('max_players_per_club', 3)
        
        # CORRECT formation requirements for 15-player squad
        requirements = {
            'Goalkeeper': 2, 
            'Defender': 5, 
            'Midfielder': 5, 
            'Forward': 3
        }
        
        # IMPROVED: Dynamic budget allocation based on remaining needs
        selected_squad = []
        selected_ids = set()
        club_counts = {}
        total_cost = 0.0
        
        # Process positions in order of priority (most expensive first for balanced approach)
        position_order = ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
        
        for position in position_order:
            needed = requirements[position]
            
            # Calculate remaining budget and players
            remaining_players = sum(requirements.values()) - len(selected_squad)
            remaining_budget = budget - total_cost
            
            if remaining_players <= 0:
                break
            
            # Set budget constraints for this position
            avg_budget_per_remaining_player = remaining_budget / remaining_players
            max_budget_for_position = avg_budget_per_remaining_player * needed * 1.5  # Allow some flexibility
            
            available_players = players_df[
                (players_df['position_name'] == position) & 
                (~players_df['id'].isin(selected_ids)) &
                (players_df['cost_millions'] <= remaining_budget)  # Can't exceed remaining budget
            ].copy()
            
            if available_players.empty:
                st.warning(f"⚠️ No available {position}s within remaining budget £{remaining_budget:.1f}m")
                continue
            
            # Sort by points per million for balanced approach
            available_players = available_players.sort_values('points_per_million', ascending=False)
            
            position_selections = 0
            for _, player in available_players.iterrows():
                if position_selections >= needed:
                    break
                
                # STRICT BUDGET CHECK
                if total_cost + player['cost_millions'] > budget:
                    continue
                
                player_team = player['team_name']
                current_team_count = club_counts.get(player_team, 0)
                
                # Check constraints
                if current_team_count < max_players_per_club:
                    selected_squad.append(player.to_dict())
                    selected_ids.add(player['id'])
                    total_cost += player['cost_millions']
                    club_counts[player_team] = current_team_count + 1
                    position_selections += 1
                    
                    # Update requirements
                    requirements[position] -= 1
        
        # Final validation
        if len(selected_squad) == 15 and total_cost <= budget:
            st.success(f"✅ Balanced squad built: £{total_cost:.1f}m / £{budget:.1f}m")
            return selected_squad
        else:
            st.error(f"❌ Balanced strategy failed: {len(selected_squad)} players, £{total_cost:.1f}m")
            return None

    def get_strategy_name(self) -> str:
        return "Balanced"

class SimpleStrategy(TeamBuildingStrategy):
    """Simple strategy that just picks cheapest valid squad"""
    
    def build_team(self, players_df: pd.DataFrame, formation: str, budget: float, **kwargs) -> Optional[List[Dict]]:
        max_players_per_club = kwargs.get('max_players_per_club', 3)
        
        requirements = {
            'Goalkeeper': 2, 
            'Defender': 5, 
            'Midfielder': 5, 
            'Forward': 3
        }
        
        selected_squad = []
        selected_ids = set()
        club_counts = {}
        total_cost = 0.0
        
        # Simple strategy: Fill each position with cheapest available players
        # Process in order to ensure budget is managed properly
        for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
            needed = requirements[position]
            
            available_players = players_df[
                (players_df['position_name'] == position) & 
                (~players_df['id'].isin(selected_ids))
            ].copy()
            
            if available_players.empty:
                continue
            
            # Sort by cost (ascending) to get cheapest first
            available_players = available_players.sort_values('cost_millions', ascending=True)
            
            position_selections = 0
            for _, player in available_players.iterrows():
                if position_selections >= needed:
                    break
                
                # STRICT BUDGET CHECK
                if total_cost + player['cost_millions'] > budget:
                    continue
                
                player_team = player['team_name']
                current_team_count = club_counts.get(player_team, 0)
                
                if current_team_count < max_players_per_club:
                    selected_squad.append(player.to_dict())
                    selected_ids.add(player['id'])
                    total_cost += player['cost_millions']
                    club_counts[player_team] = current_team_count + 1
                    position_selections += 1
        
        # Final validation
        if len(selected_squad) == 15 and total_cost <= budget:
            st.success(f"✅ Simple squad built: £{total_cost:.1f}m / £{budget:.1f}m")
            return selected_squad
        else:
            st.error(f"❌ Simple strategy failed: {len(selected_squad)} players, £{total_cost:.1f}m")
            return None

    def get_strategy_name(self) -> str:
        return "Simple"

class TeamBuilder:
    """Team builder using strategy pattern"""
    
    def __init__(self):
        self.strategies = {
            'premium_heavy': PremiumHeavyStrategy(),
            'balanced': BalancedStrategy(),
            'simple': SimpleStrategy(),  # Add the new strategy
        }
    
    def build_team(self, strategy_name: str, players_df: pd.DataFrame, formation: str, budget: float, **kwargs) -> Optional[Squad]:
        """Build team using specified strategy"""
        
        strategy = self.strategies.get(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        players_data = strategy.build_team(players_df, formation, budget, **kwargs)
        
        if players_data:
            players = [Player(data) for data in players_data]
            return Squad(players, formation)
        
        return None
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategies"""
        return list(self.strategies.keys())

class AdvancedTeamBuilder(TeamBuilder):
    """Enhanced team builder with advanced strategies"""
    
    def __init__(self):
        super().__init__()
        # Use existing strategies for now, can be extended later
    
    def build_team_with_analysis(self, strategy_name: str, players_df: pd.DataFrame, 
                               formation: str, budget: float, **kwargs) -> Tuple[Optional[Squad], Dict]:
        """Build team and return detailed analysis"""
        
        # Map enhanced_advanced to balanced for now
        if strategy_name == 'enhanced_advanced':
            strategy_name = 'balanced'
        
        # Build squad using parent class method
        squad = self.build_team(strategy_name, players_df, formation, budget, **kwargs)
        
        analysis = {}
        if squad:
            # Use the existing squad analyzer
            analyzer = SquadAnalyzer()
            analysis = analyzer.analyze_squad(squad)
            
            # Add enhanced analysis
            enhanced_analysis = self._analyze_squad_roles(squad)
            analysis.update(enhanced_analysis)
        
        return squad, analysis
    
    def _analyze_squad_roles(self, squad: Squad) -> Dict:
        """Analyze squad composition by roles"""
        role_distribution = {}
        total_consistency = 0
        total_projected_minutes = 0
        
        for player in squad.players:
            # Use position as role for now (can be enhanced later)
            role_name = player.position_name
            role_distribution[role_name] = role_distribution.get(role_name, 0) + 1
            
            # Placeholder metrics (would be calculated from enhanced data in full implementation)
            total_consistency += 0.6  # Placeholder consistency score
            total_projected_minutes += 0.8  # Placeholder projected minutes
        
        num_players = len(squad.players)
        return {
            'role_distribution': role_distribution,
            'average_consistency': total_consistency / num_players if num_players > 0 else 0,
            'average_projected_minutes': total_projected_minutes / num_players if num_players > 0 else 0,
            'squad_balance_score': self._calculate_balance_score(role_distribution)
        }
    
    def _calculate_balance_score(self, role_distribution: Dict) -> float:
        """Calculate squad balance score based on position distribution"""
        total_players = sum(role_distribution.values())
        if total_players == 0:
            return 0.0
        
        # Expected distribution for a 15-player squad
        expected_distribution = {
            'Goalkeeper': 2, 
            'Defender': 5, 
            'Midfielder': 5, 
            'Forward': 3
        }
        
        balance_score = 0
        total_expected = sum(expected_distribution.values())
        
        for position, expected in expected_distribution.items():
            actual = role_distribution.get(position, 0)
            # Calculate how close actual is to expected (normalized)
            if expected > 0:
                ratio = min(actual / expected, 1.0)  # Cap at 100%
                balance_score += ratio * (expected / total_expected)  # Weight by importance
        
        return balance_score

# ============================================================================
# 4. LISKOV SUBSTITUTION PRINCIPLE - VISUALIZATION CLASSES
# ============================================================================

class BaseVisualizer(ABC):
    """Base class for all visualizations"""
    
    @abstractmethod
    def create_visualization(self, data: pd.DataFrame, **kwargs):
        pass

class CorrelationVisualizer(BaseVisualizer):
    """Handles correlation heatmap visualization"""
    
    def create_visualization(self, data: pd.DataFrame, **kwargs):
        """Create correlation heatmap with robust data handling"""
        try:
            # Get numeric columns only
            numeric_df = data.select_dtypes(include=[np.number]).copy()
            
            # Clean each column
            clean_numeric_df = pd.DataFrame()
            for col in numeric_df.columns:
                series = DataCleaner.clean_numeric_column(numeric_df[col])
                clean_numeric_df[col] = series
            
            # Remove zero-variance columns
            variance_check = clean_numeric_df.var()
            non_zero_variance_cols = variance_check[variance_check > 0].index
            
            if len(non_zero_variance_cols) < 2:
                st.warning("⚠️ Not enough columns with variance for meaningful correlation")
                return None
            
            clean_numeric_df = clean_numeric_df[non_zero_variance_cols]
            corr = clean_numeric_df.corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                corr, 
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm', 
                ax=ax,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8}
            )
            plt.title("Feature Correlation Heatmap")
            plt.tight_layout()
            
            return fig, corr
            
        except Exception as e:
            st.error(f"❌ Error creating correlation heatmap: {str(e)}")
            return None

class PerformanceVisualizer(BaseVisualizer):
    """Handles performance scatter plots"""
    
    def create_visualization(self, data: pd.DataFrame, **kwargs):
        """Create performance visualization"""
        fig = px.scatter(
            data, 
            x='cost_millions', 
            y='total_points', 
            color='team_name',
            size='points_per_game',
            hover_name='web_name',
            title="Player Performance: Cost vs Total Points",
            labels={"cost_millions": "Cost (£m)", "total_points": "Total Points"}
        )
        return fig

class TeamPerformanceVisualizer(BaseVisualizer):
    """Handles team performance visualizations"""
    
    def create_visualization(self, data: pd.DataFrame, **kwargs):
        """Create team performance visualization"""
        team_performance = data.groupby('team_name').agg({
            'total_points': 'sum',
            'goals_scored': 'sum',
            'assists': 'sum',
            'clean_sheets': 'sum',
            'now_cost': 'mean'
        }).reset_index()
        
        top_teams = team_performance.nlargest(10, 'total_points')
        fig = px.bar(
            top_teams, 
            x='team_name', 
            y='total_points',
            title="Top Teams by Total Points",
            labels={"total_points": "Total Points", "team_name": "Team Name"}
        )
        return fig

class DistributionVisualizer(BaseVisualizer):
    """Handles distribution visualizations"""
    
    def create_visualization(self, data: pd.DataFrame, **kwargs):
        """Create distribution visualization"""
        column = kwargs.get('column', 'total_points')
        title = kwargs.get('title', f'Distribution of {column.replace("_", " ").title()}')
        
        fig = px.histogram(
            data, 
            x=column, 
            nbins=kwargs.get('bins', 30),
            title=title,
            labels={column: column.replace("_", " ").title()}
        )
        return fig

class PlayerComparisonVisualizer(BaseVisualizer):
    """Handles player comparison visualizations"""
    
    def create_visualization(self, data: pd.DataFrame, **kwargs):
        """Create player comparison radar chart"""
        selected_players = kwargs.get('selected_players', [])
        
        if not selected_players:
            return None
        
        # Filter data for selected players
        filtered_data = data[data['web_name'].isin(selected_players)]
        
        # Create radar chart comparing key metrics
        metrics = ['total_points', 'form', 'points_per_million', 'selected_by_percent']
        
        fig = go.Figure()
        
        for _, player in filtered_data.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[player[metric] for metric in metrics],
                theta=metrics,
                fill='toself',
                name=player['web_name']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True)
            ),
            showlegend=True,
            title="Player Comparison Radar Chart"
        )
        
        return fig

# ============================================================================
# INTERFACE SEGREGATION PRINCIPLE - SPECIALIZED INTERFACES
# ============================================================================

class IDataValidator(ABC):
    """Interface for data validation"""
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> bool:
        pass

class ISquadAnalyzer(ABC):
    """Interface for squad analysis"""
    
    @abstractmethod
    def analyze_squad(self, squad: Squad) -> Dict:
        pass

class IUIRenderer(ABC):
    """Interface for UI rendering"""
    
    @abstractmethod
    def render(self, data: Any) -> None:
        pass

class IRecommendationEngine(ABC):
    """Interface for recommendation engines"""
    
    @abstractmethod
    def get_recommendations(self, data: pd.DataFrame, **kwargs) -> List[Dict]:
        pass

# ============================================================================
# 5. DEPENDENCY INVERSION PRINCIPLE - CONCRETE IMPLEMENTATIONS
# ============================================================================

class SquadAnalyzer(ISquadAnalyzer):
    """Concrete implementation of squad analysis"""
    
    def analyze_squad(self, squad: Squad) -> Dict:
        """Analyze squad composition and performance metrics"""
        
        analysis = {
            'total_cost': squad.total_cost,
            'starting_xi_cost': squad.starting_xi_cost,
            'bench_cost': squad.bench_cost,
            'position_distribution': self._analyze_positions(squad),
            'team_distribution': self._analyze_teams(squad),
            'value_metrics': self._calculate_value_metrics(squad),
            'form_analysis': self._analyze_form(squad)
        }
        
        return analysis
    
    def _analyze_positions(self, squad: Squad) -> Dict:
        """Analyze position distribution"""
        positions = {}
        for player in squad.players:
            pos = player.position_name
            positions[pos] = positions.get(pos, 0) + 1
        return positions
    
    def _analyze_teams(self, squad: Squad) -> Dict:
        """Analyze team distribution"""
        teams = {}
        for player in squad.players:
            team = player.team_name
            teams[team] = teams.get(team, 0) + 1
        return teams
    
    def _calculate_value_metrics(self, squad: Squad) -> Dict:
        """Calculate value-based metrics"""
        total_points = sum(p.total_points for p in squad.players)
        total_cost = squad.total_cost
        
        return {
            'total_points': total_points,
            'points_per_million': total_points / total_cost if total_cost > 0 else 0,
            'average_player_cost': total_cost / len(squad.players) if squad.players else 0
        }
    
    def _analyze_form(self, squad: Squad) -> Dict:
        """Analyze squad form"""
        forms = [p.form for p in squad.players]
        
        if not forms:
            return {}
        
        return {
            'average_form': sum(forms) / len(forms),
            'best_form': max(forms),
            'worst_form': min(forms)
        }

class DataValidator(IDataValidator):
    """Concrete implementation of data validation"""
    
    def validate(self, data: pd.DataFrame) -> bool:
        """Validate FPL data structure and content"""
        
        required_columns = ['id', 'web_name', 'total_points', 'cost_millions']
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.warning(f"Missing required columns: {missing_columns}")
            return False
        
        # Check if data is not empty
        if data.empty:
            st.warning("Data is empty")
            return False
        
        # Check for reasonable data ranges
        if 'cost_millions' in data.columns:
            if data['cost_millions'].min() < 3.0 or data['cost_millions'].max() > 20.0:
                st.warning("Cost data seems out of reasonable range")
        
        return True

# Add these missing classes after the DataValidator class:

@dataclass
class TeamBuildingConfig:
    """Configuration for team building"""
    total_budget: float
    formation: str
    max_players_per_club: int
    strategy: str
    target_premiums: List[str]

@dataclass
class AdvancedWeights:
    """Advanced weights for team building"""
    consistency_weight: float
    role_weight: float
    fixture_weight: float
    form_weight: float
    minutes_weight: float
    bps_weight: float

@dataclass
class RiskManagementSettings:
    """Risk management settings"""
    max_differential_players: int
    min_captain_options: int
    injury_risk_threshold: float
    rotation_risk_threshold: float

class ConfigurationManager:
    """Manages configuration creation"""
    
    def create_team_config(self, total_budget: float, formation: str, 
                          max_players_per_club: int, strategy: str, 
                          target_premiums: List[str]) -> TeamBuildingConfig:
        """Create team building configuration"""
        return TeamBuildingConfig(
            total_budget=total_budget,
            formation=formation,
            max_players_per_club=max_players_per_club,
            strategy=strategy,
            target_premiums=target_premiums
        )

class FPLTeamImporter:
    """Handles importing user's current FPL team with better error handling"""
    
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api"
    
    def get_user_team(self, team_id: int, gameweek: int = None) -> Optional[Dict]:
        """Get user's team data from FPL API with improved error handling"""
        try:
            # Get current gameweek if not specified
            if gameweek is None:
                gameweek = self._get_current_gameweek()
            
            st.info(f"🔍 Trying to fetch team data for gameweek {gameweek}...")
            
            # Try multiple URL patterns and gameweeks
            team_data = self._try_multiple_gameweeks(team_id, gameweek)
            
            if team_data:
                return team_data
            else:
                st.error("❌ Could not fetch team data for any recent gameweeks")
                return None
                
        except Exception as e:
            st.error(f"Error fetching team data: {str(e)}")
            return None
    
    def _try_multiple_gameweeks(self, team_id: int, starting_gameweek: int) -> Optional[Dict]:
        """Try to fetch team data from multiple gameweeks"""
        
        # Try current gameweek and previous few gameweeks
        gameweeks_to_try = [starting_gameweek]
        
        # Add previous gameweeks (go back up to 5 gameweeks)
        for i in range(1, 6):
            if starting_gameweek - i >= 1:
                gameweeks_to_try.append(starting_gameweek - i)
        
        # Add next gameweek in case current is not ready
        if starting_gameweek + 1 <= 38:
            gameweeks_to_try.insert(1, starting_gameweek + 1)
        
        for gw in gameweeks_to_try:
            st.info(f"🔍 Trying gameweek {gw}...")
            
            # Try the standard picks URL
            picks_url = f"{self.base_url}/entry/{team_id}/event/{gw}/picks/"
            
            try:
                response = requests.get(picks_url, verify=False, timeout=30)
                
                if response.status_code == 200:
                    team_data = response.json()
                    if team_data and 'picks' in team_data and team_data['picks']:
                        st.success(f"✅ Found team data for gameweek {gw}")
                        return team_data
                    else:
                        st.warning(f"⚠️ Gameweek {gw} data is empty")
                elif response.status_code == 404:
                    st.warning(f"⚠️ Gameweek {gw} not found (404)")
                else:
                    st.warning(f"⚠️ Gameweek {gw} returned status {response.status_code}")
                    
            except Exception as e:
                st.warning(f"⚠️ Error fetching gameweek {gw}: {str(e)[:50]}...")
                continue
        
        return None
    
    def _get_current_gameweek(self) -> int:
        """Get the current gameweek with better logic"""
        try:
            response = requests.get(f"{self.base_url}/bootstrap-static/", verify=False, timeout=30)
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                
                # Find current gameweek
                for event in events:
                    if event.get('is_current', False):
                        return event['id']
                
                # If no current gameweek, find the next one
                for event in events:
                    if event.get('is_next', False):
                        return event['id']
                
                # Find most recent finished gameweek
                finished_events = [e for e in events if e.get('finished', False)]
                if finished_events:
                    latest_finished = max(finished_events, key=lambda x: x['id'])
                    return latest_finished['id']
                
                return 1  # Fallback
            else:
                return 1
        except Exception:
            return 1
    
    def get_team_info(self, team_id: int) -> Optional[Dict]:
        """Get basic team information"""
        try:
            response = requests.get(f"{self.base_url}/entry/{team_id}/", verify=False, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            st.error(f"Error fetching team info: {str(e)}")
            return None
    
    def process_team_data(self, team_data: Dict, players_df: pd.DataFrame) -> Dict:
        """Process raw team data into readable format"""
        if not team_data or 'picks' not in team_data:
            return {}
        
        processed_team = {
            'starting_xi': [],
            'bench': [],
            'captain_id': None,
            'vice_captain_id': None,
            'total_cost': 0,
            'formation': '4-4-2'
        }
        
        return processed_team  # Simplified for now

class TeamAnalyzer:
    """Analyzes imported team performance"""
    
    def analyze_team(self, team_data: Dict, players_df: pd.DataFrame) -> Dict:
        """Basic team analysis"""
        return {
            'team_summary': {'total_cost': team_data.get('total_cost', 0)},
            'position_analysis': {},
            'team_distribution': {},
            'value_analysis': {'total_points': 0, 'points_per_million': 0},
            'recommendations': []
        }

class AIPlayerRecommendationEngine(IRecommendationEngine):
    """AI-powered player recommendation engine"""
    
    def get_recommendations(self, data: pd.DataFrame, **kwargs) -> List[Dict]:
        """Get basic recommendations"""
        return []
    
    def get_similar_players(self, target_player: Dict, players_df: pd.DataFrame, 
                          n_recommendations: int = 5) -> List[Dict]:
        """Find similar players"""
        return []

# ============================================================================
# MAIN APPLICATION CLASS
# ============================================================================

class FPLAnalyticsApp:
    """Main FPL Analytics Application"""
    
    def __init__(self):
        """Initialize the application"""
        self.initialize_session_state()
        self.data_loader = DataLoader()
        self.team_builder = AdvancedTeamBuilder()
        self.config_manager = ConfigurationManager()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'players_df' not in st.session_state:
            st.session_state.players_df = pd.DataFrame()
        if 'teams_df' not in st.session_state:
            st.session_state.teams_df = pd.DataFrame()
        if 'raw_data' not in st.session_state:
            st.session_state.raw_data = {}
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
    
    def run(self):
        """Main application entry point"""
        st.set_page_config(page_title="FPL Advanced Analytics", page_icon="⚽", layout="wide")
        st.title("⚽ FPL Advanced Analytics Dashboard")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🏠 Dashboard", 
            "📊 Player Analysis", 
            "🔍 Advanced Filters",
            "📈 Visualizations",
            "🏆 Team Builder",
            "📥 Import My Team",
            "🤖 AI Recommendations"
        ])
        
        with tab1:
            self._render_enhanced_dashboard()
        
        with tab2:
            self._render_player_analysis()
        
        with tab3:
            self._render_advanced_filters()
        
        with tab4:
            self._render_visualizations()
        
        with tab5:
            self._render_team_builder()
        
        with tab6:
            self._render_import_my_team()
        
        with tab7:
            self._render_ai_recommendations()
    
    def _render_enhanced_dashboard(self):
        """Render the enhanced dashboard with all previous features"""
        st.header("🏠 FPL Analytics Dashboard")
        
        # Data loading section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📊 Data Overview")
            
        with col2:
            if st.button("🔄 Load/Refresh Data", type="primary"):
                with st.spinner("Loading FPL data..."):
                    try:
                        players_df, teams_df, raw_data = self.data_loader.load_fpl_data()
                        
                        # Store in session state
                        st.session_state.players_df = players_df
                        st.session_state.teams_df = teams_df
                        st.session_state.raw_data = raw_data
                        st.session_state.data_loaded = True
                        
                        st.success("✅ Data loaded successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error loading data: {str(e)}")
        
        # Display data info if loaded
        if st.session_state.data_loaded and not st.session_state.players_df.empty:
            
            # Key Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Players", len(st.session_state.players_df))
            with col2:
                st.metric("Total Teams", len(st.session_state.teams_df))
            with col3:
                avg_cost = st.session_state.players_df['cost_millions'].mean()
                st.metric("Avg Player Cost", f"£{avg_cost:.1f}m")
            with col4:
                total_points = st.session_state.players_df['total_points'].sum()
                st.metric("Total Points", f"{total_points:,}")
            
            st.divider()
            
            # Quick Insights Section
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 Top Performers")
                
                # Top scorers
                top_scorers = st.session_state.players_df.nlargest(5, 'total_points')[
                    ['web_name', 'team_name', 'position_name', 'total_points', 'cost_millions']
                ]
                
                for _, player in top_scorers.iterrows():
                    st.write(f"**{player['web_name']}** ({player['team_name']}) - {player['total_points']} pts (£{player['cost_millions']:.1f}m)")
            
            with col2:
                st.subheader("💰 Best Value Players")
                
                # Best value (points per million)
                best_value = st.session_state.players_df.nlargest(5, 'points_per_million')[
                    ['web_name', 'team_name', 'position_name', 'points_per_million', 'cost_millions']
                ]
                
                for _, player in best_value.iterrows():
                    st.write(f"**{player['web_name']}** ({player['team_name']}) - {player['points_per_million']:.1f} pts/£m (£{player['cost_millions']:.1f}m)")
            
            st.divider()
            
            # Position Analysis
            st.subheader("📊 Position Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']
            for i, pos in enumerate(positions):
                pos_players = st.session_state.players_df[st.session_state.players_df['position_name'] == pos]
                
                with [col1, col2, col3, col4][i]:
                    st.metric(
                        f"{pos}s",
                        len(pos_players),
                        delta=f"Avg: £{pos_players['cost_millions'].mean():.1f}m"
                    )
                    
                    # Top player in position
                    if not pos_players.empty:
                        top_player = pos_players.nlargest(1, 'total_points').iloc[0]
                        st.caption(f"Top: {top_player['web_name']} ({top_player['total_points']} pts)")
            
            st.divider()
            
            # Team Analysis
            st.subheader("🏆 Team Performance")
            
            team_performance = st.session_state.players_df.groupby('team_name').agg({
                'total_points': 'sum',
                'cost_millions': 'mean',
                'web_name': 'count'
            }).round(2).reset_index()
            
            team_performance.columns = ['Team', 'Total Points', 'Avg Cost (£m)', 'Player Count']
            team_performance = team_performance.sort_values('Total Points', ascending=False)
            
            # Display top 10 teams
            st.dataframe(team_performance.head(10), use_container_width=True)
            
            st.divider()
            
            # Quick Filters Section
            st.subheader("🔍 Quick Player Search")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                search_name = st.text_input("Search by name", placeholder="e.g., Haaland")
            
            with col2:
                search_position = st.selectbox("Filter by position", ["All"] + positions)
            
            with col3:
                max_cost = st.slider("Max cost (£m)", 3.5, 15.0, 15.0, 0.5)
            
            # Apply filters
            filtered_df = st.session_state.players_df.copy()
            
            if search_name:
                filtered_df = filtered_df[
                    filtered_df['web_name'].str.contains(search_name, case=False, na=False)
                ]
            
            if search_position != "All":
                filtered_df = filtered_df[filtered_df['position_name'] == search_position]
            
            filtered_df = filtered_df[filtered_df['cost_millions'] <= max_cost]
            
            if not filtered_df.empty:
                st.subheader(f"🎯 Search Results ({len(filtered_df)} players)")
                
                # Display results
                display_columns = ['web_name', 'team_name', 'position_name', 'cost_millions', 'total_points', 'points_per_million']
                available_columns = [col for col in display_columns if col in filtered_df.columns]
                
                result_df = filtered_df[available_columns].sort_values('total_points', ascending=False).head(10)
                st.dataframe(result_df, use_container_width=True)
            
            elif search_name or search_position != "All":
                st.info("No players match your search criteria")
            
            st.divider()
            
            # Sample Data Section
            st.subheader("📋 Sample Player Data")
            sample_columns = ['web_name', 'team_name', 'position_name', 'cost_millions', 'total_points', 'form', 'selected_by_percent']
            available_sample_columns = [col for col in sample_columns if col in st.session_state.players_df.columns]
            st.dataframe(st.session_state.players_df[available_sample_columns].head(20), use_container_width=True)
            
        else:
            st.info("👆 Click 'Load/Refresh Data' to start using the app")
            
            # Show app features when no data loaded
            st.subheader("🚀 Features Available")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("**📊 Player Analysis**\n\nAdvanced filtering and sorting of players by various metrics")
            
            with col2:
                st.info("**🏆 Team Builder**\n\nBuild optimal FPL teams using different strategies")
            
            with col3:
                st.info("**📈 Visualizations**\n\nInteractive charts and graphs for data analysis")

    def _render_player_analysis(self):
        """Render enhanced player analysis tab with advanced features"""
        st.header("📊 Advanced Player Analysis")
        
        if st.session_state.players_df.empty:
            st.info("Load data in the Dashboard tab first")
            return
        
        # Create sub-tabs for different analysis types
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs([
            "🔍 Smart Filters", "📈 Performance Metrics", "⚖️ Player Comparison", "🎯 Position Deep Dive"
        ])
        
        with analysis_tab1:
            self._render_smart_filters()
        
        with analysis_tab2:
            self._render_performance_metrics()
        
        with analysis_tab3:
            self._render_player_comparison()
        
        with analysis_tab4:
            self._render_position_deep_dive()

    def _render_smart_filters(self):
        """Render smart filtering section with intelligent recommendations"""
        st.subheader("🔍 Smart Player Filtering")
        
        # Quick filter presets
        st.write("**🚀 Quick Filter Presets:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💎 Hidden Gems", help="Low ownership, high value players"):
                st.session_state.filter_preset = "hidden_gems"
        
        with col2:
            if st.button("🔥 In-Form Players", help="Players with excellent recent form"):
                st.session_state.filter_preset = "in_form"
        
        with col3:
            if st.button("💰 Budget Options", help="Affordable players with good returns"):
                st.session_state.filter_preset = "budget"
        
        with col4:
            if st.button("⭐ Premium Players", help="High-cost, high-performing players"):
                st.session_state.filter_preset = "premium"
        
        st.divider()
        
        # Advanced filtering interface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🎯 Basic Filters**")
            
            position_filter = st.selectbox(
                "Position", 
                ["All"] + list(st.session_state.players_df['position_name'].unique()),
                key="pos_filter"
            )
            
            team_filter = st.selectbox(
                "Team", 
                ["All"] + sorted(list(st.session_state.players_df['team_name'].unique())),
                key="team_filter"
            )
            
            cost_range = st.slider(
                "Cost Range (£m)", 
                float(st.session_state.players_df['cost_millions'].min()),
                float(st.session_state.players_df['cost_millions'].max()),
                (4.0, 12.0), 
                step=0.1,
                key="cost_filter"
            )
        
        with col2:
            st.write("**📊 Performance Filters**")
            
            points_range = st.slider(
                "Total Points Range",
                int(st.session_state.players_df['total_points'].min()),
                int(st.session_state.players_df['total_points'].max()),
                (30, int(st.session_state.players_df['total_points'].max())),
                step=5,
                key="points_filter"
            )
            
            form_range = st.slider(
                "Form Range",
                float(st.session_state.players_df['form'].min()),
                float(st.session_state.players_df['form'].max()),
                (3.0, float(st.session_state.players_df['form'].max())),
                step=0.1,
                key="form_filter"
            )
            
            value_threshold = st.slider(
                "Min Points per Million",
                0.0, 20.0, 5.0, 0.5,
                key="value_filter"
            )
        
        with col3:
            st.write("**🎮 Ownership & Strategy**")
            
            ownership_range = st.slider(
                "Ownership % Range",
                float(st.session_state.players_df['selected_by_percent'].min()),
                float(st.session_state.players_df['selected_by_percent'].max()),
                (0.0, 100.0),
                step=1.0,
                key="ownership_filter"
            )
            
            # Strategy-based filters
            strategy_filter = st.selectbox(
                "Strategy Focus",
                ["All", "Differentials (Low Ownership)", "Safe Picks (High Ownership)", "Value Hunters", "Form Chasers"],
                key="strategy_filter"
            )
            
            # Availability filter
            availability = st.selectbox(
                "Player Availability",
                ["All", "Available (Not Injured)", "Doubtful", "Injured"],
                key="availability_filter"
            )
        
        # Apply preset filters if selected
        if 'filter_preset' in st.session_state:
            preset = st.session_state.filter_preset
            if preset == "hidden_gems":
                filtered_df = st.session_state.players_df[
                    (st.session_state.players_df['selected_by_percent'] < 10) &
                    (st.session_state.players_df['points_per_million'] > 6) &
                    (st.session_state.players_df['total_points'] > 30)
                ]
            elif preset == "in_form":
                filtered_df = st.session_state.players_df[
                    (st.session_state.players_df['form'] > 7) &
                    (st.session_state.players_df['total_points'] > 40)
                ]
            elif preset == "budget":
                filtered_df = st.session_state.players_df[
                    (st.session_state.players_df['cost_millions'] <= 6.0) &
                    (st.session_state.players_df['points_per_million'] > 7)
                ]
            elif preset == "premium":
                filtered_df = st.session_state.players_df[
                    (st.session_state.players_df['cost_millions'] >= 9.0) &
                    (st.session_state.players_df['total_points'] > 80)
                ]
            else:
                filtered_df = st.session_state.players_df.copy()
            
            # Clear preset after applying
            del st.session_state.filter_preset
        else:
            # Apply manual filters
            filtered_df = st.session_state.players_df.copy()
            
            # Basic filters
            if position_filter != "All":
                filtered_df = filtered_df[filtered_df['position_name'] == position_filter]
            
            if team_filter != "All":
                filtered_df = filtered_df[filtered_df['team_name'] == team_filter]
            
            # Range filters
            filtered_df = filtered_df[
                (filtered_df['cost_millions'] >= cost_range[0]) &
                (filtered_df['cost_millions'] <= cost_range[1]) &
                (filtered_df['total_points'] >= points_range[0]) &
                (filtered_df['total_points'] <= points_range[1]) &
                (filtered_df['form'] >= form_range[0]) &
                (filtered_df['form'] <= form_range[1]) &
                (filtered_df['points_per_million'] >= value_threshold) &
                (filtered_df['selected_by_percent'] >= ownership_range[0]) &
                (filtered_df['selected_by_percent'] <= ownership_range[1])
            ]
            
            # Strategy filters
            if strategy_filter == "Differentials (Low Ownership)":
                filtered_df = filtered_df[filtered_df['selected_by_percent'] < 15]
            elif strategy_filter == "Safe Picks (High Ownership)":
                filtered_df = filtered_df[filtered_df['selected_by_percent'] > 25]
            elif strategy_filter == "Value Hunters":
                filtered_df = filtered_df[filtered_df['points_per_million'] > 8]
            elif strategy_filter == "Form Chasers":
                filtered_df = filtered_df[filtered_df['form'] > 6]
        
        # Display results
        st.divider()
        
        if not filtered_df.empty:
            st.subheader(f"🎯 Filtered Results ({len(filtered_df)} players)")
            
            # Sorting options
            col1, col2, col3 = st.columns(3)
            with col1:
                sort_by = st.selectbox(
                    "Sort by",
                    ["points_per_million", "total_points", "form", "cost_millions", "selected_by_percent", "ai_score"],
                    key="sort_filter"
                )
            with col2:
                sort_order = st.selectbox("Sort order", ["Descending", "Ascending"], key="sort_order")
            with col3:
                show_count = st.selectbox("Show top", [10, 20, 50, 100], key="show_count")
            
            # Calculate AI score for better sorting
            if 'ai_score' not in filtered_df.columns:
                filtered_df = filtered_df.copy()
                filtered_df['ai_score'] = (
                    filtered_df['points_per_million'].rank(pct=True) * 0.3 +
                    filtered_df['form'].rank(pct=True) * 0.25 +
                    filtered_df['total_points'].rank(pct=True) * 0.25 +
                    (100 - filtered_df['selected_by_percent']).rank(pct=True) * 0.2
                )
            
            ascending = sort_order == "Ascending"
            display_df = filtered_df.sort_values(sort_by, ascending=ascending).head(show_count)
            
            # Enhanced display with additional metrics
            display_columns = [
                'web_name', 'position_name', 'team_name', 'cost_millions', 
                'total_points', 'form', 'points_per_million', 'selected_by_percent'
            ]
            
            if sort_by == 'ai_score':
                display_columns.append('ai_score')
            
            available_columns = [col for col in display_columns if col in display_df.columns]
            
            # Format the dataframe for better display
            display_df_formatted = display_df[available_columns].copy()
            
            # Round numerical columns
            for col in ['cost_millions', 'form', 'points_per_million', 'selected_by_percent']:
                if col in display_df_formatted.columns:
                    display_df_formatted[col] = display_df_formatted[col].round(1)
            
            if 'ai_score' in display_df_formatted.columns:
                display_df_formatted['ai_score'] = display_df_formatted['ai_score'].round(2)
            
            st.dataframe(display_df_formatted, use_container_width=True)
            
            # Quick insights about filtered results
            if len(filtered_df) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_cost = filtered_df['cost_millions'].mean()
                    st.metric("Avg Cost", f"£{avg_cost:.1f}m")
                
                with col2:
                    avg_points = filtered_df['total_points'].mean()
                    st.metric("Avg Points", f"{avg_points:.0f}")
                
                with col3:
                    avg_form = filtered_df['form'].mean()
                    st.metric("Avg Form", f"{avg_form:.1f}")
                
                with col4:
                    avg_ownership = filtered_df['selected_by_percent'].mean()
                    st.metric("Avg Ownership", f"{avg_ownership:.1f}%")
            
            # Export functionality
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Export to CSV"):
                    csv = display_df_formatted.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="fpl_filtered_players.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("🔄 Clear All Filters"):
                    # Reset filter states
                    for key in st.session_state.keys():
                        if key.endswith('_filter'):
                            del st.session_state[key]
                    st.rerun()
        
        else:
            st.warning("❌ No players match your current filters. Try adjusting the criteria.")
            
            # Suggestions for better filtering
            with st.expander("💡 Filter Suggestions"):
                st.write("**Try these adjustments:**")
                st.write("• Increase the cost range")
                st.write("• Lower the minimum points threshold")
                st.write("• Expand the form range")
                st.write("• Include more teams or positions")
                st.write("• Use the Quick Filter Presets above")

    def _render_performance_metrics(self):
        """Render detailed performance metrics analysis"""
        st.subheader("📈 Performance Metrics Analysis")
        
        # Metric selection
        col1, col2 = st.columns(2)
        
        with col1:
            primary_metric = st.selectbox(
                "Primary Metric",
                ["total_points", "points_per_million", "form", "goals_scored", "assists", "clean_sheets"],
                key="primary_metric"
            )
        
        with col2:
            secondary_metric = st.selectbox(
                "Secondary Metric",
                ["cost_millions", "selected_by_percent", "minutes", "bonus", "bps"],
                key="secondary_metric"
            )
        
        # Performance analysis by position
        st.write("**📊 Performance by Position**")
        
        for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
            position_players = st.session_state.players_df[
                st.session_state.players_df['position_name'] == position
            ].copy()
            
            if not position_players.empty:
                with st.expander(f"{position} Analysis ({len(position_players)} players)"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**🏆 Top Performers**")
                        top_performers = position_players.nlargest(5, primary_metric)[
                            ['web_name', 'team_name', primary_metric, 'cost_millions']
                        ]
                        for _, player in top_performers.iterrows():
                            st.write(f"• {player['web_name']} ({player['team_name']}) - {player[primary_metric]}")
                    
                    with col2:
                        st.write("**💰 Best Value**")
                        if primary_metric != 'points_per_million':
                            # Calculate value score for this metric
                            position_players['temp_value'] = position_players[primary_metric] / position_players['cost_millions']
                            best_value = position_players.nlargest(5, 'temp_value')[
                                ['web_name', 'team_name', 'temp_value', 'cost_millions']
                            ]
                            for _, player in best_value.iterrows():
                                st.write(f"• {player['web_name']} ({player['team_name']}) - {player['temp_value']:.1f}")
                        else:
                            best_value = position_players.nlargest(5, 'points_per_million')[
                                ['web_name', 'team_name', 'points_per_million', 'cost_millions']
                            ]
                            for _, player in best_value.iterrows():
                                st.write(f"• {player['web_name']} ({player['team_name']}) - {player['points_per_million']:.1f}")
                    
                    with col3:
                        st.write("**📊 Position Stats**")
                        avg_metric = position_players[primary_metric].mean()
                        avg_cost = position_players['cost_millions'].mean()
                        max_metric = position_players[primary_metric].max()
                        
                        st.write(f"• Avg {primary_metric}: {avg_metric:.1f}")
                        st.write(f"• Avg Cost: £{avg_cost:.1f}m")
                        st.write(f"• Best {primary_metric}: {max_metric}")
                    
                    # Performance distribution chart
                    fig = px.scatter(
                        position_players,
                        x=secondary_metric,
                        y=primary_metric,
                        color='team_name',
                        size='total_points',
                        hover_name='web_name',
                        title=f"{position} - {primary_metric} vs {secondary_metric}",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

    def _render_player_comparison(self):
        """Render advanced player comparison tools"""
        st.subheader("⚖️ Advanced Player Comparison")
        
        # Player selection for comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Select Players to Compare**")
            
            # Quick selection by position
            comparison_position = st.selectbox(
                "Filter by position for comparison",
                ["All"] + list(st.session_state.players_df['position_name'].unique()),
                key="comp_position"
            )
            
            if comparison_position != "All":
                available_players = st.session_state.players_df[
                    st.session_state.players_df['position_name'] == comparison_position
                ]['web_name'].tolist()
            else:
                available_players = st.session_state.players_df['web_name'].tolist()
            
            selected_players = st.multiselect(
                "Select players (max 6)",
                options=sorted(available_players),
                max_selections=6,
                key="selected_comparison_players"
            )
        
        with col2:
            st.write("**📊 Comparison Metrics**")
            
            comparison_metrics = st.multiselect(
                "Select metrics to compare",
                options=[
                    'total_points', 'cost_millions', 'points_per_million', 'form',
                    'selected_by_percent', 'goals_scored', 'assists', 'clean_sheets',
                    'minutes', 'bonus', 'yellow_cards'
                ],
                default=['total_points', 'cost_millions', 'points_per_million', 'form'],
                key="comparison_metrics"
            )
            
            chart_type = st.selectbox(
                "Chart Type",
                ["Radar Chart", "Bar Chart", "Table Only"],
                key="chart_type"
            )
        
        if len(selected_players) >= 2 and comparison_metrics:
            st.divider()
            
            comparison_df = st.session_state.players_df[
                st.session_state.players_df['web_name'].isin(selected_players)
            ].copy()
            
            # Display comparison table
            st.write("**📋 Detailed Comparison Table**")
            
            display_cols = ['web_name', 'position_name', 'team_name'] + comparison_metrics
            available_cols = [col for col in display_cols if col in comparison_df.columns]
            
            comparison_table = comparison_df[available_cols].round(2)
            st.dataframe(comparison_table, use_container_width=True)
            
            # Visual comparison
            if chart_type == "Radar Chart" and len(comparison_metrics) >= 3:
                st.write("**🕸️ Radar Chart Comparison**")
                
                fig = go.Figure()
                
                for _, player in comparison_df.iterrows():
                    # Normalize metrics to 0-100 scale for radar chart
                    normalized_values = []
                    for metric in comparison_metrics:
                        if metric in player.index:
                            metric_data = st.session_state.players_df[metric]
                            # Normalize to percentile (0-100)
                            percentile = (player[metric] - metric_data.min()) / (metric_data.max() - metric_data.min()) * 100
                            normalized_values.append(max(0, min(100, percentile)))
                        else:
                            normalized_values.append(0)
                    
                    fig.add_trace(go.Scatterpolar(
                        r=normalized_values,
                        theta=comparison_metrics,
                        fill='toself',
                        name=player['web_name'],
                        opacity=0.7
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )
                    ),
                    showlegend=True,
                    title="Player Comparison Radar Chart (Percentile Rankings)",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "Bar Chart":
                st.write("**📊 Bar Chart Comparison**")
                
                # Create subplots for multiple metrics
                metric_count = len(comparison_metrics)
                cols = min(2, metric_count)
                rows = (metric_count + 1) // 2
                
                fig = make_subplots(
                    rows=rows, cols=cols,
                    subplot_titles=comparison_metrics,
                    specs=[[{"type": "bar"}] * cols for _ in range(rows)]
                )
                
                for i, metric in enumerate(comparison_metrics):
                    row = i // 2 + 1
                    col = i % 2 + 1
                    
                    fig.add_trace(
                        go.Bar(
                            x=comparison_df['web_name'],
                            y=comparison_df[metric],
                            name=metric,
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(height=300 * rows, title="Player Comparison Bar Charts")
                st.plotly_chart(fig, use_container_width=True)
            
            # Comparison insights
            st.write("**🔍 Comparison Insights**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🏆 Best in Each Metric:**")
                for metric in comparison_metrics[:5]:  # Show top 5 metrics
                    if metric in comparison_df.columns:
                        best_player = comparison_df.loc[comparison_df[metric].idxmax()]
                        st.write(f"• {metric}: **{best_player['web_name']}** ({best_player[metric]})")
            
            with col2:
                st.write("**💰 Value Analysis:**")
                if 'total_points' in comparison_df.columns and 'cost_millions' in comparison_df.columns:
                    comparison_df['value_score'] = comparison_df['total_points'] / comparison_df['cost_millions']
                    best_value = comparison_df.loc[comparison_df['value_score'].idxmax()]
                    st.write(f"• Best Value: **{best_value['web_name']}** ({best_value['value_score']:.1f} pts/£m)")
                
                if 'form' in comparison_df.columns:
                    best_form = comparison_df.loc[comparison_df['form'].idxmax()]
                    st.write(f"• Best Form: **{best_form['web_name']}** ({best_form['form']:.1f})")
        
        elif len(selected_players) == 1:
            st.info("Select at least 2 players to compare")
        elif not comparison_metrics:
            st.info("Select at least one metric to compare")

    def _render_position_deep_dive(self):
        """Render position-specific deep dive analysis"""
        st.subheader("🎯 Position Deep Dive Analysis")
        
        # Position selection
        selected_position = st.selectbox(
            "Choose position for detailed analysis",
            ['Goalkeeper', 'Defender', 'Midfielder', 'Forward'],
            key="deep_dive_position"
        )
        
        position_data = st.session_state.players_df[
            st.session_state.players_df['position_name'] == selected_position
        ].copy()
        
        if position_data.empty:
            st.warning(f"No {selected_position} data available")
            return
        
        # Position overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Players", len(position_data))
        with col2:
            avg_cost = position_data['cost_millions'].mean()
            st.metric("Average Cost", f"£{avg_cost:.1f}m")
        with col3:
            avg_points = position_data['total_points'].mean()
            st.metric("Average Points", f"{avg_points:.0f}")
        with col4:
            top_scorer = position_data.loc[position_data['total_points'].idxmax()]
            st.metric("Top Scorer", f"{top_scorer['web_name']} ({top_scorer['total_points']})")
        
        st.divider()
        
        # Detailed analysis tabs
        deep_tab1, deep_tab2, deep_tab3, deep_tab4 = st.tabs([
            "💰 Price Tiers", "📊 Performance Distribution", "🏆 Team Analysis", "🔥 Form Trends"
        ])
        
        with deep_tab1:
            st.write(f"**💰 {selected_position} Price Tier Analysis**")
            
            # Create price tiers
            position_data['price_tier'] = pd.cut(
                position_data['cost_millions'],
                bins=4,
                labels=['Budget (£3.5-5.5m)', 'Mid-Range (£5.5-7.5m)', 'Premium (£7.5-10m)', 'Elite (£10m+)']
            )
            
            tier_analysis = position_data.groupby('price_tier').agg({
                'web_name': 'count',
                'total_points': ['mean', 'max'],
                'points_per_million': 'mean',
                'selected_by_percent': 'mean'
            }).round(2)
            
            tier_analysis.columns = ['Count', 'Avg Points', 'Max Points', 'Avg Value', 'Avg Ownership %']
            st.dataframe(tier_analysis, use_container_width=True)
            
            # Show best players in each tier
            for tier in position_data['price_tier'].dropna().unique():
                tier_players = position_data[position_data['price_tier'] == tier]
                if not tier_players.empty:
                    best_in_tier = tier_players.nlargest(3, 'total_points')
                    
                    st.write(f"**🌟 Best {tier} Players:**")
                    for _, player in best_in_tier.iterrows():
                        st.write(f"• {player['web_name']} ({player['team_name']}) - {player['total_points']} pts, £{player['cost_millions']:.1f}m")
        
        with deep_tab2:
            st.write(f"**📊 {selected_position} Performance Distribution**")
            
            # Performance metrics specific to position
            if selected_position == 'Goalkeeper':
                key_metrics = ['total_points', 'clean_sheets', 'saves', 'bonus']
            elif selected_position == 'Defender':
                key_metrics = ['total_points', 'clean_sheets', 'goals_scored', 'assists', 'bonus']
            elif selected_position == 'Midfielder':
                key_metrics = ['total_points', 'goals_scored', 'assists', 'bonus', 'bps']
            else:  # Forward
                key_metrics = ['total_points', 'goals_scored', 'assists', 'bonus', 'bps']
            
            available_metrics = [m for m in key_metrics if m in position_data.columns]
            
            if available_metrics:
                selected_metric = st.selectbox(
                    "Select metric for distribution analysis",
                    available_metrics,
                    key="distribution_metric"
                )
                
                # Distribution histogram
                fig = px.histogram(
                    position_data,
                    x=selected_metric,
                    nbins=20,
                    title=f"{selected_position} {selected_metric} Distribution",
                    labels={selected_metric: selected_metric.replace('_', ' ').title()}
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistical summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📈 Statistical Summary:**")
                    st.write(f"• Mean: {position_data[selected_metric].mean():.2f}")
                    st.write(f"• Median: {position_data[selected_metric].median():.2f}")
                    st.write(f"• Std Dev: {position_data[selected_metric].std():.2f}")
                    st.write(f"• Min: {position_data[selected_metric].min()}")
                    st.write(f"• Max: {position_data[selected_metric].max()}")
                
                with col2:
                    st.write("**🎯 Performance Tiers:**")
                    q75 = position_data[selected_metric].quantile(0.75)
                    q50 = position_data[selected_metric].quantile(0.50)
                    q25 = position_data[selected_metric].quantile(0.25)
                    
                    elite = len(position_data[position_data[selected_metric] >= q75])
                    good = len(position_data[(position_data[selected_metric] >= q50) & (position_data[selected_metric] < q75)])
                    average = len(position_data[(position_data[selected_metric] >= q25) & (position_data[selected_metric] < q50)])
                    below = len(position_data[position_data[selected_metric] < q25])
                    
                    st.write(f"• Elite (≥{q75:.1f}): {elite} players")
                    st.write(f"• Good ({q50:.1f}-{q75:.1f}): {good} players")
                    st.write(f"• Average ({q25:.1f}-{q50:.1f}): {average} players")
                    st.write(f"• Below Average (<{q25:.1f}): {below} players")
        
        with deep_tab3:
            st.write(f"**🏆 {selected_position} Team Analysis**")
            
            # Team performance for this position
            team_stats = position_data.groupby('team_name').agg({
                'web_name': 'count',
                'total_points': ['sum', 'mean'],
                'cost_millions': 'mean',
                'selected_by_percent': 'mean'
            }).round(2)
            
            team_stats.columns = ['Player Count', 'Total Points', 'Avg Points', 'Avg Cost', 'Avg Ownership']
            team_stats = team_stats.sort_values('Total Points', ascending=False)
            
            st.dataframe(team_stats.head(10), use_container_width=True)
            
            # Best player from each top team
            st.write("**⭐ Best Player from Top Teams:**")
            top_teams = team_stats.head(10).index
            
            for team in top_teams[:5]:  # Show top 5 teams
                team_players = position_data[position_data['team_name'] == team]
                if not team_players.empty:
                    best_player = team_players.loc[team_players['total_points'].idxmax()]
                    st.write(f"• **{team}**: {best_player['web_name']} ({best_player['total_points']} pts, £{best_player['cost_millions']:.1f}m)")
        
        with deep_tab4:
            st.write(f"**🔥 {selected_position} Form Analysis**")
            
            # Form distribution
            form_bins = pd.cut(position_data['form'], bins=5, labels=['Poor', 'Below Avg', 'Average', 'Good', 'Excellent'])
            form_counts = form_bins.value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Form distribution pie chart
                fig = px.pie(
                    values=form_counts.values,
                    names=form_counts.index,
                    title=f"{selected_position} Form Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top form players
                st.write("**🔥 Best Form Players:**")
                top_form = position_data.nlargest(10, 'form')[
                    ['web_name', 'team_name', 'form', 'total_points', 'cost_millions']
                ]
                
                for _, player in top_form.iterrows():
                    st.write(f"• {player['web_name']} ({player['team_name']}) - Form: {player['form']:.1f}")
            
            # Form vs Performance correlation
            if 'form' in position_data.columns and 'total_points' in position_data.columns:
                fig = px.scatter(
                    position_data,
                    x='form',
                    y='total_points',
                    color='cost_millions',
                    size='selected_by_percent',
                    hover_name='web_name',
                    title=f"{selected_position} Form vs Total Points",
                    labels={
                        'form': 'Current Form',
                        'total_points': 'Total Points',
                        'cost_millions': 'Cost (£m)'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)

    def _render_advanced_filters(self):
        """Render advanced filters tab"""
        st.header("🔍 Advanced Filters")
        
        if st.session_state.players_df.empty:
            st.info("Load data in the Dashboard tab first")
            return
        
        st.subheader("🎯 Advanced Player Filtering")
        
        # Multi-column layout for filters
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Performance Filters**")
            
            # Points range
            points_range = st.slider(
                "Total Points Range",
                min_value=int(st.session_state.players_df['total_points'].min()),
                max_value=int(st.session_state.players_df['total_points'].max()),
                value=(50, int(st.session_state.players_df['total_points'].max())),
                step=10
            )
            
            # Form range
            form_range = st.slider(
                "Form Range",
                min_value=float(st.session_state.players_df['form'].min()),
                max_value=float(st.session_state.players_df['form'].max()),
                value=(5.0, float(st.session_state.players_df['form'].max())),
                step=0.5
            )
            
            # Cost range
            cost_range = st.slider(
                "Cost Range (£m)",
                min_value=float(st.session_state.players_df['cost_millions'].min()),
                max_value=float(st.session_state.players_df['cost_millions'].max()),
                value=(4.0, 12.0),
                step=0.5
            )
        
        with col2:
            st.write("**⚙️ Selection Filters**")
            
            # Position multi-select
            positions = st.multiselect(
                "Positions",
                options=list(st.session_state.players_df['position_name'].unique()),
                default=list(st.session_state.players_df['position_name'].unique())
            )
            
            # Team multi-select
            teams = st.multiselect(
                "Teams",
                options=list(st.session_state.players_df['team_name'].unique()),
                default=list(st.session_state.players_df['team_name'].unique())[:10]  # Limit default selection
            )
            
            # Ownership range
            ownership_range = st.slider(
                "Ownership % Range",
                min_value=float(st.session_state.players_df['selected_by_percent'].min()),
                max_value=float(st.session_state.players_df['selected_by_percent'].max()),
                value=(0.0, 50.0),
                step=1.0
            )
        
        # Apply filters
        filtered_df = st.session_state.players_df.copy()
        
        # Apply all filters
        filtered_df = filtered_df[
            (filtered_df['total_points'] >= points_range[0]) &
            (filtered_df['total_points'] <= points_range[1]) &
            (filtered_df['form'] >= form_range[0]) &
            (filtered_df['form'] <= form_range[1]) &
            (filtered_df['cost_millions'] >= cost_range[0]) &
            (filtered_df['cost_millions'] <= cost_range[1]) &
            (filtered_df['selected_by_percent'] >= ownership_range[0]) &
            (filtered_df['selected_by_percent'] <= ownership_range[1]) &
            (filtered_df['position_name'].isin(positions)) &
            (filtered_df['team_name'].isin(teams))
        ]
        
        st.divider()
        
        # Display results
        st.subheader(f"🎯 Filtered Results ({len(filtered_df)} players)")
        
        if not filtered_df.empty:
            # Sort options
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox(
                    "Sort by",
                    ["points_per_million", "total_points", "form", "cost_millions", "selected_by_percent"]
                )
            with col2:
                sort_order = st.selectbox("Sort order", ["Descending", "Ascending"])
            
            ascending = sort_order == "Ascending"
            display_df = filtered_df.sort_values(sort_by, ascending=ascending)
            
            # Display results
            display_columns = [
                'web_name', 'position_name', 'team_name', 'cost_millions', 
                'total_points', 'form', 'points_per_million', 'selected_by_percent'
            ]
            available_columns = [col for col in display_columns if col in display_df.columns]
            
            st.dataframe(display_df[available_columns].head(50), use_container_width=True)
            
            # Export options
            if st.button("📥 Export Results to CSV"):
                csv = display_df[available_columns].to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="fpl_filtered_players.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No players match the current filter criteria. Try adjusting the filters.")

    def _render_visualizations(self):
        """Render visualizations tab"""
        st.header("📈 Data Visualizations")
        
        if st.session_state.players_df.empty:
            st.info("Load data in the Dashboard tab first")
            return
        
        # Visualization options
        viz_type = st.selectbox(
            "Choose Visualization Type",
            [
                "Cost vs Points Scatter", 
                "Team Performance", 
                "Position Distribution",
                "Form vs Ownership",
                "Value Analysis",
                "Performance Correlation"
            ]
        )
        
        st.divider()
        
        if viz_type == "Cost vs Points Scatter":
            st.subheader("💰 Player Cost vs Total Points")
            
            # Color options
            color_by = st.selectbox(
                "Color points by",
                ["position_name", "team_name", "form"]
            )
            
            fig = px.scatter(
                st.session_state.players_df,
                x='cost_millions',
                y='total_points',
                color=color_by,
                size='points_per_million',
                hover_name='web_name',
                hover_data=['team_name', 'form'],
                title="Player Cost vs Total Points",
                labels={
                    'cost_millions': 'Cost (£m)',
                    'total_points': 'Total Points',
                    'points_per_million': 'Points per Million'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Team Performance":
            st.subheader("🏆 Team Performance Analysis")
            
            metric = st.selectbox(
                "Performance Metric",
                ["total_points", "goals_scored", "assists", "clean_sheets"]
            )
            
            team_stats = st.session_state.players_df.groupby('team_name')[metric].sum().reset_index()
            team_stats = team_stats.sort_values(metric, ascending=False)
            
            fig = px.bar(
                team_stats, 
                x='team_name', 
                y=metric,
                title=f"Team Performance: {metric.replace('_', ' ').title()}",
                labels={'team_name': 'Team', metric: metric.replace('_', ' ').title()}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Position Distribution":
            st.subheader("📊 Player Distribution by Position")
            
            pos_counts = st.session_state.players_df['position_name'].value_counts()
            
            # Pie chart
            fig1 = px.pie(
                values=pos_counts.values, 
                names=pos_counts.index, 
                title="Player Count by Position"
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Average cost by position
            avg_cost = st.session_state.players_df.groupby('position_name')['cost_millions'].mean().reset_index()
            
            fig2 = px.bar(
                avg_cost,
                x='position_name',
                y='cost_millions',
                title="Average Player Cost by Position",
                labels={'position_name': 'Position', 'cost_millions': 'Average Cost (£m)'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        elif viz_type == "Form vs Ownership":
            st.subheader("🔥 Player Form vs Ownership")
            
            fig = px.scatter(
                st.session_state.players_df,
                x='selected_by_percent',
                y='form',
                color='position_name',
                size='total_points',
                hover_name='web_name',
                title="Player Form vs Ownership Percentage",
                labels={
                    'selected_by_percent': 'Ownership %',
                    'form': 'Form',
                    'total_points': 'Total Points'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Value Analysis":
            st.subheader("💎 Player Value Analysis")
            
            # Create value categories
            df_copy = st.session_state.players_df.copy()
            df_copy['value_category'] = pd.cut(
                df_copy['points_per_million'],
                bins=4,
                labels=['Low Value', 'Fair Value', 'Good Value', 'Excellent Value']
            )
            
            value_counts = df_copy['value_category'].value_counts()
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title="Player Distribution by Value Category",
                labels={'x': 'Value Category', 'y': 'Number of Players'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top value players
            st.write("**🌟 Top Value Players (Points per Million):**")
            top_value = st.session_state.players_df.nlargest(10, 'points_per_million')[
                ['web_name', 'team_name', 'position_name', 'cost_millions', 'points_per_million']
            ]
            st.dataframe(top_value, use_container_width=True)
        
        elif viz_type == "Performance Correlation":
            st.subheader("🔗 Performance Metrics Correlation")
            
            # Select numeric columns for correlation
            numeric_cols = ['total_points', 'cost_millions', 'form', 'selected_by_percent', 'points_per_million']
            available_cols = [col for col in numeric_cols if col in st.session_state.players_df.columns]
            
            if len(available_cols) >= 2:
                corr_data = st.session_state.players_df[available_cols].corr()
                
                fig = px.imshow(
                    corr_data,
                    text_auto=True,
                    aspect="auto",
                    title="Performance Metrics Correlation Heatmap",
                    color_continuous_scale="RdBu"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric columns for correlation analysis")

    def _render_team_builder(self):
        """Render enhanced team builder tab"""
        st.header("🏆 Advanced Team Builder")
        
        if st.session_state.players_df.empty:
            st.info("Load data in the Dashboard tab first")
            return
        
        # Team building configuration
        st.subheader("⚙️ Team Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            budget = st.number_input(
                "Total Budget (£m)", 
                min_value=80.0, 
                max_value=120.0, 
                value=100.0, 
                step=0.5,
                help="Your total budget for the 15-player squad"
            )
            
            formation = st.selectbox(
                "Formation", 
                ["4-3-3", "3-4-3", "4-4-2", "3-5-2", "4-5-1", "5-3-2", "5-4-1"],
                help="Choose your preferred formation (affects starting XI only)"
            )
        
        with col2:
            strategy = st.selectbox(
                "Building Strategy", 
                ["balanced", "premium_heavy", "simple"],
                help="Strategy for team selection:\n- Balanced: Mix of premium and value\n- Premium Heavy: Focus on expensive players\n- Simple: Cheapest valid team"
            )
            
            max_per_club = st.number_input(
                "Max players per club", 
                min_value=1, 
                max_value=3, 
                value=3,
                help="Maximum number of players from any single team"
            )
        
        with col3:
            st.write("**Target Premium Players (Optional)**")
            
            # Get premium players (cost >= 8.0)
            premium_options = st.session_state.players_df[
                st.session_state.players_df['cost_millions'] >= 8.0
            ].nlargest(30, 'total_points')['web_name'].tolist()
            
            premium_players = st.multiselect(
                "Select premium players to include",
                options=premium_options,
                help="These players will be prioritized in team selection"
            )
            
            # Show budget impact
            if premium_players:
                premium_cost = 0
                for player_name in premium_players:
                    player_row = st.session_state.players_df[
                        st.session_state.players_df['web_name'] == player_name
                    ]
                    if not player_row.empty:
                        premium_cost += player_row.iloc[0]['cost_millions']
                
                st.caption(f"Premium players cost: £{premium_cost:.1f}m")
                st.caption(f"Remaining budget: £{budget - premium_cost:.1f}m")
        
        st.divider()
        
        # Advanced Options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Strategy Weights")
                form_weight = st.slider("Form importance", 0.0, 1.0, 0.3, 0.1)
                value_weight = st.slider("Value importance", 0.0, 1.0, 0.4, 0.1)
                points_weight = st.slider("Total points importance", 0.0, 1.0, 0.3, 0.1)
            
            with col2:
                st.subheader("Constraints")
                min_playing_time = st.slider("Min expected minutes %", 0, 100, 60, 5)
                avoid_teams = st.multiselect(
                    "Avoid teams",
                    options=st.session_state.players_df['team_name'].unique().tolist()
                )
        
        # Build team button
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🏗️ Build Optimal Team", type="primary", use_container_width=True):
                with st.spinner("Building your optimal team..."):
                    try:
                        # Prepare kwargs
                        build_kwargs = {
                            'max_players_per_club': max_per_club,
                            'target_premiums': premium_players,
                            'form_weight': form_weight if 'form_weight' in locals() else 0.3,
                            'value_weight': value_weight if 'value_weight' in locals() else 0.4,
                            'points_weight': points_weight if 'points_weight' in locals() else 0.3,
                            'avoid_teams': avoid_teams if 'avoid_teams' in locals() else []
                        }
                        
                        squad = self.team_builder.build_team(
                            strategy,
                            st.session_state.players_df,
                            formation,
                            budget,
                            **build_kwargs
                        )
                        
                        if squad:
                            st.success("✅ Team built successfully!")
                            
                            # Store squad in session state
                            st.session_state.built_squad = squad
                            
                            # Display team results
                            self._display_built_team(squad, budget, formation)
                        
                        else:
                            st.error("❌ Failed to build team with current constraints")
                            
                            # Show helpful suggestions
                            with st.expander("💡 Suggestions to fix this"):
                                st.write("Try these adjustments:")
                                st.write("• Increase your budget")
                                st.write("• Remove some premium player requirements")
                                st.write("• Change to a simpler strategy")
                                st.write("• Increase max players per club")
                            
                    except Exception as e:
                        st.error(f"❌ Error building team: {str(e)}")
                        
                        # Show debug info in expander
                        with st.expander("🔍 Debug Information"):
                            st.code(str(e))
        
        # Display previously built team if available
        if 'built_squad' in st.session_state and st.session_state.built_squad:
            st.divider()
            st.subheader("📊 Your Current Squad")
            self._display_built_team(st.session_state.built_squad, budget, formation)

    def _display_built_team(self, squad: Squad, budget: float, formation: str):
        """Display built team with comprehensive analysis"""
        
        # Team overview metrics
        st.subheader("📊 Squad Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Cost", f"£{squad.total_cost:.1f}m")
        with col2:
            st.metric("Budget Remaining", f"£{budget - squad.total_cost:.1f}m")
        with col3:
            total_points = sum(p.total_points for p in squad.players)
            st.metric("Total Points", f"{total_points:,}")
        with col4:
            avg_points = total_points / len(squad.players) if squad.players else 0
            st.metric("Avg Points/Player", f"{avg_points:.1f}")
        
        # Squad composition
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚽ Starting XI")
            
            if squad.starting_xi:
                starting_data = []
                for player in squad.starting_xi:
                    starting_data.append({
                        'Player': player.web_name,
                        'Position': player.position_name,
                        'Team': player.team_name,
                        'Cost': f"£{player.cost_millions:.1f}m",
                        'Points': player.total_points,
                        'Form': f"{player.form:.1f}"
                    })
                
                starting_df = pd.DataFrame(starting_data)
                st.dataframe(starting_df, use_container_width=True)
                
                # Starting XI cost
                starting_cost = sum(p.cost_millions for p in squad.starting_xi)
                st.caption(f"Starting XI Cost: £{starting_cost:.1f}m")
            else:
                st.info("Starting XI will be determined based on formation")
        
        with col2:
            st.subheader("🪑 Bench")
            
            if squad.bench:
                bench_data = []
                for player in squad.bench:
                    bench_data.append({
                        'Player': player.web_name,
                        'Position': player.position_name,
                        'Team': player.team_name,
                        'Cost': f"£{player.cost_millions:.1f}m",
                        'Points': player.total_points,
                        'Form': f"{player.form:.1f}"
                    })
                
                bench_df = pd.DataFrame(bench_data)
                st.dataframe(bench_df, use_container_width=True)
                
                # Bench cost
                bench_cost = sum(p.cost_millions for p in squad.bench)
                st.caption(f"Bench Cost: £{bench_cost:.1f}m")
            else:
                # Show all squad players if bench not allocated
                st.subheader("👥 Full Squad")
                
                squad_data = []
                for player in squad.players:
                    squad_data.append({
                        'Player': player.web_name,
                        'Position': player.position_name,
                        'Team': player.team_name,
                        'Cost': f"£{player.cost_millions:.1f}m",
                        'Points': player.total_points,
                        'Form': f"{player.form:.1f}"
                    })
                
                squad_df = pd.DataFrame(squad_data)
                st.dataframe(squad_df, use_container_width=True)
        
        # Squad analysis
        st.subheader("📈 Squad Analysis")
        
        analyzer = SquadAnalyzer()
        analysis = analyzer.analyze_squad(squad)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Position Distribution:**")
            for position, count in analysis['position_distribution'].items():
                st.write(f"• {position}: {count}")
        
        with col2:
            st.write("**Team Distribution:**")
            for team, count in analysis['team_distribution'].items():
                st.write(f"• {team}: {count}")
        
        with col3:
            st.write("**Value Metrics:**")
            st.write(f"• Points/£m: {analysis['value_metrics']['points_per_million']:.1f}")
            st.write(f"• Avg Cost: £{analysis['value_metrics']['average_player_cost']:.1f}m")
            
            if analysis['form_analysis']:
                st.write(f"• Avg Form: {analysis['form_analysis']['average_form']:.1f}")

    def _render_ai_recommendations(self):
        """Render enhanced AI recommendations tab with comprehensive FPL data"""
        st.header("🤖 AI-Powered Recommendations")
        
        if st.session_state.players_df.empty:
            st.info("Load data in the Dashboard tab first")
            return
        
        # Initialize recommendation engine
        recommendation_engine = AIPlayerRecommendationEngine()
        
        # Recommendation categories
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Best Picks", "💎 Hidden Gems", "📊 Position Analysis", "🔄 Transfer Suggestions"
        ])
        
        with tab1:
            st.subheader("🎯 AI Best Picks by Position")
            
            for position in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
                st.write(f"**{position}s:**")
                
                position_players = st.session_state.players_df[
                    st.session_state.players_df['position_name'] == position
                ].copy()
                
                if not position_players.empty:
                    # Enhanced AI score calculation
                    position_players['ai_score'] = (
                        position_players['points_per_million'].rank(pct=True) * 0.25 +
                        position_players['form'].rank(pct=True) * 0.25 +
                        position_players['total_points'].rank(pct=True) * 0.2 +
                        (100 - position_players['selected_by_percent']).rank(pct=True) * 0.15 +
                        position_players.get('points_per_game', position_players['total_points']).rank(pct=True) * 0.15
                    )
                    
                    # Add expected goals/assists boost for attacking players
                    if position in ['Midfielder', 'Forward']:
                        if 'expected_goals' in position_players.columns:
                            position_players['ai_score'] += position_players['expected_goals'].rank(pct=True) * 0.1
                        if 'expected_assists' in position_players.columns:
                            position_players['ai_score'] += position_players['expected_assists'].rank(pct=True) * 0.1
                    
                    # Add clean sheet boost for defensive players
                    elif position in ['Goalkeeper', 'Defender']:
                        if 'clean_sheets' in position_players.columns:
                            position_players['ai_score'] += position_players['clean_sheets'].rank(pct=True) * 0.15
                        if 'saves' in position_players.columns and position == 'Goalkeeper':
                            position_players['ai_score'] += position_players['saves'].rank(pct=True) * 0.1
                    
                    top_picks = position_players.nlargest(3, 'ai_score')
                    
                    col1, col2, col3 = st.columns(3)
                    
                    for i, (_, player) in enumerate(top_picks.iterrows()):
                        with [col1, col2, col3][i]:
                            # Enhanced player card with more stats
                            player_info = f"**{player['web_name']}** ({player['team_name']})\n\n"
                            player_info += f"💰 Cost: £{player['cost_millions']:.1f}m\n"
                            player_info += f"📊 Points: {player['total_points']}\n"
                            player_info += f"🔥 Form: {player['form']:.1f}\n"
                            player_info += f"📈 Value: {player['points_per_million']:.1f} pts/£m\n"
                            player_info += f"👥 Ownership: {player['selected_by_percent']:.1f}%\n"
                            
                            # Add position-specific stats with safe numeric conversion
                            if position == 'Goalkeeper':
                                if 'clean_sheets' in player.index:
                                    player_info += f"🥅 Clean Sheets: {player['clean_sheets']}\n"
                                if 'saves' in player.index:
                                    player_info += f"✋ Saves: {player['saves']}\n"
                            elif position == 'Defender':
                                if 'clean_sheets' in player.index:
                                    player_info += f"🛡️ Clean Sheets: {player['clean_sheets']}\n"
                                if 'goals_scored' in player.index:
                                    player_info += f"⚽ Goals: {player['goals_scored']}\n"
                            elif position in ['Midfielder', 'Forward']:
                                if 'goals_scored' in player.index:
                                    player_info += f"⚽ Goals: {player['goals_scored']}\n"
                                if 'assists' in player.index:
                                    player_info += f"🎯 Assists: {player['assists']}\n"
                                # Safe expected goals conversion
                                if 'expected_goals' in player.index:
                                    try:
                                        xg_value = float(player['expected_goals'])
                                        player_info += f"📈 xG: {xg_value:.2f}\n"
                                    except (ValueError, TypeError):
                                        player_info += f"📈 xG: {player['expected_goals']}\n"
                            
                            # Add bonus and influence stats with safe conversion
                            if 'bonus' in player.index:
                                player_info += f"🌟 Bonus: {player['bonus']}\n"
                            if 'influence' in player.index:
                                try:
                                    influence_value = float(player['influence'])
                                    player_info += f"💪 Influence: {influence_value:.1f}\n"
                                except (ValueError, TypeError):
                                    player_info += f"💪 Influence: {player['influence']}\n"
                            
                            # Dream team status
                            if player.get('in_dreamteam', False):
                                player_info += f"⭐ In Dream Team!\n"
                            
                            st.info(player_info.strip())
                
                st.divider()
        
        with tab2:
            st.subheader("💎 Hidden Gems (Low Ownership, High Value)")
            
            # Enhanced hidden gems criteria
            hidden_gems = st.session_state.players_df[
                (st.session_state.players_df['selected_by_percent'] < 15) &
                (st.session_state.players_df['points_per_million'] > 5) &
                (st.session_state.players_df['total_points'] > 25) &
                (st.session_state.players_df['minutes'] > 500)  # Ensure they actually play
            ].copy()
            
            if not hidden_gems.empty:
                # Enhanced gem scoring
                hidden_gems['gem_score'] = (
                    hidden_gems['points_per_million'].rank(pct=True) * 0.3 +
                    hidden_gems['form'].rank(pct=True) * 0.25 +
                    (100 - hidden_gems['selected_by_percent']).rank(pct=True) * 0.2 +
                    hidden_gems['total_points'].rank(pct=True) * 0.15 +
                    hidden_gems.get('value_form', hidden_gems['form']).rank(pct=True) * 0.1
                )
                
                top_gems = hidden_gems.nlargest(6, 'gem_score')
                
                # Display gems in a grid with enhanced information
                cols = st.columns(3)
                for i, (_, player) in enumerate(top_gems.iterrows()):
                    with cols[i % 3]:
                        gem_info = f"💎 **{player['web_name']}**\n\n"
                        gem_info += f"🏟️ {player['team_name']} ({player['position_name']})\n"
                        gem_info += f"💰 £{player['cost_millions']:.1f}m\n"
                        gem_info += f"📊 {player['total_points']} pts ({player['points_per_million']:.1f} pts/£m)\n"
                        gem_info += f"👥 Only {player['selected_by_percent']:.1f}% ownership!\n"
                        gem_info += f"🔥 Form: {player['form']:.1f}\n"
                        
                        # Add minutes played
                        if 'minutes' in player.index:
                            gem_info += f"⏱️ Minutes: {player['minutes']}\n"
                        
                        # Add position-specific hidden gem stats with safe conversion
                        if player['position_name'] in ['Midfielder', 'Forward']:
                            if 'expected_goals' in player.index:
                                try:
                                    xg_value = float(player['expected_goals'])
                                    gem_info += f"📈 xG: {xg_value:.2f}\n"
                                except (ValueError, TypeError):
                                    gem_info += f"📈 xG: {player['expected_goals']}\n"
                            if 'expected_assists' in player.index:
                                try:
                                    xa_value = float(player['expected_assists'])
                                    gem_info += f"🎯 xA: {xa_value:.2f}\n"
                                except (ValueError, TypeError):
                                    gem_info += f"🎯 xA: {player['expected_assists']}\n"
                        
                        if player['position_name'] in ['Goalkeeper', 'Defender']:
                            if 'clean_sheets' in player.index:
                                gem_info += f"🛡️ CS: {player['clean_sheets']}\n"
                        
                        # Highlight if they're in good form
                        if player['form'] > 6:
                            gem_info += f"🚀 Hot streak!\n"
                        
                        st.success(gem_info.strip())
            else:
                st.info("No hidden gems found with current criteria")
        
        with tab3:
            st.subheader("📊 Position-Specific Insights")
            
            selected_position = st.selectbox(
                "Choose position for detailed analysis",
                ['Goalkeeper', 'Defender', 'Midfielder', 'Forward'],
                key="ai_position_analysis"
            )
            
            position_data = st.session_state.players_df[
                st.session_state.players_df['position_name'] == selected_position
            ].copy()
            
            if not position_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📈 Best Form Players:**")
                    best_form = position_data.nlargest(5, 'form')[
                        ['web_name', 'team_name', 'form', 'cost_millions', 'total_points']
                    ]
                    
                    for _, player in best_form.iterrows():
                        st.write(f"• **{player['web_name']}** ({player['team_name']}) - Form: {player['form']:.1f}, {player['total_points']} pts")
                
                with col2:
                    st.write("**💰 Best Value Players:**")
                    best_value = position_data.nlargest(5, 'points_per_million')[
                        ['web_name', 'team_name', 'points_per_million', 'cost_millions', 'selected_by_percent']
                    ]
                    
                    for _, player in best_value.iterrows():
                        st.write(f"• **{player['web_name']}** ({player['team_name']}) - {player['points_per_million']:.1f} pts/£m, {player['selected_by_percent']:.1f}% owned")
                
                # Position-specific analysis
                st.write("**🎯 Position-Specific Leaders:**")
                
                if selected_position == 'Goalkeeper':
                    metrics = ['clean_sheets', 'saves', 'bonus']
                    labels = ['🥅 Most Clean Sheets', '✋ Most Saves', '🌟 Most Bonus Points']
                elif selected_position == 'Defender':
                    metrics = ['clean_sheets', 'goals_scored', 'assists', 'bonus']
                    labels = ['🛡️ Most Clean Sheets', '⚽ Most Goals', '🎯 Most Assists', '🌟 Most Bonus Points']
                elif selected_position == 'Midfielder':
                    metrics = ['goals_scored', 'assists', 'bonus', 'bps']
                    labels = ['⚽ Most Goals', '🎯 Most Assists', '🌟 Most Bonus Points', '📊 Highest BPS']
                else:  # Forward
                    metrics = ['goals_scored', 'assists', 'bonus', 'bps']
                    labels = ['⚽ Most Goals', '🎯 Most Assists', '🌟 Most Bonus Points', '📊 Highest BPS']
                
                # Display leaders in each metric
                for metric, label in zip(metrics, labels):
                    if metric in position_data.columns:
                        leader = position_data.loc[position_data[metric].idxmax()]
                        st.write(f"• **{label}**: {leader['web_name']} ({leader[metric]})")
                
                # Advanced metrics for attacking players with safe conversion
                if selected_position in ['Midfielder', 'Forward']:
                    st.write("**📈 Advanced Attacking Metrics:**")
                    
                    advanced_metrics = ['expected_goals', 'expected_assists', 'expected_goal_involvements']
                    advanced_labels = ['🎯 Highest xG', '📤 Highest xA', '⚡ Highest xGI']
                    
                    for metric, label in zip(advanced_metrics, advanced_labels):
                        if metric in position_data.columns:
                            # Convert to numeric first
                            position_data_numeric = position_data.copy()
                            position_data_numeric[metric] = pd.to_numeric(position_data_numeric[metric], errors='coerce').fillna(0)
                            
                            if position_data_numeric[metric].max() > 0:
                                leader = position_data_numeric.loc[position_data_numeric[metric].idxmax()]
                                st.write(f"• **{label}**: {leader['web_name']} ({leader[metric]:.2f})")
                
                # Price distribution chart with enhanced data
                st.write("**💵 Enhanced Price vs Performance Analysis:**")
                
                # Create performance categories
                if 'points_per_game' in position_data.columns:
                    performance_metric = 'points_per_game'
                    performance_label = 'Points per Game'
                else:
                    performance_metric = 'total_points'
                    performance_label = 'Total Points'
                
                fig = px.scatter(
                    position_data,
                    x='cost_millions',
                    y=performance_metric,
                    size='minutes',
                    color='selected_by_percent',
                    hover_name='web_name',
                    hover_data=['form', 'bonus'],
                    title=f"{selected_position} Price vs {performance_label}",
                    labels={
                        'cost_millions': 'Cost (£m)', 
                        performance_metric: performance_label,
                        'selected_by_percent': 'Ownership %',
                        'minutes': 'Minutes Played'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("🔄 Advanced Transfer Suggestions")
            
            st.info("💡 **Enhanced Transfer Recommendation System**")
            
            # Enhanced user input
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_budget = st.number_input(
                    "Available transfer budget (£m)",
                    min_value=0.0,
                    max_value=20.0,
                    value=2.0,
                    step=0.1
                )
                
                transfer_position = st.selectbox(
                    "Position to upgrade",
                    ['Any', 'Goalkeeper', 'Defender', 'Midfielder', 'Forward'],
                    key="transfer_position"
                )
            
            with col2:
                max_cost = st.slider(
                    "Maximum player cost (£m)",
                    3.5, 15.0, 10.0, 0.5
                )
                
                min_form = st.slider(
                    "Minimum form",
                    0.0, 10.0, 4.0, 0.5
                )
            
            with col3:
                transfer_strategy = st.selectbox(
                    "Transfer Strategy",
                    ["Balanced", "Form Chasers", "Value Hunters", "Differentials", "Premium Focus"],
                    key="transfer_strategy"
                )
                
                min_minutes = st.slider(
                    "Minimum minutes played",
                    0, 2000, 500, 100,
                    help="Ensures players are getting regular game time"
                )
            
            # Advanced filters
            with st.expander("🔧 Advanced Transfer Filters"):
                col1, col2 = st.columns(2)
                
                with col1:
                    exclude_teams = st.multiselect(
                        "Exclude teams",
                        options=sorted(st.session_state.players_df['team_name'].unique().tolist()),
                        help="Avoid players from specific teams"
                    )
                    
                    max_ownership = st.slider(
                        "Maximum ownership %",
                        0.0, 100.0, 50.0, 5.0,
                        help="For differential picks"
                    )
                
                with col2:
                    include_news_check = st.checkbox("Include injury news check", value=True)
                    
                    min_bonus = st.slider(
                        "Minimum bonus points",
                        0, 50, 5, 5,
                        help="Players who consistently get bonus points"
                    )
            
            if st.button("🔍 Get Enhanced Transfer Recommendations", type="primary"):
                
                # Filter players based on criteria
                candidates = st.session_state.players_df.copy()
                
                # Basic filters
                if transfer_position != 'Any':
                    candidates = candidates[candidates['position_name'] == transfer_position]
                
                candidates = candidates[
                    (candidates['cost_millions'] <= max_cost) &
                    (candidates['form'] >= min_form) &
                    (candidates['minutes'] >= min_minutes) &
                    (candidates['selected_by_percent'] <= max_ownership)
                ]
                
                # Advanced filters
                if exclude_teams:
                    candidates = candidates[~candidates['team_name'].isin(exclude_teams)]
                
                if 'bonus' in candidates.columns:
                    candidates = candidates[candidates['bonus'] >= min_bonus]
                
                # Filter out injured players if requested
                if include_news_check and 'news' in candidates.columns:
                    candidates = candidates[
                        (candidates['news'].isna()) | 
                        (~candidates['news'].str.contains('injured|doubt|miss|suspended', case=False, na=False))
                    ]
                
                if not candidates.empty:
                    # Apply strategy-specific scoring
                    if transfer_strategy == "Form Chasers":
                        candidates['transfer_score'] = (
                            candidates['form'].rank(pct=True) * 0.4 +
                            candidates.get('points_per_game', candidates['total_points']/38).rank(pct=True) * 0.3 +
                            candidates['total_points'].rank(pct=True) * 0.2 +
                            candidates.get('value_form', candidates['form']).rank(pct=True) * 0.1
                        )
                    elif transfer_strategy == "Value Hunters":
                        candidates['transfer_score'] = (
                            candidates['points_per_million'].rank(pct=True) * 0.4 +
                            candidates.get('value_season', candidates['points_per_million']).rank(pct=True) * 0.25 +
                            candidates['total_points'].rank(pct=True) * 0.2 +
                            (100 - candidates['cost_millions']).rank(pct=True) * 0.15
                        )
                    elif transfer_strategy == "Differentials":
                        candidates['transfer_score'] = (
                            (100 - candidates['selected_by_percent']).rank(pct=True) * 0.35 +
                            candidates['points_per_million'].rank(pct=True) * 0.25 +
                            candidates['form'].rank(pct=True) * 0.25 +
                            candidates['total_points'].rank(pct=True) * 0.15
                        )
                    elif transfer_strategy == "Premium Focus":
                        premium_candidates = candidates[candidates['cost_millions'] >= 8.0]
                        if not premium_candidates.empty:
                            candidates = premium_candidates
                        candidates['transfer_score'] = (
                            candidates['total_points'].rank(pct=True) * 0.35 +
                            candidates['form'].rank(pct=True) * 0.25 +
                            candidates.get('points_per_game', candidates['total_points']/38).rank(pct=True) * 0.25 +
                            candidates['bonus'].rank(pct=True) * 0.15
                        )
                    else:  # Balanced
                        candidates['transfer_score'] = (
                            candidates['form'].rank(pct=True) * 0.25 +
                            candidates['points_per_million'].rank(pct=True) * 0.25 +
                            candidates['total_points'].rank(pct=True) * 0.2 +
                            (100 - candidates['selected_by_percent']).rank(pct=True) * 0.15 +
                            candidates['bonus'].rank(pct=True) * 0.15
                        )
                    
                    top_transfers = candidates.nlargest(8, 'transfer_score')
                    
                    st.write("**🎯 Top Transfer Targets:**")
                    
                    # Enhanced transfer table with comprehensive columns
                    transfer_display_cols = [
                        'web_name', 'position_name', 'team_name', 'cost_millions', 'total_points',
                        'form', 'points_per_million', 'selected_by_percent', 'minutes', 'bonus'
                    ]
                    
                    # Add position-specific columns
                    if transfer_position == 'Goalkeeper':
                        transfer_display_cols.extend(['clean_sheets', 'saves', 'goals_conceded'])
                    elif transfer_position == 'Defender':
                        transfer_display_cols.extend(['clean_sheets', 'goals_scored', 'assists'])
                    elif transfer_position in ['Midfielder', 'Forward']:
                        transfer_display_cols.extend(['goals_scored', 'assists'])
                        if 'expected_goals' in candidates.columns:
                            transfer_display_cols.append('expected_goals')
                        if 'expected_assists' in candidates.columns:
                            transfer_display_cols.append('expected_assists')
                    
                    # Add advanced metrics if available
                    advanced_cols = ['points_per_game', 'value_form', 'value_season', 'bps', 'influence', 'creativity', 'threat']
                    for col in advanced_cols:
                        if col in candidates.columns and col not in transfer_display_cols:
                            transfer_display_cols.append(col)
                    
                    transfer_display_cols.append('transfer_score')
                    
                    available_transfer_cols = [col for col in transfer_display_cols if col in top_transfers.columns]
                    
                    # Format transfer table with safe numeric conversion
                    transfer_table = top_transfers[available_transfer_cols].copy()
                    
                    # Round numeric columns for better display with safe conversion
                    numeric_round_rules = {
                        'cost_millions': 1, 'form': 1, 'points_per_million': 1, 'selected_by_percent': 1,
                        'points_per_game': 1, 'value_form': 1, 'value_season': 1, 'transfer_score': 2,
                        'expected_goals': 2, 'expected_assists': 2, 'influence': 1, 'creativity': 1, 'threat': 1
                    }
                    
                    for col, decimals in numeric_round_rules.items():
                        if col in transfer_table.columns:
                            transfer_table[col] = pd.to_numeric(transfer_table[col], errors='coerce').round(decimals)
                    
                    st.dataframe(transfer_table, use_container_width=True)
                    
                    # Detailed transfer insights for top 3 with safe conversion
                    st.write("**📊 Detailed Transfer Analysis:**")
                    
                    for i, (_, player) in enumerate(top_transfers.head(3).iterrows(), 1):
                        with st.expander(f"#{i} {player['web_name']} - Complete Analysis"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write("**📊 Core Stats:**")
                                st.metric("Cost", f"£{player['cost_millions']:.1f}m")
                                st.metric("Total Points", f"{player['total_points']}")
                                st.metric("Form", f"{player['form']:.1f}")
                                st.metric("Ownership", f"{player['selected_by_percent']:.1f}%")
                                
                                if 'points_per_game' in player.index:
                                    try:
                                        ppg_value = float(player['points_per_game'])
                                        st.metric("Points/Game", f"{ppg_value:.1f}")
                                    except (ValueError, TypeError):
                                        st.metric("Points/Game", str(player['points_per_game']))
                            
                            with col2:
                                st.write("**⚽ Performance:**")
                                
                                if player['position_name'] in ['Midfielder', 'Forward']:
                                    st.write(f"Goals: {player.get('goals_scored', 0)}")
                                    st.write(f"Assists: {player.get('assists', 0)}")
                                    if 'expected_goals' in player.index:
                                        try:
                                            xg_value = float(player['expected_goals'])
                                            st.write(f"Expected Goals: {xg_value:.2f}")
                                        except (ValueError, TypeError):
                                            st.write(f"Expected Goals: {player['expected_goals']}")
                                    if 'expected_assists' in player.index:
                                        try:
                                            xa_value = float(player['expected_assists'])
                                            st.write(f"Expected Assists: {xa_value:.2f}")
                                        except (ValueError, TypeError):
                                            st.write(f"Expected Assists: {player['expected_assists']}")
                                
                                elif player['position_name'] in ['Goalkeeper', 'Defender']:
                                    if 'clean_sheets' in player.index:
                                        st.write(f"Clean Sheets: {player['clean_sheets']}")
                                    if 'saves' in player.index and player['position_name'] == 'Goalkeeper':
                                        st.write(f"Saves: {player['saves']}")
                                    if 'goals_scored' in player.index:
                                        st.write(f"Goals: {player['goals_scored']}")
                                
                                st.write(f"Bonus Points: {player.get('bonus', 0)}")
                                st.write(f"Minutes: {player.get('minutes', 0)}")
                            
                            with col3:
                                st.write("**💡 Transfer Rationale:**")
                                reasons = []
                                
                                if player['form'] > 7:
                                    reasons.append("🔥 Excellent recent form")
                                if player['points_per_million'] > 8:
                                    reasons.append("💰 Great value for money")
                                if player['selected_by_percent'] < 15:
                                    reasons.append("💎 Low ownership differential")
                                
                                # Safe expected goals check
                                try:
                                    if float(player.get('expected_goals', 0)) > 0.3:
                                        reasons.append("⚽ High expected goals")
                                except (ValueError, TypeError):
                                    pass
                                
                                try:
                                    if float(player.get('expected_assists', 0)) > 0.3:
                                        reasons.append("🎯 High expected assists")
                                except (ValueError, TypeError):
                                    pass
                                
                                if player.get('clean_sheets', 0) > 5 and player['position_name'] in ['Goalkeeper', 'Defender']:
                                    reasons.append("🛡️ Strong defensive record")
                                if player.get('bonus', 0) > 10:
                                    reasons.append("🌟 Consistent bonus points")
                                if player.get('minutes', 0) > 1500:
                                    reasons.append("⏱️ Regular starter")
                                
                                for reason in reasons:
                                    st.write(f"• {reason}")
                                
                                # Transfer score breakdown
                                st.write(f"**🎯 AI Score: {player['transfer_score']:.2f}**")
                                
                                # News check
                                if 'news' in player.index and pd.notna(player['news']) and player['news'].strip():
                                    st.warning(f"📰 News: {player['news']}")
                                else:
                                    st.success("✅ No injury concerns")
                else:
                    st.warning("No players match your transfer criteria. Try adjusting the filters.")
                    
                    # Enhanced adjustment suggestions
                    with st.expander("💡 Suggestions to find more options"):
                        st.write("**Try these adjustments:**")
                        st.write("• Increase the maximum cost limit")
                        st.write("• Lower the minimum form requirement")
                        st.write("• Reduce the minimum minutes played")
                        st.write("• Increase the maximum ownership percentage")
                        st.write("• Remove team exclusions")
                        st.write("• Lower the minimum bonus points requirement")
                        st.write("• Change the position filter to 'Any'")
                        st.write("• Try a different transfer strategy")
            
            # Enhanced transfer tips
            with st.expander("💡 Advanced Transfer Strategy Guide"):
                st.write("""
                **Smart Transfer Strategy:**
                
                **🎯 Form-Based Transfers:**
                • Target players with 3+ consecutive good games
                • Look for underlying stats (xG, xA) supporting good form
                • Check fixture difficulty for next 4-6 gameweeks
                
                **💰 Value-Based Transfers:**
                • Focus on points per million over 7.0
                • Consider price rise potential for popular picks
                • Look for players just returning from injury at reduced price
                
                **💎 Differential Strategy:**
                • Target <10% ownership for significant rank gains
                • Check captain alternatives for big gameweeks
                • Monitor team news for rotation-prone premiums
                
                **⏰ Timing Your Transfers:**
                • Make transfers after all team news is released
                • Consider using Free Hit during blank/double gameweeks
                • Bank transfers during international breaks
                
                **📊 Data-Driven Decisions:**
                • xG/xA trends more predictive than recent goals
                • Bonus point consistency indicates underlying performance
                • Minutes played shows injury/rotation risk
                """)

    def _render_import_my_team(self):
        """Render import my team tab"""
        st.header("📥 Import My FPL Team")
        
        if st.session_state.players_df.empty:
            st.info("Load data in the Dashboard tab first")
            return
        
        # Team import section
        st.subheader("🔗 Connect Your FPL Team")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**How to find your Team ID:**")
            st.write("1. Go to the official FPL website")
            st.write("2. Navigate to 'View Gameweek History'")
            st.write("3. Your Team ID is in the URL: fantasy.premierleague.com/entry/**YOUR_ID**/history")
            
            team_id_input = st.text_input(
                "Enter your FPL Team ID",
                placeholder="e.g., 123456",
                help="Your unique FPL team identifier"
            )
            
            gameweek_input = st.number_input(
                "Gameweek (leave empty for current)",
                min_value=1,
                max_value=38,
                value=None,
                help="Specific gameweek to import, or leave empty for current"
            )
        
        with col2:
            st.info("**Why import your team?**\n\n"
                   "• See detailed analysis of your squad\n"
                   "• Get transfer recommendations\n"
                   "• Compare with optimal lineups\n"
                   "• Track your team's performance")
        
        # Import button
        if st.button("🚀 Import My Team", type="primary") and team_id_input:
            try:
                team_id = int(team_id_input)
                
                with st.spinner("Importing your FPL team..."):
                    # Initialize importer
                    importer = FPLTeamImporter()
                    
                    # Get team info
                    team_info = importer.get_team_info(team_id)
                    
                    if team_info:
                        st.success(f"✅ Found team: {team_info.get('name', 'Unknown')}")
                        
                        # Get team data
                        team_data = importer.get_user_team(team_id, gameweek_input)
                        
                        if team_data:
                            # Process team data
                            processed_team = importer.process_team_data(team_data, st.session_state.players_df)
                            
                            if processed_team:
                                # Store in session state
                                st.session_state.imported_team = processed_team
                                st.session_state.team_info = team_info
                                
                                st.success("✅ Team imported successfully!")
                                
                                # Display basic info for now
                                st.write(f"**Team Name:** {team_info.get('name', 'Unknown')}")
                                st.write(f"**Manager:** {team_info.get('player_first_name', '')} {team_info.get('player_last_name', '')}")
                                st.write(f"**Overall Rank:** {team_info.get('summary_overall_rank', 'N/A'):,}")
                            else:
                                st.error("❌ Failed to process team data")
                        else:
                            st.error("❌ Failed to fetch team data. Check your Team ID and try again.")
                    else:
                        st.error("❌ Team not found. Please check your Team ID.")
                        
            except ValueError:
                st.error("❌ Please enter a valid numeric Team ID")
            except Exception as e:
                st.error(f"❌ Error importing team: {str(e)}")
        
        # Show info about team import feature
        st.info("🚧 Full team analysis features coming soon! Currently supporting basic team import.")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point with enhanced error handling"""
    try:
        app = FPLAnalyticsApp()
        app.run()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.error("Full error details:")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
