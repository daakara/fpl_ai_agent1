"""
Transfer Planning Assistant - Advanced transfer and chip strategy recommendations
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from services.fixture_service import FixtureService


class TransferPlanningAssistant:
    """Advanced transfer planning and chip strategy assistant"""
    
    def __init__(self):
        self.fixture_service = FixtureService()
        self.transfer_cache = {}
        
        # Chip timing recommendations based on typical FPL season patterns
        self.optimal_chip_windows = {
            'wildcard1': {'start': 6, 'end': 12, 'priority': 'medium'},
            'wildcard2': {'start': 20, 'end': 28, 'priority': 'high'},
            'triple_captain': {'start': 28, 'end': 35, 'priority': 'high'},
            'bench_boost': {'start': 32, 'end': 37, 'priority': 'medium'},
            'free_hit': {'start': 18, 'end': 25, 'priority': 'medium'}
        }
    
    def generate_transfer_plan(self, team_data: Dict, players_df: pd.DataFrame, 
                             planning_weeks: int = 8) -> Dict:
        """
        Generate comprehensive multi-gameweek transfer plan
        
        Args:
            team_data: Current team data
            players_df: All players dataframe
            planning_weeks: Number of weeks to plan ahead
            
        Returns:
            Detailed transfer plan with recommendations
        """
        current_gw = team_data.get('gameweek', 20)
        picks = team_data.get('picks', [])
        bank = team_data.get('bank', 0) / 10
        
        # Analyze current squad performance
        squad_analysis = self._analyze_current_squad(picks, players_df)
        
        # Identify transfer priorities
        transfer_priorities = self._identify_transfer_priorities(squad_analysis, players_df)
        
        # Generate week-by-week plan
        weekly_plan = self._generate_weekly_transfer_plan(
            transfer_priorities, current_gw, planning_weeks, bank
        )
        
        # Chip usage recommendations
        chip_recommendations = self._recommend_chip_usage(current_gw, squad_analysis)
        
        return {
            'current_gw': current_gw,
            'squad_analysis': squad_analysis,
            'transfer_priorities': transfer_priorities,
            'weekly_plan': weekly_plan,
            'chip_recommendations': chip_recommendations,
            'total_cost': sum([week.get('cost', 0) for week in weekly_plan]),
            'expected_gain': sum([week.get('expected_points_gain', 0) for week in weekly_plan])
        }
    
    def _analyze_current_squad(self, picks: List, players_df: pd.DataFrame) -> Dict:
        """Analyze current squad strengths and weaknesses"""
        squad_players = []
        
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                squad_players.append({
                    'id': player.get('id'),
                    'name': player.get('web_name'),
                    'position': player.get('position_name'),
                    'team': player.get('team_short_name'),
                    'price': float(player.get('now_cost', 0)) / 10,
                    'form': float(player.get('form', 0)),
                    'total_points': int(player.get('total_points', 0)),
                    'ppm': float(player.get('points_per_million', 0)),
                    'ownership': float(player.get('selected_by_percent', 0)),
                    'minutes': int(player.get('minutes', 0)),
                    'is_starting': pick.get('position', 12) <= 11
                })
        
        # Categorize players by performance
        excellent = [p for p in squad_players if p['form'] > 7 and p['ppm'] > 8]
        good = [p for p in squad_players if p['form'] > 5 and p['ppm'] > 6]
        average = [p for p in squad_players if p['form'] > 3 and p['ppm'] > 4]
        poor = [p for p in squad_players if p['form'] <= 3 or p['ppm'] <= 4]
        
        return {
            'total_players': len(squad_players),
            'excellent_performers': excellent,
            'good_performers': good,
            'average_performers': average,
            'poor_performers': poor,
            'squad_value': sum([p['price'] for p in squad_players]),
            'avg_form': np.mean([p['form'] for p in squad_players]),
            'avg_ppm': np.mean([p['ppm'] for p in squad_players])
        }
    
    def _identify_transfer_priorities(self, squad_analysis: Dict, players_df: pd.DataFrame) -> List[Dict]:
        """Identify transfer priorities based on squad analysis"""
        priorities = []
        
        # High priority: Poor performers who are expensive
        for player in squad_analysis['poor_performers']:
            if player['price'] > 8 and player['is_starting']:
                priorities.append({
                    'player': player,
                    'priority': 'high',
                    'reason': 'Expensive underperformer in starting XI',
                    'urgency': 1,
                    'potential_replacements': self._find_replacement_options(
                        player, players_df, price_range=0.5
                    )
                })
        
        # Medium priority: Poor form players
        for player in squad_analysis['poor_performers']:
            if player['form'] < 3 and player['is_starting']:
                priorities.append({
                    'player': player,
                    'priority': 'medium',
                    'reason': 'Poor form affecting starting XI',
                    'urgency': 2,
                    'potential_replacements': self._find_replacement_options(
                        player, players_df, price_range=1.0
                    )
                })
        
        # Low priority: Bench improvements
        for player in squad_analysis['poor_performers']:
            if not player['is_starting'] and player['price'] < 6:
                priorities.append({
                    'player': player,
                    'priority': 'low',
                    'reason': 'Bench player with low potential',
                    'urgency': 3,
                    'potential_replacements': self._find_replacement_options(
                        player, players_df, price_range=0.3
                    )
                })
        
        # Sort by urgency
        return sorted(priorities, key=lambda x: x['urgency'])
    
    def _find_replacement_options(self, current_player: Dict, players_df: pd.DataFrame, 
                                 price_range: float = 1.0) -> List[Dict]:
        """Find suitable replacement options for a player"""
        position = current_player['position']
        current_price = current_player['price']
        
        # Filter by position and price range
        candidates = players_df[
            (players_df['position_name'] == position) &
            (players_df['now_cost'] <= (current_price + price_range) * 10) &
            (players_df['now_cost'] >= (current_price - price_range) * 10) &
            (players_df['id'] != current_player['id'])
        ].copy()
        
        if candidates.empty:
            return []
        
        # Score candidates
        candidates['replacement_score'] = (
            candidates['form'].astype(float) * 0.3 +
            candidates['points_per_million'].astype(float) * 0.4 +
            (candidates['total_points'].astype(float) / 200) * 0.2 +
            (100 - candidates['selected_by_percent'].astype(float)) / 100 * 0.1
        )
        
        # Return top 3 options
        top_candidates = candidates.nlargest(3, 'replacement_score')
        
        replacements = []
        for _, candidate in top_candidates.iterrows():
            replacements.append({
                'id': candidate.get('id'),
                'name': candidate.get('web_name'),
                'team': candidate.get('team_short_name'),
                'price': float(candidate.get('now_cost', 0)) / 10,
                'form': float(candidate.get('form', 0)),
                'ppm': float(candidate.get('points_per_million', 0)),
                'score': candidate.get('replacement_score', 0)
            })
        
        return replacements
    
    def _generate_weekly_transfer_plan(self, priorities: List[Dict], current_gw: int, 
                                     weeks: int, available_funds: float) -> List[Dict]:
        """Generate week-by-week transfer plan"""
        weekly_plan = []
        remaining_funds = available_funds
        
        for week in range(weeks):
            gw = current_gw + week
            week_plan = {
                'gameweek': gw,
                'transfers': [],
                'cost': 0,
                'expected_points_gain': 0,
                'rationale': []
            }
            
            # Determine available transfers (free transfer + any banked)
            free_transfers = 1 if week == 0 else min(2, 1 + week)
            
            # Plan transfers for this week
            transfers_made = 0
            for priority in priorities[:2]:  # Max 2 transfers per week typically
                if transfers_made >= free_transfers:
                    # Would require hits
                    hit_cost = (transfers_made - free_transfers + 1) * 4
                    if priority['urgency'] == 1 and hit_cost <= 8:  # Only take hits for urgent transfers
                        week_plan['cost'] += hit_cost
                        week_plan['rationale'].append(f"Taking {hit_cost}pt hit for urgent transfer")
                    else:
                        break
                
                best_replacement = priority['potential_replacements'][0] if priority['potential_replacements'] else None
                if best_replacement:
                    price_diff = best_replacement['price'] - priority['player']['price']
                    
                    if remaining_funds + price_diff >= 0:  # Can afford the transfer
                        week_plan['transfers'].append({
                            'out': priority['player']['name'],
                            'in': best_replacement['name'],
                            'price_change': price_diff,
                            'reason': priority['reason']
                        })
                        
                        remaining_funds += price_diff
                        transfers_made += 1
                        
                        # Estimate points gain
                        form_diff = best_replacement['form'] - priority['player']['form']
                        week_plan['expected_points_gain'] += max(form_diff * 2, 0)
            
            if week_plan['transfers']:
                weekly_plan.append(week_plan)
        
        return weekly_plan
    
    def _recommend_chip_usage(self, current_gw: int, squad_analysis: Dict) -> Dict:
        """Recommend optimal chip usage timing"""
        recommendations = {}
        
        for chip, window in self.optimal_chip_windows.items():
            if current_gw < window['end']:
                # Calculate chip suitability score
                score = self._calculate_chip_suitability(chip, current_gw, squad_analysis)
                
                recommendations[chip] = {
                    'recommended_gw': self._find_optimal_gw_for_chip(chip, current_gw, window),
                    'suitability_score': score,
                    'rationale': self._get_chip_rationale(chip, score, current_gw),
                    'priority': window['priority']
                }
        
        return recommendations
    
    def _calculate_chip_suitability(self, chip: str, current_gw: int, squad_analysis: Dict) -> float:
        """Calculate how suitable the current squad is for each chip"""
        if chip == 'wildcard1' or chip == 'wildcard2':
            # Wildcard suitable when many poor performers
            poor_ratio = len(squad_analysis['poor_performers']) / squad_analysis['total_players']
            return poor_ratio * 100
        
        elif chip == 'bench_boost':
            # Bench boost suitable when bench players are good
            bench_players = [p for p in squad_analysis['good_performers'] if not p['is_starting']]
            return len(bench_players) * 25
        
        elif chip == 'triple_captain':
            # Triple captain suitable when you have excellent captain options
            excellent_captains = [p for p in squad_analysis['excellent_performers'] if p['is_starting']]
            return len(excellent_captains) * 30
        
        elif chip == 'free_hit':
            # Free hit suitable when your team has many poor fixtures
            return 50  # Simplified - would integrate with fixture analysis
        
        return 50
    
    def _find_optimal_gw_for_chip(self, chip: str, current_gw: int, window: Dict) -> int:
        """Find optimal gameweek to use a chip within its window"""
        # Simplified implementation - would integrate with fixture analysis
        window_start = max(current_gw + 1, window['start'])
        window_end = window['end']
        
        if chip in ['triple_captain', 'bench_boost']:
            # Use during good fixture runs (simplified)
            return window_start + 2
        elif chip in ['wildcard1', 'wildcard2']:
            # Use during international breaks or early in good fixture runs
            return window_start + 1
        else:  # free_hit
            # Use during difficult fixture gameweeks
            return window_end - 2
    
    def _get_chip_rationale(self, chip: str, score: float, current_gw: int) -> str:
        """Get rationale for chip recommendation"""
        if chip == 'wildcard1':
            if score > 60:
                return "Squad has many underperformers - wildcard could provide significant improvement"
            else:
                return "Squad is performing well - consider saving wildcard for later"
        
        elif chip == 'wildcard2':
            if score > 50:
                return "Good time to restructure squad for final stretch"
            else:
                return "Current squad suitable for season end"
        
        elif chip == 'triple_captain':
            if score > 60:
                return "You have excellent captain options - great time for triple captain"
            else:
                return "Wait for better captain options or easier fixtures"
        
        elif chip == 'bench_boost':
            if score > 50:
                return "Strong bench players make bench boost valuable"
            else:
                return "Improve bench strength before using bench boost"
        
        else:  # free_hit
            return "Consider for difficult fixture gameweeks or when many players are unavailable"
    
    def get_transfer_success_probability(self, player_out: Dict, player_in: Dict) -> float:
        """Calculate probability of transfer success based on multiple factors"""
        # Form trajectory
        form_improvement = player_in['form'] - player_out['form']
        form_score = min(max(form_improvement / 5, 0), 1) * 30
        
        # Value efficiency
        ppm_improvement = player_in['ppm'] - player_out['ppm']
        value_score = min(max(ppm_improvement / 5, 0), 1) * 25
        
        # Fixture difficulty (simplified)
        fixture_score = 20  # Would integrate with actual fixture data
        
        # Minutes reliability
        minutes_score = min(player_in.get('minutes', 0) / 2000, 1) * 15
        
        # Team strength
        team_score = 10  # Would integrate with team strength data
        
        total_score = form_score + value_score + fixture_score + minutes_score + team_score
        return min(total_score, 100)