import pandas as pd
import numpy as np
import logging
import streamlit as st
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
print("Team Recommender module loaded")

__all__ = ["TeamOptimizer", "get_latest_team_recommendations", "valid_team_formation",
           "calculate_team_diversity", "calculate_fixture_difficulty_score",
           "generate_key_player_explanations", "render_team_recommendations_tab"]

class TeamOptimizer:
    """Main team optimization class"""
    
    def __init__(self, df_players, budget=100, formations=None, max_players_per_club=3,
                 substitute_counts=(1, 3), injury_penalty=0.5, hard_fixture_penalty=0.8,
                 min_ownership=0.0, form_weight=0.3, xg_weight=0.2, xa_weight=0.2,
                 ownership_weight=0.0, budget_weight=0.1, team_diversity_weight=0.1,
                 fixture_difficulty_weight=0.1, consistency_weight=0.05, bps_weight=0.05,
                 minutes_weight=0.05, clean_sheet_weight=0.05, defensive_contribution_weight=0.05,
                 debug=False, available_transfers=1, transfer_cost=4, preferred_style="balanced",
                 risk_tolerance=0.5):
        
        self.df_players = df_players.copy()
        self.budget = budget
        self.formations = formations or [(3,4,3), (4,3,3), (3,5,2), (4,4,2), (5,3,2)]
        self.max_players_per_club = max_players_per_club
        self.substitute_counts = substitute_counts
        self.injury_penalty = injury_penalty
        self.hard_fixture_penalty = hard_fixture_penalty
        self.min_ownership = min_ownership
        self.debug = debug
        self.preferred_style = preferred_style
        self.risk_tolerance = risk_tolerance
        
        # Scoring weights
        self.weights = {
            'form': form_weight,
            'xg': xg_weight,
            'xa': xa_weight,
            'ownership': ownership_weight,
            'budget': budget_weight,
            'team_diversity': team_diversity_weight,
            'fixture_difficulty': fixture_difficulty_weight,
            'consistency': consistency_weight,
            'bps': bps_weight,
            'minutes': minutes_weight,
            'clean_sheet': clean_sheet_weight,
            'defensive_contribution': defensive_contribution_weight
        }
        
        # Validate and prepare data
        self._validate_and_prepare_data()
    
    def _validate_and_prepare_data(self):
        """Validate and prepare player data for optimization"""
        required_columns = {
            'id': 0, 'web_name': 'Unknown', 'element_type': 1, 'team_name': 'Unknown',
            'now_cost': 50, 'selected_by_percent': 0, 'form': 0, 'total_points': 0,
            'expected_points_next_5': 0, 'xG_next_5': 0, 'xA_next_5': 0,
            'fixture_difficulty': 3, 'bps': 0, 'minutes': 0, 'consistency': 0,
            'clean_sheets_rate': 0, 'injury_status': 'Available'
        }
        
        # Add missing columns with default values
        for col, default_value in required_columns.items():
            if col not in self.df_players.columns:
                self.df_players[col] = default_value
        
        # Convert numeric columns
        numeric_cols = ['now_cost', 'selected_by_percent', 'form', 'total_points',
                       'expected_points_next_5', 'xG_next_5', 'xA_next_5',
                       'fixture_difficulty', 'bps', 'minutes', 'consistency']
        
        for col in numeric_cols:
            self.df_players[col] = pd.to_numeric(self.df_players[col], errors='coerce').fillna(0)
        
        # Calculate derived metrics
        self.df_players['now_cost_m'] = self.df_players['now_cost'] / 10
        self.df_players['minutes_per_game'] = self.df_players['minutes'] / max(1, self.df_players['minutes'].max() / 90)
        
        # Calculate comprehensive player scores
        self._calculate_player_scores()
    
    def _calculate_player_scores(self):
        """Calculate comprehensive player scores"""
        df = self.df_players
        
        # Base score from expected points
        df['base_score'] = df['expected_points_next_5']
        
        # Form contribution
        df['form_score'] = df['form'] * self.weights['form']
        
        # Expected goals and assists
        df['xg_score'] = df['xG_next_5'] * self.weights['xg']
        df['xa_score'] = df['xA_next_5'] * self.weights['xa']
        
        # Bonus points potential
        df['bps_score'] = df['bps'] * self.weights['bps'] * 0.1
        
        # Position-specific adjustments
        df['position_multiplier'] = 1.0
        
        # Goalkeepers
        gk_mask = df['element_type'] == 1
        df.loc[gk_mask, 'position_multiplier'] = 1.0 + df.loc[gk_mask, 'clean_sheets_rate'] * 0.2
        
        # Defenders
        def_mask = df['element_type'] == 2
        df.loc[def_mask, 'position_multiplier'] = 1.0 + df.loc[def_mask, 'clean_sheets_rate'] * 0.3
        
        # Playing style adjustments
        if self.preferred_style == "attacking":
            # Boost forwards and attacking midfielders
            fwd_mask = df['element_type'] == 4
            mid_mask = df['element_type'] == 3
            df.loc[fwd_mask, 'position_multiplier'] *= 1.2
            df.loc[mid_mask, 'position_multiplier'] *= 1.1
        elif self.preferred_style == "defensive":
            # Boost defenders and defensive midfielders
            def_mask = df['element_type'] == 2
            gk_mask = df['element_type'] == 1
            df.loc[def_mask, 'position_multiplier'] *= 1.2
            df.loc[gk_mask, 'position_multiplier'] *= 1.1
        
        # Injury and fixture penalties
        df['injury_penalty'] = np.where(
            df['injury_status'].isin(['Injured', 'Doubtful']), 
            self.injury_penalty, 1.0
        )
        
        df['fixture_penalty'] = np.where(
            df['fixture_difficulty'] >= 4, 
            self.hard_fixture_penalty, 1.0
        )
        
        # Calculate final adjusted score
        df['adjusted_score'] = (
            (df['base_score'] + df['form_score'] + df['xg_score'] + 
             df['xa_score'] + df['bps_score']) * 
            df['position_multiplier'] * df['injury_penalty'] * df['fixture_penalty']
        )
        
        # Calculate value (points per million)
        df['value'] = df['adjusted_score'] / (df['now_cost_m'] + 0.1)  # Add small epsilon to avoid division by zero
        
        self.df_players = df
    
    def optimize_team(self):
        """Main optimization method"""
        try:
            best_team = None
            best_score = -np.inf
            
            for formation in self.formations:
                team_result = self._try_formation(formation)
                if team_result and team_result['score'] > best_score:
                    best_score = team_result['score']
                    best_team = team_result
            
            return best_team
            
        except Exception as e:
            if self.debug:
                logging.error(f"Error in optimize_team: {e}")
            return None
    
    def _try_formation(self, formation):
        """Try to build a team with the given formation"""
        try:
            n_def, n_mid, n_fwd = formation
            
            # Filter players with NaN values in critical columns
            available_players = self.df_players[
                (self.df_players['selected_by_percent'] >= self.min_ownership) &
                (self.df_players['now_cost'].notna()) &
                (self.df_players['form'].notna()) &
                (self.df_players['xG_next_5'].notna()) &
                (self.df_players['xA_next_5'].notna()) &
                (self.df_players['bps'].notna()) &
                (self.df_players['minutes'].notna()) &
                (self.df_players['consistency'].notna())
            ].copy()
            
            if available_players.empty:
                return None
            
            # Select starting XI
            starting_xi = self._select_starting_xi(available_players, formation)
            if not starting_xi:
                return None
            
            # Select bench
            bench_players = self._select_bench(available_players, starting_xi)
            if not bench_players:
                return None
            
            # Combine team
            full_team = starting_xi + bench_players
            team_df = pd.DataFrame(full_team)
            
            # Add starting status
            team_df['is_starting'] = False
            team_df.iloc[:11, team_df.columns.get_loc('is_starting')] = True
            
            # Validate team
            if not self._validate_team(team_df, formation):
                return None
            
            # Calculate team metrics
            total_cost = team_df['now_cost'].sum()
            if total_cost > self.budget * 10:  # Budget is in millions, cost in 0.1m
                return None
            
            starters = team_df[team_df['is_starting']]
            team_score = starters['adjusted_score'].sum()
            
            # Select captain and vice-captain
            captain_id = starters.loc[starters['adjusted_score'].idxmax(), 'id']
            vice_starters = starters[starters['id'] != captain_id]
            vice_captain_id = vice_starters.loc[vice_starters['adjusted_score'].idxmax(), 'id'] if not vice_starters.empty else captain_id
            
            # Generate rationale
            rationale = self._generate_rationale(team_df, formation, captain_id, vice_captain_id)
            
            return {
                'team': team_df,
                'formation': formation,
                'captain': captain_id,
                'vice_captain': vice_captain_id,
                'score': team_score,
                'total_expected_points': team_score,
                'total_cost': total_cost,
                'rationale': rationale
            }
            
        except Exception as e:
            if self.debug:
                logging.error(f"Formation {formation} failed: {e}")
            return None
    
    def _select_starting_xi(self, available_players, formation):
        """Select starting XI based on formation"""
        n_def, n_mid, n_fwd = formation
        selected_players = []
        selected_ids = set()
        club_counts = {}
        
        # Position requirements
        positions = {
            1: 1,      # GK
            2: n_def,  # DEF
            3: n_mid,  # MID
            4: n_fwd   # FWD
        }
        
        # Budget allocation per position (rough estimates)
        budget_allocation = {
            1: self.budget * 0.08,  # 8% for GK
            2: self.budget * 0.35,  # 35% for DEF
            3: self.budget * 0.35,  # 35% for MID
            4: self.budget * 0.22   # 22% for FWD
        }
        
        for pos, count in positions.items():
            pos_players = available_players[available_players['element_type'] == pos].copy()
            if len(pos_players) < count:
                return None
            
            # Sort by value for this position
            pos_players = pos_players.sort_values('value', ascending=False)
            
            pos_budget = budget_allocation[pos] * 10  # Convert to 0.1m units
            pos_selected = 0
            
            for _, player in pos_players.iterrows():
                if pos_selected >= count:
                    break
                
                # Check constraints
                if player['id'] in selected_ids:
                    continue
                
                if club_counts.get(player['team_name'], 0) >= self.max_players_per_club:
                    continue
                
                # Budget check (with some flexibility)
                if pos_selected < count - 1 and player['now_cost'] > pos_budget / (count - pos_selected) * 1.5:
                    continue
                
                # Add player
                selected_players.append(player.to_dict())
                selected_ids.add(player['id'])
                club_counts[player['team_name']] = club_counts.get(player['team_name'], 0) + 1
                pos_budget -= player['now_cost']
                pos_selected += 1
            
            if pos_selected < count:
                return None
        
        return selected_players if len(selected_players) == 11 else None
    
    def _select_bench(self, available_players, starting_xi):
        """Select bench players"""
        selected_ids = {p['id'] for p in starting_xi}
        club_counts = Counter(p['team_name'] for p in starting_xi)
        
        bench_players = []
        
        # Select substitute goalkeeper (cheapest available)
        gk_subs = available_players[
            (available_players['element_type'] == 1) & 
            (~available_players['id'].isin(selected_ids))
        ].sort_values('now_cost')
        
        for _, gk in gk_subs.iterrows():
            if club_counts.get(gk['team_name'], 0) < self.max_players_per_club:
                bench_players.append(gk.to_dict())
                selected_ids.add(gk['id'])
                club_counts[gk['team_name']] += 1
                break
        
        if len(bench_players) == 0:
            return None
        
        # Select 3 outfield subs
        outfield_subs = available_players[
            (available_players['element_type'] != 1) & 
            (~available_players['id'].isin(selected_ids))
        ].sort_values(['value', 'now_cost'], ascending=[False, True])
        
        outfield_count = 0
        for _, player in outfield_subs.iterrows():
            if outfield_count >= 3:
                break
            
            if club_counts.get(player['team_name'], 0) < self.max_players_per_club:
                bench_players.append(player.to_dict())
                selected_ids.add(player['id'])
                club_counts[player['team_name']] += 1
                outfield_count += 1
        
        return bench_players if len(bench_players) == 4 else None
    
    def _validate_team(self, team_df, formation):
        """Validate team composition"""
        if len(team_df) != 15:
            return False
        
        starters = team_df[team_df['is_starting']]
        if len(starters) != 11:
            return False
        
        # Check formation
        starter_counts = starters['element_type'].value_counts()
        n_def, n_mid, n_fwd = formation
        
        if (starter_counts.get(1, 0) != 1 or 
            starter_counts.get(2, 0) != n_def or
            starter_counts.get(3, 0) != n_mid or
            starter_counts.get(4, 0) != n_fwd):
            return False
        
        # Check club limits
        club_counts = team_df['team_name'].value_counts()
        if club_counts.max() > self.max_players_per_club:
            return False
        
        return True
    
    def _generate_rationale(self, team_df, formation, captain_id, vice_captain_id):
        """Generate explanation for team selection"""
        starters = team_df[team_df['is_starting']]
        
        # Get captain name safely
        captain_name = "Unknown"
        vice_captain_name = "Unknown"
        
        try:
            captain_name = starters[starters['id'] == captain_id]['web_name'].iloc[0]
        except (IndexError, KeyError):
            pass
        
        try:
            vice_captain_name = starters[starters['id'] == vice_captain_id]['web_name'].iloc[0]
        except (IndexError, KeyError):
            pass
        
        return {
            'formation': f"Formation {formation[0]}-{formation[1]}-{formation[2]} selected for {self.preferred_style} approach",
            'budget': f"Team cost: £{team_df['now_cost'].sum()/10:.1f}m of £{self.budget}m budget",
            'captain': f"{captain_name} selected as captain for highest expected returns",
            'vice_captain': f"{vice_captain_name} chosen as vice-captain for reliability",
            'team_balance': f"Players from {starters['team_name'].nunique()} different teams for good balance",
            'style_rationale': f"Team optimized for {self.preferred_style} play with {self.risk_tolerance:.1f} risk tolerance",
            'top_performers': f"Top 3 players by expected points: {', '.join(starters.nlargest(3, 'adjusted_score')['web_name'].tolist())}"
        }

    # Update the rationale generation to include more detailed explanations:
    def _generate_enhanced_rationale(self, team, formation):
        """Generate enhanced rationale with more categories"""
        return {
            'strategy': 'Balanced approach focusing on form and fixtures',
            'formation_reasoning': f'{formation[0]}-{formation[1]}-{formation[2]} provides optimal coverage',
            'budget_reasoning': 'Strategic allocation with premium players in key positions',
            'form_analysis': 'Selected players showing consistent recent performance',
            'fixture_analysis': 'Favorable upcoming fixtures considered',
            'risk_assessment': 'Balanced risk profile with minimal single-team exposure'
        }

def get_latest_team_recommendations(
    df_players: pd.DataFrame,
    budget=100,
    formations=[(3,4,3), (4,3,3), (3,5,2), (4,4,2), (5,3,2)],
    max_players_per_club=3,
    substitute_counts=(1,3),
    injury_penalty=0.5,
    hard_fixture_penalty=0.8,
    min_ownership=0.0,
    form_weight=0.3,
    xg_weight=0.2,
    xa_weight=0.2,
    ownership_weight=0.0,
    budget_weight=0.1,
    team_diversity_weight=0.1,
    fixture_difficulty_weight=0.1,
    consistency_weight=0.05,
    bps_weight=0.05,
    minutes_weight=0.05,
    clean_sheet_weight=0.05,
    defensive_contribution_weight=0.05,
    debug=False,
    available_transfers=1,
    transfer_cost=4,
    preferred_style="balanced",
    risk_tolerance=0.5,
):
    """
    Standalone function to get team recommendations using the TeamOptimizer class.
    """
    try:
        optimizer = TeamOptimizer(
            df_players=df_players,
            budget=budget,
            formations=formations,
            max_players_per_club=max_players_per_club,
            substitute_counts=substitute_counts,
            injury_penalty=injury_penalty,
            hard_fixture_penalty=hard_fixture_penalty,
            min_ownership=min_ownership,
            form_weight=form_weight,
            xg_weight=xg_weight,
            xa_weight=xa_weight,
            ownership_weight=ownership_weight,
            budget_weight=budget_weight,
            team_diversity_weight=team_diversity_weight,
            fixture_difficulty_weight=fixture_difficulty_weight,
            consistency_weight=consistency_weight,
            bps_weight=bps_weight,
            minutes_weight=minutes_weight,
            clean_sheet_weight=clean_sheet_weight,
            defensive_contribution_weight=defensive_contribution_weight,
            debug=debug,
            available_transfers=available_transfers,
            transfer_cost=transfer_cost,
            preferred_style=preferred_style,
            risk_tolerance=risk_tolerance
        )
        
        return optimizer.optimize_team()
        
    except Exception as e:
        logging.error(f"Error in get_latest_team_recommendations: {e}")
        return None

# Helper functions for backward compatibility
def calculate_team_diversity(team_df, weight):
    """Calculate team diversity score"""
    unique_teams = team_df['team_name'].nunique()
    return unique_teams * weight

def calculate_fixture_difficulty_score(team_df, weight):
    """Calculate fixture difficulty score"""
    total_difficulty = team_df['fixture_difficulty'].sum()
    return total_difficulty * weight

def valid_team_formation(team_df, formation, n_sub_gk=1, n_sub_outfield=3):
    """Validate team formation"""
    if len(team_df) != 15:
        return False
    
    starters = team_df[team_df.get('is_starting', True)]
    if len(starters) != 11:
        return False
    
    starter_counts = starters['element_type'].value_counts()
    n_def, n_mid, n_fwd = formation
    
    return (starter_counts.get(1, 0) == 1 and 
            starter_counts.get(2, 0) == n_def and 
            starter_counts.get(3, 0) == n_mid and 
            starter_counts.get(4, 0) == n_fwd)

def generate_key_player_explanations(team_df, all_players_df, n=2):
    """Generate explanations for key players"""
    explanations = {}
    top_players = team_df.nlargest(n, 'adjusted_score') if 'adjusted_score' in team_df.columns else team_df.head(n)
    
    for _, player in top_players.iterrows():
        explanations[player['id']] = {
            'name': player.get('web_name', 'Unknown'),
            'reason': f"Selected for {player.get('expected_points_next_5', 0):.1f} expected points."
        }
    return explanations

# UI Component for rendering team recommendations
def render_team_recommendations_tab(df_players):
    """Renders the team recommendations tab with interactive features"""
    st.header("🏆 Team Recommendations")

    if df_players.empty:
        st.error("No player data available! Check your data loading process.")
        return

    # User configuration
    col1, col2 = st.columns(2)
    
    with col1:
        preferred_style = st.selectbox(
            "Preferred Playing Style",
            ["balanced", "attacking", "defensive"],
            index=0,
        )
    
    with col2:
        risk_tolerance = st.slider(
            "Risk Tolerance",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )

    # Budget and formation controls
    col3, col4 = st.columns(2)
    
    with col3:
        budget = st.slider(
            "Budget (£m)",
            min_value=80.0,
            max_value=120.0,
            value=100.0,
            step=0.5
        )
    
    with col4:
        formations = st.multiselect(
            "Allowed Formations",
            ["3-4-3", "4-3-3", "3-5-2", "4-4-2", "5-3-2"],
            default=["3-4-3", "4-3-3"]
        )

    # Convert formation strings to tuples
    formation_map = {
        "3-4-3": (3, 4, 3),
        "4-3-3": (4, 3, 3),
        "3-5-2": (3, 5, 2),
        "4-4-2": (4, 4, 2),
        "5-3-2": (5, 3, 2)
    }
    formation_tuples = [formation_map[f] for f in formations]

    if st.button("🔮 Generate Team Recommendations", type="primary"):
        if not formation_tuples:
            st.error("Please select at least one formation!")
            return

        try:
            with st.spinner("Generating team recommendations..."):
                recommendations = get_latest_team_recommendations(
                    df_players, 
                    budget=budget,
                    formations=formation_tuples,
                    preferred_style=preferred_style, 
                    risk_tolerance=risk_tolerance,
                    debug=True
                )

            if recommendations is None:
                st.warning("No team recommendations found. Try adjusting your parameters:")
                st.write("- Increase budget")
                st.write("- Try different formations")
                st.write("- Check if player data is complete")
                return

            # Display results
            team_df = recommendations["team"]
            formation = recommendations["formation"]
            captain = recommendations["captain"]
            vice_captain = recommendations["vice_captain"]
            total_expected_points = recommendations["total_expected_points"]
            total_cost = recommendations["total_cost"]
            rationale = recommendations["rationale"]

            # Team summary
            st.subheader("✅ Recommended Team")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Formation", f"{formation[0]}-{formation[1]}-{formation[2]}")
                st.metric("Total Cost", f"£{total_cost/10:.1f}m")
            
            with col2:
                st.metric("Expected Points", f"{total_expected_points:.1f}")
                st.metric("Budget Remaining", f"£{budget - total_cost/10:.1f}m")
            
            with col3:
                # Get captain names safely
                try:
                    captain_name = team_df[team_df['id'] == captain]['web_name'].iloc[0] if captain is not None else 'N/A'
                    vice_captain_name = team_df[team_df['id'] == vice_captain]['web_name'].iloc[0] if vice_captain is not None else 'N/A'
                except (IndexError, KeyError):
                    captain_name = 'N/A'
                    vice_captain_name = 'N/A'
                
                st.metric("Captain", captain_name)
                st.metric("Vice Captain", vice_captain_name)

            # Team table
            st.subheader("👥 Team Squad")
            
            # Prepare display data
            display_df = team_df.copy()
            display_df['Position'] = display_df['element_type'].map({
                1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'
            })
            display_df['Cost'] = '£' + (display_df['now_cost'] / 10).round(1).astype(str) + 'm'
            display_df['Status'] = display_df['is_starting'].map({
                True: '🟢 Starting', False: '🟡 Bench'
            })
            
            # Select columns to display
            columns_to_display = [
                "web_name", "Position", "team_name", "Cost", "Status"
            ]
            
            # Add optional columns if they exist
            if "adjusted_score" in display_df.columns:
                columns_to_display.append("adjusted_score")
            if "xG_next_5" in display_df.columns:
                columns_to_display.append("xG_next_5")
            if "xA_next_5" in display_df.columns:
                columns_to_display.append("xA_next_5")

            # Configure column display
            column_config = {
                "web_name": "Player",
                "team_name": "Team",
                "adjusted_score": st.column_config.NumberColumn("Score", format="%.1f"),
                "xG_next_5": st.column_config.NumberColumn("xG", format="%.1f"),
                "xA_next_5": st.column_config.NumberColumn("xA", format="%.1f")
            }

            st.dataframe(
                display_df[columns_to_display], 
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )

            # Rationale
            with st.expander("🧠 Team Rationale", expanded=False):
                for key, value in rationale.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")

        except Exception as e:
            st.error(f"Error generating team recommendations: {str(e)}")
            if st.checkbox("Show debug info"):
                st.exception(e)

def get_custom_team_recommendations(df_players, custom_params):
    """
    Generate team recommendations with custom parameters
    """
    try:
        # For now, return the default recommendations
        # You can enhance this later with the custom logic
        return get_latest_team_recommendations(df_players)
        
    except Exception as e:
        logging.error(f"Error in custom team recommendations: {e}")
        return None

def apply_custom_filters(df_players, custom_params):
    """Apply custom filters to player dataframe"""
    df = df_players.copy()
    
    # Form filter
    if 'min_form' in custom_params:
        df = df[pd.to_numeric(df['form'], errors='coerce') >= custom_params['min_form']]
    
    # Fixture difficulty filter
    if 'max_fixture_difficulty' in custom_params and 'fixture_difficulty_score' in df.columns:
        df = df[df['fixture_difficulty_score'] <= custom_params['max_fixture_difficulty']]
    
    # Minutes filter
    if 'min_minutes' in custom_params and 'minutes' in df.columns:
        df = df[df['minutes'] >= custom_params['min_minutes']]
    
    # Team preferences
    if custom_params.get('preferred_teams'):
        df = df[df['team_name'].isin(custom_params['preferred_teams'])]
    
    if custom_params.get('avoid_teams'):
        df = df[~df['team_name'].isin(custom_params['avoid_teams'])]
    
    # Injury filter
    if custom_params.get('exclude_injured') and 'injury_status' in df.columns:
        df = df[df['injury_status'] != 'Injured']
    
    return df

class CustomTeamOptimizer(TeamOptimizer):
    """Enhanced team optimizer with custom parameters"""
    
    def __init__(self, players_df, formation=(3,4,3), max_players_per_team=3, 
                 xg_weight=1.0, xa_weight=1.0, budget_strategy='Balanced', 
                 risk_tolerance='Moderate', captain_priority='Highest Expected Points'):
        
        super().__init__(players_df)
        self.formation = formation
        self.max_players_per_team = max_players_per_team
        self.xg_weight = xg_weight
        self.xa_weight = xa_weight
        self.budget_strategy = budget_strategy
        self.risk_tolerance = risk_tolerance
        self.captain_priority = captain_priority
    
    def calculate_custom_score(self, player):
        """Calculate player score with custom weights"""
        base_score = player.get('adjusted_score', player.get('total_points', 0))
        
        # Apply xG and xA weights
        xg_bonus = player.get('xG_next_5', 0) * self.xg_weight
        xa_bonus = player.get('xA_next_5', 0) * self.xa_weight
        
        # Risk adjustment
        risk_multiplier = self._get_risk_multiplier(player)
        
        return (base_score + xg_bonus + xa_bonus) * risk_multiplier
    
    def _get_risk_multiplier(self, player):
        """Calculate risk multiplier based on tolerance"""
        if self.risk_tolerance == 'Conservative':
            # Prefer established players
            if player.get('total_points', 0) > 100:
                return 1.1
            return 0.9
        elif self.risk_tolerance == 'Aggressive':
            # Prefer differentials and high upside
            if player.get('ownership', 10) < 5:  # Low ownership
                return 1.2
            return 1.0
        else:  # Moderate
            return 1.0

