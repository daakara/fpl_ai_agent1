import pandas as pd
import numpy as np
import logging

class IterativeTeamOptimizer:
    """Enhanced team optimizer with iterative refinement"""
    
    def __init__(self, df_players, **kwargs):
        self.df_players = df_players
        self.config = kwargs
        self.max_iterations = kwargs.get('max_iterations', 5)
        self.improvement_threshold = kwargs.get('improvement_threshold', 0.01)
        
    def optimize_team_iteratively(self):
        """Main iterative optimization loop"""
        best_team = None
        best_score = -float('inf')
        
        for iteration in range(self.max_iterations):
            # Generate initial team or refine existing one
            if iteration == 0:
                current_team = self._generate_initial_team()
            else:
                current_team = self._refine_team(current_team)
            
            if current_team is None:
                continue
                
            # Optimize bench for current starting XI
            optimized_bench = self._optimize_bench_selection(current_team)
            if optimized_bench:
                current_team.extend(optimized_bench)
            
            # Select optimal captain
            captain_info = self._optimize_captaincy_selection(current_team)
            
            # Calculate total team score
            team_score = self._calculate_team_score(current_team, captain_info)
            
            # Check for improvement
            if team_score > best_score:
                improvement = (team_score - best_score) / abs(best_score) if best_score != 0 else float('inf')
                
                if improvement < self.improvement_threshold and iteration > 0:
                    break  # Convergence reached
                    
                best_score = team_score
                best_team = {
                    'team': pd.DataFrame(current_team),
                    'captain': captain_info['captain'],
                    'vice_captain': captain_info['vice_captain'],
                    'total_score': team_score,
                    'formation': self._get_formation_from_team(current_team),
                    'iteration': iteration + 1,
                    'rationale': self._generate_iteration_rationale(current_team, captain_info, iteration)
                }
        
        return best_team
    
    def _generate_initial_team(self):
        """Generate initial team using greedy selection"""
        selected_players = []
        available_budget = self.config.get('budget', 1000)
        formation = self.config.get('formations', [(3,4,3)])[0]
        
        # Position requirements: [GK, DEF, MID, FWD]
        position_needs = [1, formation[0], formation[1], formation[2]]
        
        for pos_type, count in enumerate(position_needs, 1):
            position_players = self.df_players[
                (self.df_players['element_type'] == pos_type) & 
                (self.df_players['now_cost'] <= available_budget)
            ].copy()
            
            # Calculate value score for position
            position_players['value_score'] = self._calculate_position_value_score(position_players, pos_type)
            position_players = position_players.sort_values('value_score', ascending=False)
            
            selected_count = 0
            for _, player in position_players.iterrows():
                if selected_count >= count:
                    break
                    
                if self._can_add_player(player, selected_players, available_budget):
                    player_dict = player.to_dict()
                    player_dict['is_starting'] = True
                    selected_players.append(player_dict)
                    available_budget -= player['now_cost']
                    selected_count += 1
        
        return selected_players if len(selected_players) == 11 else None
    
    def _refine_team(self, current_team):
        """Refine current team through local search"""
        team_df = pd.DataFrame(current_team)
        starting_xi = team_df[team_df['is_starting']].copy()
        
        best_refinement = current_team.copy()
        best_score = self._calculate_starting_xi_score(starting_xi)
        
        # Try swapping each player with better alternatives
        for idx, player in starting_xi.iterrows():
            alternatives = self.df_players[
                (self.df_players['element_type'] == player['element_type']) &
                (self.df_players['id'] != player['id']) &
                (self.df_players['now_cost'] <= player['now_cost'] + 5)  # Small budget flexibility
            ].copy()
            
            alternatives['improvement_score'] = self._calculate_improvement_score(alternatives, player)
            alternatives = alternatives.sort_values('improvement_score', ascending=False).head(3)
            
            for _, alt_player in alternatives.iterrows():
                # Create test team with replacement
                test_team = current_team.copy()
                for i, p in enumerate(test_team):
                    if p['id'] == player['id']:
                        test_team[i] = alt_player.to_dict()
                        test_team[i]['is_starting'] = True
                        break
                
                # Validate and score test team
                if self._validate_team_constraints(test_team):
                    test_score = self._calculate_starting_xi_score(pd.DataFrame([p for p in test_team if p['is_starting']]))

                    if test_score > best_score:
                        best_refinement = test_team
                        best_score = test_score
        
        return best_refinement
    
    def _optimize_bench_selection(self, starting_xi):
        """Optimize bench selection for maximum value and coverage"""
        starting_ids = {p['id'] for p in starting_xi}
        starting_clubs = [p['team_name'] for p in starting_xi]
        club_counts = {club: starting_clubs.count(club) for club in set(starting_clubs)}
        
        used_budget = sum(p['now_cost'] for p in starting_xi)
        available_budget = self.config.get('budget', 1000) - used_budget
        
        bench_players = []
        
        # 1. Select substitute goalkeeper (cheapest reliable option)
        gk_candidates = self.df_players[
            (self.df_players['element_type'] == 1) &
            (~self.df_players['id'].isin(starting_ids)) &
            (self.df_players['now_cost'] <= available_budget)
        ].copy()
        
        if not gk_candidates.empty:
            # Balance cost and reliability for GK sub
            gk_candidates['bench_value'] = (
                gk_candidates['total_points'] * 0.3 +
                gk_candidates['form'] * 0.2 +
                (50 - gk_candidates['now_cost']) * 0.5  # Prefer cheaper options
            )
            
            selected_gk = gk_candidates.loc[gk_candidates['bench_value'].idxmax()]
            gk_dict = selected_gk.to_dict()
            gk_dict['is_starting'] = False
            gk_dict['bench_priority'] = 4  # Lowest priority
            bench_players.append(gk_dict)
            
            available_budget -= selected_gk['now_cost']
            starting_ids.add(selected_gk['id'])
        
        # 2. Select 3 outfield substitutes with strategic considerations
        outfield_candidates = self.df_players[
            (self.df_players['element_type'].isin([2, 3, 4])) &
            (~self.df_players['id'].isin(starting_ids)) &
            (self.df_players['now_cost'] <= available_budget)
        ].copy()
        
        if not outfield_candidates.empty:
            # Calculate bench value considering multiple factors
            outfield_candidates['bench_value'] = self._calculate_bench_value_score(
                outfield_candidates, starting_xi, club_counts
            )
            
            # Select best 3 outfield subs within budget
            selected_outfield = self._select_optimal_bench_combination(
                outfield_candidates, available_budget, 3
            )
            
            for i, (_, player) in enumerate(selected_outfield.iterrows()):
                player_dict = player.to_dict()
                player_dict['is_starting'] = False
                player_dict['bench_priority'] = i + 1  # Priority 1-3
                bench_players.append(player_dict)
        
        return bench_players if len(bench_players) == 4 else None
    
    def _calculate_bench_value_score(self, candidates, starting_xi, club_counts):
        """Calculate value score for bench players"""
        # Base playing likelihood
        candidates['playing_chance'] = candidates.apply(
            lambda x: self._estimate_playing_chance(x, starting_xi), axis=1
        )
        
        # Points when playing
        candidates['points_when_playing'] = (
            candidates['form'] * 0.4 +
            candidates['total_points'] / max(1, candidates['minutes']) * 90 * 0.3 +
            candidates['expected_points_next_5'] * 0.3
        )
        
        # Cost efficiency
        candidates['cost_efficiency'] = candidates['points_when_playing'] / (candidates['now_cost'] / 10)
        
        # Team diversity bonus
        candidates['diversity_bonus'] = candidates['team_name'].apply(
            lambda x: 2.0 if club_counts.get(x, 0) < 2 else 1.0
        )
        
        # Position flexibility bonus
        candidates['flexibility_bonus'] = candidates['element_type'].apply(
            lambda x: 1.2 if x == 3 else 1.0  # Midfielders are more flexible
        )
        
        return (
            candidates['playing_chance'] * 0.3 +
            candidates['cost_efficiency'] * 0.25 +
            candidates['diversity_bonus'] * 0.2 +
            candidates['flexibility_bonus'] * 0.15 +
            (10 - candidates['now_cost'] / 10) * 0.1  # Prefer cheaper options
        )
    
    def _select_optimal_bench_combination(self, candidates, budget, num_players):
        """Select optimal combination of bench players within budget"""
        from itertools import combinations
        
        best_combination = None
        best_score = -float('inf')
        
        # Try combinations of players within budget
        for combo in combinations(candidates.index, num_players):
            combo_players = candidates.loc[list(combo)]
            total_cost = combo_players['now_cost'].sum()
            
            if total_cost <= budget:
                combo_score = combo_players['bench_value'].sum()
                # Bonus for good budget utilization
                budget_efficiency = total_cost / budget
                combo_score *= (0.8 + 0.4 * budget_efficiency)
                
                if combo_score > best_score:
                    best_score = combo_score
                    best_combination = combo_players
        
        return best_combination if best_combination is not None else candidates.head(num_players)
    
    def _optimize_captaincy_selection(self, team_players):
        """Optimize captain and vice-captain selection"""
        starting_players = [p for p in team_players if p.get('is_starting', True)]
        
        if not starting_players:
            return {'captain': None, 'vice_captain': None}
        
        team_df = pd.DataFrame(starting_players)
        
        # Calculate captaincy scores
        team_df['captaincy_score'] = self._calculate_captaincy_score(team_df)
        
        # Sort by captaincy score
        sorted_players = team_df.sort_values('captaincy_score', ascending=False)
        
        captain = sorted_players.iloc[0] if len(sorted_players) > 0 else None
        vice_captain = sorted_players.iloc[1] if len(sorted_players) > 1 else None
        
        return {
            'captain': captain.to_dict() if captain is not None else None,
            'vice_captain': vice_captain.to_dict() if vice_captain is not None else None,
            'captaincy_rationale': self._generate_captaincy_rationale(captain, vice_captain)
        }
    
    def _calculate_captaincy_score(self, team_df):
        """Calculate captaincy score for each player"""
        # Base expected points
        base_score = team_df['expected_points_next_5'] * 0.4
        
        # Form contribution
        form_score = team_df['form'] * 0.3
        
        # Fixture difficulty (easier fixtures = higher score)
        fixture_score = (6 - team_df.get('fixture_difficulty', 3)) * 0.1
        
        # Position bias (forwards and attacking mids favored)
        position_bonus = team_df['element_type'].apply(
            lambda x: 1.2 if x == 4 else 1.1 if x == 3 else 1.0
        )
        
        # Consistency factor
        consistency_score = team_df.get('consistency', 0.5) * 0.1
        
        # Expected goals and assists for attacking players
        attacking_bonus = 0
        if 'xG_next_5' in team_df.columns:
            attacking_bonus += team_df['xG_next_5'] * 0.05
        if 'xA_next_5' in team_df.columns:
            attacking_bonus += team_df['xA_next_5'] * 0.05
        
        return (base_score + form_score + fixture_score + attacking_bonus + consistency_score) * position_bonus
    
    def _generate_captaincy_rationale(self, captain, vice_captain):
        """Generate rationale for captaincy choices"""
        rationale = {}
        
        if captain:
            rationale['captain_reason'] = (
                f"{captain['web_name']} selected as captain due to "
                f"strong form ({captain.get('form', 0):.1f}), "
                f"expected points ({captain.get('expected_points_next_5', 0):.1f}), "
                f"and favorable fixtures."
            )
        
        if vice_captain:
            rationale['vice_captain_reason'] = (
                f"{vice_captain['web_name']} chosen as vice-captain for "
                f"reliability and consistent performance "
                f"({vice_captain.get('total_points', 0)} total points)."
            )
        
        return rationale

# Update the main function to use iterative optimization
def get_latest_team_recommendations(
    df_players: pd.DataFrame,
    budget=1000,
    formations=[(3,4,3), (4,3,3), (3,5,2), (4,4,2), (5,3,2)],
    max_players_per_club=3,
    max_iterations=5,
    improvement_threshold=0.01,
    **kwargs
):
    """Enhanced team recommendations with iterative optimization"""
    
    if df_players.empty:
        return None
    
    # Validate budget
    if budget <= 0:
        raise ValueError("Budget must be a positive number")
    
    try:
        # Create iterative optimizer
        optimizer = IterativeTeamOptimizer(
            df_players=df_players,
            budget=budget,
            formations=formations,
            max_players_per_club=max_players_per_club,
            max_iterations=max_iterations,
            improvement_threshold=improvement_threshold,
            **kwargs
        )
        
        # Run iterative optimization
        result = optimizer.optimize_team_iteratively()
        
        if result is None:
            return None
        
        # Add team chemistry and final calculations
        team_df = result['team']
        chemistry_bonus = calculate_team_chemistry_bonus(team_df)
        
        # Update final result
        result.update({
            'team_chemistry_bonus': chemistry_bonus,
            'total_expected_points': team_df[team_df['is_starting']]['expected_points_next_5'].sum() + chemistry_bonus,
            'total_cost': team_df['now_cost'].sum(),
            'optimization_method': 'iterative',
            'iterations_completed': result.get('iteration', 1)
        })
        
        return result
        
    except Exception as e:
        logging.error(f"Error in get_latest_team_recommendations: {e}")
        return None

def calculate_team_chemistry_bonus(team_df):
    """Calculate team chemistry bonus based on player connections"""
    if team_df.empty:
        return 0.0
    
    # Players from same team bonus
    team_counts = team_df['team_name'].value_counts()
    same_team_bonus = sum((count - 1) * 0.1 for count in team_counts if count > 1)
    
    # League consistency bonus (players from top teams)
    top_teams = ['Manchester City', 'Arsenal', 'Liverpool', 'Chelsea', 'Manchester United', 'Newcastle United']
    top_team_players = len(team_df[team_df['team_name'].isin(top_teams)])
    top_team_bonus = min(top_team_players * 0.05, 0.3)
    
    return same_team_bonus + top_team_bonus

def calculate_team_diversity(team_df, weight):
    unique_teams = team_df['team_name'].nunique()
    return unique_teams * weight

def calculate_fixture_difficulty_score(team_df, weight):
    total_difficulty = team_df['fixture_difficulty'].sum()
    return total_difficulty * weight

def valid_team_formation(team_df, formation, n_sub_gk, n_sub_outfield):
    n_gk = (team_df['element_type'] == 1).sum()
    n_def = (team_df['element_type'] == 2).sum()
    n_mid = (team_df['element_type'] == 3).sum()
    n_fwd = (team_df['element_type'] == 4).sum()

    n_def_needed, n_mid_needed, n_fwd_needed = formation
    return (n_gk == 1 and n_def == n_def_needed and n_mid == n_mid_needed and n_fwd == n_fwd_needed)

def generate_key_player_explanations(team_df, all_players_df, n=2):
    explanations = {}
    for player in team_df.head(n).itertuples():
        player_info = all_players_df[all_players_df['id'] == player.id].iloc[0]
        explanations[player.id] = {
            'name': player_info['web_name'],
            'reason': f"Selected for {player_info['expected_points_next_5']} expected points."
        }
    return explanations

def render_team_recommendations_table(team_df, show_extended_stats=True):
    """
    Render team recommendations table with extended statistics if available
    """
    # Base columns that should always be displayed
    base_columns = [
        'web_name', 'team_name', 'element_type', 'now_cost', 'total_points', 
        'form', 'selected_by_percent', 'expected_points_next_5'
    ]
    
    # Extended statistics columns to include if available
    extended_columns = {
        'xG_next_5': 'xG Next 5',
        'xA_next_5': 'xA Next 5', 
        'expected_assists_per_90': 'xA per 90',
        'expected_goal_involvements_per_90': 'xGI per 90',
        'expected_goals_conceded_per_90': 'xGC per 90',
        'expected_goals': 'xG',
        'expected_assists': 'xA',
        'expected_goal_involvements': 'xGI',
        'expected_goals_conceded': 'xGC',
        'recoveries': 'Recoveries',
        'tackles': 'Tackles',
        'clearances_blocks_interceptions': 'CBI',
        'expected_goals_per_90': 'xG per 90'
    }
    
    # Determine which columns are actually available in the data
    available_columns = base_columns.copy()
    available_extended = {}
    
    if show_extended_stats:
        for col_key, col_display in extended_columns.items():
            if col_key in team_df.columns and not team_df[col_key].isna().all():
                available_columns.append(col_key)
                available_extended[col_key] = col_display
    
    # Create display dataframe with available columns
    display_df = team_df[available_columns].copy()
    
    # Create column name mapping for display
    column_mapping = {
        'web_name': 'Player',
        'team_name': 'Team',
        'element_type': 'Position',
        'now_cost': 'Cost (£m)',
        'total_points': 'Points',
        'form': 'Form',
        'selected_by_percent': 'Owned %',
        'expected_points_next_5': 'xPts (5GW)'
    }
    column_mapping.update(available_extended)
    
    # Format specific columns
    if 'now_cost' in display_df.columns:
        display_df['now_cost'] = (display_df['now_cost'] / 10).round(1)
    
    if 'element_type' in display_df.columns:
        position_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        display_df['element_type'] = display_df['element_type'].map(position_map)
    
    # Round numerical columns to appropriate decimal places
    numeric_columns = ['form', 'selected_by_percent', 'expected_points_next_5'] + list(available_extended.keys())
    for col in numeric_columns:
        if col in display_df.columns:
            if 'per_90' in col or col.startswith('expected_') or col in ['xG_next_5', 'xA_next_5']:
                display_df[col] = display_df[col].round(2)
            else:
                display_df[col] = display_df[col].round(1)
    
    # Rename columns for display
    display_df = display_df.rename(columns=column_mapping)
    
    # Display info about available extended stats
    if show_extended_stats and available_extended:
        st.info(f"📊 Extended stats available: {', '.join(available_extended.values())}")
    elif show_extended_stats:
        st.warning("⚠️ Extended statistics not available in current dataset")
    
    return display_df

def enhanced_team_recommendations_display(recommendations, all_players_df):
    """
    Enhanced display of team recommendations with extended statistics
    """
    if not recommendations or recommendations.get('team') is None:
        st.warning("No team recommendations available")
        return
    
    team_df = recommendations['team']
    
    # Main team overview
    st.subheader("🏆 Recommended Team")
    
    # Team summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cost = team_df['now_cost'].sum() / 10
        st.metric("Total Cost", f"£{total_cost:.1f}m")
    
    with col2:
        if 'expected_points_next_5' in team_df.columns:
            total_xpts = team_df[team_df.get('is_starting', True)]['expected_points_next_5'].sum()
            st.metric("Expected Points (5GW)", f"{total_xpts:.1f}")
    
    with col3:
        if 'xG_next_5' in team_df.columns:
            total_xg = team_df[team_df.get('is_starting', True)]['xG_next_5'].sum()
            st.metric("Team xG (5GW)", f"{total_xg:.1f}")
    
    with col4:
        if 'xA_next_5' in team_df.columns:
            total_xa = team_df[team_df.get('is_starting', True)]['xA_next_5'].sum()
            st.metric("Team xA (5GW)", f"{total_xa:.1f}")
    
    # Tabbed view for different perspectives
    tab1, tab2, tab3 = st.tabs(["🏟️ Starting XI", "🪑 Full Squad", "📈 Extended Stats"])
    
    with tab1:
        st.markdown("### Starting XI")
        starting_xi = team_df[team_df.get('is_starting', True)] if 'is_starting' in team_df.columns else team_df.head(11)
        starting_display = render_team_recommendations_table(starting_xi, show_extended_stats=False)
        st.dataframe(starting_display, use_container_width=True, hide_index=True)
    
    with tab2:
        st.markdown("### Full Squad (15 Players)")
        full_squad_display = render_team_recommendations_table(team_df, show_extended_stats=False)
        
        # Add starting status indicator
        if 'is_starting' in team_df.columns:
            full_squad_display['Status'] = team_df['is_starting'].map({True: '🏟️ Starting', False: '🪑 Bench'})
            # Reorder columns to put Status first
            cols = ['Status'] + [col for col in full_squad_display.columns if col != 'Status']
            full_squad_display = full_squad_display[cols]
        
        st.dataframe(full_squad_display, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### Extended Statistics")
        
        # Check what extended stats are available
        extended_stats_available = []
        extended_columns = [
            'xG_next_5', 'xA_next_5', 'expected_assists_per_90', 'expected_goal_involvements_per_90',
            'expected_goals_conceded_per_90', 'expected_goals', 'expected_assists', 
            'expected_goal_involvements', 'expected_goals_conceded', 'recoveries', 
            'tackles', 'clearances_blocks_interceptions', 'expected_goals_per_90'
        ]
        
        for col in extended_columns:
            if col in team_df.columns and not team_df[col].isna().all():
                extended_stats_available.append(col)
        
        if extended_stats_available:
            extended_display = render_team_recommendations_table(team_df, show_extended_stats=True)
            st.dataframe(extended_display, use_container_width=True, hide_index=True)
            
            # Position-based extended stats analysis
            st.markdown("#### 📊 Position-Based Extended Stats")
            
            position_map = {1: 'Goalkeepers', 2: 'Defenders', 3: 'Midfielders', 4: 'Forwards'}
            
            for pos_id, pos_name in position_map.items():
                pos_players = team_df[team_df['element_type'] == pos_id]
                if not pos_players.empty:
                    with st.expander(f"{pos_name} ({len(pos_players)} players)"):
                        pos_display = render_team_recommendations_table(pos_players, show_extended_stats=True)
                        st.dataframe(pos_display, use_container_width=True, hide_index=True)
                        
                        # Position-specific insights
                        if pos_id == 1 and 'expected_goals_conceded' in pos_players.columns:  # Goalkeepers
                            avg_xgc = pos_players['expected_goals_conceded'].mean()
                            st.metric("Average xGC", f"{avg_xgc:.2f}")
                        
                        elif pos_id == 2 and 'clearances_blocks_interceptions' in pos_players.columns:  # Defenders
                            avg_cbi = pos_players['clearances_blocks_interceptions'].mean()
                            st.metric("Average CBI", f"{avg_cbi:.1f}")
                        
                        elif pos_id in [3, 4] and 'expected_goal_involvements' in pos_players.columns:  # Mids/Forwards
                            avg_xgi = pos_players['expected_goal_involvements'].mean()
                            st.metric("Average xGI", f"{avg_xgi:.2f}")
        else:
            st.info("📋 Extended statistics are not available in the current dataset. The table will show standard FPL metrics.")
            
            # Show what could be available
            st.markdown("**Extended stats that could be displayed when available:**")
            potential_stats = [
                "xG Next 5", "xA Next 5", "Expected Assists per 90", "Expected Goal Involvements per 90",
                "Expected Goals Conceded per 90", "Expected Goals", "Expected Assists", 
                "Expected Goal Involvements", "Expected Goals Conceded", "Recoveries", 
                "Tackles", "Clearances/Blocks/Interceptions", "Expected Goals per 90"
            ]
            
            col1, col2 = st.columns(2)
            for i, stat in enumerate(potential_stats):
                if i % 2 == 0:
                    col1.markdown(f"• {stat}")
                else:
                    col2.markdown(f"• {stat}")