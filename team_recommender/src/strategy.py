from typing import List, Dict
import pandas as pd
import numpy as np

def optimize_bench_selection(team: pd.DataFrame, budget: float, max_bench_size: int) -> List[Dict]:
    bench_candidates = team[~team['is_starting']]
    bench_candidates = bench_candidates[bench_candidates['now_cost'] <= budget]
    bench_candidates = bench_candidates.sort_values(by='expected_points_next_5', ascending=False)

    bench_selection = bench_candidates.head(max_bench_size).to_dict(orient='records')
    return bench_selection

def select_captain(team: pd.DataFrame) -> Dict:
    starters = team[team['is_starting']]
    captain = starters.loc[starters['form'].idxmax()]
    return captain.to_dict()

def calculate_team_chemistry(team: pd.DataFrame) -> float:
    chemistry_score = 0.0
    team_counts = team['team_name'].value_counts()

    for count in team_counts:
        if count > 1:
            chemistry_score += (count - 1) * 0.1  # Bonus for each additional player from the same team

    return chemistry_score

def iterative_team_optimization(df_players: pd.DataFrame, budget: float, formations: List[tuple], max_bench_size: int) -> Dict:
    best_team = None
    best_score = -float('inf')

    for formation in formations:
        team = get_latest_team_recommendations(df_players, budget, formations=[formation])
        if team is not None:
            bench = optimize_bench_selection(team, budget, max_bench_size)
            captain = select_captain(team)
            chemistry = calculate_team_chemistry(team)

            total_score = team['expected_points_next_5'].sum() + chemistry

            if total_score > best_score:
                best_score = total_score
                best_team = {
                    'team': team,
                    'bench': bench,
                    'captain': captain,
                    'total_score': total_score
                }

    return best_team

def optimize_bench_selection_advanced(team: pd.DataFrame, budget: float, max_bench_size: int, starting_xi: pd.DataFrame) -> List[Dict]:
    """Advanced bench optimization considering injury coverage and value"""
    
    starting_ids = set(starting_xi['id'].tolist())
    starting_positions = starting_xi['element_type'].value_counts()
    
    # Available players for bench
    bench_candidates = team[~team['id'].isin(starting_ids) & (team['now_cost'] <= budget)].copy()
    
    if bench_candidates.empty:
        return []
    
    # Calculate bench value score
    bench_candidates['bench_score'] = calculate_bench_player_value(
        bench_candidates, starting_xi, budget
    )
    
    # Select optimal bench composition
    bench_selection = []
    remaining_budget = budget
    positions_covered = set()
    
    # Priority 1: Substitute goalkeeper (cheapest reliable option)
    gk_candidates = bench_candidates[bench_candidates['element_type'] == 1]
    if not gk_candidates.empty and 1 not in starting_positions:
        gk_sub = gk_candidates.loc[gk_candidates['now_cost'].idxmin()]
        bench_selection.append(gk_sub.to_dict())
        remaining_budget -= gk_sub['now_cost']
        positions_covered.add(1)
        bench_candidates = bench_candidates[bench_candidates['id'] != gk_sub['id']]
    
    # Priority 2: Outfield players with rotation potential
    outfield_candidates = bench_candidates[
        (bench_candidates['element_type'].isin([2, 3, 4])) &
        (bench_candidates['now_cost'] <= remaining_budget)
    ].sort_values('bench_score', ascending=False)
    
    for _, player in outfield_candidates.iterrows():
        if len(bench_selection) >= max_bench_size:
            break
            
        if player['now_cost'] <= remaining_budget:
            bench_selection.append(player.to_dict())
            remaining_budget -= player['now_cost']
    
    return bench_selection[:max_bench_size]

def calculate_bench_player_value(candidates: pd.DataFrame, starting_xi: pd.DataFrame, budget: float) -> pd.Series:
    """Calculate value score for potential bench players"""
    
    # Playing time likelihood
    candidates['minutes_ratio'] = candidates['minutes'] / candidates['minutes'].max()
    
    # Points per appearance
    candidates['points_per_appearance'] = candidates['total_points'] / np.maximum(1, candidates['minutes'] / 90)
    
    # Value for money
    candidates['value_score'] = candidates['points_per_appearance'] / (candidates['now_cost'] / 10)
    
    # Form momentum
    candidates['form_score'] = pd.to_numeric(candidates['form'], errors='coerce').fillna(0)
    
    # Position scarcity in starting XI
    starting_position_counts = starting_xi['element_type'].value_counts()
    candidates['position_scarcity'] = candidates['element_type'].map(
        lambda x: 1.0 + (1.0 / max(1, starting_position_counts.get(x, 1)))
    )
    
    # Calculate composite bench score
    bench_score = (
        candidates['value_score'] * 0.3 +
        candidates['form_score'] * 0.25 +
        candidates['minutes_ratio'] * 0.2 +
        candidates['position_scarcity'] * 0.15 +
        (budget / candidates['now_cost']) * 0.1  # Budget efficiency
    )
    
    return bench_score

def select_captain_advanced(team: pd.DataFrame, gameweek_data: Dict = None) -> Dict:
    """Advanced captain selection considering multiple factors"""
    
    starters = team[team.get('is_starting', True)]
    if starters.empty:
        return {"captain": None, "reasoning": "No starting players available"}
    
    # Calculate captaincy scores
    captaincy_scores = []
    
    for _, player in starters.iterrows():
        score = calculate_captaincy_score_advanced(player, gameweek_data)
        captaincy_scores.append({
            'player': player.to_dict(),
            'score': score,
            'factors': get_captaincy_factors(player, gameweek_data)
        })
    
    # Sort by score and select captain
    captaincy_scores.sort(key=lambda x: x['score'], reverse=True)
    
    captain_choice = captaincy_scores[0] if captaincy_scores else None
    vice_captain_choice = captaincy_scores[1] if len(captaincy_scores) > 1 else None
    
    return {
        "captain": captain_choice,
        "vice_captain": vice_captain_choice,
        "reasoning": generate_captaincy_reasoning(captain_choice, vice_captain_choice),
        "alternatives": captaincy_scores[2:5]  # Show top alternatives
    }

def calculate_captaincy_score_advanced(player: pd.Series, gameweek_data: Dict = None) -> float:
    """Calculate advanced captaincy score for a player"""
    
    # Base expected points (most important factor)
    base_score = float(player.get('expected_points_next_5', 0)) / 5  # Per gameweek
    
    # Form factor (recent performance)
    form_factor = float(player.get('form', 0)) * 0.2
    
    # Position multiplier (attackers favored)
    position_multipliers = {1: 0.5, 2: 0.7, 3: 0.9, 4: 1.0}
    position_factor = position_multipliers.get(player.get('element_type', 3), 0.8)
    
    # Fixture difficulty (easier = better for captain)
    fixture_difficulty = player.get('fixture_difficulty', 3)
    fixture_factor = max(0, (5 - fixture_difficulty) * 0.1)
    
    # Consistency factor
    consistency = player.get('consistency', 0.5)
    consistency_factor = consistency * 0.15
    
    # Goal/assist potential
    goal_potential = float(player.get('xG_next_5', 0)) * 0.1
    assist_potential = float(player.get('xA_next_5', 0)) * 0.08
    
    # Calculate final score
    total_score = (
        base_score * position_factor +
        form_factor +
        fixture_factor +
        consistency_factor +
        goal_potential +
        assist_potential
    )
    
    return round(total_score, 2)

def get_captaincy_factors(player: pd.Series, gameweek_data: Dict = None) -> Dict:
    """Get detailed factors contributing to captaincy score"""
    return {
        'expected_points': player.get('expected_points_next_5', 0),
        'form': player.get('form', 0),
        'position': {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(player.get('element_type'), 'Unknown'),
        'fixture_difficulty': player.get('fixture_difficulty', 3),
        'consistency': player.get('consistency', 0.5),
        'xG': player.get('xG_next_5', 0),
        'xA': player.get('xA_next_5', 0),
        'total_points': player.get('total_points', 0)
    }

def generate_captaincy_reasoning(captain_choice: Dict, vice_captain_choice: Dict) -> str:
    """Generate human-readable reasoning for captaincy choices"""
    if not captain_choice:
        return "No suitable captain found"
    
    captain = captain_choice['player']
    reasoning = f"**Captain: {captain['web_name']}** - "
    
    factors = captain_choice['factors']
    reasons = []
    
    if factors['expected_points'] > 5:
        reasons.append(f"high expected points ({factors['expected_points']:.1f})")
    
    if factors['form'] > 4:
        reasons.append(f"excellent form ({factors['form']:.1f})")
    
    if factors['fixture_difficulty'] <= 2:
        reasons.append("easy fixtures")
    
    if factors['xG'] > 1:
        reasons.append(f"strong goal threat (xG: {factors['xG']:.1f})")
    
    reasoning += ", ".join(reasons) if reasons else "best overall option"
    
    if vice_captain_choice:
        vc = vice_captain_choice['player']
        reasoning += f". **Vice-Captain: {vc['web_name']}** - reliable backup option"
    
    return reasoning

def iterative_team_optimization_enhanced(
    df_players: pd.DataFrame, 
    budget: float, 
    formations: List[tuple], 
    max_iterations: int = 5,
    convergence_threshold: float = 0.01
) -> Dict:
    """Enhanced iterative optimization with multiple strategies"""
    
    best_team = None
    best_score = -float('inf')
    iteration_history = []
    
    for iteration in range(max_iterations):
        # Try different optimization strategies per iteration
        if iteration == 0:
            # Initial greedy selection
            strategy = "greedy"
            team = optimize_team_greedy(df_players, budget, formations[0])
        elif iteration == 1:
            # Value-based optimization
            strategy = "value_based"
            team = optimize_team_value_based(df_players, budget, formations)
        else:
            # Local search improvements on best team
            strategy = "local_search"
            team = optimize_team_local_search(best_team, df_players, budget)
        
        if team is None:
            continue
        
        # Optimize bench for current team
        starting_xi = team['team'][team['team']['is_starting']]
        bench = optimize_bench_selection_advanced(
            df_players, 
            budget - starting_xi['now_cost'].sum(),
            4,
            starting_xi
        )
        
        # Add bench to team
        full_team = starting_xi.to_dict('records') + bench
        
        # Select optimal captain
        captain_info = select_captain_advanced(pd.DataFrame(full_team))
        
        # Calculate team score
        team_score = calculate_comprehensive_team_score(full_team, captain_info)
        
        iteration_history.append({
            'iteration': iteration + 1,
            'strategy': strategy,
            'score': team_score,
            'improvement': team_score - best_score if best_score != -float('inf') else 0
        })
        
        # Check for improvement
        if team_score > best_score:
            improvement_ratio = (team_score - best_score) / abs(best_score) if best_score != 0 else float('inf')
            
            best_score = team_score
            best_team = {
                'team': pd.DataFrame(full_team),
                'captain_info': captain_info,
                'total_score': team_score,
                'iteration': iteration + 1,
                'strategy_used': strategy,
                'formation': formations[0],  # Use first formation for now
                'optimization_history': iteration_history.copy()
            }
            
            # Check convergence
            if iteration > 0 and improvement_ratio < convergence_threshold:
                break
    
    return best_team

def calculate_comprehensive_team_score(team_players: List[Dict], captain_info: Dict) -> float:
    """Calculate comprehensive team score including all factors"""
    
    starting_players = [p for p in team_players if p.get('is_starting', True)]
    bench_players = [p for p in team_players if not p.get('is_starting', False)]
    
    # Starting XI expected points
    starting_score = sum(p.get('expected_points_next_5', 0) for p in starting_players)
    
    # Captain bonus (double points for captain)
    captain_bonus = 0
    if captain_info.get('captain'):
        captain_expected = captain_info['captain']['player'].get('expected_points_next_5', 0)
        captain_bonus = captain_expected  # Additional points from captaincy
    
    # Bench value (weighted by likelihood of playing)
    bench_score = sum(p.get('expected_points_next_5', 0) * 0.1 for p in bench_players)
    
    # Team chemistry bonus
    team_df = pd.DataFrame(team_players)
    chemistry_bonus = calculate_team_chemistry(team_df)
    
    # Budget efficiency bonus
    total_cost = sum(p.get('now_cost', 0) for p in team_players)
    budget_efficiency = max(0, (1000 - total_cost) * 0.001)  # Small bonus for unused budget
    
    total_score = (
        starting_score +
        captain_bonus +
        bench_score +
        chemistry_bonus +
        budget_efficiency
    )
    
    return round(total_score, 2)