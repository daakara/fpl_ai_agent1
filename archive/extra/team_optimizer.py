import pandas as pd
import numpy as np
import pulp

def build_team_lp(
    df_players,
    budget=100,
    formation=(4,3,3),
    max_players_per_club=3,
    max_budget_per_player=15,
    n_sub_gk=1,
    n_sub_outfield=3,
    injury_penalty=0.5,
    hard_fixture_penalty=0.8,
    min_ownership=0.0,
    form_weight=0.3,
    xg_weight=0.4,
    xa_weight=0.3,
    ownership_weight=0.2,
    debug=False,
):
    """
    Build optimal FPL team with linear integer programming.
    """

    GK, DEF, MID, FWD = 1, 2, 3, 4

    df = df_players.copy()

    # Precompute and fill numeric columns
    df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0)
    df["expected_points_next_5"] = pd.to_numeric(df["expected_points_next_5"], errors="coerce").fillna(0)
    df["xG_next_5"] = pd.to_numeric(df.get("xG_next_5", 0), errors="coerce").fillna(0)
    df["xA_next_5"] = pd.to_numeric(df.get("xA_next_5", 0), errors="coerce").fillna(0)
    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0)
    df["now_cost"] = pd.to_numeric(df["now_cost"], errors="coerce").fillna(0)
    df["now_cost_m"] = df["now_cost"] / 10

    # Filter early
    df = df[(df["selected_by_percent"] >= min_ownership) & (df["now_cost_m"] <= max_budget_per_player)].copy()

    # Penalties
    df["injury_penalty_factor"] = np.where(
        df["injury_status"].isin(["Injured", "Doubtful"]), injury_penalty, 1.0)
    df["fixture_penalty_factor"] = np.where(
        df["fixture_difficulty"] == "Hard", hard_fixture_penalty,
        np.where(df["fixture_difficulty"] == "Medium", 1.0, 1.1),
    )

    df["adjusted_expected_points"] = (
        df["expected_points_next_5"] * df["injury_penalty_factor"] * df["fixture_penalty_factor"]
    )

    # Combined score
    df["combined_score"] = (
        (1 - form_weight) * df["adjusted_expected_points"]
        + form_weight * df["form"]
        + xg_weight * df["xG_next_5"]
        + xa_weight * df["xA_next_5"]
        + ownership_weight * df["selected_by_percent"]
    )

    if debug:
        print(f"Players after filtering: {len(df)}")

    # Create LP model
    model = pulp.LpProblem("FPL_Team_Optimization", pulp.LpMaximize)

    # Player selection binary variables
    player_vars = {pid: pulp.LpVariable(f"x_{pid}", cat="Binary") for pid in df.index}

    # Objective: maximize total combined score
    model += pulp.lpSum(df.loc[pid, "combined_score"] * player_vars[pid] for pid in df.index)

    # Constraints

    # 1) Exactly 15 players selected
    model += pulp.lpSum(player_vars[pid] for pid in df.index) == 15, "TotalPlayers15"

    # 2) Formation constraints for starters (assume starters = 11)
    # Because we don't distinguish starters/subs in LP easily without complexity,
    # enforce formation counts for starting 11, plus subs being remainder (4 players)
    # Approximate by formation + bench constraint combos:

    # We ensure total GK count is 2 (1 starter + 1 bench)
    model += pulp.lpSum(player_vars[pid] for pid in df.index if df.loc[pid, "element_type"] == GK) == n_sub_gk + 1, "GK_count"

    # Defenders count (starters + bench)
    # Bench outfield = 3 subs
    model += pulp.lpSum(player_vars[pid] for pid in df.index if df.loc[pid, "element_type"] == DEF) >= formation[0], "DEF_starters_min"
    model += pulp.lpSum(player_vars[pid] for pid in df.index if df.loc[pid, "element_type"] == DEF) <= formation[0] + n_sub_outfield, "DEF_max"

    # Midfielders
    model += pulp.lpSum(player_vars[pid] for pid in df.index if df.loc[pid, "element_type"] == MID) >= formation[1], "MID_starters_min"
    model += pulp.lpSum(player_vars[pid] for pid in df.index if df.loc[pid, "element_type"] == MID) <= formation[1] + n_sub_outfield, "MID_max"

    # Forwards
    model += pulp.lpSum(player_vars[pid] for pid in df.index if df.loc[pid, "element_type"] == FWD) >= formation[2], "FWD_starters_min"
    model += pulp.lpSum(player_vars[pid] for pid in df.index if df.loc[pid, "element_type"] == FWD) <= formation[2] + n_sub_outfield, "FWD_max"

    # 3) Maximum 3 players from a single team
    teams = df["team_name"].unique()
    for team_name in teams:
        model += pulp.lpSum(player_vars[pid] for pid in df.index if df.loc[pid, "team_name"] == team_name) <= max_players_per_club, f"MaxPlayers_{team_name}"

    # 4) Total cost must be within budget
    model += pulp.lpSum(df.loc[pid, "now_cost_m"] * player_vars[pid] for pid in df.index) <= budget, "BudgetConstraint"

    # Solve the model
    status = model.solve(pulp.PULP_CBC_CMD(msg=0))

    if debug:
        print(f"Solver Status: {pulp.LpStatus[status]}")

    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError("No valid team found under constraints via LP solver")

    # Extract selected players
    selected_pids = [pid for pid in df.index if player_vars[pid].varValue > 0.9]
    team_df = df.loc[selected_pids].copy()
    
    # Set "is_starting" to True for players above threshold in formation positions (approximate: top 11 by combined score)
    team_df = team_df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    team_df["is_starting"] = False
    # Assign starters based on formation counts
    starters_count = 11
    team_df.loc[:starters_count-1, "is_starting"] = True

    # Choose captain & vice captain as top 2 starters by adjusted_expected_points
    starters = team_df[team_df["is_starting"]].sort_values("adjusted_expected_points", ascending=False)
    captain = starters.iloc[0].name if not starters.empty else None
    vice_captain = starters.iloc[1].name if len(starters) > 1 else None

    if debug:
        total_cost = team_df["now_cost_m"].sum()
        total_points = team_df.loc[team_df["is_starting"], "adjusted_expected_points"].sum()
        print(f"Total cost: {total_cost:.1f}m")
        print(f"Total expected points (starters): {total_points:.2f}")
        print("Selected players:")
        for _, p in team_df.iterrows():
            print(f"{p['web_name']} - {p['team_name']} - {p['element_type']} - cost: {p['now_cost_m']:.1f}m - combined_score: {p['combined_score']:.2f}")

    return {
        "team": team_df.reset_index(drop=True),
        "formation": formation,
        "captain": captain,
        "vice_captain": vice_captain,
        "total_expected_points": team_df.loc[team_df["is_starting"], "adjusted_expected_points"].sum(),
        "total_cost": team_df["now_cost_m"].sum(),
    }

def preprocess_players(df):
    """
    Preprocess the raw FPL player DataFrame to ensure numeric columns are clean,
    missing columns are created if necessary, and all needed fields exist with 
    appropriate defaults and types.

    Args:
        df (pd.DataFrame): Raw player data

    Returns:
        pd.DataFrame: Preprocessed player data ready for team optimization
    """
    df = df.copy()

    # Ensure selected_by_percent numeric and fill missing
    if "selected_by_percent" in df.columns:
        df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce").fillna(0)
    else:
        df["selected_by_percent"] = pd.Series(0, index=df.index)

    # expected_points_next_5 either exists or create as (form * 5)
    if "expected_points_next_5" not in df.columns:
        if "form" in df.columns:
            df["expected_points_next_5"] = df["form"].apply(lambda x: float(x) if pd.notna(x) else 0) * 5
        else:
            df["expected_points_next_5"] = pd.Series(0, index=df.index)
    df["expected_points_next_5"] = pd.to_numeric(df["expected_points_next_5"], errors="coerce").fillna(0)

    # xG_next_5 numeric or zero Series
    if "xG_next_5" in df.columns:
        df["xG_next_5"] = pd.to_numeric(df["xG_next_5"], errors="coerce").fillna(0)
    else:
        df["xG_next_5"] = pd.Series(0, index=df.index)

    # xA_next_5 numeric or zero Series
    if "xA_next_5" in df.columns:
        df["xA_next_5"] = pd.to_numeric(df["xA_next_5"], errors="coerce").fillna(0)
    else:
        df["xA_next_5"] = pd.Series(0, index=df.index)

    # form numeric or zero Series
    if "form" in df.columns:
        df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0)
    else:
        df["form"] = pd.Series(0, index=df.index)

    # now_cost numeric or zero Series
    if "now_cost" in df.columns:
        df["now_cost"] = pd.to_numeric(df["now_cost"], errors="coerce").fillna(0)
    else:
        df["now_cost"] = pd.Series(0, index=df.index)

    # Compute easier cost representation in millions
    df["now_cost_m"] = df["now_cost"] / 10

    # team_name fill missing with "Unknown"
    if "team_name" not in df.columns:
        df["team_name"] = pd.Series("Unknown", index=df.index)
    else:
        df["team_name"] = df["team_name"].fillna("Unknown")

    # Optional: injury_status fill missing if used downstream
    if "injury_status" not in df.columns:
        df["injury_status"] = pd.Series("", index=df.index)
    else:
        df["injury_status"] = df["injury_status"].fillna("")

    # Optional: fixture_difficulty fill missing if used downstream
    if "fixture_difficulty" not in df.columns:
        df["fixture_difficulty"] = pd.Series("Medium", index=df.index)
    else:
        df["fixture_difficulty"] = df["fixture_difficulty"].fillna("Medium")

    return df
