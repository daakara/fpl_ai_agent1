import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from thefuzz import process
import logging
from typing import Optional

def fuzzy_match_player(player_name: str, player_list: pd.Series, threshold: int = 85) -> Optional[str]:
    if player_list.empty:
        return None
    match, score = process.extractOne(player_name, player_list.dropna().unique())
    return match if score >= threshold else None

def prepare_ml_data_from_scrapes(ffscout_lineups, fbref_stats) -> pd.DataFrame:
    players_data = []
    if fbref_stats is None or fbref_stats.empty:
        print("FBref stats DataFrame empty, returning empty ML data.")
        return pd.DataFrame()

    fbref_names = fbref_stats['Player'].astype(str).tolist()

    for lineup in ffscout_lineups:
        for player_name in lineup:
            matched_name = fuzzy_match_player(player_name, fbref_stats['Player'])
            if not matched_name:
                print(f"No FBref match found for player: {player_name}")
                continue
            row = fbref_stats[fbref_stats['Player'] == matched_name].iloc[0]

            # Check for required columns before extracting values
            required_cols = ['Gls', 'Ast', 'MP']
            missing_cols = [col for col in required_cols if col not in row.index]
            if missing_cols:
                print(f"Missing expected columns {missing_cols} for player {matched_name}, skipping.")
                continue

            features = {
                'name': player_name,
                'xG': row.get('Gls', 0),
                'xA': row.get('Ast', 0),
                'form': row.get('MP', 0),
                'price': 6.0,  # Placeholder: best if from official API
                'points_last3': row.get('Gls', 0)  # Use goals as proxy
            }
            players_data.append(features)

    if not players_data:
        print("No valid player data extracted after matching.")
        return pd.DataFrame()

    df = pd.DataFrame(players_data)
    df.fillna(0, inplace=True)
    print(f"Prepared ML data with shape: {df.shape}")
    return df

def train_player_selection_model(historic_data: pd.DataFrame) -> RandomForestClassifier:
    features = ['xG', 'xA', 'form', 'price', 'points_last3']
    X = historic_data[features]
    y = historic_data.get('target', pd.Series([1]*len(X)))
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    print("Trained RandomForestClassifier model.")
    return clf

def predict_good_picks(current_players: pd.DataFrame, clf: RandomForestClassifier) -> pd.DataFrame:
    features = ['xG', 'xA', 'form', 'price', 'points_last3']
    X_new = current_players[features]
    probabilities = clf.predict_proba(X_new)[:, 1]
    results = current_players.copy()
    results['pick_probability'] = probabilities
    results_sorted = results.sort_values(by='pick_probability', ascending=False)
    print("Predicted pick probabilities for current players.")
    return results_sorted
