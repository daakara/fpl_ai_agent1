import pandas as pd

def load_players_data(filepath):
    """
    Load player data from a CSV file into a pandas DataFrame.
    
    Parameters:
    - filepath: str, path to the CSV file containing player data.
    
    Returns:
    - DataFrame containing player data.
    """
    return pd.read_csv(filepath)

def load_fixtures_data(filepath):
    """
    Load fixture data from a CSV file into a pandas DataFrame.
    
    Parameters:
    - filepath: str, path to the CSV file containing fixture data.
    
    Returns:
    - DataFrame containing fixture data.
    """
    return pd.read_csv(filepath)

def preprocess_players_data(df):
    """
    Preprocess player data DataFrame.
    
    Parameters:
    - df: DataFrame containing player data.
    
    Returns:
    - DataFrame after preprocessing.
    """
    # Convert cost to numeric and handle any errors
    df['now_cost'] = pd.to_numeric(df['now_cost'], errors='coerce')
    
    # Fill missing values in injury status
    df['injury_status'] = df['injury_status'].fillna('Fit')
    
    return df

def preprocess_fixtures_data(df):
    """
    Preprocess fixture data DataFrame.
    
    Parameters:
    - df: DataFrame containing fixture data.
    
    Returns:
    - DataFrame after preprocessing.
    """
    # Convert fixture difficulty to numeric and handle any errors
    df['fixture_difficulty'] = pd.to_numeric(df['fixture_difficulty'], errors='coerce')
    
    return df