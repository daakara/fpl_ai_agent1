import requests
import pandas as pd
from io import StringIO

def fetch_expected_points(url="https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/expected_points.csv"):
    """
    Download expected points CSV dataset from GitHub (public community source).
    Returns DataFrame with columns including 'name' (player names) and 'expected_points'.
    """
    r = requests.get(url, verify=False, timeout=15)
    r.raise_for_status()
    csv_data = StringIO(r.text)
    df = pd.read_csv(csv_data)
    # Expected columns: 'name', 'expected_points', etc.
    return df

if __name__ == "__main__":
    df = fetch_expected_points()
    print(df.head())
