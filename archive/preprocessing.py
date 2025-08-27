import pandas as pd

def preprocess_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the raw FPL player DataFrame to ensure numeric columns are clean,
    missing columns are created if necessary, and all needed fields exist with
    appropriate defaults and types. This function uses pd.to_numeric with
    errors='coerce' to handle invalid data gracefully.

    Args:
        df (pd.DataFrame): Raw player data

    Returns:
        pd.DataFrame: Preprocessed player data ready for use in the application.
    """
    df = df.copy()

    # List of columns to convert to numeric, filling NaNs with 0
    numeric_cols = [
        "selected_by_percent", "form", "now_cost",
        "expected_points_next_5", "xG_next_5", "xA_next_5"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            df[col] = 0.0

    # If expected points are missing, estimate from form
    if "expected_points_next_5" not in df.columns or df["expected_points_next_5"].sum() == 0:
        df["expected_points_next_5"] = df["form"] *
