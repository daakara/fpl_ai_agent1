def recommend_chip_strategy(fixtures, team_state, chips_available):
    """Suggest optimal chip timing based on DGWs, blanks, fixtures."""
    # Pseudocode: Replace with actual fixture analysis logic
    advice = []
    if "Wildcard" in chips_available:
        advice.append("Wildcard: Use when many key players have bad fixtures or you need multiple changes.")
    if "Triple Captain" in chips_available:
        # Look for double gameweeks
        if fixtures["DGW"].any():
            advice.append("Triple Captain: Play in a double gameweek for your star attacker.")
    if "Bench Boost" in chips_available and fixtures["BenchStrength"].max() > 8:
        advice.append("Bench Boost: Use in weeks where all bench players have easy fixtures and are likely to start.")
    if "Free Hit" in chips_available and fixtures["Blanks"].any():
        advice.append("Free Hit: Save for major blank gameweeeks to field 11 players.")
    return advice
def fetch_chip_data() -> Dict:
    """
    Fetches chip data from the official FPL API.
    SSL verification is disabled temporarily for debugging.
    """
    response = requests.get(FPL_CHIP_URL, timeout=10, verify=False)
