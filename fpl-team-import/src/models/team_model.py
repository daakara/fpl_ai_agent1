import streamlit as st
from filters import FilterManager
from data_loader import load_player_data
from fpl_myteam import load_my_fpl_team  # Import the function to load FPL team data

def main():
    st.title("Fantasy Football Team Selection")

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Player Filters", "Import FPL Team"])

    with tab1:
        # Load player data
        players_df = load_player_data()

        # Initialize filter manager
        filter_manager = FilterManager(players_df)

        # Display filters
        st.sidebar.header("Filters")
        price_range = st.sidebar.slider("Price Range", 0, 15, (5, 10))
        selected_teams = st.sidebar.multiselect("Select Teams", options=players_df['team_name'].unique())
        selected_positions = st.sidebar.multiselect("Select Positions", options=players_df['element_type'].unique())
        fixture_difficulty = st.sidebar.selectbox("Fixture Difficulty", options=["All", "Easy", "Medium", "Hard"])

        # Apply filters
        filtered_players = filter_manager.apply_filters(price_range, selected_teams, selected_positions, fixture_difficulty)

        # Display filtered players
        st.subheader("Filtered Players")
        st.dataframe(filtered_players)

    with tab2:
        # Import user's FPL team
        st.header("Import Your FPL Team")
        team_id = st.text_input("Enter your FPL Team ID:")
        
        if st.button("Load Team"):
            if team_id:
                try:
                    team_data = load_my_fpl_team(int(team_id))  # Load the team data using the provided ID
                    st.success("Team loaded successfully!")
                    st.json(team_data)  # Display the team data in JSON format
                except Exception as e:
                    st.error(f"Error loading team: {e}")
            else:
                st.warning("Please enter a valid Team ID.")

if __name__ == "__main__":
    main()