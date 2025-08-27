"""
UI Components for FPL Analytics Application
Following Single Responsibility Principle - each component has one clear purpose
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

from models.domain_models import Player, Team, FDRData, UserTeam, AnalysisConfig, AnalysisType


class IUIComponent(ABC):
    """Base interface for UI components"""
    
    @abstractmethod
    def render(self, **kwargs) -> None:
        """Render the component"""
        pass


class ExplanationComponent(IUIComponent):
    """Reusable component for rendering explanations"""
    
    def __init__(self, title: str, content: str, expanded: bool = False):
        self.title = title
        self.content = content
        self.expanded = expanded
    
    def render(self, **kwargs) -> None:
        """Render explanation in expandable section"""
        with st.expander(self.title, expanded=self.expanded):
            st.markdown(self.content)


class MetricsDisplayComponent(IUIComponent):
    """Component for displaying key metrics"""
    
    def render(self, metrics: List[Dict], **kwargs) -> None:
        """Render metrics in columns"""
        if not metrics:
            return
        
        cols = st.columns(len(metrics))
        for i, metric in enumerate(metrics):
            with cols[i]:
                st.metric(
                    metric['label'],
                    metric['value'],
                    metric.get('delta', None),
                    metric.get('help', None)
                )


class FDRHeatmapComponent(IUIComponent):
    """Component for rendering FDR heatmaps"""
    
    def __init__(self, fdr_colors: Dict[int, str]):
        self.fdr_colors = fdr_colors
    
    def render(self, fdr_data: List[FDRData], fdr_type: str = 'combined', **kwargs) -> None:
        """Render FDR heatmap"""
        if not fdr_data:
            st.warning("No FDR data available for heatmap")
            return
        
        # Convert to DataFrame for pivot operations
        df = pd.DataFrame([
            {
                'team_short_name': fdr.team_short_name,
                'fixture_number': fdr.fixture_number,
                'opponent_short_name': fdr.opponent_short_name,
                f'{fdr_type}_fdr': getattr(fdr, f'{fdr_type}_fdr', fdr.combined_fdr)
            }
            for fdr in fdr_data
        ])
        
        if df.empty:
            st.warning("No data available for heatmap")
            return
        
        # Create pivot table
        pivot_data = df.pivot_table(
            index='team_short_name',
            columns='fixture_number',
            values=f'{fdr_type}_fdr',
            aggfunc='first'
        )
        
        # Create hover text
        hover_text = df.pivot_table(
            index='team_short_name',
            columns='fixture_number',
            values='opponent_short_name',
            aggfunc='first'
        )
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=[f'GW{i}' for i in pivot_data.columns],
            y=pivot_data.index,
            text=hover_text.values,
            colorscale=[
                [0.0, self.fdr_colors.get(1, '#00FF87')],
                [0.25, self.fdr_colors.get(2, '#01FF70')],
                [0.5, self.fdr_colors.get(3, '#FFDC00')],
                [0.75, self.fdr_colors.get(4, '#FF851B')],
                [1.0, self.fdr_colors.get(5, '#FF4136')]
            ],
            zmin=1, zmax=5,
            hovertemplate='<b>%{y}</b><br>Fixture %{x}<br>FDR: %{z}<br>vs %{text}<extra></extra>',
            colorbar=dict(
                title="FDR",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["Very Easy", "Easy", "Average", "Hard", "Very Hard"]
            )
        ))
        
        fig.update_layout(
            title=f'{fdr_type.title()} FDR - Next 5 Fixtures',
            xaxis_title="Fixture Number",
            yaxis_title="Team",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)


class PlayerTableComponent(IUIComponent):
    """Component for displaying player data tables"""
    
    def render(self, players: List[Player], config: Optional[AnalysisConfig] = None, **kwargs) -> None:
        """Render player data table"""
        if not players:
            st.warning("No player data available")
            return
        
        # Convert players to DataFrame
        player_data = []
        for player in players:
            player_dict = {
                'Player': player.web_name,
                'Position': player.position_name,
                'Team': player.team_short_name,
                'Price': f"£{player.cost_millions:.1f}m",
                'Points': player.total_points,
                'PPG': f"{player.points_per_game:.1f}",
                'Form': f"{player.form:.1f}",
                'PPM': f"{player.points_per_million:.1f}",
                'Selected': f"{player.selected_by_percent:.1f}%"
            }
            
            # Add position-specific stats
            if player.position_name == "GK":
                player_dict['Saves'] = player.saves
                player_dict['CS'] = player.clean_sheets
            elif player.position_name == "DEF":
                player_dict['CS'] = player.clean_sheets
                player_dict['Goals'] = player.goals_scored
                player_dict['Assists'] = player.assists
            else:  # MID, FWD
                player_dict['Goals'] = player.goals_scored
                player_dict['Assists'] = player.assists
            
            player_data.append(player_dict)
        
        df = pd.DataFrame(player_data)
        
        # Configure columns
        column_config = {
            "Price": st.column_config.TextColumn("Price", help="Current market price"),
            "Points": st.column_config.NumberColumn("Points", help="Total points this season"),
            "PPG": st.column_config.TextColumn("PPG", help="Points per game"),
            "Form": st.column_config.TextColumn("Form", help="Form over last 5 games"),
            "PPM": st.column_config.TextColumn("PPM", help="Points per million"),
            "Selected": st.column_config.TextColumn("Ownership", help="Percentage ownership")
        }
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )


class TeamFormationComponent(IUIComponent):
    """Component for displaying team formation and squad"""
    
    def render(self, user_team: UserTeam, players: List[Player], **kwargs) -> None:
        """Render team formation display"""
        if not user_team.picks or not players:
            st.warning("No team data available")
            return
        
        # Create player lookup
        player_lookup = {p.id: p for p in players}
        
        # Group players by position
        formation = {'GK': [], 'DEF': [], 'MID': [], 'FWD': []}
        
        starting_eleven = user_team.get_starting_eleven()
        
        for pick in starting_eleven:
            player = player_lookup.get(pick.element)
            if not player:
                continue
            
            position = player.position_name
            if position in formation:
                formation[position].append({
                    'player': player,
                    'pick': pick
                })
        
        # Display formation
        formation_str = f"{len(formation['GK'])}-{len(formation['DEF'])}-{len(formation['MID'])}-{len(formation['FWD'])}"
        st.info(f"**Formation:** {formation_str}")
        
        # Display by position
        for position, position_players in formation.items():
            if position_players:
                st.subheader(f"{position}")
                cols = st.columns(len(position_players))
                
                for i, player_data in enumerate(position_players):
                    with cols[i]:
                        player = player_data['player']
                        pick = player_data['pick']
                        
                        # Player card
                        captain_indicator = "👑" if pick.is_captain else "🅒" if pick.is_vice_captain else ""
                        st.write(f"**{captain_indicator} {player.web_name}**")
                        st.write(f"£{player.cost_millions:.1f}m | {player.total_points} pts")
                        st.write(f"Form: {player.form:.1f}")


class AnalysisConfigComponent(IUIComponent):
    """Component for analysis configuration controls"""
    
    def render(self, **kwargs) -> AnalysisConfig:
        """Render analysis configuration controls and return config"""
        with st.expander("⚙️ Analysis Settings", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                gameweeks_ahead = st.slider("Gameweeks to analyze:", 1, 15, 5)
                show_colors = st.checkbox("Show color coding", value=True)
            
            with col2:
                fdr_threshold = st.slider("Good fixture threshold:", 1.0, 4.0, 2.5, 0.1)
                show_opponents = st.checkbox("Show opponent names", value=True)
            
            with col3:
                sort_by = st.selectbox("Sort teams by:", [
                    "Combined FDR", "Attack FDR", "Defense FDR", 
                    "Form-Adjusted FDR", "Alphabetical"
                ])
                ascending_sort = st.checkbox("Ascending order", value=True)
            
            with col4:
                use_form_adjustment = st.checkbox("Use Form Adjustment", value=True)
                form_weight = st.slider("Form Impact Weight:", 0.0, 1.0, 0.3, 0.1)
        
        # Analysis type selection
        analysis_type_str = st.selectbox(
            "🎯 Analysis Focus:",
            ["All Fixtures", "Home Only", "Away Only", "Next 3 Fixtures", "Fixture Congestion Periods"]
        )
        
        analysis_type = AnalysisType(analysis_type_str)
        
        return AnalysisConfig(
            gameweeks_ahead=gameweeks_ahead,
            fdr_threshold=fdr_threshold,
            form_weight=form_weight,
            analysis_type=analysis_type,
            use_form_adjustment=use_form_adjustment,
            show_opponents=show_opponents,
            show_colors=show_colors,
            sort_ascending=ascending_sort
        )


class StatisticalChartsComponent(IUIComponent):
    """Component for statistical analysis charts"""
    
    def render(self, fdr_data: List[FDRData], **kwargs) -> None:
        """Render statistical analysis charts"""
        if not fdr_data:
            st.warning("No FDR data available for analysis")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'team': fdr.team_short_name,
                'combined_fdr': fdr.combined_fdr,
                'attack_fdr': fdr.attack_fdr,
                'defense_fdr': fdr.defense_fdr
            }
            for fdr in fdr_data
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # FDR Distribution
            fdr_counts = df['combined_fdr'].value_counts().sort_index()
            
            fig_dist = go.Figure(data=[
                go.Bar(
                    x=fdr_counts.index,
                    y=fdr_counts.values,
                    marker_color=['#00FF87', '#01FF70', '#FFDC00', '#FF851B', '#FF4136'][:len(fdr_counts)]
                )
            ])
            
            fig_dist.update_layout(
                title="FDR Distribution",
                xaxis_title="FDR Rating",
                yaxis_title="Number of Fixtures",
                height=400
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Team Variance Analysis
            team_stats = df.groupby('team')['combined_fdr'].agg(['mean', 'std']).round(2)
            team_stats = team_stats.sort_values('std', ascending=False).head(10)
            
            fig_variance = go.Figure()
            
            fig_variance.add_trace(go.Scatter(
                x=team_stats['mean'],
                y=team_stats['std'],
                mode='markers+text',
                text=team_stats.index,
                textposition="top center",
                marker=dict(size=10, color='blue'),
                name='Teams'
            ))
            
            fig_variance.update_layout(
                title="FDR Consistency Analysis",
                xaxis_title="Average FDR",
                yaxis_title="FDR Standard Deviation",
                height=400
            )
            
            st.plotly_chart(fig_variance, use_container_width=True)


class LoadingStateComponent(IUIComponent):
    """Component for managing loading states"""
    
    def __init__(self, message: str = "Loading..."):
        self.message = message
    
    def render(self, **kwargs) -> None:
        """Render loading state"""
        with st.spinner(self.message):
            pass
    
    def render_with_progress(self, steps: List[str]) -> None:
        """Render loading with progress steps"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, step in enumerate(steps):
            status_text.text(step)
            progress_bar.progress((i + 1) / len(steps))
        
        status_text.text("Complete!")


class ErrorDisplayComponent(IUIComponent):
    """Component for displaying errors"""
    
    def render(self, error_message: str, error_type: str = "error", **kwargs) -> None:
        """Render error message"""
        if error_type == "error":
            st.error(f"❌ {error_message}")
        elif error_type == "warning":
            st.warning(f"⚠️ {error_message}")
        elif error_type == "info":
            st.info(f"ℹ️ {error_message}")
        else:
            st.write(error_message)


# Legacy components for backward compatibility
def player_card(player, index):
    pid = player["id"]
    web_name = player["web_name"]
    code = player["code"]
    team_name = player["team_name"]
    element_type = player["element_type"]
    now_cost = player["now_cost"]
    total_points = player["total_points"]
    form = player["form"]
    expected_points_next_5 = player["expected_points_next_5"]
    
    # Add xG and xA if available
    xg = player.get("expected_goals", "N/A")  # Use get() to handle missing keys
    xa = player.get("expected_assists", "N/A")
    
    # Construct the player image URL
    image_url = f"https://resources.premierleague.com/premierleague/photos/players/110x140/p{code}.png"

    col1, col2 = st.columns([1, 3])  # Adjust column widths as needed

    with col1:
        st.image(image_url, width=110)  # Display player image

    with col2:
        st.markdown(f"**{web_name}** ({team_name})")
        st.markdown(f"Position: {element_type}")
        st.markdown(f"Cost: £{now_cost / 10:.1f}m")
        st.markdown(f"Total Points: {total_points}")
        st.markdown(f"Form: {form}")
        st.markdown(f"Expected Points (Next 5): {expected_points_next_5:.2f}")
        
        # Display xG and xA
        st.markdown(f"xG: {xg}")
        st.markdown(f"xA: {xa}")

def display_player_on_pitch(player, on_bench=False):
    """Displays a single player on the virtual pitch."""
    name = player['web_name']
    team_name = player['team_name']
    points = player.get('total_points', 0)
    
    captain_symbol = ""
    if player.get('is_captain'):
        captain_symbol = " (C)"
    elif player.get('is_vice_captain'):
        captain_symbol = " (V)"

    background_color = "#e8e8e8" if on_bench else "#f9f9f9"
    
    # Tooltip implementation using HTML title attribute
    tooltip_text = f"Team: {team_name}\nPoints: {points}"
    st.markdown(f"""
    <div style="text-align: center; background-color: {background_color}; padding: 5px; border-radius: 5px; margin: 2px; border: 1px solid #ddd;"
         title="{tooltip_text}">
        <b>{name}{captain_symbol}</b><br>
        <small style="color: #555;">{team_name}</small><br>
        <small>Pts: {points}</small>
    </div>
    """, unsafe_allow_html=True)

def display_team_pitch(picks_data, players_df):
    """Renders the team in a pitch-like formation."""
    if not picks_data or not picks_data.get("picks"):
        st.info("No team picks available for the current gameweek.")
        return

    player_ids = [p['element'] for p in picks_data['picks']]
    picks_df = pd.DataFrame(picks_data['picks'])
    
    team_details = players_df[players_df['id'].isin(player_ids)].copy()
    team_details = team_details.merge(picks_df, left_on='id', right_on='element', how="left")

    starters = team_details[team_details['position'] <= 11].sort_values('element_type')
    bench = team_details[team_details['position'] > 11].sort_values('position')

    # Pitch styling
    st.markdown("""
    <div style="background-color:#00ff85; color:black; padding: 20px; border-radius: 10px; border: 2px solid #008000;">
    """, unsafe_allow_html=True)

    # Display by position
    for pos_id in [1, 2, 3, 4]:  # GK, DEF, MID, FWD
        players_in_pos = starters[starters['element_type'] == pos_id]
        if not players_in_pos.empty:
            cols = st.columns(len(players_in_pos))
            for i, (_, player) in enumerate(players_in_pos.iterrows()):
                with cols[i]:
                    display_player_on_pitch(player)

    st.markdown("</div>", unsafe_allow_html=True)

    # Display bench
    st.markdown("---")
    st.subheader("Bench")
    if len(bench) > 0:
        cols = st.columns(len(bench))
        for i, (_, player) in enumerate(bench.iterrows()):
            with cols[i]:
                display_player_on_pitch(player, on_bench=True)

def display_transfers(transfers, players_df):
    """Displays transfer history in a clean table."""
    if not transfers:
        st.info("No transfers made this season.")
        return

    transfers_df = pd.DataFrame(transfers)
    player_names = players_df.set_index('id')['web_name']
    transfers_df['Player In'] = transfers_df['element_in'].map(player_names).fillna('Unknown')
    transfers_df['Player Out'] = transfers_df['element_out'].map(player_names).fillna('Unknown')
    transfers_df['Gameweek'] = transfers_df['event']
    transfers_df['Time'] = pd.to_datetime(transfers_df['time']).dt.strftime('%Y-%m-%d')

    st.dataframe(
        transfers_df[['Gameweek', 'Player In', 'Player Out', 'Time']].sort_values('Gameweek', ascending=False),
        hide_index=True,
        use_container_width=True
    )

def display_chips(chips):
    """Displays the status of available FPL chips."""
    st.subheader("Chip Status")
    if not chips:
        st.info("Chip information is currently unavailable.")
        return

    chip_map = {
        'wildcard': 'Wildcard',
        '3xc': 'Triple Captain',
        'bboost': 'Bench Boost',
        'freehit': 'Free Hit'
    }

    played_chips = {chip['name']: chip['event'] for chip in chips}
    for chip_code, chip_name in chip_map.items():
        if chip_code in played_chips:
            st.markdown(f"**{chip_name}**: ✅ Played (GW{played_chips[chip_code]})")
        else:
            st.markdown(f"**{chip_name}**: 🟢 Available")

def enhanced_team_recommendations_controls():
    """Render enhanced controls for team recommendations with comprehensive filters"""
    st.markdown("### 🎛️ Team Configuration")
    
    # Create expandable sections for different control groups
    with st.expander("⚽ Playing Style & Formation", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            preferred_style = st.selectbox(
                "Playing Style",
                ["balanced", "attacking", "defensive"],
                help="Choose your preferred tactical approach"
            )
            
        with col2:
            risk_tolerance = st.slider(
                "Risk Tolerance",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Higher values favor differential picks"
            )
            
        with col3:
            formations = st.multiselect(
                "Allowed Formations",
                ["3-4-3", "4-3-3", "3-5-2", "4-4-2", "5-3-2"],
                default=["3-4-3", "4-3-3"],
                help="Select formations to consider"
            )
    
    with st.expander("💰 Budget & Team Constraints", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            budget = st.slider(
                "Total Budget (£m)",
                min_value=80.0,
                max_value=120.0,
                value=100.0,
                step=0.5,
                help="Maximum budget for your team"
            )
            
            max_players_per_club = st.slider(
                "Max Players per Club",
                min_value=1,
                max_value=5,
                value=3,
                help="Maximum players from any single team"
            )
            
            available_transfers = st.number_input(
                "Available Transfers",
                min_value=0,
                max_value=15,
                value=1,
                help="Number of free transfers available"
            )
            
        with col2:
            min_ownership = st.slider(
                "Minimum Ownership %",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=1.0,
                help="Filter out players below this ownership"
            )
            
            max_ownership = st.slider(
                "Maximum Ownership %",
                min_value=0.0,
                max_value=100.0,
                value=100.0,
                step=1.0,
                help="Filter out players above this ownership (for differentials)"
            )
            
            min_minutes = st.slider(
                "Minimum Minutes Played",
                min_value=0,
                max_value=3000,
                value=500,
                step=100,
                help="Filter players with insufficient playing time"
            )
    
    with st.expander("🎯 Performance Filters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Form & Points Filters**")
            min_form = st.slider(
                "Minimum Form",
                min_value=0.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                help="Filter players below this form rating"
            )
            
            min_total_points = st.slider(
                "Minimum Total Points",
                min_value=0,
                max_value=300,
                value=20,
                step=10,
                help="Filter players with low total points"
            )
            
            min_points_per_game = st.slider(
                "Minimum Points per Game",
                min_value=0.0,
                max_value=15.0,
                value=2.0,
                step=0.5,
                help="Filter players with low PPG"
            )
            
        with col2:
            st.markdown("**Expected Performance Filters**")
            min_xg = st.slider(
                "Minimum xG (Expected Goals)",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.5,
                help="Filter forwards/midfielders below this xG"
            )
            
            min_xa = st.slider(
                "Minimum xA (Expected Assists)",
                min_value=0.0,
                max_value=15.0,
                value=0.0,
                step=0.5,
                help="Filter midfielders/forwards below this xA"
            )
            
            min_expected_points = st.slider(
                "Minimum Expected Points (Next 5)",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=2.0,
                help="Filter players with low expected points"
            )
    
    with st.expander("💵 Position-Based Budget Constraints", expanded=False):
        st.markdown("**Set maximum budget per position:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_gk_budget = st.slider(
                "Max GK Budget (£m)",
                min_value=4.0,
                max_value=12.0,
                value=10.0,
                step=0.5
            )
            
        with col2:
            max_def_budget = st.slider(
                "Max DEF Budget (£m)",
                min_value=15.0,
                max_value=40.0,
                value=30.0,
                step=1.0
            )
            
        with col3:
            max_mid_budget = st.slider(
                "Max MID Budget (£m)",
                min_value=20.0,
                max_value=50.0,
                value=40.0,
                step=1.0
            )
            
        with col4:
            max_fwd_budget = st.slider(
                "Max FWD Budget (£m)",
                min_value=15.0,
                max_value=40.0,
                value=30.0,
                step=1.0
            )
    
    with st.expander("🏥 Injury & Availability Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            exclude_injured = st.checkbox(
                "Exclude Injured Players",
                value=True,
                help="Remove players with injury status"
            )
            
            exclude_suspended = st.checkbox(
                "Exclude Suspended Players",
                value=True,
                help="Remove players with suspension"
            )
            
        with col2:
            min_chance_of_playing = st.slider(
                "Minimum Chance of Playing %",
                min_value=0,
                max_value=100,
                value=75,
                step=5,
                help="Filter players with low chance of playing"
            )
    
    with st.expander("📊 Advanced Weights & Preferences", expanded=False):
        st.markdown("**Customize how different metrics are weighted:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            form_weight = st.slider("Form Weight", 0.0, 1.0, 0.3, 0.05)
            xg_weight = st.slider("xG Weight", 0.0, 1.0, 0.2, 0.05)
            xa_weight = st.slider("xA Weight", 0.0, 1.0, 0.2, 0.05)
            fixture_weight = st.slider("Fixture Weight", 0.0, 1.0, 0.1, 0.05)
            consistency_weight = st.slider("Consistency Weight", 0.0, 1.0, 0.05, 0.01)
            
        with col2:
            bps_weight = st.slider("BPS Weight", 0.0, 1.0, 0.05, 0.01)
            ownership_weight = st.slider("Ownership Weight", 0.0, 1.0, 0.0, 0.05)
            team_diversity_weight = st.slider("Team Diversity Weight", 0.0, 1.0, 0.1, 0.05)
            value_weight = st.slider("Value (PPG/Price) Weight", 0.0, 1.0, 0.15, 0.05)
            recent_form_weight = st.slider("Recent Form Weight", 0.0, 1.0, 0.2, 0.05)
    
    with st.expander("🎲 Differential & Strategy Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            differential_focus = st.checkbox(
                "Focus on Differentials",
                value=False,
                help="Prioritize low-owned players for rank climbing"
            )
            
            template_safe = st.checkbox(
                "Template Safe Team",
                value=False,
                help="Include popular template players"
            )
            
            rotation_friendly = st.checkbox(
                "Rotation Friendly",
                value=False,
                help="Prioritize players with good fixture rotation"
            )
            
        with col2:
            captaincy_focus = st.selectbox(
                "Captaincy Strategy",
                ["balanced", "aggressive", "safe"],
                help="Choose captaincy approach"
            )
            
            bench_strategy = st.selectbox(
                "Bench Strategy",
                ["cheap", "playing", "rotation"],
                help="Choose bench composition strategy"
            )
    
    return {
        'preferred_style': preferred_style,
        'risk_tolerance': risk_tolerance,
        'formations': formations,
        'budget': budget,
        'max_players_per_club': max_players_per_club,
        'min_ownership': min_ownership,
        'max_ownership': max_ownership,
        'available_transfers': available_transfers,
        'min_form': min_form,
        'min_total_points': min_total_points,
        'min_points_per_game': min_points_per_game,
        'min_xg': min_xg,
        'min_xa': min_xa,
        'min_expected_points': min_expected_points,
        'min_minutes': min_minutes,
        'max_gk_budget': max_gk_budget,
        'max_def_budget': max_def_budget,
        'max_mid_budget': max_mid_budget,
        'max_fwd_budget': max_fwd_budget,
        'exclude_injured': exclude_injured,
        'exclude_suspended': exclude_suspended,
        'min_chance_of_playing': min_chance_of_playing,
        'form_weight': form_weight,
        'xg_weight': xg_weight,
        'xa_weight': xa_weight,
        'fixture_weight': fixture_weight,
        'consistency_weight': consistency_weight,
        'bps_weight': bps_weight,
        'ownership_weight': ownership_weight,
        'team_diversity_weight': team_diversity_weight,
        'value_weight': value_weight,
        'recent_form_weight': recent_form_weight,
        'differential_focus': differential_focus,
        'template_safe': template_safe,
        'rotation_friendly': rotation_friendly,
        'captaincy_focus': captaincy_focus,
        'bench_strategy': bench_strategy
    }

def apply_player_filters(df_players, config):
    """Apply comprehensive filters to player data based on user configuration"""
    filtered_df = df_players.copy()
    
    # Basic performance filters
    filtered_df = filtered_df[
        (filtered_df['selected_by_percent'] >= config['min_ownership']) &
        (filtered_df['selected_by_percent'] <= config['max_ownership']) &
        (filtered_df['form'] >= config['min_form']) &
        (filtered_df['total_points'] >= config['min_total_points'])
    ]
    
    # Points per game filter (handle division by zero)
    if 'minutes' in filtered_df.columns:
        filtered_df['games_played'] = (filtered_df['minutes'] / 90).round()
        filtered_df['points_per_game'] = filtered_df['total_points'] / filtered_df['games_played'].replace(0, 1)
        filtered_df = filtered_df[filtered_df['points_per_game'] >= config['min_points_per_game']]
    
    # Expected performance filters
    if 'xG_next_5' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['xG_next_5'] >= config['min_xg']]
    
    if 'xA_next_5' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['xA_next_5'] >= config['min_xa']]
    
    if 'expected_points_next_5' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['expected_points_next_5'] >= config['min_expected_points']]
    
    # Minutes filter
    if 'minutes' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['minutes'] >= config['min_minutes']]
    
    # Injury and availability filters
    if config['exclude_injured'] and 'chance_of_playing_this_round' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['chance_of_playing_this_round'].isna()) |
            (filtered_df['chance_of_playing_this_round'] >= config['min_chance_of_playing'])
        ]
    
    if config['exclude_suspended'] and 'status' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['status'] != 's']  # 's' typically means suspended
    
    return filtered_df

def advanced_team_analytics_dashboard(team_df, config):
    """Enhanced analytics dashboard with filter-specific insights"""
    st.markdown("### 📈 Advanced Team Analytics")
    
    # Key metrics row with filter insights
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_value = team_df['now_cost'].sum() / 10
        budget_remaining = config['budget'] - total_value
        st.metric("Team Value", f"£{total_value:.1f}m", f"£{budget_remaining:.1f}m ITB")
    
    with col2:
        avg_form = team_df[team_df['is_starting']]['form'].mean()
        st.metric("Avg Form", f"{avg_form:.1f}", "📈")
    
    with col3:
        if 'xG_next_5' in team_df.columns:
            total_xg = team_df[team_df['is_starting']]['xG_next_5'].sum()
            st.metric("Team xG (Next 5)", f"{total_xg:.1f}", "⚽")
        else:
            st.metric("Team xG", "N/A", "⚽")
    
    with col4:
        if 'xA_next_5' in team_df.columns:
            total_xa = team_df[team_df['is_starting']]['xA_next_5'].sum()
            st.metric("Team xA (Next 5)", f"{total_xa:.1f}", "🎯")
        else:
            st.metric("Team xA", "N/A", "🎯")
    
    with col5:
        ownership_avg = team_df[team_df['is_starting']]['selected_by_percent'].mean()
        ownership_status = "🔥 Template" if ownership_avg > 30 else "💎 Differential"
        st.metric("Avg Ownership", f"{ownership_avg:.1f}%", ownership_status)
    
    # Filter compliance check
    st.markdown("### ✅ Filter Compliance")
    compliance_col1, compliance_col2, compliance_col3 = st.columns(3)
    
    with compliance_col1:
        # Budget compliance by position
        position_budgets = {
            1: team_df[team_df['element_type'] == 1]['now_cost'].sum() / 10,
            2: team_df[team_df['element_type'] == 2]['now_cost'].sum() / 10,
            3: team_df[team_df['element_type'] == 3]['now_cost'].sum() / 10,
            4: team_df[team_df['element_type'] == 4]['now_cost'].sum() / 10
        }
        
        budget_limits = {
            1: config['max_gk_budget'],
            2: config['max_def_budget'],
            3: config['max_mid_budget'],
            4: config['max_fwd_budget']
        }
        
        st.markdown("**Position Budget Compliance:**")
        for pos, pos_name in {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.items():
            actual = position_budgets[pos]
            limit = budget_limits[pos]
            status = "✅" if actual <= limit else "❌"
            st.markdown(f"{status} {pos_name}: £{actual:.1f}m / £{limit:.1f}m")
    
    with compliance_col2:
        # Ownership compliance
        high_ownership = len(team_df[team_df['selected_by_percent'] > config['max_ownership']])
        low_ownership = len(team_df[team_df['selected_by_percent'] < config['min_ownership']])
        
        st.markdown("**Ownership Compliance:**")
        st.markdown(f"✅ Within range: {15 - high_ownership - low_ownership}/15")
        if high_ownership > 0:
            st.markdown(f"❌ Above max: {high_ownership}")
        if low_ownership > 0:
            st.markdown(f"❌ Below min: {low_ownership}")
    
    with compliance_col3:
        # Performance compliance
        low_form = len(team_df[team_df['form'] < config['min_form']])
        low_points = len(team_df[team_df['total_points'] < config['min_total_points']])
        
        st.markdown("**Performance Compliance:**")
        st.markdown(f"✅ Good form: {15 - low_form}/15")
        st.markdown(f"✅ Min points: {15 - low_points}/15")
    
    # Enhanced charts section
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # xG vs xA scatter plot for starters
        if 'xG_next_5' in team_df.columns and 'xA_next_5' in team_df.columns:
            st.markdown("**Expected Performance Matrix**")
            starters = team_df[team_df['is_starting']]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                starters['xG_next_5'], 
                starters['xA_next_5'], 
                s=starters['now_cost'] * 2,  # Size by cost
                c=starters['element_type'], 
                cmap='viridis',
                alpha=0.7
            )
            
            # Add player labels
            for _, player in starters.iterrows():
                ax.annotate(
                    player['web_name'], 
                    (player['xG_next_5'], player['xA_next_5']),
                    fontsize=8,
                    ha='center'
                )
            
            ax.set_xlabel('Expected Goals (Next 5)')
            ax.set_ylabel('Expected Assists (Next 5)')
            ax.set_title('Team Expected Performance')
            plt.colorbar(scatter, label='Position')
            st.pyplot(fig)
        else:
            st.info("xG/xA data not available for visualization")
    
    with chart_col2:
        # Value analysis
        st.markdown("**Value Analysis**")
        if 'points_per_game' in team_df.columns:
            team_df['value_score'] = (team_df['points_per_game'] * 10) / team_df['now_cost']
            value_by_position = team_df.groupby('element_type')['value_score'].mean()
            position_names = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
            value_by_position.index = value_by_position.index.map(position_names)
            st.bar_chart(value_by_position)
        else:
            st.info("Value analysis requires points per game data")

def render_enhanced_team_recommendations_tab(data_manager):
    """Enhanced team recommendations tab with comprehensive filters"""
    st.header("🏆 Enhanced Team Recommendations")
    
    # Main control panel with all filters
    user_config = enhanced_team_recommendations_controls()
    
    # Apply filters to player data before generating recommendations
    if st.button("🔮 Generate Team Recommendations", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing players and generating optimal teams..."):
            try:
                # Apply filters to player data
                filtered_players = apply_player_filters(data_manager.df_players, user_config)
                
                if filtered_players.empty:
                    st.error("❌ No players match your filter criteria. Please adjust your filters.")
                    return
                
                st.info(f"✅ {len(filtered_players)} players match your criteria (from {len(data_manager.df_players)} total)")
                
                # Convert formations format
                formation_map = {
                    "3-4-3": (3, 4, 3),
                    "4-3-3": (4, 3, 3),
                    "3-5-2": (3, 5, 2),
                    "4-4-2": (4, 4, 2),
                    "5-3-2": (5, 3, 2)
                }
                formations = [formation_map[f] for f in user_config['formations']]
                
                # Generate multiple team options
                teams = {}
                for i, style in enumerate(['balanced', 'attacking', 'defensive']):
                    if style == user_config['preferred_style'] or i == 0:
                        recommendations = get_latest_team_recommendations(
                            filtered_players,  # Use filtered data
                            budget=user_config['budget'],
                            formations=formations,
                            max_players_per_club=user_config['max_players_per_club'],
                            min_ownership=user_config['min_ownership'],
                            available_transfers=user_config['available_transfers'],
                            preferred_style=style,
                            risk_tolerance=user_config['risk_tolerance'],
                            form_weight=user_config['form_weight'],
                            xg_weight=user_config['xg_weight'],
                            xa_weight=user_config['xa_weight'],
                            fixture_difficulty_weight=user_config['fixture_weight'],
                            consistency_weight=user_config['consistency_weight'],
                            bps_weight=user_config['bps_weight'],
                            ownership_weight=user_config['ownership_weight'],
                            team_diversity_weight=user_config['team_diversity_weight']
                        )
                        
                        if recommendations:
                            teams[f"{style.title()} Team"] = recommendations
                
                if not teams:
                    st.error("❌ Could not generate team recommendations. Try relaxing your constraints.")
                    return
                
                # Store in session state for persistence
                st.session_state['generated_teams'] = teams
                st.session_state['user_config'] = user_config
                
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")
                st.exception(e)  # Show full traceback in debug mode
                return
    
    # Display results if available
    if 'generated_teams' in st.session_state and st.session_state['generated_teams']:
        teams = st.session_state['generated_teams']
        config = st.session_state['user_config']
        
        # Team selection dropdown
        selected_team_name = st.selectbox(
            "🎯 Select Team to View",
            list(teams.keys()),
            help="Choose which generated team to analyze in detail"
        )
        
        selected_team = teams[selected_team_name]
        
        # Multi-tab view for different aspects
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏟️ Formation", "📊 Analytics", "📅 Fixtures", "💡 Insights", "⚖️ Compare"
        ])
        
        with tab1:
            enhanced_team_visualization(selected_team['team'], selected_team['formation'])
            
            # Quick team summary with filter insights
            st.markdown("### 📋 Team Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Formation:** {selected_team['formation'][0]}-{selected_team['formation'][1]}-{selected_team['formation'][2]}")
                st.markdown(f"**Total Cost:** £{selected_team['total_cost']/10:.1f}m")
                st.markdown(f"**Budget Remaining:** £{config['budget'] - selected_team['total_cost']/10:.1f}m")
            
            with col2:
                captain_name = selected_team['captain'].get('web_name', 'Unknown')
                vice_captain_name = selected_team['vice_captain'].get('web_name', 'Unknown')
                st.markdown(f"**Captain:** {captain_name}")
                st.markdown(f"**Vice Captain:** {vice_captain_name}")
            
            with col3:
                st.markdown(f"**Expected Points:** {selected_team['total_expected_points']:.1f}")
                st.markdown(f"**Risk Level:** {selected_team.get('risk_level', 'Medium')}")
                
                # Show filter compliance status
                team_df = selected_team['team']
                avg_ownership = team_df['selected_by_percent'].mean()
                compliance = "✅ Compliant" if config['min_ownership'] <= avg_ownership <= config['max_ownership'] else "⚠️ Check Filters"
                st.markdown(f"**Filter Status:** {compliance}")
        
        with tab2:
            advanced_team_analytics_dashboard(selected_team['team'], config)
            
            # Enhanced player table with filter-relevant columns
            st.markdown("### 👥 Detailed Player Analysis")
            
            display_df = selected_team['team'].copy()
            display_df['Position'] = display_df['element_type'].map({1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'})
            display_df['Cost'] = '£' + (display_df['now_cost'] / 10).round(1).astype(str) + 'm'
            display_df['Status'] = display_df['is_starting'].map({True: '🟢 Starting', False: '🟡 Bench'})
            display_df['Form Rating'] = display_df['form'].apply(lambda x: '⭐' * min(int(x), 5))
            
            # Add filter-specific columns
            columns_to_show = [
                'web_name', 'Position', 'team_name', 'Cost', 'Status', 
                'adjusted_expected_points', 'Form Rating', 'selected_by_percent'
            ]
            
            if 'xG_next_5' in display_df.columns:
                columns_to_show.append('xG_next_5')
            if 'xA_next_5' in display_df.columns:
                columns_to_show.append('xA_next_5')
            if 'points_per_game' in display_df.columns:
                columns_to_show.append('points_per_game')
            
            column_config = {
                'web_name': 'Player',
                'team_name': 'Team',
                'adjusted_expected_points': st.column_config.NumberColumn('Expected Points', format="%.1f"),
                'selected_by_percent': st.column_config.NumberColumn('Ownership %', format="%.1f%%"),
                'xG_next_5': st.column_config.NumberColumn('xG (Next 5)', format="%.1f"),
                'xA_next_5': st.column_config.NumberColumn('xA (Next 5)', format="%.1f"),
                'points_per_game': st.column_config.NumberColumn('PPG', format="%.1f")
            }
            
            st.dataframe(
                display_df[columns_to_show],
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )
        
        with tab3:
            fixture_analysis_widget(selected_team['team'])
        
        with tab4:
            # Enhanced insights with filter explanations
            st.markdown("### 🧠 AI Insights & Filter Impact")
            
            # Filter impact analysis
            st.markdown("#### 🎯 Filter Impact Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Applied Filters:**")
                if config['min_ownership'] > 0:
                    st.markdown(f"• Min Ownership: {config['min_ownership']}%")
                if config['max_ownership'] < 100:
                    st.markdown(f"• Max Ownership: {config['max_ownership']}%")
                if config['min_form'] > 0:
                    st.markdown(f"• Min Form: {config['min_form']}")
                if config['min_xg'] > 0:
                    st.markdown(f"• Min xG: {config['min_xg']}")
                if config['min_xa'] > 0:
                    st.markdown(f"• Min xA: {config['min_xa']}")
            
            with col2:
                st.markdown("**Strategy Flags:**")
                if config['differential_focus']:
                    st.markdown("• 💎 Differential Focus Enabled")
                if config['template_safe']:
                    st.markdown("• 🔒 Template Safe Mode")
                if config['rotation_friendly']:
                    st.markdown("• 🔄 Rotation Friendly")
            
            # Standard rationale
            rationale = selected_team.get('rationale', {})
            insight_sections = [
                ('🎯 Team Strategy', ['formation', 'playing_style']),
                ('💰 Budget Analysis', ['budget', 'budget_efficiency']),
                ('👑 Captaincy', ['captain']),
                ('⚖️ Risk Assessment', ['risk_assessment']),
                ('📅 Fixture Considerations', ['fixture_difficulty']),
                ('🔄 Transfer Strategy', ['transfer_strategy'])
            ]
            
            for section_title, keys in insight_sections:
                with st.expander(section_title, expanded=False):
                    for key in keys:
                        if key in rationale:
                            st.markdown(f"**{key.replace('_', ' ').title()}:** {rationale[key]}")
        
        with tab5:
            # Enhanced comparison with filter metrics
            if len(teams) > 1:
                display_team_comparison(teams)
                
                # Filter comparison table
                st.markdown("### 📊 Filter Compliance Comparison")
                filter_comparison = []
                for team_name, team_data in teams.items():
                    team_df = team_data['team']
                    filter_comparison.append({
                        'Team': team_name,
                        'Avg Ownership': f"{team_df['selected_by_percent'].mean():.1f}%",
                        'Avg Form': f"{team_df['form'].mean():.1f}",
                        'Total xG': f"{team_df.get('xG_next_5', pd.Series([0])).sum():.1f}",
                        'Total xA': f"{team_df.get('xA_next_5', pd.Series([0])).sum():.1f}",
                        'Budget Used': f"£{team_data['total_cost']/10:.1f}m"
                    })
                
                comparison_df = pd.DataFrame(filter_comparison)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            else:
                st.info("Generate multiple teams with different styles to enable comparison.")
    
    else:
        # Enhanced initial state with filter preview
        st.info("👆 Configure your preferences and filters above, then click 'Generate Team Recommendations' to get started!")
        
        # Show filter summary
        with st.expander("📋 Current Filter Summary", expanded=True):
            if 'user_config' in locals():
                st.markdown("**Active Filters:**")
                st.markdown(f"• Budget: £{user_config.get('budget', 100):.1f}m")
                st.markdown(f"• Max players per club: {user_config.get('max_players_per_club', 3)}")
                st.markdown(f"• Ownership range: {user_config.get('min_ownership', 0):.1f}% - {user_config.get('max_ownership', 100):.1f}%")
                st.markdown(f"• Min form: {user_config.get('min_form', 0):.1f}")
                if user_config.get('min_xg', 0) > 0:
                    st.markdown(f"• Min xG: {user_config['min_xg']:.1f}")
                if user_config.get('min_xa', 0) > 0:
                    st.markdown(f"• Min xA: {user_config['min_xa']:.1f}")
        
        # Enhanced how it works section
        with st.expander("💡 How Enhanced Filtering Works", expanded=True):
            st.markdown("""
            **The Enhanced Team Recommender now includes:**
            
            1. **🎯 Performance Filters** - xG, xA, form, points per game thresholds
            2. **💰 Budget Controls** - Total budget plus position-specific limits
            3. **👥 Ownership Management** - Min/max ownership for template vs differential strategies
            4. **🏥 Availability Filters** - Exclude injured, suspended, or unlikely to play
            5. **📊 Advanced Analytics** - Filter compliance tracking and impact analysis
            6. **🎲 Strategy Options** - Differential focus, template safe, rotation friendly
            
            **Pro Tips:**
            - Use low max ownership (20-30%) for differential teams
            - Set higher min xG/xA for attacking strategies  
            - Enable rotation friendly for easier transfers
            - Monitor filter compliance in the analytics tab
            """)