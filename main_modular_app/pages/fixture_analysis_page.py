"""
Enhanced Fixture Analysis Page - Comprehensive fixture difficulty ratings and team analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from services.fixture_service import FixtureService


class FixtureAnalysisPage:
    """Handles comprehensive fixture analysis functionality"""
    
    def __init__(self):
        self.fixture_service = FixtureService()
    
    def render(self):
        """Main render method for fixture analysis page"""
        st.header("🎯 Fixture Difficulty Analysis")
        
        # Comprehensive explanation
        with st.expander("📚 What is Fixture Difficulty Analysis?", expanded=False):
            st.markdown("""
            **Fixture Difficulty Rating (FDR)** is a crucial tool for FPL success that helps you identify:
            
            🎯 **Core Concepts:**
            - **Easy Fixtures**: Target players from teams facing weaker opponents
            - **Difficult Fixtures**: Consider transferring out players facing strong teams
            - **Home vs Away**: Home advantage typically makes fixtures easier
            - **Form Impact**: Recent team performance affects fixture difficulty
            
            📊 **How to Use This Analysis:**
            - **Green (1-2)**: Excellent fixtures - Strong targets for transfers IN
            - **Yellow (3)**: Average fixtures - Neutral, monitor closely  
            - **Red (4-5)**: Difficult fixtures - Consider transfers OUT
            
            🎮 **Strategic Applications:**
            - **Transfer Planning**: Target players from teams with upcoming easy fixtures
            - **Captain Selection**: Choose captains facing the weakest opponents
            - **Squad Rotation**: Plan bench players around difficult fixture periods
            """)
        
        # Check if we have basic player data
        if not st.session_state.get('data_loaded', False):
            st.info("Please load FPL data first from the sidebar to begin fixture analysis.")
            return
        
        df = st.session_state.players_df
        
        if df.empty:
            st.warning("No player data available for fixture analysis.")
            return
        
        # Main fixture analysis tabs
        main_tabs = st.tabs([
            "🎯 Overall Difficulty", 
            "⚔️ Attack vs Defense", 
            "👑 Captain Analysis",
            "🔄 Transfer Targets",
            "📊 Team Comparison",
            "📈 Advanced Analysis"
        ])
        
        with main_tabs[0]:
            self._display_overall_fixture_difficulty(df)
        
        with main_tabs[1]:
            self._display_attack_defense_analysis(df)
        
        with main_tabs[2]:
            self._display_captain_fixture_analysis(df)
        
        with main_tabs[3]:
            self._display_fixture_transfer_targets(df)
        
        with main_tabs[4]:
            self._display_team_fixture_comparison(df)
        
        with main_tabs[5]:
            self._display_advanced_fixture_analysis(df)
    
    def _display_overall_fixture_difficulty(self, df):
        """Display overall fixture difficulty for next 5 games"""
        st.subheader("🎯 Overall Fixture Difficulty (Next 5 Games)")
        
        # Get unique teams
        all_teams = df['team_short_name'].unique() if 'team_short_name' in df.columns else []
        
        # Calculate fixture difficulty for each team
        team_fixtures = {}
        for team in all_teams:
            if pd.notna(team):
                fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team, 5)
                team_fixtures[team] = fixtures
        
        if not team_fixtures:
            st.warning("No fixture data available. Using simplified analysis based on team strength.")
            self._render_simplified_fixture_analysis(df)
            return
        
        # Display overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate average difficulty across all teams
        all_difficulties = []
        for team_data in team_fixtures.values():
            all_difficulties.extend([f['difficulty'] for f in team_data['fixtures']])
        
        avg_difficulty = np.mean(all_difficulties) if all_difficulties else 3
        
        with col1:
            difficulty_color = "🟢" if avg_difficulty <= 2.5 else "🟡" if avg_difficulty <= 3.5 else "🔴"
            st.metric("League Avg Difficulty", f"{difficulty_color} {avg_difficulty:.1f}")
        
        with col2:
            easy_fixtures = len([d for d in all_difficulties if d <= 2])
            st.metric("Easy Fixtures", f"🟢 {easy_fixtures}")
        
        with col3:
            hard_fixtures = len([d for d in all_difficulties if d >= 4])
            st.metric("Hard Fixtures", f"🔴 {hard_fixtures}")
        
        with col4:
            best_team = min(team_fixtures.keys(), key=lambda t: team_fixtures[t]['average_difficulty']) if team_fixtures else "N/A"
            st.metric("Best Fixtures", f"🎯 {best_team}")
        
        # Detailed team-by-team breakdown
        st.subheader("📋 Team-by-Team Fixture Breakdown")
        
        # Sort teams by difficulty (easiest first)
        sorted_teams = sorted(team_fixtures.items(), key=lambda x: x[1]['average_difficulty'])
        
        for team_name, fixtures_data in sorted_teams:
            with st.expander(f"⚽ {team_name} - {fixtures_data['rating']} Fixtures ({fixtures_data['average_difficulty']:.1f} avg)"):
                
                # Show next 5 fixtures
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Next 5 Fixtures:**")
                    for fixture in fixtures_data['fixtures']:
                        home_away = "🏠 vs" if fixture['home'] else "✈️ @"
                        difficulty_emoji = "🟢" if fixture['difficulty'] <= 2 else "🟡" if fixture['difficulty'] == 3 else "🔴"
                        st.write(f"GW{fixture['gameweek']}: {home_away} {fixture['opponent']} {difficulty_emoji} ({fixture['difficulty_text']})")
                
                with col2:
                    st.metric("Average Difficulty", f"{fixtures_data['average_difficulty']:.1f}")
                    st.metric("Total Difficulty", fixtures_data['total_difficulty'])
                    
                    # Show top players from this team
                    team_players = df[df['team_short_name'] == team_name]
                    if not team_players.empty:
                        top_players = team_players.nlargest(3, 'total_points')
                        st.write("**Top Players:**")
                        for _, player in top_players.iterrows():
                            st.write(f"• {player.get('web_name', 'Unknown')}")
        
        # Fixture difficulty heatmap
        st.subheader("🔥 Fixture Difficulty Heatmap")
        
        # Create difficulty matrix
        heatmap_data = []
        gameweeks = list(range(1, 6))  # Next 5 gameweeks
        
        for team_name, fixtures_data in team_fixtures.items():
            team_row = [team_name]
            for i in range(5):
                if i < len(fixtures_data['fixtures']):
                    difficulty = fixtures_data['fixtures'][i]['difficulty']
                    team_row.append(difficulty)
                else:
                    team_row.append(3)  # Default neutral difficulty
            heatmap_data.append(team_row)
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data, columns=['Team'] + [f'GW+{i+1}' for i in range(5)])
            
            # Display as styled dataframe
            def style_difficulty(val):
                if isinstance(val, str):  # Team name column
                    return ''
                elif val <= 2:
                    return 'background-color: #90EE90'  # Light green
                elif val == 3:
                    return 'background-color: #FFFFE0'  # Light yellow  
                else:
                    return 'background-color: #FFB6C1'  # Light red
            
            styled_df = heatmap_df.style.applymap(style_difficulty)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Legend
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("🟢 **Easy (1-2)**: Target for transfers in")
            with col2:
                st.write("🟡 **Average (3)**: Monitor closely")
            with col3:
                st.write("🔴 **Hard (4-5)**: Consider transfers out")
    
    def _display_attack_defense_analysis(self, df):
        """Display attacking and defensive fixture analysis"""
        st.subheader("⚔️ Attacking vs Defensive Fixture Analysis")
        
        with st.expander("📚 Understanding Attack vs Defense Analysis", expanded=False):
            st.markdown("""
            **Attacking Fixtures**: How easy it is for players to score/assist
            - Consider opponent's **defensive strength**
            - Target players facing weak defenses
            
            **Defensive Fixtures**: How likely defenders/GKs are to get clean sheets
            - Consider opponent's **attacking strength** 
            - Target defenders facing weak attacks
            """)
        
        # Get all teams
        all_teams = df['team_short_name'].unique() if 'team_short_name' in df.columns else []
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚔️ Best Attacking Fixtures")
            st.write("*Teams facing weak defenses*")
            
            attacking_scores = []
            for team in all_teams:
                if pd.notna(team):
                    fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team, 5)
                    # For attacking, we want weak defenses (easier to score against)
                    attacking_score = 6 - fixtures['average_difficulty']  # Invert for attacking
                    attacking_scores.append({
                        'team': team,
                        'attacking_score': attacking_score,
                        'fixtures': fixtures
                    })
            
            # Sort by attacking score (best first)
            attacking_scores.sort(key=lambda x: x['attacking_score'], reverse=True)
            
            for team_data in attacking_scores[:10]:
                team_name = team_data['team']
                score = team_data['attacking_score']
                
                score_color = "🟢" if score >= 3.5 else "🟡" if score >= 2.5 else "🔴"
                
                with st.expander(f"{score_color} {team_name} - Attacking Score: {score:.1f}"):
                    # Show top attacking players from this team
                    team_players = df[df['team_short_name'] == team_name]
                    if not team_players.empty:
                        attacking_players = team_players[team_players['position_name'].isin(['Midfielder', 'Forward'])]
                        if not attacking_players.empty:
                            top_attackers = attacking_players.nlargest(3, 'total_points')
                            st.write("**Top Attacking Options:**")
                            for _, player in top_attackers.iterrows():
                                st.write(f"• {player.get('web_name', 'Unknown')} ({player.get('position_name', 'Unknown')})")
        
        with col2:
            st.subheader("🛡️ Best Defensive Fixtures")
            st.write("*Teams facing weak attacks*")
            
            defensive_scores = []
            for team in all_teams:
                if pd.notna(team):
                    fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team, 5)
                    # For defending, we want weak attacks (easier to keep clean sheets)
                    defensive_score = 6 - fixtures['average_difficulty']  # Invert for defensive
                    defensive_scores.append({
                        'team': team,
                        'defensive_score': defensive_score,
                        'fixtures': fixtures
                    })
            
            # Sort by defensive score (best first)
            defensive_scores.sort(key=lambda x: x['defensive_score'], reverse=True)
            
            for team_data in defensive_scores[:10]:
                team_name = team_data['team']
                score = team_data['defensive_score']
                
                score_color = "🟢" if score >= 3.5 else "🟡" if score >= 2.5 else "🔴"
                
                with st.expander(f"{score_color} {team_name} - Defensive Score: {score:.1f}"):
                    # Show top defensive players from this team
                    team_players = df[df['team_short_name'] == team_name]
                    if not team_players.empty:
                        defensive_players = team_players[team_players['position_name'].isin(['Goalkeeper', 'Defender'])]
                        if not defensive_players.empty:
                            top_defenders = defensive_players.nlargest(3, 'total_points')
                            st.write("**Top Defensive Options:**")
                            for _, player in top_defenders.iterrows():
                                st.write(f"• {player.get('web_name', 'Unknown')} ({player.get('position_name', 'Unknown')})")
        
        # Combined recommendation
        st.subheader("🎯 Combined Fixture Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("""
            **🟢 Excellent Fixtures**
            - Strong attacking options
            - Good clean sheet potential
            - Consider captaincy
            """)
        
        with col2:
            st.info("""
            **🟡 Mixed Fixtures**
            - Some good, some difficult
            - Monitor team news
            - Backup options ready
            """)
        
        with col3:
            st.error("""
            **🔴 Difficult Fixtures**
            - Consider bench/transfer
            - Avoid captaincy
            - Look for alternatives
            """)
    
    def _display_captain_fixture_analysis(self, df):
        """Analyze fixtures for captaincy decisions"""
        st.subheader("👑 Captain Fixture Analysis")
        
        # Get potential captains (high-scoring non-GKs)
        captain_candidates = []
        
        # Filter for good captain options
        if 'position_name' in df.columns and 'total_points' in df.columns:
            non_gks = df[df['position_name'] != 'Goalkeeper']
            top_scorers = non_gks.nlargest(20, 'total_points')  # Top 20 scorers
            
            for _, player in top_scorers.iterrows():
                team_name = player.get('team_short_name', 'UNK')
                
                # Get fixture difficulty for this player's team
                fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team_name, 1)  # Next fixture only
                
                captain_candidates.append({
                    'name': player.get('web_name', 'Unknown'),
                    'team': team_name,
                    'position': player.get('position_name', 'Unknown'),
                    'form': float(player.get('form', 0)),
                    'total_points': int(player.get('total_points', 0)),
                    'ownership': float(player.get('selected_by_percent', 0)),
                    'fixture_difficulty': fixtures['average_difficulty'],
                    'next_opponent': fixtures['fixtures'][0]['opponent'] if fixtures['fixtures'] else 'TBD',
                    'is_home': fixtures['fixtures'][0]['home'] if fixtures['fixtures'] else True
                })
        
        if captain_candidates:
            # Calculate captain score (form + fixture ease + points)
            for candidate in captain_candidates:
                fixture_score = 6 - candidate['fixture_difficulty']  # Invert difficulty
                form_score = candidate['form']
                points_score = candidate['total_points'] / 100  # Normalize
                
                candidate['captain_score'] = (fixture_score * 0.4 + form_score * 0.4 + points_score * 0.2)
            
            # Sort by captain score
            captain_candidates.sort(key=lambda x: x['captain_score'], reverse=True)
            
            st.write("**📊 Captain Options Ranked by Fixture + Form:**")
            
            for i, candidate in enumerate(captain_candidates[:10], 1):
                home_away = "🏠 vs" if candidate['is_home'] else "✈️ @"
                difficulty_emoji = "🟢" if candidate['fixture_difficulty'] <= 2 else "🟡" if candidate['fixture_difficulty'] == 3 else "🔴"
                
                # Captain recommendation level
                if candidate['captain_score'] >= 7:
                    rec_level = "🔥 Excellent"
                elif candidate['captain_score'] >= 5.5:
                    rec_level = "👍 Good"
                elif candidate['captain_score'] >= 4:
                    rec_level = "⚖️ Average"
                else:
                    rec_level = "⚠️ Risky"
                
                with st.expander(f"{i}. {candidate['name']} - {rec_level} ({candidate['captain_score']:.1f})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Next Fixture", f"{home_away} {candidate['next_opponent']} {difficulty_emoji}")
                        st.write(f"Fixture Difficulty: {candidate['fixture_difficulty']:.1f}")
                    
                    with col2:
                        st.metric("Form", f"{candidate['form']:.1f}")
                        st.metric("Total Points", candidate['total_points'])
                    
                    with col3:
                        st.metric("Ownership", f"{candidate['ownership']:.1f}%")
                        
                        # Risk/reward analysis
                        if candidate['ownership'] > 50:
                            st.write("🛡️ **Safe pick** - High ownership")
                        elif candidate['ownership'] < 15:
                            st.write("💎 **Differential** - Low ownership")
                        else:
                            st.write("⚖️ **Balanced** - Medium ownership")
        
        else:
            st.warning("No captain candidates found")
        
        # Captain strategy tips
        st.subheader("💡 Captain Strategy Tips")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **🎯 This Gameweek:**
            - Prioritize easy fixtures (1-2 difficulty)
            - Consider home advantage
            - Check for rotation risk
            - Monitor team news
            """)
        
        with col2:
            st.warning("""
            **🔮 Long-term Planning:**
            - Look ahead 2-3 gameweeks
            - Plan around difficult fixtures
            - Consider differential captains
            - Track form trends
            """)
    
    def _display_fixture_transfer_targets(self, df):
        """Identify transfer targets based on fixtures"""
        st.subheader("🔄 Fixture-Based Transfer Targets")
        
        # Get all teams with their fixture difficulties
        all_teams = df['team_short_name'].unique() if 'team_short_name' in df.columns else []
        
        team_fixture_scores = []
        
        for team in all_teams:
            if pd.notna(team):
                fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team, 5)
                fixture_score = 6 - fixtures['average_difficulty']  # Higher = better fixtures
                
                team_fixture_scores.append({
                    'team': team,
                    'fixture_score': fixture_score,
                    'avg_difficulty': fixtures['average_difficulty'],
                    'rating': fixtures['rating']
                })
        
        # Sort by fixture quality (best first)
        team_fixture_scores.sort(key=lambda x: x['fixture_score'], reverse=True)
        
        transfer_tabs = st.tabs(["🎯 Best Fixtures", "⚠️ Worst Fixtures", "💎 Differentials"])
        
        with transfer_tabs[0]:
            st.subheader("🟢 Teams with Best Fixtures")
            st.write("*Consider players from these teams*")
            
            best_teams = team_fixture_scores[:8]
            
            for team_data in best_teams:
                team_name = team_data['team']
                
                # Get best players from this team
                team_players = df[df['team_short_name'] == team_name]
                if not team_players.empty:
                    # Get top players by points
                    top_players = team_players.nlargest(5, 'total_points')
                    
                    with st.expander(f"🟢 {team_name} - {team_data['rating']} Fixtures ({team_data['avg_difficulty']:.1f})"):
                        st.write("**🎯 Top Transfer Targets:**")
                        
                        for _, player in top_players.iterrows():
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.write(f"**{player.get('web_name', 'Unknown')}**")
                                st.write(f"{player.get('position_name', 'Unknown')}")
                            
                            with col2:
                                st.write(f"£{float(player.get('now_cost', 0))/10:.1f}m")
                                st.write(f"Form: {player.get('form', 0)}")
                            
                            with col3:
                                st.write(f"{player.get('total_points', 0)} pts")
                                ppm = float(player.get('points_per_million', 0))
                                st.write(f"PPM: {ppm:.1f}")
                            
                            with col4:
                                ownership = float(player.get('selected_by_percent', 0))
                                st.write(f"{ownership:.1f}% owned")
                                
                                if ownership < 10:
                                    st.write("💎 Differential")
                                elif ownership > 50:
                                    st.write("🛡️ Template")
                                else:
                                    st.write("⚖️ Balanced")
        
        with transfer_tabs[1]:
            st.subheader("🔴 Teams with Worst Fixtures")
            st.write("*Consider transferring out players from these teams*")
            
            worst_teams = team_fixture_scores[-8:]
            
            for team_data in worst_teams:
                team_name = team_data['team']
                
                # Get popular players from this team (likely in many squads)
                team_players = df[df['team_short_name'] == team_name]
                if not team_players.empty:
                    # Get most owned players
                    popular_players = team_players.nlargest(3, 'selected_by_percent')
                    
                    with st.expander(f"🔴 {team_name} - {team_data['rating']} Fixtures ({team_data['avg_difficulty']:.1f})"):
                        st.write("**⚠️ Consider Transferring Out:**")
                        
                        for _, player in popular_players.iterrows():
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"**{player.get('web_name', 'Unknown')}**")
                                st.write(f"{player.get('position_name', 'Unknown')}")
                            
                            with col2:
                                st.write(f"£{float(player.get('now_cost', 0))/10:.1f}m")
                                st.write(f"{player.get('total_points', 0)} pts")
                            
                            with col3:
                                ownership = float(player.get('selected_by_percent', 0))
                                st.write(f"{ownership:.1f}% owned")
                                
                                if ownership > 30:
                                    st.warning("High ownership - many will keep")
                                else:
                                    st.info("Good time to transfer out")
        
        with transfer_tabs[2]:
            st.subheader("💎 Differential Opportunities")
            st.write("*Low ownership players with good fixtures*")
            
            # Find differential players (low ownership) with good fixtures
            differential_candidates = []
            
            for team_data in team_fixture_scores[:12]:  # Top 12 teams by fixtures
                if team_data['fixture_score'] >= 3:  # Only good fixtures
                    team_name = team_data['team']
                    team_players = df[df['team_short_name'] == team_name]
                    
                    if not team_players.empty:
                        # Find players with <15% ownership and decent points
                        differentials = team_players[
                            (team_players['selected_by_percent'] < 15) & 
                            (team_players['total_points'] > 50)
                        ]
                        
                        for _, player in differentials.iterrows():
                            differential_candidates.append({
                                'name': player.get('web_name', 'Unknown'),
                                'team': team_name,
                                'position': player.get('position_name', 'Unknown'),
                                'price': float(player.get('now_cost', 0)) / 10,
                                'points': player.get('total_points', 0),
                                'form': float(player.get('form', 0)),
                                'ownership': float(player.get('selected_by_percent', 0)),
                                'fixture_score': team_data['fixture_score']
                            })
            
            # Sort by combined differential score
            for candidate in differential_candidates:
                candidate['differential_score'] = (
                    candidate['fixture_score'] * 0.3 +
                    candidate['form'] * 0.3 +
                    (candidate['points'] / 100) * 0.2 +
                    (15 - candidate['ownership']) * 0.2  # Lower ownership = higher score
                )
            
            differential_candidates.sort(key=lambda x: x['differential_score'], reverse=True)
            
            for candidate in differential_candidates[:10]:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**{candidate['name']}**")
                    st.write(f"{candidate['position']} ({candidate['team']})")
                
                with col2:
                    st.write(f"£{candidate['price']:.1f}m")
                    st.write(f"Form: {candidate['form']:.1f}")
                
                with col3:
                    st.write(f"{candidate['points']} points")
                    st.write(f"Fixtures: {candidate['fixture_score']:.1f}")
                
                with col4:
                    st.write(f"{candidate['ownership']:.1f}% owned")
                    st.success("💎 Differential")
    
    def _display_team_fixture_comparison(self, df):
        """Compare fixture difficulty between teams"""
        st.subheader("📊 Team Fixture Comparison")
        
        # Team selection for comparison
        all_teams = sorted(df['team_short_name'].unique()) if 'team_short_name' in df.columns else []
        
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select First Team", all_teams, index=0 if all_teams else None)
        
        with col2:
            team2 = st.selectbox("Select Second Team", all_teams, index=1 if len(all_teams) > 1 else 0)
        
        if team1 and team2 and team1 != team2:
            # Compare fixtures between teams
            comparison = self.fixture_service.compare_fixture_run(team1, team2, 5)
            
            # Display comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"⚽ {team1}")
                team1_data = comparison['team1']['data']
                
                st.metric("Average Difficulty", f"{team1_data['average_difficulty']:.1f}")
                st.metric("Fixture Rating", team1_data['rating'])
                
                st.write("**Next 5 Fixtures:**")
                for fixture in team1_data['fixtures']:
                    home_away = "🏠 vs" if fixture['home'] else "✈️ @"
                    difficulty_emoji = "🟢" if fixture['difficulty'] <= 2 else "🟡" if fixture['difficulty'] == 3 else "🔴"
                    st.write(f"GW{fixture['gameweek']}: {home_away} {fixture['opponent']} {difficulty_emoji}")
            
            with col2:
                st.subheader(f"⚽ {team2}")
                team2_data = comparison['team2']['data']
                
                st.metric("Average Difficulty", f"{team2_data['average_difficulty']:.1f}")
                st.metric("Fixture Rating", team2_data['rating'])
                
                st.write("**Next 5 Fixtures:**")
                for fixture in team2_data['fixtures']:
                    home_away = "🏠 vs" if fixture['home'] else "✈️ @"
                    difficulty_emoji = "🟢" if fixture['difficulty'] <= 2 else "🟡" if fixture['difficulty'] == 3 else "🔴"
                    st.write(f"GW{fixture['gameweek']}: {home_away} {fixture['opponent']} {difficulty_emoji}")
            
            # Recommendation
            st.subheader("🎯 Recommendation")
            recommended_team = comparison['recommendation']
            
            if recommended_team == team1:
                st.success(f"✅ **{team1}** has easier fixtures than {team2}")
            else:
                st.success(f"✅ **{team2}** has easier fixtures than {team1}")
            
            # Show players from both teams
            st.subheader("👥 Players Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{team1} Players:**")
                team1_players = df[df['team_short_name'] == team1].nlargest(5, 'total_points')
                for _, player in team1_players.iterrows():
                    st.write(f"• {player.get('web_name', 'Unknown')} ({player.get('position_name', 'Unknown')}) - £{float(player.get('now_cost', 0))/10:.1f}m")
            
            with col2:
                st.write(f"**{team2} Players:**")
                team2_players = df[df['team_short_name'] == team2].nlargest(5, 'total_points')
                for _, player in team2_players.iterrows():
                    st.write(f"• {player.get('web_name', 'Unknown')} ({player.get('position_name', 'Unknown')}) - £{float(player.get('now_cost', 0))/10:.1f}m")
        
        else:
            st.info("Please select two different teams to compare their fixtures.")
    
    def _display_advanced_fixture_analysis(self, df):
        """Display advanced fixture analysis including the original simplified version"""
        st.subheader("📈 Advanced Fixture Analysis")
        
        # Sub-tabs for different types of advanced analysis
        advanced_tabs = st.tabs([
            "📊 Team Strength Analysis", 
            "🏠 Home vs Away", 
            "📈 Form-Based Fixtures",
            "🎯 Transfer Recommendations"
        ])
        
        with advanced_tabs[0]:
            self._render_team_strength_analysis(df)
        
        with advanced_tabs[1]:
            self._render_home_away_analysis(df)
        
        with advanced_tabs[2]:
            self._render_form_based_analysis(df)
        
        with advanced_tabs[3]:
            self._render_fixture_transfer_recommendations(df)
    
    def _render_simplified_fixture_analysis(self, df):
        """Render simplified fixture analysis using available data"""
        
        # Analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Team Strength Analysis", 
            "🏠 Home vs Away", 
            "📈 Form-Based Fixtures",
            "🎯 Transfer Recommendations"
        ])
        
        with tab1:
            self._render_team_strength_analysis(df)
        
        with tab2:
            self._render_home_away_analysis(df)
        
        with tab3:
            self._render_form_based_analysis(df)
        
        with tab4:
            self._render_fixture_transfer_recommendations(df)
    
    def _render_team_strength_analysis(self, df):
        """Analyze team strength for fixture difficulty estimation"""
        st.subheader("📊 Team Strength Analysis")
        st.info("💡 Stronger teams = harder fixtures when playing against them")
        
        if df.empty:
            st.warning("No data available for team strength analysis")
            return
        
        # Calculate team strength metrics
        team_metrics = df.groupby('team_short_name').agg({
            'total_points': 'sum',
            'goals_scored': 'sum' if 'goals_scored' in df.columns else 'count',
            'clean_sheets': 'sum' if 'clean_sheets' in df.columns else 'count',
            'form': 'mean' if 'form' in df.columns else 'count'
        }).reset_index()
        
        # Calculate team strength score
        if 'total_points' in team_metrics.columns:
            team_metrics['strength_score'] = (
                team_metrics['total_points'] / team_metrics['total_points'].max() * 100
            ).round(1)
            
            team_metrics = team_metrics.sort_values('strength_score', ascending=False)
            
            # Display team strength
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 Strongest Teams (Hardest to face)")
                strong_teams = team_metrics.head(10)
                for _, team in strong_teams.iterrows():
                    st.write(f"🔴 **{team['team_short_name']}**: {team['strength_score']:.1f} strength")
            
            with col2:
                st.subheader("📉 Weaker Teams (Easier to face)")
                weak_teams = team_metrics.tail(10)
                for _, team in weak_teams.iterrows():
                    st.write(f"🟢 **{team['team_short_name']}**: {team['strength_score']:.1f} strength")
            
            # Team strength visualization
            fig = px.bar(
                team_metrics, 
                x='team_short_name', 
                y='strength_score',
                title="Team Strength Rankings",
                color='strength_score',
                color_continuous_scale='RdYlGn_r'
            )
            
            fig.update_layout(
                xaxis_title="Team",
                yaxis_title="Strength Score",
                height=500,
                xaxis={'categoryorder': 'total descending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Insufficient data for team strength calculation")
    
    def _render_home_away_analysis(self, df):
        """Analyze home vs away performance"""
        st.subheader("🏠 Home vs Away Performance")
        st.info("📍 Teams typically perform better at home - use this for fixture planning")
        
        # Since we don't have fixture data, we'll use team strength as proxy
        if 'team_short_name' in df.columns and 'total_points' in df.columns:
            team_stats = df.groupby('team_short_name').agg({
                'total_points': 'sum',
                'form': 'mean' if 'form' in df.columns else 'count'
            }).reset_index()
            
            # Simulate home advantage (typically 0.3-0.5 points boost)
            team_stats['home_strength'] = team_stats['total_points'] * 1.15
            team_stats['away_strength'] = team_stats['total_points'] * 0.9
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏠 Best Home Teams")
                st.write("*Teams likely to perform well at home*")
                home_teams = team_stats.nlargest(8, 'home_strength')
                for _, team in home_teams.iterrows():
                    st.write(f"🟢 **{team['team_short_name']}**: Strong at home")
            
            with col2:
                st.subheader("✈️ Best Away Teams") 
                st.write("*Teams that travel well*")
                away_teams = team_stats.nlargest(8, 'away_strength')
                for _, team in away_teams.iterrows():
                    st.write(f"🟡 **{team['team_short_name']}**: Good away form")
            
            # Home vs Away comparison chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Home Strength',
                x=team_stats['team_short_name'],
                y=team_stats['home_strength'],
                marker_color='lightgreen'
            ))
            
            fig.add_trace(go.Bar(
                name='Away Strength',
                x=team_stats['team_short_name'], 
                y=team_stats['away_strength'],
                marker_color='lightcoral'
            ))
            
            fig.update_layout(
                title='Home vs Away Performance Comparison',
                xaxis_title='Team',
                yaxis_title='Estimated Strength',
                barmode='group',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Home/Away analysis requires team and points data")
    
    def _render_form_based_analysis(self, df):
        """Analyze current form for fixture difficulty"""
        st.subheader("📈 Form-Based Fixture Analysis")
        st.info("🔥 Teams in good form are harder to face - adjust your transfers accordingly")
        
        if 'form' not in df.columns:
            st.warning("Form data not available - using total points as proxy")
            if 'total_points' in df.columns:
                # Use total points as form proxy
                df = df.copy()
                df['form'] = df['total_points'] / 20  # Approximate form from total points
            else:
                st.error("No suitable data for form analysis")
                return
        
        # Team form analysis
        team_form = df.groupby('team_short_name').agg({
            'form': 'mean',
            'total_points': 'sum' if 'total_points' in df.columns else 'count'
        }).reset_index()
        
        team_form = team_form.sort_values('form', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔥 Teams in Hot Form")
            st.write("*Avoid facing these teams*")
            hot_teams = team_form.head(8)
            for _, team in hot_teams.iterrows():
                st.write(f"🔴 **{team['team_short_name']}**: {team['form']:.1f} form")
        
        with col2:
            st.subheader("❄️ Teams in Poor Form")
            st.write("*Target players facing these teams*")
            cold_teams = team_form.tail(8)
            for _, team in cold_teams.iterrows():
                st.write(f"🟢 **{team['team_short_name']}**: {team['form']:.1f} form")
        
        # Form distribution
        fig = px.histogram(
            team_form, 
            x='form',
            nbins=10,
            title="Team Form Distribution",
            labels={'form': 'Average Form', 'count': 'Number of Teams'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Form vs Points correlation
        if 'total_points' in team_form.columns:
            fig_scatter = px.scatter(
                team_form,
                x='form',
                y='total_points',
                text='team_short_name',
                title="Form vs Total Points Correlation",
                labels={'form': 'Average Form', 'total_points': 'Total Team Points'}
            )
            
            fig_scatter.update_traces(textposition="top center")
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    def _render_fixture_transfer_recommendations(self, df):
        """Provide transfer recommendations based on fixture analysis"""
        st.subheader("🎯 Fixture-Based Transfer Recommendations")
        st.info("💡 Strategic recommendations based on team strength and form analysis")
        
        if df.empty:
            st.warning("No data available for recommendations")
            return
        
        # Calculate recommendation scores
        team_analysis = df.groupby('team_short_name').agg({
            'total_points': 'sum',
            'form': 'mean' if 'form' in df.columns else 'count',
            'selected_by_percent': 'mean' if 'selected_by_percent' in df.columns else 'count'
        }).reset_index()
        
        if 'total_points' in team_analysis.columns:
            # Calculate fixture attractiveness (lower = better fixtures ahead)
            team_analysis['fixture_attractiveness'] = (
                100 - (team_analysis['total_points'] / team_analysis['total_points'].max() * 100)
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Teams to Target")
                st.write("*Players from these teams likely have easier fixtures*")
                
                # Teams with poor opponents (high fixture attractiveness)
                target_teams = team_analysis.nlargest(8, 'fixture_attractiveness')
                
                for _, team in target_teams.iterrows():
                    # Get best players from this team
                    team_players = df[df['team_short_name'] == team['team_short_name']]
                    if not team_players.empty:
                        best_player = team_players.nlargest(1, 'total_points').iloc[0]
                        st.write(f"🟢 **{team['team_short_name']}**: Consider {best_player['web_name']}")
            
            with col2:
                st.subheader("⚠️ Teams to Avoid")
                st.write("*Players from these teams likely face difficult fixtures*")
                
                # Teams with strong opponents (low fixture attractiveness) 
                avoid_teams = team_analysis.nsmallest(8, 'fixture_attractiveness')
                
                for _, team in avoid_teams.iterrows():
                    # Get popular players from this team
                    team_players = df[df['team_short_name'] == team['team_short_name']]
                    if not team_players.empty:
                        if 'selected_by_percent' in team_players.columns:
                            popular_player = team_players.nlargest(1, 'selected_by_percent').iloc[0]
                        else:
                            popular_player = team_players.nlargest(1, 'total_points').iloc[0]
                        st.write(f"🔴 **{team['team_short_name']}**: Consider selling {popular_player['web_name']}")
        
        # Transfer timing recommendations
        st.subheader("⏰ Transfer Timing Strategy")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **🚀 Immediate Targets**
            - Players from weak teams
            - Good form + easy fixtures
            - Price rises expected
            """)
        
        with col2:
            st.warning("""
            **⏳ Monitor Closely**
            - Form vs fixture conflict
            - Injury concerns
            - Rotation risks
            """)
        
        with col3:
            st.error("""
            **❌ Avoid This Week**
            - Strong opposition ahead
            - Poor recent form
            - High risk of benching
            """)
        
        # Simple fixture difficulty matrix
        st.subheader("📊 Quick Fixture Difficulty Guide")
        
        difficulty_guide = pd.DataFrame({
            'Opponent Strength': ['Very Strong', 'Strong', 'Average', 'Weak', 'Very Weak'],
            'Home Fixture': ['🔴 Very Hard', '🟠 Hard', '🟡 Average', '🟢 Easy', '🟢 Very Easy'],
            'Away Fixture': ['🔴 Extremely Hard', '🔴 Very Hard', '🟠 Hard', '🟡 Average', '🟢 Easy'],
            'Recommendation': ['Avoid', 'Consider Out', 'Monitor', 'Consider In', 'Strong Target']
        })
        
        st.dataframe(difficulty_guide, use_container_width=True, hide_index=True)

