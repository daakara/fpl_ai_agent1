"""
Enhanced Fixture Analysis Page - Comprehensive fixture difficulty ratings and team analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict
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
        """Display overall fixture difficulty for current + next 4 games (5 total)"""
        st.subheader("🎯 Fixture Difficulty Heatmap (Current GW + Next 4)")
        
        with st.expander("📚 Understanding the Heatmap", expanded=False):
            st.markdown("""
            **Fixture Difficulty Heatmap** shows the next 5 gameweeks for all teams:
            
            🎯 **Current GW + Next 4**: Comprehensive 5-gameweek outlook
            📊 **Color Coding**: Green = Easy, Yellow = Average, Red = Difficult
            ⚔️ **Attack vs Defense**: Separate analysis for attacking and defensive players
            🏠 **Home Advantage**: Considered in difficulty calculations
            """)
        
        # Get unique teams
        all_teams = df['team_short_name'].unique() if 'team_short_name' in df.columns else []
        
        # Fixture difficulty tabs
        difficulty_tabs = st.tabs(["📊 Overall Difficulty", "⚔️ Attack Difficulty", "🛡️ Defense Difficulty"])
        
        with difficulty_tabs[0]:
            self._display_overall_difficulty_heatmap(df, all_teams)
        
        with difficulty_tabs[1]:
            self._display_attack_difficulty_heatmap(df, all_teams)
        
        with difficulty_tabs[2]:
            self._display_defense_difficulty_heatmap(df, all_teams)
    
    def _display_overall_difficulty_heatmap(self, df, all_teams):
        """Display overall fixture difficulty heatmap"""
        st.subheader("📊 Overall Fixture Difficulty")
        
        # Calculate fixture difficulty for each team
        team_fixtures = {}
        api_status = {'success': 0, 'fallback': 0, 'error': 0}
        
        for team in all_teams:
            if pd.notna(team):
                fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team, 5)
                team_fixtures[team] = fixtures
                
                # Track API status
                if fixtures.get('is_fallback'):
                    api_status['fallback'] += 1
                else:
                    api_status['success'] += 1
        
        if not team_fixtures:
            st.warning("No fixture data available. Using simplified analysis based on team strength.")
            self._render_simplified_fixture_analysis(df)
            return
        
        # Display data source status
        total_teams = api_status['success'] + api_status['fallback']
        if total_teams > 0:
            success_rate = api_status['success'] / total_teams * 100
            
            if success_rate == 100:
                st.success(f"✅ **Official FPL API Data** - All {total_teams} teams loaded from official API")
            elif success_rate >= 50:
                st.warning(f"⚠️ **Mixed Data Sources** - {api_status['success']} teams from API, {api_status['fallback']} teams using fallback data")
            else:
                st.error(f"❌ **Limited API Access** - Only {api_status['success']} teams from API, {api_status['fallback']} teams using fallback data")
        
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
        
        # Create fixture difficulty heatmap
        st.subheader("🔥 Fixture Difficulty Heatmap")
        
        # Create difficulty matrix
        heatmap_data = []
        current_gw = self._get_current_gameweek()
        gameweek_labels = [f"GW{current_gw + i}" for i in range(5)]
        
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
            heatmap_df = pd.DataFrame(heatmap_data, columns=['Team'] + gameweek_labels)
            
            # Display as styled dataframe
            def style_difficulty(val):
                if isinstance(val, str):  # Team name column
                    return ''
                elif val <= 2:
                    return 'background-color: #90EE90; color: black'  # Light green
                elif val == 3:
                    return 'background-color: #FFFFE0; color: black'  # Light yellow  
                else:
                    return 'background-color: #FFB6C1; color: black'  # Light red
            
            styled_df = heatmap_df.style.applymap(style_difficulty)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            # Legend and insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success("🟢 **Easy (1-2)**: Target for transfers in")
            with col2:
                st.info("🟡 **Average (3)**: Monitor closely")
            with col3:
                st.error("🔴 **Hard (4-5)**: Consider transfers out")
        
        # Detailed team-by-team breakdown
        self._display_team_fixture_breakdown(team_fixtures, df)
    
    def _display_attack_difficulty_heatmap(self, df, all_teams):
        """Display attacking fixture difficulty heatmap"""
        st.subheader("⚔️ Attack Fixture Difficulty")
        st.info("💡 **Lower difficulty = Better for attacking players** (Forwards, Midfielders)")
        
        # Calculate attacking fixture difficulty for each team
        team_attack_fixtures = {}
        for team in all_teams:
            if pd.notna(team):
                fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team, 5)
                # For attacking, we analyze how strong the opponent's defense is
                attack_fixtures = self._calculate_attack_difficulty(team, fixtures)
                team_attack_fixtures[team] = attack_fixtures
        
        if not team_attack_fixtures:
            st.warning("No fixture data available for attack analysis.")
            return
        
        # Display attack-specific metrics
        col1, col2, col3, col4 = st.columns(4)
        
        attack_difficulties = []
        for team_data in team_attack_fixtures.values():
            attack_difficulties.extend([f['attack_difficulty'] for f in team_data['fixtures']])
        
        avg_attack_difficulty = np.mean(attack_difficulties) if attack_difficulties else 3
        
        with col1:
            difficulty_color = "🟢" if avg_attack_difficulty <= 2.5 else "🟡" if avg_attack_difficulty <= 3.5 else "🔴"
            st.metric("Avg Attack Difficulty", f"{difficulty_color} {avg_attack_difficulty:.1f}")
        
        with col2:
            easy_attack = len([d for d in attack_difficulties if d <= 2])
            st.metric("Easy Attack Fixtures", f"🟢 {easy_attack}")
        
        with col3:
            hard_attack = len([d for d in attack_difficulties if d >= 4])
            st.metric("Hard Attack Fixtures", f"🔴 {hard_attack}")
        
        with col4:
            best_attack_team = min(team_attack_fixtures.keys(), 
                                 key=lambda t: team_attack_fixtures[t]['avg_attack_difficulty']) if team_attack_fixtures else "N/A"
            st.metric("Best Attack Fixtures", f"⚔️ {best_attack_team}")
        
        # Create attack difficulty heatmap
        st.subheader("⚔️ Attack Difficulty Heatmap")
        
        heatmap_data = []
        current_gw = self._get_current_gameweek()
        gameweek_labels = [f"GW{current_gw + i}" for i in range(5)]
        
        for team_name, fixtures_data in team_attack_fixtures.items():
            team_row = [team_name]
            for i in range(5):
                if i < len(fixtures_data['fixtures']):
                    difficulty = fixtures_data['fixtures'][i]['attack_difficulty']
                    team_row.append(difficulty)
                else:
                    team_row.append(3)
            heatmap_data.append(team_row)
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data, columns=['Team'] + gameweek_labels)
            
            def style_attack_difficulty(val):
                if isinstance(val, str):
                    return ''
                elif val <= 2:
                    return 'background-color: #90EE90; color: black'
                elif val == 3:
                    return 'background-color: #FFFFE0; color: black'
                else:
                    return 'background-color: #FFB6C1; color: black'
            
            styled_df = heatmap_df.style.applymap(style_attack_difficulty)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Attack-specific recommendations
        self._display_attack_recommendations(team_attack_fixtures, df)
    
    def _display_defense_difficulty_heatmap(self, df, all_teams):
        """Display defensive fixture difficulty heatmap"""
        st.subheader("🛡️ Defense Fixture Difficulty")
        st.info("💡 **Lower difficulty = Better for defensive players** (Goalkeepers, Defenders)")
        
        # Calculate defensive fixture difficulty for each team
        team_defense_fixtures = {}
        for team in all_teams:
            if pd.notna(team):
                fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team, 5)
                # For defending, we analyze how strong the opponent's attack is
                defense_fixtures = self._calculate_defense_difficulty(team, fixtures)
                team_defense_fixtures[team] = defense_fixtures
        
        if not team_defense_fixtures:
            st.warning("No fixture data available for defense analysis.")
            return
        
        # Display defense-specific metrics
        col1, col2, col3, col4 = st.columns(4)
        
        defense_difficulties = []
        for team_data in team_defense_fixtures.values():
            defense_difficulties.extend([f['defense_difficulty'] for f in team_data['fixtures']])
        
        avg_defense_difficulty = np.mean(defense_difficulties) if defense_difficulties else 3
        
        with col1:
            difficulty_color = "🟢" if avg_defense_difficulty <= 2.5 else "🟡" if avg_defense_difficulty <= 3.5 else "🔴"
            st.metric("Avg Defense Difficulty", f"{difficulty_color} {avg_defense_difficulty:.1f}")
        
        with col2:
            easy_defense = len([d for d in defense_difficulties if d <= 2])
            st.metric("Easy Defense Fixtures", f"🟢 {easy_defense}")
        
        with col3:
            hard_defense = len([d for d in defense_difficulties if d >= 4])
            st.metric("Hard Defense Fixtures", f"🔴 {hard_defense}")
        
        with col4:
            best_defense_team = min(team_defense_fixtures.keys(), 
                                  key=lambda t: team_defense_fixtures[t]['avg_defense_difficulty']) if team_defense_fixtures else "N/A"
            st.metric("Best Defense Fixtures", f"🛡️ {best_defense_team}")
        
        # Create defense difficulty heatmap
        st.subheader("🛡️ Defense Difficulty Heatmap")
        
        heatmap_data = []
        current_gw = self._get_current_gameweek()
        gameweek_labels = [f"GW{current_gw + i}" for i in range(5)]
        
        for team_name, fixtures_data in team_defense_fixtures.items():
            team_row = [team_name]
            for i in range(5):
                if i < len(fixtures_data['fixtures']):
                    difficulty = fixtures_data['fixtures'][i]['defense_difficulty']
                    team_row.append(difficulty)
                else:
                    team_row.append(3)
            heatmap_data.append(team_row)
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data, columns=['Team'] + gameweek_labels)
            
            def style_defense_difficulty(val):
                if isinstance(val, str):
                    return ''
                elif val <= 2:
                    return 'background-color: #90EE90; color: black'
                elif val == 3:
                    return 'background-color: #FFFFE0; color: black'
                else:
                    return 'background-color: #FFB6C1; color: black'
            
            styled_df = heatmap_df.style.applymap(style_defense_difficulty)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Defense-specific recommendations
        self._display_defense_recommendations(team_defense_fixtures, df)
    
    def _calculate_attack_difficulty(self, team_name: str, fixtures_data: Dict) -> Dict:
        """Calculate attack-specific difficulty based on opponent defensive strength using official FPL data"""
        attack_fixtures = []
        total_attack_difficulty = 0
        
        for fixture in fixtures_data['fixtures']:
            opponent = fixture['opponent']
            is_home = fixture['home']
            
            # Get opponent's defensive strength from official FPL API data
            opponent_strength = self.fixture_service.get_team_attack_defense_strength(opponent)
            
            # Use official FPL defensive strength ratings (1-5 scale)
            if is_home:
                opponent_defense = opponent_strength.get('defense_away', 3)  # Opponent defending away
            else:
                opponent_defense = opponent_strength.get('defense_home', 3)  # Opponent defending at home
            
            # Calculate attack difficulty using FPL's scale
            # Higher defense rating = harder to score against
            attack_difficulty = opponent_defense
            
            # Apply home advantage adjustment
            if is_home:
                attack_difficulty = max(1, attack_difficulty - 0.3)  # Easier to score at home
            else:
                attack_difficulty = min(5, attack_difficulty + 0.3)  # Harder to score away
            
            attack_fixtures.append({
                **fixture,
                'attack_difficulty': round(attack_difficulty, 1),
                'opponent_defense_strength': opponent_defense,
                'opponent_team_strength': opponent_strength
            })
            
            total_attack_difficulty += attack_difficulty
        
        avg_attack_difficulty = total_attack_difficulty / len(attack_fixtures) if attack_fixtures else 3
        
        return {
            'fixtures': attack_fixtures,
            'avg_attack_difficulty': avg_attack_difficulty,
            'total_attack_difficulty': total_attack_difficulty,
            'attack_rating': self._get_attack_rating(avg_attack_difficulty)
        }
    
    def _calculate_defense_difficulty(self, team_name: str, fixtures_data: Dict) -> Dict:
        """Calculate defense-specific difficulty based on opponent attacking strength using official FPL data"""
        defense_fixtures = []
        total_defense_difficulty = 0
        
        for fixture in fixtures_data['fixtures']:
            opponent = fixture['opponent']
            is_home = fixture['home']
            
            # Get opponent's attacking strength from official FPL API data
            opponent_strength = self.fixture_service.get_team_attack_defense_strength(opponent)
            
            # Use official FPL attacking strength ratings (1-5 scale)
            if is_home:
                opponent_attack = opponent_strength.get('attack_away', 3)  # Opponent attacking away
            else:
                opponent_attack = opponent_strength.get('attack_home', 3)  # Opponent attacking at home
            
            # Calculate defense difficulty using FPL's scale
            # Higher attack rating = harder to defend against
            defense_difficulty = opponent_attack
            
            # Apply home advantage adjustment for defending
            if is_home:
                defense_difficulty = max(1, defense_difficulty - 0.4)  # Easier to defend at home
            else:
                defense_difficulty = min(5, defense_difficulty + 0.4)  # Harder to defend away
            
            defense_fixtures.append({
                **fixture,
                'defense_difficulty': round(defense_difficulty, 1),
                'opponent_attack_strength': opponent_attack,
                'opponent_team_strength': opponent_strength
            })
            
            total_defense_difficulty += defense_difficulty
        
        avg_defense_difficulty = total_defense_difficulty / len(defense_fixtures) if defense_fixtures else 3
        
        return {
            'fixtures': defense_fixtures,
            'avg_defense_difficulty': avg_defense_difficulty,
            'total_defense_difficulty': total_defense_difficulty,
            'defense_rating': self._get_defense_rating(avg_defense_difficulty)
        }
    
    def _get_attack_rating(self, avg_difficulty: float) -> str:
        """Get attack rating based on average difficulty"""
        if avg_difficulty <= 2:
            return "Excellent Attack Fixtures"
        elif avg_difficulty <= 2.5:
            return "Good Attack Fixtures"
        elif avg_difficulty <= 3.5:
            return "Average Attack Fixtures"
        elif avg_difficulty <= 4:
            return "Difficult Attack Fixtures"
        else:
            return "Very Difficult Attack Fixtures"
    
    def _get_defense_rating(self, avg_difficulty: float) -> str:
        """Get defense rating based on average difficulty"""
        if avg_difficulty <= 2:
            return "Excellent Defense Fixtures"
        elif avg_difficulty <= 2.5:
            return "Good Defense Fixtures"
        elif avg_difficulty <= 3.5:
            return "Average Defense Fixtures"
        elif avg_difficulty <= 4:
            return "Difficult Defense Fixtures"
        else:
            return "Very Difficult Defense Fixtures"
    
    def _get_current_gameweek(self) -> int:
        """Get current gameweek number"""
        # Try to get from FPL API service first
        try:
            from services.enhanced_fpl_api_service import EnhancedFPLAPIService
            api_service = EnhancedFPLAPIService()
            current_gw = api_service.get_current_gameweek()
            if current_gw and current_gw > 0:
                return current_gw
        except Exception:
            pass
        
        # Try to get from session state
        if 'current_gameweek' in st.session_state:
            gw = st.session_state.get('current_gameweek')
            if gw and gw > 0:
                return gw
        
        # Return gameweek 4 as the current default instead of 20
        return 4
    
    def _display_team_fixture_breakdown(self, team_fixtures, df):
        """Display detailed team-by-team fixture breakdown"""
        st.subheader("📋 Team-by-Team Fixture Breakdown")
        
        # Sort teams by difficulty (easiest first)
        sorted_teams = sorted(team_fixtures.items(), key=lambda x: x[1]['average_difficulty'])
        
        for team_name, fixtures_data in sorted_teams:
            with st.expander(f"⚽ {team_name} - {fixtures_data['rating']} ({fixtures_data['average_difficulty']:.1f} avg)"):
                
                # Show next 5 fixtures
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Next 5 Fixtures:**")
                    current_gw = self._get_current_gameweek()
                    for i, fixture in enumerate(fixtures_data['fixtures']):
                        home_away = "🏠 vs" if fixture['home'] else "✈️ @"
                        difficulty_emoji = "🟢" if fixture['difficulty'] <= 2 else "🟡" if fixture['difficulty'] == 3 else "🔴"
                        gw_num = current_gw + i
                        st.write(f"GW{gw_num}: {home_away} {fixture['opponent']} {difficulty_emoji} ({fixture['difficulty_text']})")
                
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
    
    def _display_attack_recommendations(self, team_attack_fixtures, df):
        """Display attack-specific recommendations"""
        st.subheader("⚔️ Attack Fixture Recommendations")
        
        # Sort teams by attack difficulty (easiest first)
        sorted_attack_teams = sorted(team_attack_fixtures.items(), 
                                   key=lambda x: x[1]['avg_attack_difficulty'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("🟢 **Best Attack Fixtures**")
            st.write("*Target attacking players from these teams*")
            
            for team_name, fixtures_data in sorted_attack_teams[:5]:
                team_players = df[df['team_short_name'] == team_name]
                if not team_players.empty:
                    # Get top attacking players
                    attacking_players = team_players[team_players['position_name'].isin(['Midfielder', 'Forward'])]
                    if not attacking_players.empty:
                        top_attacker = attacking_players.nlargest(1, 'total_points').iloc[0]
                        st.write(f"• **{team_name}** ({fixtures_data['avg_attack_difficulty']:.1f}) - {top_attacker['web_name']}")
        
        with col2:
            st.error("🔴 **Difficult Attack Fixtures**")
            st.write("*Consider benching attacking players from these teams*")
            
            for team_name, fixtures_data in sorted_attack_teams[-5:]:
                team_players = df[df['team_short_name'] == team_name]
                if not team_players.empty:
                    attacking_players = team_players[team_players['position_name'].isin(['Midfielder', 'Forward'])]
                    if not attacking_players.empty:
                        top_attacker = attacking_players.nlargest(1, 'total_points').iloc[0]
                        st.write(f"• **{team_name}** ({fixtures_data['avg_attack_difficulty']:.1f}) - {top_attacker['web_name']}")
    
    def _display_defense_recommendations(self, team_defense_fixtures, df):
        """Display defense-specific recommendations"""
        st.subheader("🛡️ Defense Fixture Recommendations")
        
        # Sort teams by defense difficulty (easiest first)
        sorted_defense_teams = sorted(team_defense_fixtures.items(), 
                                    key=lambda x: x[1]['avg_defense_difficulty'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("🟢 **Best Defense Fixtures**")
            st.write("*Target defensive players from these teams*")
            
            for team_name, fixtures_data in sorted_defense_teams[:5]:
                team_players = df[df['team_short_name'] == team_name]
                if not team_players.empty:
                    # Get top defensive players
                    defensive_players = team_players[team_players['position_name'].isin(['Goalkeeper', 'Defender'])]
                    if not defensive_players.empty:
                        top_defender = defensive_players.nlargest(1, 'total_points').iloc[0]
                        st.write(f"• **{team_name}** ({fixtures_data['avg_defense_difficulty']:.1f}) - {top_defender['web_name']}")
        
        with col2:
            st.error("🔴 **Difficult Defense Fixtures**")
            st.write("*Consider benching defensive players from these teams*")
            
            for team_name, fixtures_data in sorted_defense_teams[-5:]:
                team_players = df[df['team_short_name'] == team_name]
                if not team_players.empty:
                    defensive_players = team_players[team_players['position_name'].isin(['Goalkeeper', 'Defender'])]
                    if not defensive_players.empty:
                        top_defender = defensive_players.nlargest(1, 'total_points').iloc[0]
                        st.write(f"• **{team_name}** ({fixtures_data['avg_defense_difficulty']:.1f}) - {top_defender['web_name']}")
    
    def _display_attack_defense_analysis(self, df):
        """Display attacking and defensive fixture analysis using official FPL data"""
        st.subheader("⚔️ Attacking vs Defensive Fixture Analysis")
        
        with st.expander("📚 Understanding Attack vs Defense Analysis", expanded=False):
            st.markdown("""
            **Attacking Fixtures**: How easy it is for players to score/assist
            - Uses **official FPL defensive strength ratings**
            - Target players facing weak defenses
            
            **Defensive Fixtures**: How likely defenders/GKs are to get clean sheets
            - Uses **official FPL attacking strength ratings**
            - Target defenders facing weak attacks
            
            ✅ **Data Source**: Official Fantasy Premier League API team strength values
            """)
        
        # Get all teams
        all_teams = df['team_short_name'].unique() if 'team_short_name' in df.columns else []
        
        # Track API data usage
        api_teams_count = 0
        fallback_teams_count = 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚔️ Best Attacking Fixtures")
            st.write("*Teams facing weak defenses (official FPL ratings)*")
            
            attacking_scores = []
            for team in all_teams:
                if pd.notna(team):
                    fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team, 5)
                    
                    # Track data source
                    if fixtures.get('is_fallback'):
                        fallback_teams_count += 1
                    else:
                        api_teams_count += 1
                    
                    # Calculate attacking score using real opponent defensive strength
                    attacking_score = 0
                    valid_fixtures = 0
                    
                    for fixture in fixtures['fixtures']:
                        opponent = fixture['opponent']
                        is_home = fixture['home']
                        
                        # Get opponent's defensive strength from FPL API
                        opponent_strength = self.fixture_service.get_team_attack_defense_strength(opponent)
                        
                        if is_home:
                            opponent_defense = opponent_strength.get('defense_away', 3)
                        else:
                            opponent_defense = opponent_strength.get('defense_home', 3)
                        
                        # Lower defense = better for attacking (invert scale)
                        fixture_attacking_score = 6 - opponent_defense
                        attacking_score += fixture_attacking_score
                        valid_fixtures += 1
                    
                    if valid_fixtures > 0:
                        attacking_score = attacking_score / valid_fixtures
                    else:
                        attacking_score = 3  # Neutral
                    
                    attacking_scores.append({
                        'team': team,
                        'attacking_score': attacking_score,
                        'fixtures': fixtures
                    })
            
            # Display data source status
            total_teams = api_teams_count + fallback_teams_count
            if total_teams > 0:
                api_percentage = (api_teams_count / total_teams) * 100
                if api_percentage == 100:
                    st.success("✅ Using official FPL defensive strength data")
                elif api_percentage >= 50:
                    st.warning(f"⚠️ {api_teams_count}/{total_teams} teams using FPL API data")
                else:
                    st.error(f"❌ Limited API access: {api_teams_count}/{total_teams} teams")
            
            # Sort by attacking score (best first)
            attacking_scores.sort(key=lambda x: x['attacking_score'], reverse=True)
            
            for team_data in attacking_scores[:10]:
                team_name = team_data['team']
                score = team_data['attacking_score']
                
                score_color = "🟢" if score >= 3.5 else "🟡" if score >= 2.5 else "🔴"
                
                with st.expander(f"{score_color} {team_name} - Attacking Score: {score:.1f}"):
                    # Show next opponents with their defensive ratings
                    for fixture in team_data['fixtures']['fixtures'][:3]:
                        opponent = fixture['opponent']
                        opponent_strength = self.fixture_service.get_team_attack_defense_strength(opponent)
                        home_away = "🏠 vs" if fixture['home'] else "✈️ @"
                        
                        if fixture['home']:
                            def_rating = opponent_strength.get('defense_away', 3)
                        else:
                            def_rating = opponent_strength.get('defense_home', 3)
                        
                        st.write(f"GW{fixture['gameweek']}: {home_away} {opponent} (Def: {def_rating}/5)")
                    
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
            st.write("*Teams facing weak attacks (official FPL ratings)*")
            
            defensive_scores = []
            for team in all_teams:
                if pd.notna(team):
                    fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team, 5)
                    
                    # Calculate defensive score using real opponent attacking strength
                    defensive_score = 0
                    valid_fixtures = 0
                    
                    for fixture in fixtures['fixtures']:
                        opponent = fixture['opponent']
                        is_home = fixture['home']
                        
                        # Get opponent's attacking strength from FPL API
                        opponent_strength = self.fixture_service.get_team_attack_defense_strength(opponent)
                        
                        if is_home:
                            opponent_attack = opponent_strength.get('attack_away', 3)
                        else:
                            opponent_attack = opponent_strength.get('attack_home', 3)
                        
                        # Lower attack = better for defending (invert scale)
                        fixture_defensive_score = 6 - opponent_attack
                        defensive_score += fixture_defensive_score
                        valid_fixtures += 1
                    
                    if valid_fixtures > 0:
                        defensive_score = defensive_score / valid_fixtures
                    else:
                        defensive_score = 3  # Neutral
                    
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
                    # Show next opponents with their attacking ratings
                    for fixture in team_data['fixtures']['fixtures'][:3]:
                        opponent = fixture['opponent']
                        opponent_strength = self.fixture_service.get_team_attack_defense_strength(opponent)
                        home_away = "🏠 vs" if fixture['home'] else "✈️ @"
                        
                        if fixture['home']:
                            att_rating = opponent_strength.get('attack_away', 3)
                        else:
                            att_rating = opponent_strength.get('attack_home', 3)
                        
                        st.write(f"GW{fixture['gameweek']}: {home_away} {opponent} (Att: {att_rating}/5)")
                    
                    # Show top defensive players from this team
                    team_players = df[df['team_short_name'] == team_name]
                    if not team_players.empty:
                        defensive_players = team_players[team_players['position_name'].isin(['Goalkeeper', 'Defender'])]
                        if not defensive_players.empty:
                            top_defenders = defensive_players.nlargest(3, 'total_points')
                            st.write("**Top Defensive Options:**")
                            for _, player in top_defenders.iterrows():
                                st.write(f"• {player.get('web_name', 'Unknown')} ({player.get('position_name', 'Unknown')})")
        
        # Combined recommendation using official data
        st.subheader("🎯 Combined Fixture Recommendations (FPL API Data)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success("""
            **🟢 Excellent Fixtures**
            - Weak opponent defense/attack (FPL ratings 1-2)
            - Strong clean sheet potential
            - Consider captaincy
            """)
        
        with col2:
            st.info("""
            **🟡 Mixed Fixtures**
            - Average opponent strength (FPL ratings 3)
            - Monitor team news
            - Backup options ready
            """)
        
        with col3:
            st.error("""
            **🔴 Difficult Fixtures**
            - Strong opponent defense/attack (FPL ratings 4-5)
            - Avoid captaincy
            - Look for alternatives
            """)

    def _display_captain_fixture_analysis(self, df):
        """Analyze fixtures for captaincy decisions using official FPL API data"""
        st.subheader("👑 Captain Fixture Analysis")
        
        with st.expander("📚 Understanding Captain Analysis with FPL Data", expanded=False):
            st.markdown("""
            **Captain Analysis** uses official FPL data to rank captaincy options:
            
            🎯 **Factors Considered:**
            - **Official FPL fixture difficulty** (1-5 scale)
            - **Player form** and total points
            - **Ownership percentage** for risk/reward analysis
            - **Home vs Away** advantage from actual fixtures
            
            ✅ **Data Source**: Official Fantasy Premier League API
            """)
        
        # Get potential captains (high-scoring non-GKs)
        captain_candidates = []
        api_fixture_count = 0
        fallback_fixture_count = 0
        
        # Filter for good captain options
        if 'position_name' in df.columns and 'total_points' in df.columns:
            non_gks = df[df['position_name'] != 'Goalkeeper']
            top_scorers = non_gks.nlargest(20, 'total_points')  # Top 20 scorers
            
            for _, player in top_scorers.iterrows():
                team_name = player.get('team_short_name', 'UNK')
                
                # Get fixture difficulty for this player's team using official FPL API
                fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team_name, 1)  # Next fixture only
                
                # Track data source
                if fixtures.get('is_fallback'):
                    fallback_fixture_count += 1
                else:
                    api_fixture_count += 1
                
                # Get next fixture details
                next_fixture = fixtures['fixtures'][0] if fixtures['fixtures'] else None
                
                if next_fixture:
                    # Get opponent strength data from FPL API
                    opponent_strength = self.fixture_service.get_team_attack_defense_strength(next_fixture['opponent'])
                    
                    # Calculate fixture attractiveness for attacking players
                    if next_fixture['home']:
                        opponent_defense = opponent_strength.get('defense_away', 3)
                        venue_bonus = 0.5  # Home advantage
                    else:
                        opponent_defense = opponent_strength.get('defense_home', 3)
                        venue_bonus = -0.2  # Away disadvantage
                    
                    # Convert to attractiveness score (lower defense = higher attractiveness)
                    fixture_attractiveness = (6 - opponent_defense) + venue_bonus
                else:
                    fixture_attractiveness = 3
                    opponent_defense = 3
                
                captain_candidates.append({
                    'name': player.get('web_name', 'Unknown'),
                    'team': team_name,
                    'position': player.get('position_name', 'Unknown'),
                    'form': float(player.get('form', 0)),
                    'total_points': int(player.get('total_points', 0)),
                    'ownership': float(player.get('selected_by_percent', 0)),
                    'fixture_difficulty': fixtures['average_difficulty'],
                    'fixture_attractiveness': fixture_attractiveness,
                    'next_opponent': next_fixture['opponent'] if next_fixture else 'TBD',
                    'opponent_defense': opponent_defense,
                    'is_home': next_fixture['home'] if next_fixture else True,
                    'kickoff_time': next_fixture.get('kickoff_time') if next_fixture else None
                })
        
        # Display data source status
        total_fixtures = api_fixture_count + fallback_fixture_count
        if total_fixtures > 0:
            api_percentage = (api_fixture_count / total_fixtures) * 100
            
            if api_percentage == 100:
                st.success("✅ **Captain analysis using official FPL fixture data**")
            elif api_percentage >= 50:
                st.warning(f"⚠️ **Mixed data sources**: {api_fixture_count}/{total_fixtures} using FPL API")
            else:
                st.error(f"❌ **Limited API access**: {api_fixture_count}/{total_fixtures} using FPL API")
        
        if captain_candidates:
            # Calculate captain score using official data
            for candidate in captain_candidates:
                fixture_score = candidate['fixture_attractiveness']
                form_score = candidate['form']
                points_score = candidate['total_points'] / 100  # Normalize
                
                # Weight: fixture 40%, form 40%, historical points 20%
                candidate['captain_score'] = (fixture_score * 0.4 + form_score * 0.4 + points_score * 0.2)
            
            # Sort by captain score
            captain_candidates.sort(key=lambda x: x['captain_score'], reverse=True)
            
            st.write("**📊 Captain Rankings (Using Official FPL Data):**")
            
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
                
                with st.expander(f"{i}. {candidate['name']} - {rec_level} (Score: {candidate['captain_score']:.1f})"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Next Fixture", f"{home_away} {candidate['next_opponent']} {difficulty_emoji}")
                        st.write(f"**FPL Difficulty:** {candidate['fixture_difficulty']:.1f}/5")
                        st.write(f"**Opponent Defense:** {candidate['opponent_defense']}/5")
                    
                    with col2:
                        st.metric("Form", f"{candidate['form']:.1f}")
                        st.metric("Total Points", candidate['total_points'])
                        st.write(f"**Fixture Score:** {candidate['fixture_attractiveness']:.1f}")
                    
                    with col3:
                        st.metric("Ownership", f"{candidate['ownership']:.1f}%")
                        
                        # Risk/reward analysis
                        if candidate['ownership'] > 50:
                            st.write("🛡️ **Safe pick** - High ownership")
                        elif candidate['ownership'] < 15:
                            st.write("💎 **Differential** - Low ownership")
                        else:
                            st.write("⚖️ **Balanced** - Medium ownership")
                        
                        # Show kickoff time if available
                        if candidate.get('kickoff_time'):
                            st.write(f"**Kickoff:** {candidate['kickoff_time']}")
        
        else:
            st.warning("No captain candidates found")
        
        # Captain strategy tips with FPL context
        st.subheader("💡 Captain Strategy Tips (FPL Data)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **🎯 This Gameweek:**
            - Prioritize FPL difficulty 1-2 fixtures
            - Consider opponent defensive weakness
            - Check for rotation risk in team news
            - Monitor home advantage
            """)
        
        with col2:
            st.warning("""
            **🔮 Long-term Planning:**
            - Look ahead 2-3 gameweeks in FPL fixtures
            - Plan around DGW/BGW schedules
            - Consider differential captains vs template
            - Track form trends vs fixture difficulty
            """)

    def _display_fixture_transfer_targets(self, df):
        """Identify transfer targets based on official FPL fixture data"""
        st.subheader("🔄 Fixture-Based Transfer Targets")
        
        with st.expander("📚 Understanding Transfer Target Analysis", expanded=False):
            st.markdown("""
            **Transfer Target Analysis** uses official FPL data to identify optimal transfers:
            
            🎯 **Analysis Factors:**
            - **Official FPL fixture difficulty ratings** for next 5 gameweeks
            - **Player ownership** for differential opportunities
            - **Form and points** for performance validation
            - **Price and value** for budget optimization
            
            ✅ **Data Source**: Official Fantasy Premier League API
            """)
        
        # Get all teams with their fixture difficulties using official FPL data
        all_teams = df['team_short_name'].unique() if 'team_short_name' in df.columns else []
        
        team_fixture_scores = []
        api_teams = 0
        fallback_teams = 0
        
        for team in all_teams:
            if pd.notna(team):
                # Get official FPL fixture data
                fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team, 5)
                
                # Track data source
                if fixtures.get('is_fallback'):
                    fallback_teams += 1
                else:
                    api_teams += 1
                
                # Calculate fixture score (higher = better fixtures)
                fixture_score = 6 - fixtures['average_difficulty']
                
                team_fixture_scores.append({
                    'team': team,
                    'fixture_score': fixture_score,
                    'avg_difficulty': fixtures['average_difficulty'],
                    'rating': fixtures['rating'],
                    'fixtures_data': fixtures,
                    'is_api_data': not fixtures.get('is_fallback', False)
                })
        
        # Display data source status
        total_teams = api_teams + fallback_teams
        if total_teams > 0:
            api_percentage = (api_teams / total_teams) * 100
            
            if api_percentage == 100:
                st.success(f"✅ **Official FPL fixture data** - All {total_teams} teams using API")
            elif api_percentage >= 50:
                st.warning(f"⚠️ **Mixed data sources** - {api_teams}/{total_teams} teams using FPL API")
            else:
                st.error(f"❌ **Limited API access** - {api_teams}/{total_teams} teams using FPL API")
        
        # Sort by fixture quality (best first)
        team_fixture_scores.sort(key=lambda x: x['fixture_score'], reverse=True)
        
        transfer_tabs = st.tabs(["🎯 Best Fixtures", "⚠️ Worst Fixtures", "💎 Differentials", "📊 FPL Insights"])
        
        with transfer_tabs[0]:
            st.subheader("🟢 Teams with Best Fixtures (Official FPL)")
            st.write("*Consider players from these teams - based on official fixture difficulty*")
            
            best_teams = team_fixture_scores[:8]
            
            for team_data in best_teams:
                team_name = team_data['team']
                data_source_indicator = "🟢 API" if team_data['is_api_data'] else "🟡 Est."
                
                # Get best players from this team
                team_players = df[df['team_short_name'] == team_name]
                if not team_players.empty:
                    # Get top players by points
                    top_players = team_players.nlargest(5, 'total_points')
                    
                    with st.expander(f"{data_source_indicator} {team_name} - {team_data['rating']} Fixtures (FPL: {team_data['avg_difficulty']:.1f})"):
                        
                        # Show next 3 fixtures with official data
                        st.write("**📅 Next 3 Official Fixtures:**")
                        for fixture in team_data['fixtures_data']['fixtures'][:3]:
                            home_away = "🏠 vs" if fixture['home'] else "✈️ @"
                            difficulty_emoji = "🟢" if fixture['difficulty'] <= 2 else "🟡" if fixture['difficulty'] == 3 else "🔴"
                            st.write(f"GW{fixture['gameweek']}: {home_away} {fixture['opponent']} {difficulty_emoji} (FPL: {fixture['difficulty']})")
                        
                        st.write("**🎯 Top Transfer Targets:**")
                        
                        for _, player in top_players.iterrows():
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.write(f"**{player.get('web_name', 'Unknown')}**")
                                st.write(f"{player.get('position_name', 'Unknown')}")
                            
                            with col2:
                                price = float(player.get('now_cost', 0))/10
                                st.write(f"£{price:.1f}m")
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
            st.subheader("🔴 Teams with Worst Fixtures (Official FPL)")
            st.write("*Consider transferring out players from these teams*")
            
            worst_teams = team_fixture_scores[-8:]
            
            for team_data in worst_teams:
                team_name = team_data['team']
                data_source_indicator = "🟢 API" if team_data['is_api_data'] else "🟡 Est."
                
                # Get popular players from this team (likely in many squads)
                team_players = df[df['team_short_name'] == team_name]
                if not team_players.empty:
                    # Get most owned players
                    popular_players = team_players.nlargest(3, 'selected_by_percent')
                    
                    with st.expander(f"{data_source_indicator} {team_name} - {team_data['rating']} Fixtures (FPL: {team_data['avg_difficulty']:.1f})"):
                        
                        # Show next 3 difficult fixtures
                        st.write("**📅 Upcoming Difficult Fixtures:**")
                        for fixture in team_data['fixtures_data']['fixtures'][:3]:
                            home_away = "🏠 vs" if fixture['home'] else "✈️ @"
                            difficulty_emoji = "🟢" if fixture['difficulty'] <= 2 else "🟡" if fixture['difficulty'] == 3 else "🔴"
                            st.write(f"GW{fixture['gameweek']}: {home_away} {fixture['opponent']} {difficulty_emoji} (FPL: {fixture['difficulty']})")
                        
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
            st.subheader("💎 Differential Opportunities (FPL Data)")
            st.write("*Low ownership players with excellent fixtures*")
            
            # Find differential players (low ownership) with good fixtures using official data
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
                                'fixture_score': team_data['fixture_score'],
                                'fpl_difficulty': team_data['avg_difficulty'],
                                'is_api_data': team_data['is_api_data']
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
                data_indicator = "🟢" if candidate['is_api_data'] else "🟡"
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**{candidate['name']}** {data_indicator}")
                    st.write(f"{candidate['position']} ({candidate['team']})")
                
                with col2:
                    st.write(f"£{candidate['price']:.1f}m")
                    st.write(f"Form: {candidate['form']:.1f}")
                
                with col3:
                    st.write(f"{candidate['points']} points")
                    st.write(f"FPL Diff: {candidate['fpl_difficulty']:.1f}")
                
                with col4:
                    st.write(f"{candidate['ownership']:.1f}% owned")
                    st.success("💎 Differential")
        
        with transfer_tabs[3]:
            st.subheader("📊 FPL Fixture Insights")
            st.write("*Data-driven insights from official FPL fixtures*")
            
            # Create insights based on official data
            if api_teams > 0:
                # Calculate insights from API teams only
                api_team_data = [t for t in team_fixture_scores if t['is_api_data']]
                
                if api_team_data:
                    difficulties = [t['avg_difficulty'] for t in api_team_data]
                    avg_league_difficulty = sum(difficulties) / len(difficulties)
                    
                    easy_fixture_teams = [t for t in api_team_data if t['avg_difficulty'] <= 2.5]
                    hard_fixture_teams = [t for t in api_team_data if t['avg_difficulty'] >= 3.5]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info(f"""
                        **📈 League Fixture Analysis:**
                        - **Average Difficulty:** {avg_league_difficulty:.1f}/5
                        - **Easy Fixture Teams:** {len(easy_fixture_teams)}
                        - **Hard Fixture Teams:** {len(hard_fixture_teams)}
                        - **Data Coverage:** {api_teams}/{total_teams} teams
                        """)
                    
                    with col2:
                        st.success(f"""
                        **🎯 Transfer Strategy:**
                        - Target teams with <2.5 difficulty
                        - Avoid teams with >3.5 difficulty  
                        - Monitor fixture swings
                        - Plan 2-3 gameweeks ahead
                        """)

    def _display_team_fixture_comparison(self, df):
        """Compare fixture difficulty between teams using official FPL API data"""
        st.subheader("📊 Team Fixture Comparison")
        
        with st.expander("📚 Understanding Team Comparison", expanded=False):
            st.markdown("""
            **Team Fixture Comparison** allows you to directly compare upcoming fixtures between any two teams:
            
            🎯 **Comparison Features:**
            - **Official FPL fixture difficulty** for both teams
            - **Side-by-side fixture analysis** for next 5 gameweeks
            - **Player recommendations** from both teams
            - **Transfer decision support** based on fixture comparison
            
            ✅ **Data Source**: Official Fantasy Premier League API
            """)
        
        # Team selection for comparison
        all_teams = sorted(df['team_short_name'].unique()) if 'team_short_name' in df.columns else []
        
        if len(all_teams) < 2:
            st.warning("Need at least 2 teams to perform comparison.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select First Team", all_teams, index=0 if all_teams else None, key="team1_select")
        
        with col2:
            team2 = st.selectbox("Select Second Team", all_teams, index=1 if len(all_teams) > 1 else 0, key="team2_select")
        
        if team1 and team2 and team1 != team2:
            # Compare fixtures between teams using official FPL data
            comparison = self.fixture_service.compare_fixture_run(team1, team2, 5)
            
            # Track data source quality
            team1_is_api = not comparison['team1']['data'].get('is_fallback', False)
            team2_is_api = not comparison['team2']['data'].get('is_fallback', False)
            
            # Display data source status
            if team1_is_api and team2_is_api:
                st.success("✅ **Both teams using official FPL fixture data**")
            elif team1_is_api or team2_is_api:
                st.warning("⚠️ **Mixed data sources** - some teams using FPL API, others using estimates")
            else:
                st.error("❌ **Limited API access** - using estimated fixture data")
            
            # Display comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"⚽ {team1}")
                team1_data = comparison['team1']['data']
                data_indicator1 = "🟢 API" if team1_is_api else "🟡 Est."
                
                st.metric("Average Difficulty", f"{team1_data['average_difficulty']:.1f}/5")
                st.metric("Fixture Rating", team1_data['rating'])
                st.write(f"**Data Source:** {data_indicator1}")
                
                st.write("**Next 5 Fixtures:**")
                for fixture in team1_data['fixtures']:
                    home_away = "🏠 vs" if fixture['home'] else "✈️ @"
                    difficulty_emoji = "🟢" if fixture['difficulty'] <= 2 else "🟡" if fixture['difficulty'] == 3 else "🔴"
                    st.write(f"GW{fixture['gameweek']}: {home_away} {fixture['opponent']} {difficulty_emoji} (FPL: {fixture['difficulty']})")
            
            with col2:
                st.subheader(f"⚽ {team2}")
                team2_data = comparison['team2']['data']
                data_indicator2 = "🟢 API" if team2_is_api else "🟡 Est."
                
                st.metric("Average Difficulty", f"{team2_data['average_difficulty']:.1f}/5")
                st.metric("Fixture Rating", team2_data['rating'])
                st.write(f"**Data Source:** {data_indicator2}")
                
                st.write("**Next 5 Fixtures:**")
                for fixture in team2_data['fixtures']:
                    home_away = "🏠 vs" if fixture['home'] else "✈️ @"
                    difficulty_emoji = "🟢" if fixture['difficulty'] <= 2 else "🟡" if fixture['difficulty'] == 3 else "🔴"
                    st.write(f"GW{fixture['gameweek']}: {home_away} {fixture['opponent']} {difficulty_emoji} (FPL: {fixture['difficulty']})")
            
            # Recommendation based on official FPL data
            st.subheader("🎯 Fixture Comparison Result")
            recommended_team = comparison['recommendation']
            
            difficulty_diff = abs(team1_data['average_difficulty'] - team2_data['average_difficulty'])
            
            if difficulty_diff < 0.3:
                st.info(f"⚖️ **Similar fixtures** - Both {team1} and {team2} have comparable difficulty ({team1_data['average_difficulty']:.1f} vs {team2_data['average_difficulty']:.1f})")
            elif recommended_team == team1:
                st.success(f"✅ **{team1}** has significantly easier fixtures than {team2} (Difference: {difficulty_diff:.1f})")
            else:
                st.success(f"✅ **{team2}** has significantly easier fixtures than {team1} (Difference: {difficulty_diff:.1f})")
            
            # Show players from both teams with transfer recommendations
            st.subheader("👥 Player Comparison & Transfer Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{team1} Players:**")
                team1_players = df[df['team_short_name'] == team1].nlargest(5, 'total_points')
                
                if not team1_players.empty:
                    for _, player in team1_players.iterrows():
                        price = float(player.get('now_cost', 0))/10
                        ownership = float(player.get('selected_by_percent', 0))
                        
                        # Add recommendation based on fixture comparison
                        if recommended_team == team1:
                            rec_emoji = "🎯"
                            rec_text = "Target"
                        elif difficulty_diff < 0.3:
                            rec_emoji = "⚖️"
                            rec_text = "Monitor"
                        else:
                            rec_emoji = "⚠️"
                            rec_text = "Consider out"
                        
                        st.write(f"{rec_emoji} **{player.get('web_name', 'Unknown')}** ({player.get('position_name', 'Unknown')}) - £{price:.1f}m | {ownership:.1f}% owned | *{rec_text}*")
                else:
                    st.write("No players found for this team")
            
            with col2:
                st.write(f"**{team2} Players:**")
                team2_players = df[df['team_short_name'] == team2].nlargest(5, 'total_points')
                
                if not team2_players.empty:
                    for _, player in team2_players.iterrows():
                        price = float(player.get('now_cost', 0))/10
                        ownership = float(player.get('selected_by_percent', 0))
                        
                        # Add recommendation based on fixture comparison
                        if recommended_team == team2:
                            rec_emoji = "🎯"
                            rec_text = "Target"
                        elif difficulty_diff < 0.3:
                            rec_emoji = "⚖️"
                            rec_text = "Monitor"
                        else:
                            rec_emoji = "⚠️"
                            rec_text = "Consider out"
                        
                        st.write(f"{rec_emoji} **{player.get('web_name', 'Unknown')}** ({player.get('position_name', 'Unknown')}) - £{price:.1f}m | {ownership:.1f}% owned | *{rec_text}*")
                else:
                    st.write("No players found for this team")
            
            # Advanced comparison insights
            st.subheader("📈 Advanced Comparison Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Count easy vs hard fixtures for each team
                team1_easy = len([f for f in team1_data['fixtures'] if f['difficulty'] <= 2])
                team1_hard = len([f for f in team1_data['fixtures'] if f['difficulty'] >= 4])
                
                st.info(f"""
                **{team1} Breakdown:**
                - Easy fixtures (1-2): {team1_easy}
                - Hard fixtures (4-5): {team1_hard}
                - Rating: {team1_data['rating']}
                """)
            
            with col2:
                team2_easy = len([f for f in team2_data['fixtures'] if f['difficulty'] <= 2])
                team2_hard = len([f for f in team2_data['fixtures'] if f['difficulty'] >= 4])
                
                st.info(f"""
                **{team2} Breakdown:**
                - Easy fixtures (1-2): {team2_easy}
                - Hard fixtures (4-5): {team2_hard}
                - Rating: {team2_data['rating']}
                """)
            
            with col3:
                # Transfer strategy recommendation
                if recommended_team == team1:
                    strategy = f"Consider transferring {team2} players to {team1} players"
                elif recommended_team == team2:
                    strategy = f"Consider transferring {team1} players to {team2} players"
                else:
                    strategy = "Fixtures are similar - consider other factors"
                
                st.success(f"""
                **Transfer Strategy:**
                {strategy}
                
                **Best Time:** Plan transfers before difficult fixture runs begin
                """)
        
        else:
            st.info("Please select two different teams to compare their fixtures.")
            
            # Show some example comparisons when no teams selected
            if len(all_teams) >= 4:
                st.subheader("💡 Suggested Comparisons")
                st.write("Try comparing these teams based on their typical fixture difficulty patterns:")
                
                sample_teams = all_teams[:4]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Popular Comparisons:**")
                    st.write(f"• {sample_teams[0]} vs {sample_teams[1]}")
                    st.write(f"• {sample_teams[2]} vs {sample_teams[3]}")
                
                with col2:
                    st.write("**Analysis Tips:**")
                    st.write("• Compare teams before making transfers")
                    st.write("• Look for fixture swings (easy to hard)")
                    st.write("• Consider both short and long-term fixtures")

    def _display_advanced_fixture_analysis(self, df):
        """Display advanced fixture analysis including enhanced analytics"""
        st.subheader("📈 Advanced Fixture Analysis")
        
        with st.expander("📚 Understanding Advanced Analysis", expanded=False):
            st.markdown("""
            **Advanced Fixture Analysis** provides deeper insights using multiple data sources:
            
            🎯 **Advanced Features:**
            - **Team strength trending** over time
            - **Home vs Away performance** analysis  
            - **Form-based fixture adjustments**
            - **Statistical transfer recommendations**
            
            📊 **Data Integration**: Combines official FPL data with calculated analytics
            """)
        
        # Sub-tabs for different types of advanced analysis
        advanced_tabs = st.tabs([
            "📊 Team Strength Analysis", 
            "🏠 Home vs Away", 
            "📈 Form-Based Fixtures",
            "🎯 Statistical Recommendations"
        ])
        
        with advanced_tabs[0]:
            self._render_team_strength_analysis(df)
        
        with advanced_tabs[1]:
            self._render_home_away_analysis(df)
        
        with advanced_tabs[2]:
            self._render_form_based_analysis(df)
        
        with advanced_tabs[3]:
            self._render_statistical_recommendations(df)

    def _render_statistical_recommendations(self, df):
        """Provide statistical transfer recommendations based on fixture analysis"""
        st.subheader("🎯 Statistical Transfer Recommendations")
        st.info("💡 Data-driven recommendations combining fixture difficulty with player performance")
        
        if df.empty:
            st.warning("No data available for statistical analysis")
            return
        
        # Get all teams with their fixture data
        all_teams = df['team_short_name'].unique() if 'team_short_name' in df.columns else []
        
        # Calculate comprehensive team scores
        team_analysis = []
        
        for team in all_teams:
            if pd.notna(team):
                # Get fixture data
                fixtures = self.fixture_service.get_upcoming_fixtures_difficulty(team, 5)
                
                # Get team players
                team_players = df[df['team_short_name'] == team]
                
                if not team_players.empty:
                    # Calculate team metrics
                    avg_points = team_players['total_points'].mean()
                    avg_form = team_players['form'].mean() if 'form' in team_players.columns else 0
                    avg_ownership = team_players['selected_by_percent'].mean() if 'selected_by_percent' in team_players.columns else 0
                    avg_price = team_players['now_cost'].mean() / 10 if 'now_cost' in team_players.columns else 0
                    
                    # Fixture score (lower difficulty = higher score)
                    fixture_score = 6 - fixtures['average_difficulty']
                    
                    # Combined recommendation score
                    rec_score = (
                        fixture_score * 0.4 +  # Fixture difficulty weight
                        (avg_form / 10) * 0.3 +  # Form weight
                        (avg_points / 100) * 0.2 +  # Historical performance weight
                        (1 / avg_price if avg_price > 0 else 0) * 0.1  # Value weight
                    )
                    
                    team_analysis.append({
                        'team': team,
                        'fixture_score': fixture_score,
                        'avg_difficulty': fixtures['average_difficulty'],
                        'avg_points': avg_points,
                        'avg_form': avg_form,
                        'avg_ownership': avg_ownership,
                        'avg_price': avg_price,
                        'rec_score': rec_score,
                        'rating': fixtures['rating'],
                        'is_api_data': not fixtures.get('is_fallback', False)
                    })
        
        if not team_analysis:
            st.warning("No team analysis data available")
            return
        
        # Sort by recommendation score
        team_analysis.sort(key=lambda x: x['rec_score'], reverse=True)
        
        # Display recommendations in categories
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("🎯 **Top Transfer Targets**")
            st.write("*Teams with best fixture + performance combination*")
            
            top_teams = team_analysis[:6]
            for i, team_data in enumerate(top_teams, 1):
                data_indicator = "🟢" if team_data['is_api_data'] else "🟡"
                
                with st.expander(f"{i}. {data_indicator} {team_data['team']} - Score: {team_data['rec_score']:.2f}"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**Fixture Rating:** {team_data['rating']}")
                        st.write(f"**Avg Difficulty:** {team_data['avg_difficulty']:.1f}/5")
                        st.write(f"**Avg Form:** {team_data['avg_form']:.1f}")
                    
                    with col_b:
                        st.write(f"**Avg Points:** {team_data['avg_points']:.0f}")
                        st.write(f"**Avg Price:** £{team_data['avg_price']:.1f}m")
                        st.write(f"**Avg Ownership:** {team_data['avg_ownership']:.1f}%")
                    
                    # Show top 3 players from this team
                    team_players = df[df['team_short_name'] == team_data['team']]
                    if not team_players.empty:
                        top_players = team_players.nlargest(3, 'total_points')
                        st.write("**Recommended Players:**")
                        for _, player in top_players.iterrows():
                            price = float(player.get('now_cost', 0))/10
                            st.write(f"• {player.get('web_name', 'Unknown')} - £{price:.1f}m")
        
        with col2:
            st.error("⚠️ **Teams to Avoid**")
            st.write("*Teams with poor fixture + performance combination*")
            
            bottom_teams = team_analysis[-6:]
            for i, team_data in enumerate(bottom_teams, 1):
                data_indicator = "🟢" if team_data['is_api_data'] else "🟡"
                
                with st.expander(f"{i}. {data_indicator} {team_data['team']} - Score: {team_data['rec_score']:.2f}"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write(f"**Fixture Rating:** {team_data['rating']}")
                        st.write(f"**Avg Difficulty:** {team_data['avg_difficulty']:.1f}/5")
                        st.write(f"**Avg Form:** {team_data['avg_form']:.1f}")
                    
                    with col_b:
                        st.write(f"**Avg Points:** {team_data['avg_points']:.0f}")
                        st.write(f"**Avg Price:** £{team_data['avg_price']:.1f}m")
                        st.write(f"**Avg Ownership:** {team_data['avg_ownership']:.1f}%")
                    
                    # Show popular players to consider transferring out
                    team_players = df[df['team_short_name'] == team_data['team']]
                    if not team_players.empty:
                        if 'selected_by_percent' in team_players.columns:
                            popular_players = team_players.nlargest(3, 'selected_by_percent')
                        else:
                            popular_players = team_players.nlargest(3, 'total_points')
                        
                        st.write("**Consider Transferring Out:**")
                        for _, player in popular_players.iterrows():
                            price = float(player.get('now_cost', 0))/10
                            ownership = float(player.get('selected_by_percent', 0))
                            st.write(f"• {player.get('web_name', 'Unknown')} - £{price:.1f}m ({ownership:.1f}% owned)")
        
        # Overall transfer strategy
        st.subheader("📋 Overall Transfer Strategy")
        
        # Calculate league averages
        avg_rec_score = sum(t['rec_score'] for t in team_analysis) / len(team_analysis)
        avg_fixture_diff = sum(t['avg_difficulty'] for t in team_analysis) / len(team_analysis)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("League Avg Rec Score", f"{avg_rec_score:.2f}")
            st.metric("League Avg Difficulty", f"{avg_fixture_diff:.1f}/5")
        
        with col2:
            excellent_teams = len([t for t in team_analysis if t['rec_score'] > avg_rec_score + 0.5])
            poor_teams = len([t for t in team_analysis if t['rec_score'] < avg_rec_score - 0.5])
            
            st.metric("Excellent Opportunities", excellent_teams)
            st.metric("Teams to Avoid", poor_teams)
        
        with col3:
            api_coverage = len([t for t in team_analysis if t['is_api_data']]) / len(team_analysis) * 100
            st.metric("API Data Coverage", f"{api_coverage:.0f}%")
            
            if api_coverage == 100:
                st.success("🟢 Full API Coverage")
            elif api_coverage >= 70:
                st.warning("🟡 Good API Coverage")
            else:
                st.error("🔴 Limited API Coverage")

