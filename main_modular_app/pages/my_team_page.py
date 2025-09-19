"""
My Team Page - Handles FPL team import and analysis functionality
"""
import streamlit as st
import pandas as pd
from services.fpl_data_service import FPLDataService


class MyTeamPage:
    """Handles My FPL Team functionality"""
    
    def __init__(self):
        self.data_service = FPLDataService()
    
    def render(self):
        """Main render method for My Team page"""
        st.header("👤 My FPL Team")
        
        # Check if team is loaded
        if not st.session_state.get('my_team_loaded', False):
            self._render_team_import_section()
            return
        
        # Display loaded team
        team_data = st.session_state.my_team_data
        
        # Team overview
        st.subheader(f"🏆 {team_data.get('entry_name', 'Your Team')} (ID: {st.session_state.my_team_id})")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Rank", f"{team_data.get('summary_overall_rank', 'N/A'):,}")
        
        with col2:
            st.metric("Total Points", f"{team_data.get('summary_overall_points', 'N/A'):,}")
        
        with col3:
            st.metric("Gameweek Rank", f"{team_data.get('summary_event_rank', 'N/A'):,}")
        
        with col4:
            st.metric("Team Value", f"£{team_data.get('value', 1000)/10:.1f}m")
        
        # Team analysis tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "👥 Current Squad", 
            "📊 Performance Analysis", 
            "💡 Recommendations",
            "⭐ Starting XI Optimizer",
            "🎯 SWOT Analysis",
            "📈 Advanced Analytics",
            "🔄 Transfer Planning",
            "📊 Performance Comparison",
            "⚽ Fixture Analysis"
        ])
        
        with tab1:
            self._display_current_squad(team_data)
        
        with tab2:
            self._display_performance_analysis(team_data)
        
        with tab3:
            self._display_recommendations(team_data)
        
        with tab4:
            self._display_starting_xi_optimizer(team_data)
        
        with tab5:
            self._display_swot_analysis(team_data)
        
        with tab6:
            self._display_advanced_analytics(team_data)
        
        with tab7:
            self._display_transfer_planning(team_data)
        
        with tab8:
            self._display_performance_comparison(team_data)
        
        with tab9:
            self._display_fixture_analysis(team_data)
        
        # Reset team button
        if st.button("🔄 Load Different Team"):
            st.session_state.my_team_loaded = False
            st.rerun()
    
    def _render_team_import_section(self):
        """Render team import interface"""
        st.subheader("📥 Import Your FPL Team")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            team_id = st.text_input(
                "Enter your FPL Team ID:",
                placeholder="e.g., 1234567",
                help="Find your team ID in your FPL team URL"
            )
            
            # Get current gameweek for selection
            current_gw = self.data_service.get_current_gameweek()
            gameweeks = list(range(1, min(current_gw + 1, 39)))
            
            selected_gw = st.selectbox(
                "Select Gameweek:",
                gameweeks,
                index=min(current_gw - 1, 37) if current_gw else 0,
                help="Choose which gameweek's team to analyze"
            )
        
        with col2:
            if st.button("🔄 Load My Team", type="primary"):
                if team_id:
                    self._load_team_data(team_id, selected_gw)
                else:
                    st.warning("Please enter a team ID")
        
        # Instructions
        with st.expander("💡 How to find your Team ID", expanded=False):
            st.markdown("""
            **Step 1:** Go to the official FPL website and log in
            
            **Step 2:** Navigate to your team page
            
            **Step 3:** Look at the URL - it will look like:
            `https://fantasy.premierleague.com/entry/1234567/event/15`
            
            **Step 4:** Your Team ID is the number after `/entry/` (in this example: 1234567)
            
            **Note:** Your team must be public for this to work. You can change this in your FPL account settings.
            """)
    
    def _load_team_data(self, team_id, gameweek):
        """Load team data from FPL API"""
        with st.spinner("Loading your team..."):
            team_data = self.data_service.load_team_data(team_id, gameweek)
            
            if team_data:
                st.session_state.my_team_data = team_data
                st.session_state.my_team_id = team_id
                st.session_state.my_team_gameweek = gameweek
                st.session_state.my_team_loaded = True
                st.success("✅ Team loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Could not load team. Please check your team ID.")
    
    def _display_current_squad(self, team_data):
        """Display current squad with player details"""
        st.subheader("👥 Current Squad")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load player data to see detailed squad analysis")
            return
        
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available")
            return
        
        players_df = st.session_state.players_df
        
        # Match picks with player data
        squad_data = []
        formation_counts = {'Goalkeeper': 0, 'Defender': 0, 'Midfielder': 0, 'Forward': 0}
        
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                position = player.get('position_name', 'Unknown')
                
                # Count for formation
                if position in formation_counts:
                    formation_counts[position] += 1
                
                squad_data.append({
                    'Player': player.get('web_name', 'Unknown'),
                    'Position': position,
                    'Team': player.get('team_short_name', 'UNK'),
                    'Price': f"£{player.get('cost_millions', 0):.1f}m",
                    'Points': player.get('total_points', 0),
                    'Form': f"{player.get('form', 0):.1f}",
                    'Status': '(C)' if pick.get('is_captain') else '(VC)' if pick.get('is_vice_captain') else '',
                    'Playing': '✅' if pick.get('position', 12) <= 11 else '🪑'
                })
        
        if squad_data:
            # Formation display
            formation_str = f"{formation_counts['Goalkeeper']}-{formation_counts['Defender']}-{formation_counts['Midfielder']}-{formation_counts['Forward']}"
            st.info(f"**Formation:** {formation_str}")
            
            # Squad table
            squad_df = pd.DataFrame(squad_data)
            st.dataframe(
                squad_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Price": st.column_config.TextColumn("Price"),
                    "Points": st.column_config.NumberColumn("Points"),
                    "Form": st.column_config.TextColumn("Form"),
                    "Playing": st.column_config.TextColumn("Starting?")
                }
            )
            
            # Squad statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_value = sum([float(row['Price'].replace('£', '').replace('m', '')) for row in squad_data])
                st.metric("Squad Value", f"£{total_value:.1f}m")
            
            with col2:
                total_points = sum([row['Points'] for row in squad_data])
                st.metric("Total Points", f"{total_points:,}")
            
            with col3:
                avg_form = sum([float(row['Form']) for row in squad_data]) / len(squad_data)
                st.metric("Average Form", f"{avg_form:.1f}")
            
            with col4:
                starting_players = len([row for row in squad_data if row['Playing'] == '✅'])
                st.metric("Starting XI", f"{starting_players}/11")
    
    def _display_performance_analysis(self, team_data):
        """Display performance analysis"""
        st.subheader("📊 Performance Analysis")
        
        # Basic performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**📈 Season Performance**")
            total_points = team_data.get('summary_overall_points', 0)
            overall_rank = team_data.get('summary_overall_rank', 0)
            
            current_gw = team_data.get('gameweek', 1)
            avg_points_per_gw = total_points / max(current_gw, 1)
            
            st.metric("Total Points", f"{total_points:,}")
            st.metric("Points/GW", f"{avg_points_per_gw:.1f}")
            st.metric("Overall Rank", f"{overall_rank:,}" if overall_rank else "N/A")
        
        with col2:
            st.write("**🎯 Recent Performance**")
            gw_points = team_data.get('summary_event_points', 0)
            gw_rank = team_data.get('summary_event_rank', 0)
            
            # Performance indicators
            avg_gw_points = 50  # League average
            performance = "🔥 Excellent" if gw_points >= 70 else "👍 Good" if gw_points >= avg_gw_points else "📈 Below Average"
            
            st.metric("GW Points", f"{gw_points}")
            st.metric("GW Rank", f"{gw_rank:,}" if gw_rank else "N/A")
            st.metric("Performance", performance)
        
        with col3:
            st.write("**💰 Team Value**")
            team_value = team_data.get('value', 1000) / 10
            bank = team_data.get('bank', 0) / 10
            total_budget = team_value + bank
            
            st.metric("Team Value", f"£{team_value:.1f}m")
            st.metric("In Bank", f"£{bank:.1f}m")
            st.metric("Total Budget", f"£{total_budget:.1f}m")
        
        # Performance insights
        st.write("**💡 Performance Insights**")
        
        if overall_rank:
            total_players = 8000000  # Approximate
            percentile = (1 - (overall_rank / total_players)) * 100
            
            if percentile >= 90:
                st.success("🏆 Elite performance - Top 10% of all managers!")
            elif percentile >= 75:
                st.info("🥇 Excellent performance - Top 25% of all managers!")
            elif percentile >= 50:
                st.info("👍 Above average performance - Top 50% of all managers")
            else:
                st.warning("📈 Room for improvement - Focus on consistency and transfers")
    
    def _display_recommendations(self, team_data):
        """Display recommendations for the team"""
        st.subheader("💡 Team Recommendations")
        
        # Basic recommendations based on available data
        recommendations = []
        
        # Bank analysis
        bank = team_data.get('bank', 0) / 10
        if bank > 2:
            recommendations.append({
                'type': 'info',
                'title': 'Unused Funds',
                'message': f"You have £{bank:.1f}m in the bank. Consider upgrading a player to improve your team."
            })
        
        # Recent performance
        gw_points = team_data.get('summary_event_points', 0)
        if gw_points < 40:
            recommendations.append({
                'type': 'warning',
                'title': 'Low Gameweek Score',
                'message': "Your recent gameweek score was below average. Consider reviewing your captain choice and active players."
            })
        
        # Overall rank
        overall_rank = team_data.get('summary_overall_rank', 0)
        if overall_rank and overall_rank > 1000000:
            recommendations.append({
                'type': 'info',
                'title': 'Rank Improvement',
                'message': "Focus on consistent captain choices and popular template players to improve your rank."
            })
        
        # Display recommendations
        if recommendations:
            for rec in recommendations:
                if rec['type'] == 'success':
                    st.success(f"✅ **{rec['title']}**: {rec['message']}")
                elif rec['type'] == 'warning':
                    st.warning(f"⚠️ **{rec['title']}**: {rec['message']}")
                else:
                    st.info(f"💡 **{rec['title']}**: {rec['message']}")
        else:
            st.success("🎉 Your team looks good! Keep monitoring form and fixtures for optimal transfers.")
        
        # General advice
        st.write("**📚 General FPL Tips:**")
        st.write("• Monitor player form and upcoming fixtures")
        st.write("• Use your free transfer each gameweek")
        st.write("• Plan your chip usage strategically")
        st.write("• Consider differential picks to climb ranks")
        st.write("• Stay active in the transfer market")
    
    def _display_starting_xi_optimizer(self, team_data):
        """Display Starting XI Optimizer with comprehensive analysis"""
        st.subheader("⭐ Starting XI Optimizer")
        
        # Explanation section
        with st.expander("📚 How the Starting XI Optimizer Works", expanded=False):
            st.markdown("""
            The **Starting XI Optimizer** uses advanced algorithms to suggest your best possible lineup based on:
            
            🎯 **Key Factors:**
            - **Form Rating**: Recent performance trends (last 5 games)
            - **Fixture Difficulty**: Opposition strength analysis
            - **Expected Points**: Predicted performance this gameweek
            - **Minutes Played**: Reliability and rotation risk
            - **Price Per Point**: Value efficiency
            - **Team Strength**: Overall team performance indicators
            
            ⚽ **Optimization Strategy:**
            - Ensures valid formation (1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD)
            - Maximizes expected points while considering form
            - Suggests captain based on highest expected returns
            - Identifies bench players and substitution priorities
            """)
        
        if not st.session_state.get('data_loaded', False):
            st.warning("⚠️ Please load FPL player data first to enable the Starting XI Optimizer")
            return
        
        picks = team_data.get('picks', [])
        if not picks:
            st.warning("No squad data available for optimization")
            return
        
        players_df = st.session_state.players_df
        
        # Settings for optimization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            formation_pref = st.selectbox(
                "🏗️ Preferred Formation",
                ["Auto-Select", "3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"],
                help="Choose formation or let AI auto-select the best"
            )
        
        with col2:
            optimization_focus = st.selectbox(
                "🎯 Optimization Focus",
                ["Balanced", "Form-Heavy", "Fixture-Based", "Conservative", "Differential"],
                help="Adjust the strategy focus"
            )
        
        with col3:
            risk_tolerance = st.slider(
                "⚖️ Risk Tolerance",
                min_value=1, max_value=10, value=5,
                help="1=Safe picks, 10=High risk/reward"
            )
        
        # Analyze squad and generate recommendations
        if st.button("🚀 Optimize Starting XI", type="primary"):
            with st.spinner("Analyzing your squad and optimizing lineup..."):
                optimized_lineup = self._optimize_starting_eleven(
                    team_data, players_df, formation_pref, optimization_focus, risk_tolerance
                )
                
                if optimized_lineup:
                    self._display_optimized_lineup(optimized_lineup)
                else:
                    st.error("❌ Could not optimize lineup. Please check your squad data.")
    
    def _optimize_starting_eleven(self, team_data, players_df, formation_pref, optimization_focus, risk_tolerance):
        """Core optimization algorithm for Starting XI"""
        picks = team_data.get('picks', [])
        
        # Get squad players with enhanced stats
        squad_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0].to_dict()
                player['pick_data'] = pick
                player['optimization_score'] = self._calculate_optimization_score(
                    player, optimization_focus, risk_tolerance
                )
                squad_players.append(player)
        
        if len(squad_players) < 15:
            return None
        
        # Group by position
        positions = {
            'GK': [p for p in squad_players if p.get('position_name') == 'Goalkeeper'],
            'DEF': [p for p in squad_players if p.get('position_name') == 'Defender'],
            'MID': [p for p in squad_players if p.get('position_name') == 'Midfielder'],
            'FWD': [p for p in squad_players if p.get('position_name') == 'Forward']
        }
        
        # Sort each position by optimization score
        for pos in positions:
            positions[pos].sort(key=lambda x: x['optimization_score'], reverse=True)
        
        # Determine formation
        if formation_pref == "Auto-Select":
            formation = self._determine_optimal_formation(positions)
        else:
            formation = self._parse_formation(formation_pref)
        
        # Select starting XI
        starting_xi = {
            'GK': positions['GK'][:1] if positions['GK'] else [],
            'DEF': positions['DEF'][:formation['DEF']] if len(positions['DEF']) >= formation['DEF'] else positions['DEF'],
            'MID': positions['MID'][:formation['MID']] if len(positions['MID']) >= formation['MID'] else positions['MID'],
            'FWD': positions['FWD'][:formation['FWD']] if len(positions['FWD']) >= formation['FWD'] else positions['FWD']
        }
        
        # Calculate bench (remaining players)
        bench_players = []
        for pos in positions:
            if pos == 'GK':
                bench_players.extend(positions[pos][1:])  # Backup GK
            elif pos == 'DEF':
                bench_players.extend(positions[pos][formation['DEF']:])
            elif pos == 'MID':
                bench_players.extend(positions[pos][formation['MID']:])
            elif pos == 'FWD':
                bench_players.extend(positions[pos][formation['FWD']:])
        
        # Sort bench by optimization score
        bench_players.sort(key=lambda x: x['optimization_score'], reverse=True)
        
        # Suggest captain and vice-captain
        all_starters = []
        for pos_players in starting_xi.values():
            all_starters.extend(pos_players)
        
        captain_candidates = sorted(all_starters, key=lambda x: x['optimization_score'], reverse=True)
        
        return {
            'formation': formation,
            'starting_xi': starting_xi,
            'bench': bench_players[:4],  # First 4 bench players
            'captain': captain_candidates[0] if captain_candidates else None,
            'vice_captain': captain_candidates[1] if len(captain_candidates) > 1 else None,
            'total_predicted_points': sum([p['optimization_score'] for p in all_starters])
        }
    
    def _calculate_optimization_score(self, player, optimization_focus, risk_tolerance):
        """Calculate optimization score based on multiple factors"""
        score = 0
        
        # Helper function to safely convert to float
        def safe_float(value, default=0.0):
            try:
                if value is None or value == '':
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default
        
        # Helper function to safely convert to int
        def safe_int(value, default=0):
            try:
                if value is None or value == '':
                    return default
                return int(float(value))  # Convert via float first to handle string decimals
            except (ValueError, TypeError):
                return default
        
        # Base score from total points (normalized)
        total_points = safe_float(player.get('total_points', 0))
        score += (total_points / 300) * 30  # Max 30 points from total points
        
        # Form component (0-20 points)
        form = safe_float(player.get('form', 0))
        if optimization_focus == "Form-Heavy":
            score += (form / 10) * 25
        else:
            score += (form / 10) * 15
        
        # Points per game reliability (0-15 points)
        ppg = safe_float(player.get('points_per_game', 0))
        if ppg > 0:
            score += min(ppg * 2, 15)
        
        # Minutes played reliability (0-10 points)
        minutes = safe_int(player.get('minutes', 0))
        if minutes > 1500:  # Regular starter
            score += 10
        elif minutes > 1000:
            score += 7
        elif minutes > 500:
            score += 4
        
        # Value efficiency (0-10 points)
        ppm = safe_float(player.get('points_per_million', 0))
        if ppm > 8:
            score += 10
        elif ppm > 6:
            score += 7
        elif ppm > 4:
            score += 4
        
        # Team strength factor (0-5 points)
        # Estimate based on clean sheets, goals scored
        team_strength = 0
        clean_sheets = safe_int(player.get('clean_sheets', 0))
        goals_scored = safe_int(player.get('goals_scored', 0))
        
        if clean_sheets > 8:  # Good defensive team
            team_strength += 2
        if goals_scored > 5:  # Good attacking player
            team_strength += 3
        score += min(team_strength, 5)
        
        # Risk adjustment based on tolerance
        if risk_tolerance < 5:  # Conservative
            # Prefer established players
            if total_points > 100:
                score += 5
        else:  # Higher risk tolerance
            # Prefer differentials
            ownership = safe_float(player.get('selected_by_percent', 50))
            if ownership < 15:  # Differential
                score += (10 - risk_tolerance) * 1.5
        
        # Fixture difficulty (simplified - based on team strength)
        if optimization_focus == "Fixture-Based":
            # This is a simplified fixture analysis
            # In a full implementation, you'd have actual fixture data
            score += 5  # Placeholder for fixture analysis
        
        return max(score, 0)  # Ensure non-negative score
    
    def _determine_optimal_formation(self, positions):
        """Determine the best formation based on player quality"""
        formations = [
            {'DEF': 3, 'MID': 4, 'FWD': 3},
            {'DEF': 3, 'MID': 5, 'FWD': 2},
            {'DEF': 4, 'MID': 3, 'FWD': 3},
            {'DEF': 4, 'MID': 4, 'FWD': 2},
            {'DEF': 4, 'MID': 5, 'FWD': 1},
            {'DEF': 5, 'MID': 3, 'FWD': 2},
            {'DEF': 5, 'MID': 4, 'FWD': 1}
        ]
        
        best_formation = None
        best_score = 0
        
        for formation in formations:
            if (len(positions['DEF']) >= formation['DEF'] and 
                len(positions['MID']) >= formation['MID'] and 
                len(positions['FWD']) >= formation['FWD']):
                
                # Calculate total score for this formation
                score = 0
                score += sum([p['optimization_score'] for p in positions['DEF'][:formation['DEF']]])
                score += sum([p['optimization_score'] for p in positions['MID'][:formation['MID']]])
                score += sum([p['optimization_score'] for p in positions['FWD'][:formation['FWD']]])
                
                if score > best_score:
                    best_score = score
                    best_formation = formation
        
        return best_formation or {'DEF': 4, 'MID': 4, 'FWD': 2}  # Default fallback
    
    def _parse_formation(self, formation_str):
        """Parse formation string into position counts"""
        if formation_str == "Auto-Select":
            return {'DEF': 4, 'MID': 4, 'FWD': 2}
        
        parts = formation_str.split('-')
        if len(parts) == 3:
            return {
                'DEF': int(parts[0]),
                'MID': int(parts[1]),
                'FWD': int(parts[2])
            }
        return {'DEF': 4, 'MID': 4, 'FWD': 2}  # Default
    
    def _display_optimized_lineup(self, optimized_lineup):
        """Display the optimized lineup with detailed breakdown"""
        st.success("✅ **Optimization Complete!**")
        
        formation = optimized_lineup['formation']
        formation_str = f"{formation['DEF']}-{formation['MID']}-{formation['FWD']}"
        
        # Overall recommendations
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏗️ Optimal Formation", formation_str)
        
        with col2:
            predicted_points = optimized_lineup['total_predicted_points']
            st.metric("📊 Predicted Points", f"{predicted_points:.1f}")
        
        with col3:
            captain = optimized_lineup['captain']
            captain_name = captain['web_name'] if captain else "N/A"
            st.metric("👑 Captain", captain_name)
        
        with col4:
            vice_captain = optimized_lineup['vice_captain']
            vc_name = vice_captain['web_name'] if vice_captain else "N/A"
            st.metric("🥈 Vice Captain", vc_name)
        
        # Detailed lineup breakdown
        st.subheader("🎯 Optimized Starting XI")
        
        starting_xi = optimized_lineup['starting_xi']
        
        # Display by position
        for pos_name, display_name in [('GK', '🥅 Goalkeeper'), ('DEF', '🛡️ Defenders'), ('MID', '⚽ Midfielders'), ('FWD', '🎯 Forwards')]:
            if starting_xi[pos_name]:
                st.write(f"**{display_name}**")
                
                for player in starting_xi[pos_name]:
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 1])
                    
                    with col1:
                        captain_badge = " 👑" if player == optimized_lineup['captain'] else " 🥈" if player == optimized_lineup['vice_captain'] else ""
                        st.write(f"**{player['web_name']}**{captain_badge}")
                    
                    with col2:
                        st.write(f"{player.get('team_short_name', 'UNK')}")
                    
                    with col3:
                        st.write(f"Form: {player.get('form', 0):.1f}")
                    
                    with col4:
                        st.write(f"£{player.get('cost_millions', 0):.1f}m")
                    
                    with col5:
                        score = player.get('optimization_score', 0)
                        st.write(f"{score:.1f}")
                
                st.divider()
        
        # Bench recommendations
        st.subheader("🪑 Recommended Bench")
        
        bench = optimized_lineup['bench']
        if bench:
            for i, player in enumerate(bench, 1):
                col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
                
                with col1:
                    st.write(f"{i}.")
                
                with col2:
                    st.write(f"**{player['web_name']}** ({player.get('position_name', 'UNK')})")
                
                with col3:
                    st.write(f"{player.get('team_short_name', 'UNK')}")
                
                with col4:
                    st.write(f"Score: {player.get('optimization_score', 0):.1f}")
        
        # Strategic insights
        st.subheader("💡 Strategic Insights")
        
        insights = []
        
        # Captain analysis
        captain = optimized_lineup['captain']
        if captain:
            captain_form = captain.get('form', 0)
            if captain_form > 7:
                insights.append("🔥 Your suggested captain is in excellent form!")
            elif captain_form < 4:
                insights.append("⚠️ Consider alternative captain options - current suggestion has poor form")
        
        # Formation analysis
        if formation['DEF'] >= 5:
            insights.append("🛡️ Defensive formation - good for teams expecting clean sheets")
        elif formation['FWD'] >= 3:
            insights.append("⚔️ Attacking formation - prioritizing goal-scoring potential")
        
        # Risk analysis
        risky_players = [p for pos_players in starting_xi.values() for p in pos_players if p.get('selected_by_percent', 50) < 10]
        if len(risky_players) >= 2:
            insights.append(f"💎 {len(risky_players)} differential picks in starting XI - high risk/reward strategy")
        
        # Value analysis
        total_value = sum([p.get('cost_millions', 0) for pos_players in starting_xi.values() for p in pos_players])
        if total_value < 75:
            insights.append("💰 Budget-friendly lineup leaves room for upgrades")
        elif total_value > 85:
            insights.append("💸 Premium-heavy lineup - ensure these players deliver")
        
        # Display insights
        if insights:
            for insight in insights:
                st.info(insight)
        else:
            st.success("✅ Well-balanced team selection!")
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Copy Lineup to Clipboard"):
                lineup_text = self._generate_lineup_text(optimized_lineup)
                st.code(lineup_text, language="text")
                st.success("Lineup formatted for sharing!")
        
        with col2:
            if st.button("🔄 Re-optimize with Different Settings"):
                st.rerun()
    
    def _generate_lineup_text(self, optimized_lineup):
        """Generate text representation of the lineup"""
        formation = optimized_lineup['formation']
        starting_xi = optimized_lineup['starting_xi']
        
        lines = []
        lines.append(f"🏗️ Formation: {formation['DEF']}-{formation['MID']}-{formation['FWD']}")
        lines.append("")
        
        for pos_name, display_name in [('GK', 'GK'), ('DEF', 'DEF'), ('MID', 'MID'), ('FWD', 'FWD')]:
            if starting_xi[pos_name]:
                players = [p['web_name'] for p in starting_xi[pos_name]]
                lines.append(f"{display_name}: {', '.join(players)}")
        
        lines.append("")
        captain = optimized_lineup['captain']
        vice_captain = optimized_lineup['vice_captain']
        
        if captain:
            lines.append(f"👑 Captain: {captain['web_name']}")
        if vice_captain:
            lines.append(f"🥈 Vice Captain: {vice_captain['web_name']}")
        
        return "\n".join(lines)
    
    def _display_swot_analysis(self, team_data):
        """Display comprehensive SWOT analysis for the team"""
        st.subheader("🎯 Team SWOT Analysis")
        
        with st.expander("📚 What is SWOT Analysis?", expanded=False):
            st.markdown("""
            **SWOT Analysis** is a strategic planning technique that evaluates four key aspects:
            
            - **🔥 Strengths**: Internal positive factors that give you advantages
            - **⚠️ Weaknesses**: Internal negative factors that need improvement
            - **🚀 Opportunities**: External positive factors you can exploit
            - **⛔ Threats**: External negative factors that could harm your performance
            
            This helps you make informed transfer and tactical decisions.
            """)
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load player data to enable detailed SWOT analysis")
            return
        
        # Generate SWOT analysis
        swot_data = self._generate_swot_analysis(team_data)
        
        # Display SWOT in a 2x2 grid
        col1, col2 = st.columns(2)
        
        with col1:
            # Strengths
            st.markdown("### 🔥 **STRENGTHS**")
            st.markdown("*Internal positive factors*")
            
            if swot_data['strengths']:
                for strength in swot_data['strengths']:
                    st.success(f"✅ {strength}")
            else:
                st.info("🔍 Analyzing your team strengths...")
            
            st.markdown("---")
            
            # Weaknesses
            st.markdown("### ⚠️ **WEAKNESSES**")
            st.markdown("*Internal areas for improvement*")
            
            if swot_data['weaknesses']:
                for weakness in swot_data['weaknesses']:
                    st.error(f"❌ {weakness}")
            else:
                st.info("🔍 No major weaknesses identified")
        
        with col2:
            # Opportunities
            st.markdown("### 🚀 **OPPORTUNITIES**")
            st.markdown("*External factors to exploit*")
            
            if swot_data['opportunities']:
                for opportunity in swot_data['opportunities']:
                    st.info(f"💡 {opportunity}")
            else:
                st.info("🔍 Looking for opportunities...")
            
            st.markdown("---")
            
            # Threats
            st.markdown("### ⛔ **THREATS**")
            st.markdown("*External risks to mitigate*")
            
            if swot_data['threats']:
                for threat in swot_data['threats']:
                    st.warning(f"⚠️ {threat}")
            else:
                st.info("🔍 No immediate threats detected")
        
        # Strategic recommendations based on SWOT
        st.markdown("---")
        st.subheader("📋 Strategic Action Plan")
        
        action_plan = self._generate_swot_action_plan(swot_data)
        
        strategy_tabs = st.tabs(["🎯 SO Strategy", "🔧 WO Strategy", "🛡️ ST Strategy", "⚡ WT Strategy"])
        
        with strategy_tabs[0]:
            st.markdown("**Strength-Opportunity (SO)**: *Use strengths to capitalize on opportunities*")
            for action in action_plan.get('so_strategy', []):
                st.success(f"🎯 {action}")
        
        with strategy_tabs[1]:
            st.markdown("**Weakness-Opportunity (WO)**: *Overcome weaknesses by exploiting opportunities*")
            for action in action_plan.get('wo_strategy', []):
                st.info(f"🔧 {action}")
        
        with strategy_tabs[2]:
            st.markdown("**Strength-Threat (ST)**: *Use strengths to avoid threats*")
            for action in action_plan.get('st_strategy', []):
                st.warning(f"🛡️ {action}")
        
        with strategy_tabs[3]:
            st.markdown("**Weakness-Threat (WT)**: *Minimize weaknesses and avoid threats*")
            for action in action_plan.get('wt_strategy', []):
                st.error(f"⚡ {action}")
    
    def _generate_swot_analysis(self, team_data):
        """Generate SWOT analysis based on team data"""
        players_df = st.session_state.players_df
        picks = team_data.get('picks', [])
        
        strengths = []
        weaknesses = []
        opportunities = []
        threats = []
        
        # Get team players data
        team_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                team_players.append({
                    'player': player,
                    'pick': pick
                })
        
        if not team_players:
            return {'strengths': [], 'weaknesses': [], 'opportunities': [], 'threats': []}
        
        # Analyze strengths
        total_points = team_data.get('summary_overall_points', 0)
        overall_rank = team_data.get('summary_overall_rank', 0)
        team_value = team_data.get('value', 1000) / 10
        bank = team_data.get('bank', 0) / 10
        
        # Performance-based strengths
        if overall_rank and overall_rank < 500000:
            strengths.append("Strong overall rank - Top 500K performance")
        
        if total_points > 1500:
            strengths.append("High-scoring team with consistent point generation")
        
        # Player quality strengths
        high_form_players = [tp for tp in team_players if float(tp['player'].get('form', 0)) > 7]
        if len(high_form_players) >= 3:
            strengths.append(f"{len(high_form_players)} players in excellent form (7+ rating)")
        
        premium_players = [tp for tp in team_players if float(tp['player'].get('now_cost', 0)) > 100]
        if len(premium_players) >= 2:
            strengths.append(f"Strong premium player base ({len(premium_players)} players >£10m)")
        
        # Value efficiency
        efficient_players = [tp for tp in team_players if float(tp['player'].get('points_per_million', 0)) > 8]
        if len(efficient_players) >= 5:
            strengths.append("Excellent value efficiency across squad")
        
        # Financial strength
        if bank > 1.5:
            strengths.append(f"Strong financial position (£{bank:.1f}m in bank)")
        
        # Analyze weaknesses
        # Poor form players
        poor_form_players = [tp for tp in team_players if float(tp['player'].get('form', 0)) < 3]
        if len(poor_form_players) >= 2:
            weaknesses.append(f"{len(poor_form_players)} players in poor form (<3 rating)")
        
        # Injury concerns
        injured_players = [tp for tp in team_players if tp['player'].get('status') in ['i', 'd']]
        if injured_players:
            weaknesses.append(f"{len(injured_players)} players with injury/availability concerns")
        
        # Overpriced players
        overpriced = [tp for tp in team_players if float(tp['player'].get('points_per_million', 0)) < 4]
        if len(overpriced) >= 3:
            weaknesses.append("Several players offering poor value for money")
        
        # Bench strength
        bench_players = [tp for tp in team_players if tp['pick'].get('position', 12) > 11]
        weak_bench = [bp for bp in bench_players if float(bp['player'].get('total_points', 0)) < 20]
        if len(weak_bench) >= 2:
            weaknesses.append("Weak bench with limited playing potential")
        
        # No free transfers
        if bank < 0.5 and team_data.get('summary_event_points', 0) < 40:
            weaknesses.append("Limited transfer flexibility with low funds")
        
        # Analyze opportunities
        # Market opportunities
        if bank > 2:
            opportunities.append("Significant funds available for squad upgrades")
        
        # Form opportunities
        rising_players = [tp for tp in team_players if float(tp['player'].get('form', 0)) > 6 and float(tp['player'].get('selected_by_percent', 50)) < 15]
        if rising_players:
            opportunities.append(f"Own {len(rising_players)} differential players in good form")
        
        # Fixture opportunities (simplified)
        opportunities.append("Upcoming fixture analysis reveals potential captain options")
        opportunities.append("Transfer market offers value picks for upcoming gameweeks")
        
        # Position-specific opportunities
        starting_xi = [tp for tp in team_players if tp['pick'].get('position', 12) <= 11]
        if len(starting_xi) < 11:
            opportunities.append("Opportunity to optimize starting XI selection")
        
        # Analyze threats
        # Price change threats
        high_ownership = [tp for tp in team_players if float(tp['player'].get('selected_by_percent', 0)) > 50]
        if len(high_ownership) >= 5:
            threats.append("Heavy reliance on template players - limited differential advantage")
        
        # Form decline threats
        declining_players = [tp for tp in team_players if float(tp['player'].get('form', 0)) < 4 and float(tp['player'].get('now_cost', 0)) > 80]
        if declining_players:
            threats.append(f"{len(declining_players)} expensive players underperforming")
        
        # Fixture difficulty threats
        threats.append("Difficult fixtures ahead for key players")
        
        # Competition threats
        if overall_rank and overall_rank > 2000000:
            threats.append("Current rank requires significant improvement to reach targets")
        
        # Squad balance threats
        gk_count = len([tp for tp in team_players if tp['player'].get('position_name') == 'Goalkeeper'])
        if gk_count < 2:
            threats.append("Goalkeeper shortage - squad balance risk")
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'opportunities': opportunities,
            'threats': threats
        }
    
    def _generate_swot_action_plan(self, swot_data):
        """Generate strategic action plan based on SWOT analysis"""
        return {
            'so_strategy': [
                "Leverage strong performers to target high-value differentials",
                "Use financial flexibility to secure premium assets in form",
                "Maintain successful strategies while exploring new opportunities"
            ],
            'wo_strategy': [
                "Address poor form players through strategic transfers",
                "Improve squad depth by targeting reliable bench options",
                "Transform value weaknesses into opportunity through smart picks"
            ],
            'st_strategy': [
                "Protect strong rank by avoiding risky template picks",
                "Use premium assets as captaincy insurance",
                "Maintain squad balance to weather difficult periods"
            ],
            'wt_strategy': [
                "Minimize exposure to declining expensive assets",
                "Avoid panic transfers that could worsen position",
                "Focus on fundamental improvements rather than quick fixes"
            ]
        }
    
    def _display_advanced_analytics(self, team_data):
        """Display advanced analytics and insights"""
        st.subheader("📈 Advanced Team Analytics")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load player data to enable advanced analytics")
            return
        
        players_df = st.session_state.players_df
        picks = team_data.get('picks', [])
        
        # Team composition analysis
        analytics_tabs = st.tabs([
            "🏗️ Squad Composition", 
            "📊 Performance Metrics", 
            "💰 Value Analysis",
            "🎯 Risk Assessment",
            "📈 Trend Analysis"
        ])
        
        with analytics_tabs[0]:
            self._display_squad_composition_analysis(team_data, players_df, picks)
        
        with analytics_tabs[1]:
            self._display_performance_metrics_analysis(team_data, players_df, picks)
        
        with analytics_tabs[2]:
            self._display_value_analysis(team_data, players_df, picks)
        
        with analytics_tabs[3]:
            self._display_risk_assessment(team_data, players_df, picks)
        
        with analytics_tabs[4]:
            self._display_trend_analysis(team_data, players_df, picks)
    
    def _display_squad_composition_analysis(self, team_data, players_df, picks):
        """Analyze squad composition and balance"""
        st.subheader("🏗️ Squad Composition Analysis")
        
        # Get team players
        team_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                team_players.append({
                    'position': player.get('position_name', 'Unknown'),
                    'price': float(player.get('now_cost', 0)) / 10,
                    'ownership': float(player.get('selected_by_percent', 0)),
                    'form': float(player.get('form', 0)),
                    'points': int(player.get('total_points', 0)),
                    'team': player.get('team_short_name', 'UNK')
                })
        
        if not team_players:
            st.warning("No team data available for analysis")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Position distribution
            st.write("**Position Distribution**")
            position_counts = {}
            for player in team_players:
                pos = player['position']
                position_counts[pos] = position_counts.get(pos, 0) + 1
            
            for pos, count in position_counts.items():
                st.metric(f"{pos}s", count)
        
        with col2:
            # Price distribution
            st.write("**Price Tiers**")
            budget_players = len([p for p in team_players if p['price'] < 6])
            mid_players = len([p for p in team_players if 6 <= p['price'] < 9])
            premium_players = len([p for p in team_players if p['price'] >= 9])
            
            st.metric("Budget (<£6m)", budget_players)
            st.metric("Mid-tier (£6-9m)", mid_players) 
            st.metric("Premium (£9m+)", premium_players)
        
        # Team diversity
        st.write("**Team Diversity**")
        teams_represented = len(set([p['team'] for p in team_players]))
        st.metric("Teams Represented", f"{teams_represented}/20")
        
        if teams_represented < 10:
            st.warning("⚠️ Low team diversity - consider spreading across more teams")
        elif teams_represented > 15:
            st.success("✅ Excellent team diversity")
        else:
            st.info("👍 Good team diversity")
    
    def _display_performance_metrics_analysis(self, team_data, players_df, picks):
        """Analyze performance metrics"""
        st.subheader("📊 Performance Metrics")
        
        # Calculate key metrics
        total_points = team_data.get('summary_overall_points', 0)
        current_gw = team_data.get('gameweek', 1) or 1
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_ppg = total_points / max(current_gw, 1)
            st.metric("Points Per Gameweek", f"{avg_ppg:.1f}")
            
            # Performance rating
            if avg_ppg >= 65:
                st.success("🏆 Elite Performance")
            elif avg_ppg >= 55:
                st.info("🥇 Above Average")
            elif avg_ppg >= 45:
                st.warning("📈 Below Average")
            else:
                st.error("🆘 Needs Improvement")
        
        with col2:
            # Consistency score (simplified)
            gw_points = team_data.get('summary_event_points', 0)
            consistency = min(gw_points / max(avg_ppg, 1), 2) * 50
            st.metric("Consistency Score", f"{consistency:.0f}%")
        
        with col3:
            # Rank improvement potential
            rank = team_data.get('summary_overall_rank', 0)
            if rank:
                percentile = (1 - (rank / 8000000)) * 100
                st.metric("Percentile", f"{percentile:.1f}%")
        
        # Performance trends
        st.write("**Performance Insights**")
        
        insights = []
        if avg_ppg > 60:
            insights.append("🔥 Strong scoring rate - maintain current strategy")
        elif avg_ppg < 45:
            insights.append("📈 Scoring below average - consider tactical changes")
        
        if gw_points > avg_ppg * 1.2:
            insights.append("🚀 Recent form exceeding season average")
        elif gw_points < avg_ppg * 0.8:
            insights.append("⚠️ Recent dip in form - monitor closely")
        
        for insight in insights:
            st.info(insight)
    
    def _display_value_analysis(self, team_data, players_df, picks):
        """Analyze team value and efficiency"""
        st.subheader("💰 Value Analysis")
        
        team_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                ppm = float(player.get('points_per_million', 0))
                team_players.append({
                    'name': player.get('web_name', 'Unknown'),
                    'price': float(player.get('now_cost', 0)) / 10,
                    'points': int(player.get('total_points', 0)),
                    'ppm': ppm,
                    'efficiency_rating': 'Excellent' if ppm > 8 else 'Good' if ppm > 6 else 'Average' if ppm > 4 else 'Poor'
                })
        
        if team_players:
            col1, col2 = st.columns(2)
            
            with col1:
                # Value metrics
                total_value = sum([p['price'] for p in team_players])
                avg_ppm = sum([p['ppm'] for p in team_players]) / len(team_players)
                
                st.metric("Squad Value", f"£{total_value:.1f}m")
                st.metric("Average PPM", f"{avg_ppm:.1f}")
                
                bank = team_data.get('bank', 0) / 10
                st.metric("Available Funds", f"£{bank:.1f}m")
            
            with col2:
                # Efficiency breakdown
                excellent = len([p for p in team_players if p['ppm'] > 8])
                good = len([p for p in team_players if 6 < p['ppm'] <= 8])
                average = len([p for p in team_players if 4 < p['ppm'] <= 6])
                poor = len([p for p in team_players if p['ppm'] <= 4])
                
                st.write("**Value Efficiency**")
                st.metric("Excellent (>8 PPM)", excellent)
                st.metric("Good (6-8 PPM)", good)
                st.metric("Average (4-6 PPM)", average)
                st.metric("Poor (<4 PPM)", poor)
            
            # Value recommendations
            if poor >= 3:
                st.warning("⚠️ Multiple players offering poor value - consider transfers")
            elif excellent >= 8:
                st.success("✅ Excellent value across squad")
            else:
                st.info("👍 Reasonable value efficiency")
    
    def _display_risk_assessment(self, team_data, players_df, picks):
        """Assess team risk factors"""
        st.subheader("🎯 Risk Assessment")
        
        risk_factors = []
        risk_score = 0
        
        # Analyze risk factors
        team_players = []
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                team_players.append({
                    'ownership': float(player.get('selected_by_percent', 0)),
                    'form': float(player.get('form', 0)),
                    'price': float(player.get('now_cost', 0)) / 10,
                    'minutes': int(player.get('minutes', 0)),
                    'status': player.get('status', 'a')
                })
        
        if team_players:
            # Template risk
            high_ownership = len([p for p in team_players if p['ownership'] > 50])
            if high_ownership >= 6:
                risk_factors.append("High template exposure - limited differential advantage")
                risk_score += 2
            
            # Form risk
            poor_form = len([p for p in team_players if p['form'] < 3])
            if poor_form >= 3:
                risk_factors.append(f"{poor_form} players in poor form")
                risk_score += 1
            
            # Rotation risk
            rotation_risk = len([p for p in team_players if p['minutes'] < 1000])
            if rotation_risk >= 4:
                risk_factors.append("High rotation risk in squad")
                risk_score += 1
            
            # Injury risk
            injury_risk = len([p for p in team_players if p['status'] != 'a'])
            if injury_risk >= 2:
                risk_factors.append("Multiple players with fitness concerns")
                risk_score += 2
            
            # Financial risk
            bank = team_data.get('bank', 0) / 10
            if bank < 0.5:
                risk_factors.append("Limited financial flexibility")
                risk_score += 1
            
            # Display risk assessment
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if risk_score <= 2:
                    st.success(f"🟢 Low Risk ({risk_score}/7)")
                elif risk_score <= 4:
                    st.warning(f"🟡 Medium Risk ({risk_score}/7)")
                else:
                    st.error(f"🔴 High Risk ({risk_score}/7)")
            
            with col2:
                if risk_factors:
                    st.write("**Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.success("✅ No significant risk factors identified")
    
    def _display_trend_analysis(self, team_data, players_df, picks):
        """Analyze trends and momentum"""
        st.subheader("📈 Trend Analysis")
        
        # Performance trend
        total_points = team_data.get('summary_overall_points', 0)
        gw_points = team_data.get('summary_event_points', 0)
        current_gw = team_data.get('gameweek', 1) or 1
        
        avg_ppg = total_points / max(current_gw, 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Performance Momentum**")
            
            if gw_points > avg_ppg * 1.2:
                st.success("🚀 Strong upward momentum")
                momentum = "Positive"
            elif gw_points < avg_ppg * 0.8:
                st.warning("📉 Concerning downward trend")
                momentum = "Negative"
            else:
                st.info("➡️ Stable performance")
                momentum = "Stable"
            
            st.metric("Momentum", momentum)
        
        with col2:
            st.write("**Form Trends**")
            
            team_players = []
            for pick in picks:
                player_info = players_df[players_df['id'] == pick['element']]
                if not player_info.empty:
                    player = player_info.iloc[0]
                    team_players.append(float(player.get('form', 0)))
            
            if team_players:
                avg_form = sum(team_players) / len(team_players)
                improving_form = len([f for f in team_players if f > 6])
                
                st.metric("Average Form", f"{avg_form:.1f}")
                st.metric("Players in Form", f"{improving_form}/15")
        
        # Trend insights
        st.write("**Trend Insights**")
        
        insights = []
        
        # Overall trend analysis
        rank = team_data.get('summary_overall_rank', 0)
        if momentum == "Positive":
            insights.append("📈 Team momentum suggests continued improvement")
        elif momentum == "Negative":
            insights.append("⚠️ Declining performance requires attention")
        
        # Form trend analysis
        if team_players and sum(team_players) / len(team_players) > 5.5:
            insights.append("🔥 Strong squad form indicates good transfer choices")
        elif team_players and sum(team_players) / len(team_players) < 4:
            insights.append("📉 Poor squad form suggests need for transfers")
        
        # Rank trend estimation
        if rank and rank < 1000000:
            insights.append("🎯 Well-positioned for strong finish")
        elif rank and rank > 3000000:
            insights.append("💪 Significant improvement potential available")
        
        for insight in insights:
            st.info(insight)
    
    def _display_transfer_planning(self, team_data):
        """Display comprehensive transfer planning assistant"""
        st.subheader("🔄 Transfer Planning Assistant")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load player data to enable transfer planning")
            return
        
        try:
            from services.transfer_planning_service import TransferPlanningAssistant
            from services.fixture_service import FixtureService
            
            planner = TransferPlanningAssistant()
            players_df = st.session_state.players_df
            
            # Planning settings
            col1, col2, col3 = st.columns(3)
            
            with col1:
                planning_weeks = st.selectbox(
                    "🗓️ Planning Horizon",
                    [4, 6, 8, 10, 12],
                    index=2,
                    help="Number of gameweeks to plan ahead"
                )
            
            with col2:
                strategy_focus = st.selectbox(
                    "🎯 Strategy Focus",
                    ["Balanced", "Conservative", "Aggressive", "Value-Focused", "Rank-Climbing"],
                    help="Choose your transfer strategy approach"
                )
            
            with col3:
                hit_tolerance = st.slider(
                    "💰 Hit Tolerance",
                    min_value=0, max_value=12, value=4,
                    help="Maximum points hit you're willing to take"
                )
            
            if st.button("🚀 Generate Transfer Plan", type="primary"):
                with st.spinner("Analyzing your squad and generating transfer plan..."):
                    transfer_plan = planner.generate_transfer_plan(
                        team_data, players_df, planning_weeks
                    )
                    
                    # Display plan overview
                    st.subheader("📊 Transfer Plan Overview")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Planning Period", f"{planning_weeks} weeks")
                    
                    with col2:
                        total_transfers = sum(len(week.get('transfers', [])) for week in transfer_plan['weekly_plan'])
                        st.metric("Total Transfers", total_transfers)
                    
                    with col3:
                        st.metric("Expected Cost", f"-{transfer_plan['total_cost']}pts")
                    
                    with col4:
                        st.metric("Expected Gain", f"+{transfer_plan['expected_gain']:.1f}pts")
                    
                    # Weekly breakdown
                    st.subheader("📅 Week-by-Week Plan")
                    
                    for week in transfer_plan['weekly_plan']:
                        with st.expander(f"Gameweek {week['gameweek']} - {len(week['transfers'])} transfers"):
                            if week['transfers']:
                                for transfer in week['transfers']:
                                    col1, col2, col3 = st.columns([2, 1, 2])
                                    
                                    with col1:
                                        st.write(f"**OUT:** {transfer['out']}")
                                    
                                    with col2:
                                        price_change = transfer['price_change']
                                        color = "red" if price_change < 0 else "green"
                                        st.markdown(f"<span style='color:{color}'>£{price_change:+.1f}m</span>", unsafe_allow_html=True)
                                    
                                    with col3:
                                        st.write(f"**IN:** {transfer['in']}")
                                    
                                    st.write(f"*Reason: {transfer['reason']}*")
                                    st.divider()
                                
                                if week['cost'] > 0:
                                    st.warning(f"⚠️ This week requires a {week['cost']}pt hit")
                                
                                if week['rationale']:
                                    st.info(f"💡 {'; '.join(week['rationale'])}")
                            else:
                                st.info("No transfers recommended this week")
                    
                    # Chip recommendations
                    st.subheader("🎴 Chip Usage Strategy")
                    
                    chip_recs = transfer_plan['chip_recommendations']
                    
                    for chip, rec in chip_recs.items():
                        chip_name = chip.replace('_', ' ').title()
                        
                        col1, col2, col3 = st.columns([2, 1, 2])
                        
                        with col1:
                            st.write(f"**{chip_name}**")
                        
                        with col2:
                            st.metric("Recommended GW", rec['recommended_gw'])
                        
                        with col3:
                            score = rec['suitability_score']
                            if score > 70:
                                st.success(f"Excellent timing ({score:.0f}%)")
                            elif score > 50:
                                st.info(f"Good timing ({score:.0f}%)")
                            else:
                                st.warning(f"Consider waiting ({score:.0f}%)")
                        
                        st.write(f"*{rec['rationale']}*")
                        st.divider()
                    
                    # Transfer priorities
                    st.subheader("🎯 Transfer Priorities")
                    
                    priorities = transfer_plan['transfer_priorities']
                    
                    for i, priority in enumerate(priorities[:5], 1):
                        player = priority['player']
                        
                        priority_color = {
                            'high': 'red',
                            'medium': 'orange', 
                            'low': 'gray'
                        }.get(priority['priority'], 'gray')
                        
                        st.markdown(f"**{i}. {player['name']}** "
                                   f"<span style='color:{priority_color}'>[{priority['priority'].upper()} PRIORITY]</span>",
                                   unsafe_allow_html=True)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"*{priority['reason']}*")
                            st.write(f"Current: £{player['price']:.1f}m, Form: {player['form']:.1f}, PPM: {player['ppm']:.1f}")
                        
                        with col2:
                            replacements = priority['potential_replacements']
                            if replacements:
                                best_replacement = replacements[0]
                                st.write(f"**Best Alternative:**")
                                st.write(f"{best_replacement['name']} (£{best_replacement['price']:.1f}m)")
                                st.write(f"Score: {best_replacement['score']:.1f}")
                        
                        st.divider()
            
            # Transfer success calculator
            st.subheader("🎲 Transfer Success Calculator")
            
            st.write("Estimate the probability of transfer success:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Player Out**")
                out_form = st.number_input("Form Rating", 0.0, 10.0, 4.0, key="out_form")
                out_ppm = st.number_input("Points per Million", 0.0, 15.0, 5.0, key="out_ppm")
                out_minutes = st.number_input("Minutes Played", 0, 3500, 1500, key="out_minutes")
            
            with col2:
                st.write("**Player In**")
                in_form = st.number_input("Form Rating", 0.0, 10.0, 6.0, key="in_form")
                in_ppm = st.number_input("Points per Million", 0.0, 15.0, 7.0, key="in_ppm")
                in_minutes = st.number_input("Minutes Played", 0, 3500, 2000, key="in_minutes")
            
            if st.button("Calculate Success Probability"):
                player_out = {'form': out_form, 'ppm': out_ppm, 'minutes': out_minutes}
                player_in = {'form': in_form, 'ppm': in_ppm, 'minutes': in_minutes}
                
                success_prob = planner.get_transfer_success_probability(player_out, player_in)
                
                if success_prob > 80:
                    st.success(f"🎯 High success probability: {success_prob:.1f}%")
                elif success_prob > 60:
                    st.info(f"👍 Good success probability: {success_prob:.1f}%")
                elif success_prob > 40:
                    st.warning(f"⚠️ Moderate success probability: {success_prob:.1f}%")
                else:
                    st.error(f"❌ Low success probability: {success_prob:.1f}%")
        
        except ImportError:
            st.warning("🚧 Transfer planning service not available. Enable advanced features to access this functionality.")
            
            # Basic transfer advice
            st.info("💡 **Basic Transfer Tips:**")
            st.write("• Use your free transfer each gameweek")
            st.write("• Avoid taking hits unless for urgent transfers")
            st.write("• Plan transfers around fixture swings") 
            st.write("• Monitor player price changes")
            st.write("• Consider form and injury status")
    
    def _display_performance_comparison(self, team_data):
        """Display performance comparison against benchmarks"""
        st.subheader("📊 Performance Comparison")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load player data to enable performance comparison")
            return
        
        try:
            from services.performance_comparison_service import PerformanceComparisonService
            
            comparison_service = PerformanceComparisonService()
            players_df = st.session_state.players_df
            
            with st.spinner("Analyzing your performance against benchmarks..."):
                comparison_data = comparison_service.generate_performance_comparison(
                    team_data, players_df
                )
            
            # Current performance overview
            st.subheader("🎯 Your Current Performance")
            
            metrics = comparison_data['current_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Points", f"{metrics['total_points']:,}")
            
            with col2:
                st.metric("Points per GW", f"{metrics['avg_ppg']:.1f}")
            
            with col3:
                st.metric("Percentile", f"{metrics['percentile']:.1f}%")
            
            with col4:
                st.metric("Current GW", metrics['current_gw'])
            
            # Benchmark comparisons
            comparison_tabs = st.tabs(["🏆 vs Top 10K", "📊 vs Average", "📈 Historical", "🔍 Squad Analysis"])
            
            with comparison_tabs[0]:
                st.subheader("🏆 Comparison vs Top 10K Managers")
                
                top_10k = comparison_data['top_10k_comparison']
                
                for metric_name, metric_data in top_10k['metrics'].items():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{metric_name.replace('_', ' ').title()}**")
                    
                    with col2:
                        st.metric("Your Score", f"{metric_data['your_score']:.1f}")
                    
                    with col3:
                        diff = metric_data['difference']
                        benchmark_val = metric_data['benchmark']
                        
                        if metric_data['performance'] == 'above':
                            st.success(f"+{diff:.1f} vs {benchmark_val:.1f}")
                        else:
                            st.error(f"{diff:.1f} vs {benchmark_val:.1f}")
                
                # Overall rating
                overall_rating = top_10k['overall_rating']
                percentage = overall_rating['percentage']
                
                if percentage >= 75:
                    st.success(f"🏆 Excellent! You're performing like a top 10K manager ({percentage:.0f}% of metrics above benchmark)")
                elif percentage >= 50:
                    st.info(f"👍 Good performance, approaching top 10K levels ({percentage:.0f}% of metrics above benchmark)")
                else:
                    st.warning(f"📈 Room for improvement to reach top 10K standards ({percentage:.0f}% of metrics above benchmark)")
            
            with comparison_tabs[1]:
                st.subheader("📊 Comparison vs Overall Average")
                
                overall = comparison_data['overall_comparison']
                
                for metric_name, metric_data in overall['metrics'].items():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{metric_name.replace('_', ' ').title()}**")
                    
                    with col2:
                        st.metric("Your Score", f"{metric_data['your_score']:.1f}")
                    
                    with col3:
                        diff = metric_data['difference']
                        benchmark_val = metric_data['benchmark']
                        
                        if metric_data['performance'] == 'above':
                            st.success(f"+{diff:.1f} vs {benchmark_val:.1f}")
                        else:
                            st.error(f"{diff:.1f} vs {benchmark_val:.1f}")
                
                # Overall rating vs average
                overall_rating = overall['overall_rating']
                percentage = overall_rating['percentage']
                
                if percentage >= 80:
                    st.success(f"🎉 Outstanding! Well above average performance ({percentage:.0f}% of metrics above benchmark)")
                elif percentage >= 60:
                    st.info(f"✅ Above average performance ({percentage:.0f}% of metrics above benchmark)")
                else:
                    st.warning(f"📈 Below average - focus on improvement ({percentage:.0f}% of metrics above benchmark)")
            
            with comparison_tabs[2]:
                st.subheader("📈 Historical Performance Analysis")
                
                historical = comparison_data['historical_analysis']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Current Season Phase:** {historical['current_phase']}")
                    st.metric("Your Avg PPG", f"{historical['your_avg_ppg']:.1f}")
                    st.metric("Historical Avg", f"{historical['historical_avg']:.1f}")
                
                with col2:
                    diff = historical['difference']
                    if historical['performance'] == 'above':
                        st.success(f"📈 +{diff:.1f} points above historical average")
                    else:
                        st.warning(f"📉 {diff:.1f} points below historical average")
                
                st.info(f"💡 **Phase Analysis:** {historical['phase_analysis']}")
                
                # Trend analysis
                trends = comparison_data['trend_analysis']
                
                st.write("**Performance Trends**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    momentum = trends['momentum_description']
                    st.write(momentum)
                
                with col2:
                    st.metric("Projected Total", f"{trends['projected_total']:.0f} pts")
                
                with col3:
                    st.write(trends['trajectory_description'])
            
            with comparison_tabs[3]:
                st.subheader("🔍 Squad Composition Analysis")
                
                squad_comp = comparison_data['squad_comparison']
                
                if 'error' not in squad_comp:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Squad Metrics**")
                        st.metric("Average Price", f"£{squad_comp['avg_price']:.1f}m")
                        st.metric("Average Ownership", f"{squad_comp['avg_ownership']:.1f}%")
                        st.metric("Average Form", f"{squad_comp['avg_form']:.1f}")
                    
                    with col2:
                        st.write("**Squad Composition**")
                        st.metric("Premium Players", squad_comp['premium_count'])
                        st.metric("Template Players", squad_comp['template_count'])
                        st.metric("Differentials", squad_comp['differential_count'])
                    
                    # Squad style
                    style = squad_comp['squad_style']
                    style_descriptions = {
                        'Template Heavy': "🔄 Following popular picks - safe but limited upside",
                        'Differential Heavy': "💎 High risk/reward - potential for big rank swings",
                        'Premium Heavy': "💰 Expensive squad - needs strong returns",
                        'Balanced': "⚖️ Well-balanced approach - good risk management",
                        'Standard': "📊 Standard composition - room for optimization"
                    }
                    
                    st.info(f"**Squad Style: {style}**")
                    st.write(style_descriptions.get(style, "Standard squad composition"))
                else:
                    st.warning("Squad composition analysis not available")
            
            # Recommendations
            st.subheader("💡 Performance Recommendations")
            
            recommendations = comparison_data['recommendations']
            
            if recommendations:
                for rec in recommendations:
                    st.info(rec)
            else:
                st.success("🎉 No immediate performance concerns identified!")
        
        except ImportError:
            st.warning("🚧 Performance comparison service not available.")
            
            # Basic performance insights
            total_points = team_data.get('summary_overall_points', 0)
            current_gw = team_data.get('gameweek', 1) or 1
            avg_ppg = total_points / max(current_gw, 1)
            
            st.info(f"💡 **Quick Analysis:** You're averaging {avg_ppg:.1f} points per gameweek")
            
            if avg_ppg >= 60:
                st.success("🏆 Excellent performance!")
            elif avg_ppg >= 50:
                st.info("👍 Above average performance")
            else:
                st.warning("📈 Room for improvement")
    
    def _display_fixture_analysis(self, team_data):
        """Display fixture analysis for team players"""
        st.subheader("⚽ Fixture Analysis")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load player data to enable fixture analysis")
            return
        
        try:
            from services.fixture_service import FixtureService
            
            fixture_service = FixtureService()
            players_df = st.session_state.players_df
            picks = team_data.get('picks', [])
            
            # Analysis settings
            col1, col2 = st.columns(2)
            
            with col1:
                analysis_weeks = st.selectbox(
                    "📅 Analysis Period",
                    [3, 5, 8, 10],
                    index=1,
                    help="Number of upcoming gameweeks to analyze"
                )
            
            with col2:
                focus_position = st.selectbox(
                    "🎯 Position Focus",
                    ["All Positions", "Goalkeeper", "Defender", "Midfielder", "Forward"],
                    help="Focus analysis on specific position"
                )
            
            if st.button("🔍 Analyze Fixtures", type="primary"):
                with st.spinner("Analyzing fixture difficulty for your squad..."):
                    
                    # Get team players with fixture analysis
                    fixture_analysis = []
                    
                    for pick in picks:
                        player_info = players_df[players_df['id'] == pick['element']]
                        if not player_info.empty:
                            player = player_info.iloc[0]
                            team_short = player.get('team_short_name', 'UNK')
                            position = player.get('position_name', 'Unknown')
                            
                            # Filter by position if specified
                            if focus_position != "All Positions" and position != focus_position:
                                continue
                            
                            # Get fixture difficulty for this team
                            fixtures = fixture_service.get_upcoming_fixtures_difficulty(
                                team_short, analysis_weeks
                            )
                            
                            fixture_analysis.append({
                                'player': player,
                                'pick': pick,
                                'fixtures': fixtures,
                                'is_starting': pick.get('position', 12) <= 11
                            })
                    
                    if fixture_analysis:
                        # Overall fixture summary
                        st.subheader("📊 Fixture Difficulty Summary")
                        
                        all_difficulties = []
                        starting_difficulties = []
                        
                        for analysis in fixture_analysis:
                            avg_diff = analysis['fixtures']['average_difficulty']
                            all_difficulties.append(avg_diff)
                            if analysis['is_starting']:
                                starting_difficulties.append(avg_diff)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            squad_avg = sum(all_difficulties) / len(all_difficulties)
                            st.metric("Squad Average", f"{squad_avg:.1f}/5")
                            
                            if squad_avg <= 2.5:
                                st.success("🟢 Favorable fixtures")
                            elif squad_avg <= 3.5:
                                st.info("🟡 Mixed fixtures")
                            else:
                                st.warning("🔴 Difficult fixtures")
                        
                        with col2:
                            if starting_difficulties:
                                starting_avg = sum(starting_difficulties) / len(starting_difficulties)
                                st.metric("Starting XI Average", f"{starting_avg:.1f}/5")
                        
                        with col3:
                            easy_fixtures = len([d for d in all_difficulties if d <= 2])
                            st.metric("Easy Fixtures", f"{easy_fixtures} players")
                        
                        # Player-by-player breakdown
                        st.subheader("👥 Player Fixture Breakdown")
                        
                        # Sort by fixture difficulty (easiest first)
                        fixture_analysis.sort(key=lambda x: x['fixtures']['average_difficulty'])
                        
                        for analysis in fixture_analysis:
                            player = analysis['player']
                            fixtures = analysis['fixtures']
                            is_starting = analysis['is_starting']
                            
                            with st.expander(f"{'⭐' if is_starting else '🪑'} {player.get('web_name', 'Unknown')} "
                                           f"({player.get('team_short_name', 'UNK')}) - "
                                           f"{fixtures['rating']} fixtures"):
                                
                                col1, col2 = st.columns([2, 3])
                                
                                with col1:
                                    st.write(f"**Position:** {player.get('position_name', 'Unknown')}")
                                    st.write(f"**Form:** {player.get('form', 0):.1f}")
                                    st.write(f"**Price:** £{player.get('now_cost', 0)/10:.1f}m")
                                    st.metric("Avg Difficulty", f"{fixtures['average_difficulty']:.1f}/5")
                                
                                with col2:
                                    st.write("**Upcoming Fixtures:**")
                                    
                                    for fixture in fixtures['fixtures']:
                                        difficulty = fixture['difficulty']
                                        opponent = fixture['opponent']
                                        home = "vs" if fixture['home'] else "@"
                                        
                                        # Color code difficulty
                                        if difficulty <= 2:
                                            color = "green"
                                        elif difficulty <= 3:
                                            color = "orange"
                                        else:
                                            color = "red"
                                        
                                        st.markdown(f"GW{fixture['gameweek']}: {home} {opponent} "
                                                   f"<span style='color:{color}'>({fixture['difficulty_text']})</span>",
                                                   unsafe_allow_html=True)
                        
                        # Captain recommendations
                        st.subheader("👑 Captain Recommendations")
                        
                        # Find players with easiest upcoming fixtures
                        captain_candidates = [
                            analysis for analysis in fixture_analysis 
                            if analysis['is_starting'] and analysis['fixtures']['average_difficulty'] <= 3
                        ]
                        
                        captain_candidates.sort(key=lambda x: (
                            x['fixtures']['average_difficulty'],
                            -float(x['player'].get('form', 0))
                        ))
                        
                        if captain_candidates:
                            for i, candidate in enumerate(captain_candidates[:3], 1):
                                player = candidate['player']
                                fixtures = candidate['fixtures']
                                
                                col1, col2, col3 = st.columns([2, 1, 2])
                                
                                with col1:
                                    st.write(f"**{i}. {player.get('web_name')}**")
                                    st.write(f"{player.get('team_short_name')} - {player.get('position_name')}")
                                
                                with col2:
                                    st.metric("Fixture Rating", fixtures['rating'])
                                    st.write(f"Form: {player.get('form', 0):.1f}")
                                
                                with col3:
                                    reasoning = []
                                    if fixtures['average_difficulty'] <= 2:
                                        reasoning.append("🟢 Very favorable fixtures")
                                    if float(player.get('form', 0)) > 6:
                                        reasoning.append("🔥 Excellent form")
                                    if float(player.get('selected_by_percent', 0)) > 30:
                                        reasoning.append("👥 Popular choice")
                                    
                                    for reason in reasoning:
                                        st.write(reason)
                                
                                st.divider()
                        else:
                            st.info("No standout captain options based on fixtures")
                        
                        # Transfer recommendations based on fixtures
                        st.subheader("🔄 Fixture-Based Transfer Suggestions")
                        
                        # Find players with very difficult fixtures
                        difficult_fixtures = [
                            analysis for analysis in fixture_analysis
                            if analysis['fixtures']['average_difficulty'] >= 4
                        ]
                        
                        if difficult_fixtures:
                            st.warning("⚠️ **Players with Difficult Fixtures:**")
                            
                            for analysis in difficult_fixtures:
                                player = analysis['player']
                                fixtures = analysis['fixtures']
                                is_starting = analysis['is_starting']
                                
                                priority = "HIGH" if is_starting else "MEDIUM"
                                
                                st.write(f"• **{player.get('web_name')}** ({player.get('team_short_name')}) "
                                        f"- {fixtures['rating']} fixtures [{priority} priority]")
                        else:
                            st.success("✅ No players with particularly difficult fixture runs")
                        
                        # Best fixture opportunities not in squad
                        st.subheader("💡 Market Opportunities")
                        
                        # Sample fixture opportunities (would be calculated from full player database)
                        st.info("🔍 **Teams with Excellent Fixtures:**")
                        st.write("• Brighton - 3 consecutive home games")
                        st.write("• Brentford - Favorable fixture swing")
                        st.write("• Sheffield United - Easy upcoming opponents")
                        st.write("*Consider players from these teams for transfers*")
                    
                    else:
                        st.warning("No players found for fixture analysis")
            
            # Quick fixture comparison tool
            st.subheader("⚖️ Team Fixture Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                team1 = st.selectbox(
                    "Team 1",
                    ["ARS", "MCI", "LIV", "CHE", "MUN", "TOT", "NEW", "BRI", "AVL", "WHU"],
                    help="Select first team to compare"
                )
            
            with col2:
                team2 = st.selectbox(
                    "Team 2", 
                    ["ARS", "MCI", "LIV", "CHE", "MUN", "TOT", "NEW", "BRI", "AVL", "WHU"],
                    index=1,
                    help="Select second team to compare"
                )
            
            if st.button("Compare Fixtures"):
                comparison = fixture_service.compare_fixture_run(team1, team2, 5)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**{team1} Fixtures**")
                    team1_data = comparison['team1']['data']
                    st.metric("Average Difficulty", f"{team1_data['average_difficulty']:.1f}/5")
                    st.write(f"Rating: {team1_data['rating']}")
                
                with col2:
                    st.write(f"**{team2} Fixtures**")
                    team2_data = comparison['team2']['data']
                    st.metric("Average Difficulty", f"{team2_data['average_difficulty']:.1f}/5")
                    st.write(f"Rating: {team2_data['rating']}")
                
                recommended_team = comparison['recommendation']
                st.info(f"💡 **Recommendation:** {recommended_team} has better upcoming fixtures")
        
        except ImportError:
            st.warning("🚧 Fixture analysis service not available.")
            
            # Basic fixture advice
            st.info("💡 **Basic Fixture Tips:**")
            st.write("• Monitor upcoming fixture difficulty")
            st.write("• Plan transfers around fixture swings")
            st.write("• Consider double gameweeks for chips")
            st.write("• Avoid players with blank gameweeks")
            st.write("• Captain players with favorable fixtures")

