"""
My Team Page - Handles FPL team import and analysis functionality
"""
import streamlit as st
import pandas as pd
import numpy as np
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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "👥 Current Squad", 
            "📊 Performance Analysis", 
            "💡 Recommendations",
            "⭐ Starting XI Optimizer",
            "🎯 SWOT Analysis",
            "📈 Advanced Analytics",
            "🔄 Transfer Planning",
            "📊 Performance Comparison"
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
        """Display comprehensive fixture analysis for the team"""
        st.subheader("⚽ Fixture Analysis")
        
        if not st.session_state.get('data_loaded', False):
            st.warning("Load player data to enable detailed fixture analysis")
            return
        
        # Import fixture service
        from services.fixture_service import FixtureService
        fixture_service = FixtureService()
        
        players_df = st.session_state.players_df
        picks = team_data.get('picks', [])
        
        if not picks:
            st.warning("No team data available for fixture analysis")
            return
        
        # Fixture analysis tabs
        fixture_tabs = st.tabs([
            "🎯 Overall Difficulty", 
            "⚔️ Attack vs Defense", 
            "👑 Captain Analysis",
            "🔄 Transfer Targets",
            "📊 Team Comparison"
        ])
        
        with fixture_tabs[0]:
            self._display_overall_fixture_difficulty(team_data, players_df, fixture_service)
        
        with fixture_tabs[1]:
            self._display_attack_defense_analysis(team_data, players_df, fixture_service)
        
        with fixture_tabs[2]:
            self._display_captain_fixture_analysis(team_data, players_df, fixture_service)
        
        with fixture_tabs[3]:
            self._display_fixture_transfer_targets(team_data, players_df, fixture_service)
        
        with fixture_tabs[4]:
            self._display_team_fixture_comparison(team_data, players_df, fixture_service)
    
    def _display_overall_fixture_difficulty(self, team_data, players_df, fixture_service):
        """Display overall fixture difficulty for next 5 games"""
        st.subheader("🎯 Overall Fixture Difficulty (Next 5 Games)")
        
        picks = team_data.get('picks', [])
        
        # Get team fixture difficulties
        team_fixtures = {}
        
        # Get unique teams in squad
        squad_teams = set()
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                team_name = player_info.iloc[0].get('team_short_name', 'UNK')
                squad_teams.add(team_name)
        
        # Calculate fixture difficulty for each team
        for team in squad_teams:
            fixtures = fixture_service.get_upcoming_fixtures_difficulty(team, 5)
            team_fixtures[team] = fixtures
        
        # Display overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate average difficulty across all squad teams
        all_difficulties = []
        for team_data_fix in team_fixtures.values():
            all_difficulties.extend([f['difficulty'] for f in team_data_fix['fixtures']])
        
        avg_difficulty = np.mean(all_difficulties) if all_difficulties else 3
        
        with col1:
            difficulty_color = "🟢" if avg_difficulty <= 2.5 else "🟡" if avg_difficulty <= 3.5 else "🔴"
            st.metric("Squad Avg Difficulty", f"{difficulty_color} {avg_difficulty:.1f}")
        
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
                    
                    # Get players from this team in squad
                    team_players_in_squad = []
                    for pick in picks:
                        player_info = players_df[players_df['id'] == pick['element']]
                        if not player_info.empty and player_info.iloc[0].get('team_short_name') == team_name:
                            team_players_in_squad.append(player_info.iloc[0].get('web_name', 'Unknown'))
                    
                    if team_players_in_squad:
                        st.write("**Your Players:**")
                        for player in team_players_in_squad:
                            st.write(f"• {player}")
        
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
    
    def _display_attack_defense_analysis(self, team_data, players_df, fixture_service):
        """Display attacking and defensive fixture analysis"""
        st.subheader("⚔️ Attacking vs Defensive Fixture Analysis")
        
        with st.expander("📚 Understanding Attack vs Defense Analysis", expanded=False):
            st.markdown("""
            **Attacking Fixtures**: How easy it is for your players to score/assist
            - Consider opponent's **defensive strength**
            - Target players facing weak defenses
            
            **Defensive Fixtures**: How likely your defenders/GKs are to get clean sheets
            - Consider opponent's **attacking strength** 
            - Target defenders facing weak attacks
            """)
        
        picks = team_data.get('picks', [])
        
        # Separate attacking and defensive players
        attacking_players = []
        defensive_players = []
        
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                position = player.get('position_name', 'Unknown')
                team_name = player.get('team_short_name', 'UNK')
                
                player_data = {
                    'name': player.get('web_name', 'Unknown'),
                    'team': team_name,
                    'position': position
                }
                
                if position in ['Midfielder', 'Forward']:
                    attacking_players.append(player_data)
                elif position in ['Goalkeeper', 'Defender']:
                    defensive_players.append(player_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚔️ Attacking Fixture Analysis")
            st.write("*How easy is it for your attacking players to score?*")
            
            if attacking_players:
                # Group by team and analyze
                attacking_teams = {}
                for player in attacking_players:
                    team = player['team']
                    if team not in attacking_teams:
                        attacking_teams[team] = []
                    attacking_teams[team].append(player)
                
                for team_name, players in attacking_teams.items():
                    # Get team's attacking fixture difficulty
                    team_strength = fixture_service.get_team_attack_defense_strength(team_name)
                    fixtures = fixture_service.get_upcoming_fixtures_difficulty(team_name, 5)
                    
                    # Calculate attacking fixture score (lower opponent defense = easier)
                    attacking_score = 5 - fixtures['average_difficulty']  # Invert for attacking
                    
                    score_color = "🟢" if attacking_score >= 3.5 else "🟡" if attacking_score >= 2.5 else "🔴"
                    
                    with st.expander(f"{score_color} {team_name} - Attacking Score: {attacking_score:.1f}"):
                        st.write("**Your Players:**")
                        for player in players:
                            st.write(f"• {player['name']} ({player['position']})")
                        
                        st.write("**Next 5 Fixtures (Attacking Perspective):**")
                        for fixture in fixtures['fixtures']:
                            # For attacking, we want weak defenses (easier to score against)
                            opponent_defense = fixture_service.get_team_attack_defense_strength(fixture['opponent'])['defense']
                            attacking_difficulty = min(5, max(1, opponent_defense / 20))  # Scale to 1-5
                            
                            home_away = "🏠 vs" if fixture['home'] else "✈️ @"
                            diff_emoji = "🟢" if attacking_difficulty <= 2 else "🟡" if attacking_difficulty == 3 else "🔴"
                            st.write(f"GW{fixture['gameweek']}: {home_away} {fixture['opponent']} {diff_emoji}")
            else:
                st.info("No attacking players in your squad")
        
        with col2:
            st.subheader("🛡️ Defensive Fixture Analysis")
            st.write("*How likely are your defenders to get clean sheets?*")
            
            if defensive_players:
                # Group by team and analyze
                defensive_teams = {}
                for player in defensive_players:
                    team = player['team']
                    if team not in defensive_teams:
                        defensive_teams[team] = []
                    defensive_teams[team].append(player)
                
                for team_name, players in defensive_teams.items():
                    # Get team's defensive fixture difficulty
                    fixtures = fixture_service.get_upcoming_fixtures_difficulty(team_name, 5)
                    
                    # Calculate defensive fixture score (lower opponent attack = easier clean sheets)
                    defensive_score = 5 - fixtures['average_difficulty']  # Invert for defensive
                    
                    score_color = "🟢" if defensive_score >= 3.5 else "🟡" if defensive_score >= 2.5 else "🔴"
                    
                    with st.expander(f"{score_color} {team_name} - Defensive Score: {defensive_score:.1f}"):
                        st.write("**Your Players:**")
                        for player in players:
                            st.write(f"• {player['name']} ({player['position']})")
                        
                        st.write("**Next 5 Fixtures (Defensive Perspective):**")
                        for fixture in fixtures['fixtures']:
                            # For defending, we want weak attacks (easier to keep clean sheets)
                            opponent_attack = fixture_service.get_team_attack_defense_strength(fixture['opponent'])['attack']
                            defensive_difficulty = min(5, max(1, opponent_attack / 20))  # Scale to 1-5
                            
                            home_away = "🏠 vs" if fixture['home'] else "✈️ @"
                            diff_emoji = "🟢" if defensive_difficulty <= 2 else "🟡" if defensive_difficulty == 3 else "🔴"
                            st.write(f"GW{fixture['gameweek']}: {home_away} {fixture['opponent']} {diff_emoji}")
            else:
                st.info("No defensive players in your squad")
        
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
    
    def _display_captain_fixture_analysis(self, team_data, players_df, fixture_service):
        """Analyze fixtures for captaincy decisions"""
        st.subheader("👑 Captain Fixture Analysis")
        
        picks = team_data.get('picks', [])
        
        # Get potential captains (non-GKs with good stats)
        captain_candidates = []
        
        for pick in picks:
            player_info = players_df[players_df['id'] == pick['element']]
            if not player_info.empty:
                player = player_info.iloc[0]
                position = player.get('position_name', 'Unknown')
                
                # Skip goalkeepers for captaincy
                if position != 'Goalkeeper':
                    team_name = player.get('team_short_name', 'UNK')
                    
                    # Get fixture difficulty for this player's team
                    fixtures = fixture_service.get_upcoming_fixtures_difficulty(team_name, 1)  # Next fixture only
                    
                    captain_candidates.append({
                        'name': player.get('web_name', 'Unknown'),
                        'team': team_name,
                        'position': position,
                        'form': float(player.get('form', 0)),
                        'total_points': int(player.get('total_points', 0)),
                        'ownership': float(player.get('selected_by_percent', 0)),
                        'fixture_difficulty': fixtures['average_difficulty'],
                        'next_opponent': fixtures['fixtures'][0]['opponent'] if fixtures['fixtures'] else 'TBD',
                        'is_home': fixtures['fixtures'][0]['home'] if fixtures['fixtures'] else True
                    })
        
        if captain_candidates:
            # Sort by combined score (form + fixture ease)
            for candidate in captain_candidates:
                # Calculate captain score (higher = better captain option)
                fixture_score = 6 - candidate['fixture_difficulty']  # Invert difficulty (5=best, 1=worst)
                form_score = candidate['form']
                points_score = candidate['total_points'] / 100  # Normalize
                
                candidate['captain_score'] = (fixture_score * 0.4 + form_score * 0.4 + points_score * 0.2)
            
            captain_candidates.sort(key=lambda x: x['captain_score'], reverse=True)
            
            st.write("**📊 Captain Options Ranked by Fixture + Form:**")
            
            for i, candidate in enumerate(captain_candidates[:8], 1):
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
    
    def _display_fixture_transfer_targets(self, team_data, players_df, fixture_service):
        """Identify transfer targets based on fixtures"""
        st.subheader("🔄 Fixture-Based Transfer Targets")
        
        # Get all teams with good upcoming fixtures
        all_teams = players_df['team_short_name'].unique() if 'team_short_name' in players_df.columns else []
        
        team_fixture_scores = []
        
        for team in all_teams:
            if pd.notna(team):
                fixtures = fixture_service.get_upcoming_fixtures_difficulty(team, 5)
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
            
            for team_data_fix in best_teams:
                team_name = team_data_fix['team']
                
                # Get best players from this team
                team_players = players_df[players_df['team_short_name'] == team_name]
                if not team_players.empty:
                    # Get top players by points/form
                    top_players = team_players.nlargest(5, 'total_points')
                    
                    with st.expander(f"🟢 {team_name} - {team_data_fix['rating']} Fixtures ({team_data_fix['avg_difficulty']:.1f})"):
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
            
            for team_data_fix in worst_teams:
                team_name = team_data_fix['team']
                
                # Get popular players from this team (likely in many squads)
                team_players = players_df[players_df['team_short_name'] == team_name]
                if not team_players.empty:
                    # Get most owned players
                    popular_players = team_players.nlargest(3, 'selected_by_percent')
                    
                    with st.expander(f"🔴 {team_name} - {team_data_fix['rating']} Fixtures ({team_data_fix['avg_difficulty']:.1f})"):
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
            
            for team_data_fix in team_fixture_scores[:12]:  # Top 12 teams by fixtures
                if team_data_fix['fixture_score'] >= 3:  # Only good fixtures
                    team_name = team_data_fix['team']
                    team_players = players_df[players_df['team_short_name'] == team_name]
                    
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
                                'fixture_score': team_data_fix['fixture_score']
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
    
    def _display_team_fixture_comparison(self, team_data, players_df, fixture_service):
        """Compare fixture difficulty between teams"""
        st.subheader("📊 Team Fixture Comparison")
        
        # Team selection for comparison
        all_teams = sorted(players_df['team_short_name'].unique()) if 'team_short_name' in players_df.columns else []
        
        col1, col2 = st.columns(2)
        
        with col1:
            team1 = st.selectbox("Select First Team", all_teams, index=0 if all_teams else None)
        
        with col2:
            team2 = st.selectbox("Select Second Team", all_teams, index=1 if len(all_teams) > 1 else 0)
        
        if team1 and team2 and team1 != team2:
            # Compare fixtures between teams
            comparison = fixture_service.compare_fixture_run(team1, team2, 5)
            
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
                team1_players = players_df[players_df['team_short_name'] == team1].nlargest(5, 'total_points')
                for _, player in team1_players.iterrows():
                    st.write(f"• {player.get('web_name', 'Unknown')} ({player.get('position_name', 'Unknown')}) - £{float(player.get('now_cost', 0))/10:.1f}m")
            
            with col2:
                st.write(f"**{team2} Players:**")
                team2_players = players_df[players_df['team_short_name'] == team2].nlargest(5, 'total_points')
                for _, player in team2_players.iterrows():
                    st.write(f"• {player.get('web_name', 'Unknown')} ({player.get('position_name', 'Unknown')}) - £{float(player.get('now_cost', 0))/10:.1f}m")
        
        else:
            st.info("Please select two different teams to compare their fixtures.")

