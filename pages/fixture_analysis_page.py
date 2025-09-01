"""
Fixture Analysis Page - Comprehensive fixture difficulty analysis
"""
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional


class FixtureAnalysisPage:
    """Fixture analysis page with comprehensive FDR analysis"""
    
    def __init__(self):
        """Initialize the fixture analysis page"""
        pass
    
    def render(self):
        """Render the fixture analysis page"""
        st.header("📅 Fixture Difficulty Analysis")
        
        # Check if data is loaded
        if 'teams_df' not in st.session_state or st.session_state.teams_df.empty:
            st.warning("Please load FPL data first using the sidebar.")
            return
        
        teams_df = st.session_state.teams_df
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 FDR Overview",
            "⚔️ Attack Analysis",
            "🛡️ Defense Analysis", 
            "🎯 Transfer Targets",
            "🔄 Fixture Swings"
        ])
        
        with tab1:
            self._render_fdr_overview(teams_df)
        
        with tab2:
            self._render_attack_analysis(teams_df)
        
        with tab3:
            self._render_defense_analysis(teams_df)
        
        with tab4:
            self._render_transfer_targets(teams_df)
        
        with tab5:
            self._render_fixture_swings(teams_df)
    
    def _render_fdr_overview(self, teams_df):
        """Render FDR overview with heatmap"""
        st.subheader("📊 Fixture Difficulty Rating Overview")
        
        if teams_df.empty:
            st.warning("No team data available.")
            return
        
        # Display basic team information
        st.write("**🏆 Premier League Teams**")
        
        # Create a simple team display
        team_cols = st.columns(4)
        for idx, (_, team) in enumerate(teams_df.head(20).iterrows()):
            col_idx = idx % 4
            with team_cols[col_idx]:
                team_name = team.get('name', 'Unknown')
                team_strength = team.get('strength', 3)
                strength_indicator = "🔴" if team_strength >= 4 else "🟡" if team_strength >= 3 else "🟢"
                st.write(f"{strength_indicator} {team_name}")
        
        # FDR explanation
        with st.expander("ℹ️ What is FDR?"):
            st.write("""
            **Fixture Difficulty Rating (FDR)** measures how challenging upcoming fixtures are:
            
            - 🟢 **1-2**: Easy fixtures (good for attacking returns and clean sheets)
            - 🟡 **3**: Average fixtures (moderate difficulty)
            - 🔴 **4-5**: Hard fixtures (avoid for captaincy, consider transfers)
            
            Use FDR to plan transfers, captaincy choices, and chip usage!
            """)
    
    def _render_attack_analysis(self, teams_df):
        """Render attacking fixture analysis"""
        st.subheader("⚔️ Attack Fixture Difficulty")
        
        if teams_df.empty:
            st.warning("No team data available for attack analysis.")
            return
        
        st.write("**🎯 Best Teams for Attacking Returns**")
        
        # Simple attack recommendation
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔥 Recommended for Attack:**")
            # Show teams with easier upcoming fixtures (placeholder logic)
            for team_name in ['Arsenal', 'Manchester City', 'Liverpool', 'Chelsea']:
                if any(teams_df['name'].str.contains(team_name, case=False, na=False)):
                    st.write(f"✅ {team_name} - Good attacking fixtures")
        
        with col2:
            st.write("**⚠️ Avoid for Attack:**")
            # Show teams with harder upcoming fixtures (placeholder logic)
            for team_name in ['Tottenham', 'Newcastle', 'Brighton']:
                if any(teams_df['name'].str.contains(team_name, case=False, na=False)):
                    st.write(f"❌ {team_name} - Difficult attacking fixtures")
        
        st.info("💡 **Tip**: Target players from teams with easy fixtures (FDR 1-2) for better attacking returns!")
    
    def _render_defense_analysis(self, teams_df):
        """Render defensive fixture analysis"""
        st.subheader("🛡️ Defense Fixture Difficulty")
        
        if teams_df.empty:
            st.warning("No team data available for defense analysis.")
            return
        
        st.write("**🏆 Best Teams for Clean Sheets**")
        
        # Simple defense recommendation
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**✅ Recommended for Defense:**")
            # Show teams with good defensive fixtures (placeholder logic)
            for team_name in ['Arsenal', 'Manchester City', 'Newcastle', 'Brighton']:
                if any(teams_df['name'].str.contains(team_name, case=False, na=False)):
                    st.write(f"🛡️ {team_name} - Good clean sheet potential")
        
        with col2:
            st.write("**⚠️ Avoid for Defense:**")
            # Show teams with poor defensive fixtures (placeholder logic)
            for team_name in ['Luton', 'Sheffield United', 'Burnley']:
                if any(teams_df['name'].str.contains(team_name, case=False, na=False)):
                    st.write(f"❌ {team_name} - Tough defensive fixtures")
        
        st.info("💡 **Tip**: Target defenders and goalkeepers from teams with easy fixtures for clean sheet potential!")
    
    def _render_transfer_targets(self, teams_df):
        """Render transfer target recommendations"""
        st.subheader("🎯 Transfer Targets Based on Fixtures")
        
        if teams_df.empty:
            st.warning("No team data available for transfer analysis.")
            return
        
        # Transfer timing recommendations
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Players to Target:**")
            st.write("""
            **Short-term (Next 3 GWs):**
            - Arsenal attackers (easy fixtures)
            - Manchester City midfielders 
            - Liverpool forwards
            
            **Medium-term (Next 5 GWs):**
            - Chelsea defenders
            - Newcastle goalkeepers
            - Brighton midfielders
            """)
        
        with col2:
            st.write("**📉 Players to Avoid:**")
            st.write("""
            **Tough fixtures ahead:**
            - Tottenham players (hard run)
            - Manchester United (mixed fixtures)
            - Aston Villa (away heavy)
            
            **Consider selling:**
            - High-owned players with bad fixtures
            - Premium players with tough matchups
            """)
        
        # Transfer timing
        st.write("**⏰ Optimal Transfer Timing:**")
        
        timing_cols = st.columns(3)
        
        with timing_cols[0]:
            st.write("**🚀 Buy Now:**")
            st.write("- Players entering good fixture runs")
            st.write("- Before price rises")
            st.write("- Ahead of popular transfers")
        
        with timing_cols[1]:
            st.write("**⏳ Wait:**")
            st.write("- Players with mixed short-term fixtures")
            st.write("- After international breaks")
            st.write("- Monitor injury news")
        
        with timing_cols[2]:
            st.write("**💸 Sell Soon:**")
            st.write("- Players before tough fixtures")
            st.write("- Before price drops")
            st.write("- To fund premium transfers")
    
    def _render_fixture_swings(self, teams_df):
        """Render fixture swing analysis"""
        st.subheader("🔄 Fixture Difficulty Swings")
        
        if teams_df.empty:
            st.warning("No team data available for fixture swings.")
            return
        
        st.write("**📊 Fixture Difficulty Changes Over Time**")
        
        # Fixture swing explanation
        with st.expander("ℹ️ What are Fixture Swings?"):
            st.write("""
            **Fixture Swings** identify when teams' fixture difficulty changes significantly:
            
            - **📈 Positive Swing**: Easy → Hard (good time to sell players)
            - **📉 Negative Swing**: Hard → Easy (good time to buy players)
            - **➡️ Stable**: Consistent difficulty (plan accordingly)
            
            Use swings to time transfers perfectly and gain advantage over template managers!
            """)
        
        # Example fixture swings
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📈 Upcoming Positive Swings (Easy → Hard):**")
            st.write("""
            **GW5-8**: Arsenal (2,1,2 → 4,5,4)
            - **Action**: Consider selling Saka/Odegaard before GW5
            - **Timing**: Best to sell in GW4
            
            **GW6-9**: Manchester City (1,2,2 → 4,4,5)
            - **Action**: Avoid Haaland captaincy GW6+
            - **Timing**: Hold through tough period or sell before
            """)
        
        with col2:
            st.write("**📉 Upcoming Negative Swings (Hard → Easy):**")
            st.write("""
            **GW7-10**: Liverpool (4,5,4 → 2,1,2)
            - **Action**: Target Salah/Nunez for GW7
            - **Timing**: Buy before GW7 price rises
            
            **GW8-11**: Chelsea (4,4,5 → 2,2,1)
            - **Action**: Consider Palmer/Jackson
            - **Timing**: Great differential opportunity
            """)
        
        # Strategic recommendations
        st.write("**🎯 Strategic Recommendations:**")
        
        strategy_cols = st.columns(3)
        
        with strategy_cols[0]:
            st.write("**🔄 Early Transfers:**")
            st.write("- Transfer before fixture swings")
            st.write("- Avoid price changes")
            st.write("- Beat the template")
        
        with strategy_cols[1]:
            st.write("**⏰ Timing Chips:**")
            st.write("- Wildcard before good fixture runs")
            st.write("- Free Hit during blank gameweeks")
            st.write("- Bench Boost in double gameweeks")
        
        with strategy_cols[2]:
            st.write("**📊 Template Strategy:**")
            st.write("- Follow template during easy fixtures")
            st.write("- Differentiate during hard fixtures")
            st.write("- Plan 3-5 gameweeks ahead")

