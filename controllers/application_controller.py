"""
Application Controller for FPL Analytics Application
Following SOLID principles - orchestrates components with dependency injection
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from models.domain_models import (
    Player, Team, FDRData, UserTeam, AnalysisConfig, AIConfig, 
    AnalysisType, ReliabilityStatus
)
from services.data_services import IDataLoader, IFDRCalculator
from services.visualization_services import (
    IVisualizationService, FDRVisualizationService, 
    PlayerVisualizationService, TeamComparisonService
)
from components.ui_components import (
    IUIComponent, ExplanationComponent, MetricsDisplayComponent,
    FDRHeatmapComponent, PlayerTableComponent, TeamFormationComponent,
    AnalysisConfigComponent, LoadingStateComponent, ErrorDisplayComponent
)


class IApplicationController(ABC):
    """Base interface for application controllers"""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the application"""
        pass
    
    @abstractmethod
    def render_tab(self, tab_name: str) -> None:
        """Render specific tab"""
        pass


class FPLApplicationController(IApplicationController):
    """Main application controller implementing dependency injection"""
    
    def __init__(self, 
                 data_loader: IDataLoader,
                 fdr_calculator: IFDRCalculator,
                 fdr_viz_service: FDRVisualizationService,
                 player_viz_service: PlayerVisualizationService,
                 team_comparison_service: TeamComparisonService):
        
        # Injected dependencies
        self.data_loader = data_loader
        self.fdr_calculator = fdr_calculator
        self.fdr_viz_service = fdr_viz_service
        self.player_viz_service = player_viz_service
        self.team_comparison_service = team_comparison_service
        
        # UI Components
        self.explanation_component = ExplanationComponent
        self.metrics_component = MetricsDisplayComponent()
        self.fdr_heatmap_component = FDRHeatmapComponent({
            1: '#00FF87', 2: '#01FF70', 3: '#FFDC00', 
            4: '#FF851B', 5: '#FF4136'
        })
        self.player_table_component = PlayerTableComponent()
        self.team_formation_component = TeamFormationComponent()
        self.config_component = AnalysisConfigComponent()
        self.loading_component = LoadingStateComponent()
        self.error_component = ErrorDisplayComponent()
        
        # Application state
        self.players: List[Player] = []
        self.teams: List[Team] = []
        self.fdr_data: List[FDRData] = []
        self.user_team: Optional[UserTeam] = None
        self.current_config: Optional[AnalysisConfig] = None
        
        # Cache for performance
        self._cache: Dict[str, Any] = {}
    
    def initialize(self) -> None:
        """Initialize the application with data loading"""
        try:
            self.loading_component.render_with_progress([
                "Loading FPL data...",
                "Processing player information...", 
                "Calculating fixture difficulty...",
                "Preparing analysis tools..."
            ])
            
            # Load core data
            self.players = self.data_loader.load_players()
            self.teams = self.data_loader.load_teams() 
            self.fdr_data = self.fdr_calculator.calculate_fdr(self.teams)
            
            # Try to load user team (optional)
            try:
                self.user_team = self.data_loader.load_user_team()
            except Exception:
                self.user_team = None
            
        except Exception as e:
            self.error_component.render(
                f"Failed to initialize application: {str(e)}", 
                "error"
            )
            raise
    
    def render_tab(self, tab_name: str) -> None:
        """Render specific tab with error handling"""
        try:
            if tab_name == "fdr_analysis":
                self._render_fdr_analysis_tab()
            elif tab_name == "player_analysis":
                self._render_player_analysis_tab()
            elif tab_name == "team_analysis":
                self._render_team_analysis_tab()
            elif tab_name == "my_team":
                self._render_my_team_tab()
            elif tab_name == "ai_assistant":
                self._render_ai_assistant_tab()
            else:
                self.error_component.render(
                    f"Unknown tab: {tab_name}", 
                    "warning"
                )
        except Exception as e:
            self.error_component.render(
                f"Error rendering {tab_name}: {str(e)}", 
                "error"
            )
    
    def _render_fdr_analysis_tab(self) -> None:
        """Render FDR analysis tab"""
        # Tab explanation
        explanation = self.explanation_component(
            "🎯 What is FDR Analysis?",
            """
            **Fixture Difficulty Rating (FDR)** helps you identify the best times to captain, 
            transfer in, or bench players based on upcoming fixture difficulty.
            
            **How to use this analysis:**
            - **Green fixtures (1-2)**: Prime captaincy and transfer targets
            - **Yellow fixtures (3)**: Moderate difficulty, decent options
            - **Orange/Red fixtures (4-5)**: Consider benching or avoiding
            
            **Pro Tips:**
            - Look for consistent runs of good fixtures (3+ gameweeks)
            - Consider both attack and defense FDR for different player types
            - Form-adjusted FDR accounts for recent team performance
            """,
            expanded=False
        )
        explanation.render()
        
        # Configuration
        config = self.config_component.render()
        self.current_config = config
        
        # Filter FDR data based on config
        filtered_fdr = self._filter_fdr_data(self.fdr_data, config)
        
        # Key metrics
        if filtered_fdr:
            easiest_team = min(filtered_fdr, key=lambda x: x.combined_fdr)
            hardest_team = max(filtered_fdr, key=lambda x: x.combined_fdr)
            
            metrics = [
                {
                    'label': 'Best Fixtures',
                    'value': easiest_team.team_short_name,
                    'delta': f"FDR: {easiest_team.combined_fdr:.1f}",
                    'help': 'Team with easiest upcoming fixtures'
                },
                {
                    'label': 'Worst Fixtures', 
                    'value': hardest_team.team_short_name,
                    'delta': f"FDR: {hardest_team.combined_fdr:.1f}",
                    'help': 'Team with hardest upcoming fixtures'
                },
                {
                    'label': 'Teams Analyzed',
                    'value': len(set(fdr.team_short_name for fdr in filtered_fdr)),
                    'help': 'Number of teams in analysis'
                },
                {
                    'label': 'Fixtures Analyzed',
                    'value': len(filtered_fdr),
                    'help': 'Total fixtures in analysis period'
                }
            ]
            
            self.metrics_component.render(metrics)
        
        # FDR Heatmap
        self.fdr_heatmap_component.render(filtered_fdr, 'combined')
        
        # Additional visualizations
        if filtered_fdr:
            # FDR Distribution Chart
            fdr_chart = self.fdr_viz_service.create_chart(filtered_fdr, 'bar')
            import streamlit as st
            st.plotly_chart(fdr_chart, use_container_width=True)
            
            # Attack vs Defense Scatter
            scatter_chart = self.fdr_viz_service.create_chart(filtered_fdr, 'scatter')
            st.plotly_chart(scatter_chart, use_container_width=True)
    
    def _render_player_analysis_tab(self) -> None:
        """Render player analysis tab"""
        # Tab explanation
        explanation = self.explanation_component(
            "⚽ Player Performance Analysis",
            """
            **Deep dive into player statistics** to identify value picks, captaincy options, and transfer targets.
            
            **Key Metrics Explained:**
            - **PPG**: Points per game - consistency indicator
            - **PPM**: Points per million - value indicator  
            - **Form**: Average points in last 5 games
            - **xG/xA**: Expected goals/assists - underlying performance
            
            **How to use this analysis:**
            - Filter by position to compare like-for-like
            - Look for high PPM players for value transfers
            - Check form for captaincy decisions
            - Use xG/xA to identify players due a goal/assist
            """,
            expanded=False
        )
        explanation.render()
        
        # Player table with filters
        self.player_table_component.render(self.players, self.current_config)
        
        # Player visualizations
        import streamlit as st
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Points vs Price scatter
            scatter_chart = self.player_viz_service.create_chart(
                self.players, 'scatter', 
                x_metric='cost_millions', 
                y_metric='total_points'
            )
            st.plotly_chart(scatter_chart, use_container_width=True)
        
        with col2:
            # Top scorers bar chart
            bar_chart = self.player_viz_service.create_chart(
                self.players, 'bar',
                metric='total_points',
                top_n=15
            )
            st.plotly_chart(bar_chart, use_container_width=True)
        
        # Position analysis
        position_chart = self.player_viz_service.create_chart(
            self.players, 'position_analysis'
        )
        st.plotly_chart(position_chart, use_container_width=True)
    
    def _render_team_analysis_tab(self) -> None:
        """Render team analysis tab"""
        # Tab explanation
        explanation = self.explanation_component(
            "🏟️ Team Strength Analysis",
            """
            **Compare team strengths** across different scenarios to inform transfer decisions.
            
            **Strength Ratings Explained:**
            - **Overall**: General team quality (1-6 scale)
            - **Attack**: Offensive threat level
            - **Defense**: Defensive solidity
            - **Home/Away**: Performance splits by venue
            
            **Strategic Applications:**
            - Target players from strong attacking teams
            - Avoid defenders from weak defensive teams
            - Consider home/away splits for captaincy
            - Look for strength vs fixture combinations
            """,
            expanded=False
        )
        explanation.render()
        
        # Team comparison radar chart
        radar_chart = self.team_comparison_service.create_chart(
            self.teams[:6], 'radar'  # Limit to 6 teams for readability
        )
        
        import streamlit as st
        st.plotly_chart(radar_chart, use_container_width=True)
        
        # Team strength bar chart  
        bar_chart = self.team_comparison_service.create_chart(
            self.teams, 'bar_comparison',
            metric='strength_overall_home'
        )
        st.plotly_chart(bar_chart, use_container_width=True)
    
    def _render_my_team_tab(self) -> None:
        """Render my team tab"""
        import streamlit as st
        
        if not self.user_team:
            st.warning("No team data available. Please log in or check your team ID.")
            return
        
        # Tab explanation
        explanation = self.explanation_component(
            "👥 Your FPL Team Analysis",
            """
            **Analyze your current team** and get personalized insights for optimization.
            
            **Team Analysis Features:**
            - Formation and player breakdown
            - Player performance metrics
            - Upcoming fixture analysis
            - Transfer recommendations
            - Captaincy suggestions
            
            **Optimization Tips:**
            - Look for players with poor upcoming fixtures
            - Identify bench players who could start
            - Consider form drops for transfer targets
            - Plan transfers around blank/double gameweeks
            """,
            expanded=False
        )
        explanation.render()
        
        # Team formation display
        self.team_formation_component.render(self.user_team, self.players)
        
        # Team metrics
        total_value = sum(
            next((p.cost_millions for p in self.players if p.id == pick.element), 0)
            for pick in self.user_team.picks
        )
        
        total_points = sum(
            next((p.total_points for p in self.players if p.id == pick.element), 0)
            for pick in self.user_team.picks
        )
        
        metrics = [
            {
                'label': 'Total Value',
                'value': f"£{total_value:.1f}m",
                'help': 'Current squad value'
            },
            {
                'label': 'Total Points',
                'value': str(total_points),
                'help': 'Combined points from all players'
            },
            {
                'label': 'In The Bank',
                'value': f"£{self.user_team.bank_balance / 10:.1f}m",
                'help': 'Remaining budget'
            },
            {
                'label': 'Transfers Made',
                'value': str(self.user_team.total_transfers),
                'help': 'Total transfers this season'
            }
        ]
        
        self.metrics_component.render(metrics)
    
    def _render_ai_assistant_tab(self) -> None:
        """Render AI assistant tab"""
        # Tab explanation  
        explanation = self.explanation_component(
            "🤖 AI-Powered FPL Assistant",
            """
            **Get intelligent FPL advice** powered by advanced AI analysis of your team and the latest data.
            
            **AI Assistant Features:**
            - Personalized transfer recommendations
            - Captaincy suggestions with reasoning
            - Chip strategy advice
            - Fixture analysis and planning
            - Market trends and differential picks
            
            **How it works:**
            - AI analyzes 1000+ data points per player
            - Considers your team context and constraints
            - Provides reasoning for all recommendations
            - Updates advice based on latest form and fixtures
            """,
            expanded=False
        )
        explanation.render()
        
        import streamlit as st
        st.info("AI Assistant functionality will be implemented with the enhanced components.")
    
    def _filter_fdr_data(self, fdr_data: List[FDRData], config: AnalysisConfig) -> List[FDRData]:
        """Filter FDR data based on analysis configuration"""
        if not config:
            return fdr_data
        
        filtered = []
        for fdr in fdr_data:
            # Filter by gameweeks
            if fdr.fixture_number > config.gameweeks_ahead:
                continue
            
            # Filter by analysis type
            if config.analysis_type == AnalysisType.HOME_ONLY and not fdr.is_home:
                continue
            if config.analysis_type == AnalysisType.AWAY_ONLY and fdr.is_home:
                continue
            if config.analysis_type == AnalysisType.NEXT_3_FIXTURES and fdr.fixture_number > 3:
                continue
            
            # Filter by FDR threshold
            if fdr.combined_fdr > config.fdr_threshold + 1.5:  # Allow some tolerance
                continue
            
            filtered.append(fdr)
        
        return filtered
    
    def get_cache(self, key: str) -> Any:
        """Get cached value"""
        return self._cache.get(key)
    
    def set_cache(self, key: str, value: Any) -> None:
        """Set cached value"""
        self._cache[key] = value
    
    def clear_cache(self) -> None:
        """Clear all cached values"""
        self._cache.clear()


class DependencyContainer:
    """Container for dependency injection setup"""
    
    @staticmethod
    def create_application_controller() -> FPLApplicationController:
        """Factory method to create fully configured application controller"""
        from services.data_services import FPLAPIDataLoader, FDRCalculatorService
        
        # Create service instances
        data_loader = FPLAPIDataLoader()
        fdr_calculator = FDRCalculatorService()
        fdr_viz_service = FDRVisualizationService()
        player_viz_service = PlayerVisualizationService()
        team_comparison_service = TeamComparisonService()
        
        # Create and return controller with injected dependencies
        return FPLApplicationController(
            data_loader=data_loader,
            fdr_calculator=fdr_calculator,
            fdr_viz_service=fdr_viz_service,
            player_viz_service=player_viz_service,
            team_comparison_service=team_comparison_service
        )
