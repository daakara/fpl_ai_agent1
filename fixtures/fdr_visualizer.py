"""
FDR Visualizer - Creates visualizations for fixture difficulty ratings
"""
import plotly.graph_objects as go
import pandas as pd


class FDRVisualizer:
    """Creates visualizations for fixture difficulty ratings"""
    
    def __init__(self):
        self.fdr_colors = {
            1: '#0072B2',  # Blue (Very Easy)
            2: '#009E73',  # Green (Easy)
            3: '#F0E442',  # Yellow (Average)
            4: '#E69F00',  # Orange (Hard)
            5: '#D55E00'   # Red (Very Hard)
        }
    
    def create_fdr_heatmap(self, fixtures_df: pd.DataFrame, fdr_type: str = 'combined') -> go.Figure:
        """Create FDR heatmap visualization"""
        if fixtures_df.empty:
            return go.Figure()
        
        fdr_column = f'{fdr_type}_fdr'
        
        if fdr_column not in fixtures_df.columns:
            return go.Figure()
        
        # Create pivot table for heatmap
        try:
            pivot_data = fixtures_df.pivot_table(
                index='team_short_name',
                columns='fixture_number', 
                values=fdr_column,
                aggfunc='first'
            )
        except Exception as e:
            # Fallback if pivot fails
            return go.Figure()
        
        if pivot_data.empty:
            return go.Figure()
        
        # Create color scale
        colorscale = [
            [0.0, self.fdr_colors[1]], 
            [0.25, self.fdr_colors[2]], 
            [0.5, self.fdr_colors[3]], 
            [0.75, self.fdr_colors[4]], 
            [1.0, self.fdr_colors[5]]
        ]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=[f'GW{i}' for i in pivot_data.columns],
            y=pivot_data.index,
            colorscale=colorscale,
            zmin=1, 
            zmax=5,
            text=pivot_data.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>Fixture %{x}<br>FDR: %{z}<extra></extra>',
            colorbar=dict(
                title="FDR",
                titleside="right",
                tickmode="array",
                tickvals=[1, 2, 3, 4, 5],
                ticktext=["Very Easy", "Easy", "Average", "Hard", "Very Hard"]
            )
        ))
        
        fig.update_layout(
            title=f'{fdr_type.title()} FDR - Next 5 Fixtures',
            xaxis_title="Fixture Number",
            yaxis_title="Team",
            height=600,
            font=dict(size=12)
        )
        
        return fig
    
    def create_fdr_bar_chart(self, fixtures_df: pd.DataFrame, fdr_type: str = 'combined') -> go.Figure:
        """Create bar chart of average FDR by team"""
        if fixtures_df.empty:
            return go.Figure()
        
        fdr_column = f'{fdr_type}_fdr'
        
        if fdr_column not in fixtures_df.columns:
            return go.Figure()
        
        # Calculate average FDR by team
        avg_fdr = fixtures_df.groupby('team_short_name')[fdr_column].mean().sort_values()
        
        # Create colors based on FDR values
        colors = [self._get_fdr_color(fdr) for fdr in avg_fdr.values]
        
        fig = go.Figure(data=[
            go.Bar(
                x=avg_fdr.index,
                y=avg_fdr.values,
                marker_color=colors,
                text=[f'{fdr:.1f}' for fdr in avg_fdr.values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f'Average {fdr_type.title()} FDR by Team',
            xaxis_title="Team",
            yaxis_title="Average FDR",
            height=500,
            yaxis=dict(range=[1, 5])
        )
        
        return fig
    
    def create_fdr_distribution(self, fixtures_df: pd.DataFrame, fdr_type: str = 'combined') -> go.Figure:
        """Create distribution chart of FDR values"""
        if fixtures_df.empty:
            return go.Figure()
        
        fdr_column = f'{fdr_type}_fdr'
        
        if fdr_column not in fixtures_df.columns:
            return go.Figure()
        
        # Get FDR distribution
        fdr_counts = fixtures_df[fdr_column].value_counts().sort_index()
        
        # Create colors
        colors = [self.fdr_colors.get(fdr, '#808080') for fdr in fdr_counts.index]
        
        fig = go.Figure(data=[
            go.Bar(
                x=fdr_counts.index,
                y=fdr_counts.values,
                marker_color=colors,
                text=fdr_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f'{fdr_type.title()} FDR Distribution',
            xaxis_title="FDR Rating",
            yaxis_title="Number of Fixtures",
            height=400,
            xaxis=dict(tickmode='array', tickvals=[1, 2, 3, 4, 5])
        )
        
        return fig
    
    def _get_fdr_color(self, fdr_value: float) -> str:
        """Get color for FDR value"""
        # Round to nearest integer for color mapping
        fdr_int = round(fdr_value)
        return self.fdr_colors.get(fdr_int, '#808080')