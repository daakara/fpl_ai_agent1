import streamlit as st

class StyleManager:
    """Manages the styling of the Streamlit app."""

    def __init__(self):
        self.primary_color = "#007BFF"  # Example primary color
        self.secondary_color = "#28A745"  # Example secondary color
        self.font = "sans-serif"  # Example font

    def load_styles(self):
        """Loads custom CSS styles into the Streamlit app."""
        st.markdown(
            f"""
            <style>
            /* Example styles - customize these */
            body {{
                font-family: {self.font};
                font-size: 16px; /* Default font size */
            }}
            .streamlit-expanderHeader {{
                color: {self.primary_color};
                font-weight: bold;
            }}
            .big-font {{
                font-size: 1.2em !important; /* Relative font size */
            }}
            /* Media query for smaller screens */
            @media (max-width: 768px) {{
                body {{
                    font-size: 14px; /* Smaller font size on smaller screens */
                }}
                .streamlit-expanderHeader {{
                    font-size: 16px !important;
                }}
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Add new CSS classes for the enhanced visualizations:
    def load_enhanced_team_styles(self):
        """Load enhanced styles for team recommendations"""
        enhanced_css = """
        <style>
        .formation-container {
            background: linear-gradient(45deg, #2E8B57, #228B22);
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        
        .risk-metric-high {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 10px;
            margin: 5px 0;
        }
        
        .risk-metric-low {
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 10px;
            margin: 5px 0;
        }
        
        .position-tab {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        </style>
        """
        st.markdown(enhanced_css, unsafe_allow_html=True)