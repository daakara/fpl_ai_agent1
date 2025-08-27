class FDRVisualizer:
    """Visualizes Fixture Difficulty Ratings (FDR) for teams based on attack and defense."""

    def __init__(self, fdr_analyzer):
        self.fdr_analyzer = fdr_analyzer

    def display_attack_fdr(self):
        """Display the attack fixture difficulty ratings."""
        attack_fdr = self.fdr_analyzer.calculate_attack_fdr()
        # Code to create a chart or table for attack FDR
        # Example: st.bar_chart(attack_fdr)

    def display_defense_fdr(self):
        """Display the defense fixture difficulty ratings."""
        defense_fdr = self.fdr_analyzer.calculate_defense_fdr()
        # Code to create a chart or table for defense FDR
        # Example: st.bar_chart(defense_fdr)

    def render_fdr(self):
        """Render both attack and defense FDR visualizations."""
        self.display_attack_fdr()
        self.display_defense_fdr()