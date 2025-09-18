"""
FDR Analyzer - Calculates fixture difficulty ratings for FPL teams
"""
import pandas as pd
import numpy as np


class FDRAnalyzer:
    """Analyzes fixture difficulty ratings for FPL teams"""
    
    def __init__(self):
        self.fdr_colors = {
            1: '#0072B2',  # Blue (Very Easy)
            2: '#009E73',  # Green (Easy)  
            3: '#F0E442',  # Yellow (Average)
            4: '#E69F00',  # Orange (Hard)
            5: '#D55E00'   # Red (Very Hard)
        }
        self.fdr_labels = {
            1: 'Very Easy', 2: 'Easy', 3: 'Average', 4: 'Hard', 5: 'Very Hard'
        }
    
    def calculate_attack_fdr(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate attack-based FDR"""
        if fixtures_df.empty:
            return pd.DataFrame()
        
        attack_fdr_df = fixtures_df.copy()
        
        def get_attack_difficulty(row):
            # Get opponent's defensive strength
            if row['is_home']:
                defence_strength = row.get('opponent_strength_home', 1200)
            else:
                defence_strength = row.get('opponent_strength_away', 1200)
            
            # Convert strength values to FDR scale (1-5)
            # Higher defensive strength = harder for attackers = higher FDR
            if defence_strength >= 1400:
                return 5  # Very hard
            elif defence_strength >= 1350:
                return 4  # Hard
            elif defence_strength >= 1300:
                return 3  # Average
            elif defence_strength >= 1250:
                return 2  # Easy
            else:
                return 1  # Very easy
        
        attack_fdr_df['attack_fdr'] = attack_fdr_df.apply(get_attack_difficulty, axis=1)
        return attack_fdr_df
    
    def calculate_defense_fdr(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate defense-based FDR"""
        if fixtures_df.empty:
            return pd.DataFrame()
        
        defense_fdr_df = fixtures_df.copy()
        
        def get_defense_difficulty(row):
            # Get opponent's attacking strength
            if row['is_home']:
                attack_strength = row.get('opponent_strength_away', 1200)
            else:
                attack_strength = row.get('opponent_strength_home', 1200)
            
            # Convert strength values to FDR scale (1-5)
            # Higher attacking strength = harder for defenders = higher FDR
            if attack_strength >= 1400:
                return 5  # Very hard
            elif attack_strength >= 1350:
                return 4  # Hard
            elif attack_strength >= 1300:
                return 3  # Average
            elif attack_strength >= 1250:
                return 2  # Easy
            else:
                return 1  # Very easy
        
        defense_fdr_df['defense_fdr'] = defense_fdr_df.apply(get_defense_difficulty, axis=1)
        return defense_fdr_df
    
    def calculate_combined_fdr(self, fixtures_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate combined FDR (attack + defense weighted average)"""
        attack_fdr_df = self.calculate_attack_fdr(fixtures_df)
        defense_fdr_df = self.calculate_defense_fdr(fixtures_df)
        
        if attack_fdr_df.empty or defense_fdr_df.empty:
            return pd.DataFrame()
        
        combined_df = attack_fdr_df.copy()
        combined_df['defense_fdr'] = defense_fdr_df['defense_fdr']
        
        # Calculate weighted average (60% attack, 40% defense)
        combined_df['combined_fdr'] = (
            (combined_df['attack_fdr'] * 0.6 + combined_df['defense_fdr'] * 0.4)
        ).round().astype(int)
        
        # Ensure FDR is within 1-5 range
        combined_df['combined_fdr'] = combined_df['combined_fdr'].clip(1, 5)
        
        return combined_df
    
    def apply_form_adjustment(self, fixtures_df: pd.DataFrame, form_weight: float = 0.3) -> pd.DataFrame:
        """Apply form-based adjustments to FDR calculations"""
        if fixtures_df.empty or form_weight <= 0:
            return fixtures_df
        
        # This would require team form data - placeholder implementation
        # In production, integrate with team performance metrics
        adjusted_df = fixtures_df.copy()
        
        # Example: Teams in good form get slightly easier FDR
        # Teams in poor form get slightly harder FDR
        # This is a simplified implementation
        
        return adjusted_df
    
    def get_fdr_color(self, fdr_value: int) -> str:
        """Get color for FDR value"""
        return self.fdr_colors.get(fdr_value, '#808080')
    
    def get_fdr_label(self, fdr_value: int) -> str:
        """Get label for FDR value"""
        return self.fdr_labels.get(fdr_value, 'Unknown')