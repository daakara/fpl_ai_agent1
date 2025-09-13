"""
Advanced Data Export/Import System for FPL Analytics
"""
import pandas as pd
import streamlit as st
import json
import csv
import io
import zipfile
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pickle
import base64
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch


@dataclass
class ExportConfig:
    """Configuration for data exports"""
    include_players: bool = True
    include_team_analysis: bool = True
    include_recommendations: bool = True
    include_charts: bool = True
    format_type: str = "comprehensive"  # comprehensive, summary, raw_data
    date_range: Optional[tuple] = None


class FPLDataExporter:
    """Advanced data export system for FPL analytics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)
    
    def export_to_csv(self, data: pd.DataFrame, filename: str) -> str:
        """Export DataFrame to CSV format"""
        try:
            file_path = self.export_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            data.to_csv(file_path, index=False)
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {e}")
            return ""
    
    def export_to_excel(self, data_dict: Dict[str, pd.DataFrame], filename: str) -> str:
        """Export multiple DataFrames to Excel with multiple sheets"""
        try:
            file_path = self.export_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                for sheet_name, df in data_dict.items():
                    # Clean sheet name for Excel compatibility
                    clean_sheet_name = sheet_name.replace('/', '_').replace('\\', '_')[:31]
                    df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
            
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Error exporting to Excel: {e}")
            return ""
    
    def export_to_json(self, data: Union[Dict, List], filename: str) -> str:
        """Export data to JSON format"""
        try:
            file_path = self.export_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
            
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {e}")
            return ""
    
    def create_team_analysis_report(self, team_data: Dict, players_df: pd.DataFrame, 
                                  recommendations: Optional[Dict] = None) -> str:
        """Create comprehensive team analysis report"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Prepare report data
            report_data = {
                'metadata': {
                    'export_date': datetime.now().isoformat(),
                    'team_id': team_data.get('team_data', {}).get('id', 0),
                    'team_name': team_data.get('team_data', {}).get('entry_name', 'Unknown'),
                    'gameweek': team_data.get('statistics', {}).get('gameweek', 1)
                },
                'team_summary': team_data.get('team_data', {}),
                'squad_analysis': {
                    'total_players': len(team_data.get('picks', [])),
                    'total_value': team_data.get('statistics', {}).get('total_cost_millions', 0),
                    'total_points': team_data.get('statistics', {}).get('total_points', 0),
                    'average_ownership': team_data.get('statistics', {}).get('average_ownership', 0)
                },
                'player_details': [],
                'recommendations': recommendations or {}
            }
            
            # Add player details
            for pick in team_data.get('picks', []):
                player_info = {
                    'name': pick.web_name,
                    'team': pick.team_name,
                    'position': pick.position_name,
                    'price': pick.now_cost / 10.0,
                    'total_points': pick.total_points,
                    'form': pick.form,
                    'ownership': pick.selected_by_percent,
                    'squad_position': 'Starting XI' if pick.position <= 11 else 'Bench',
                    'is_captain': pick.is_captain,
                    'is_vice_captain': pick.is_vice_captain
                }
                report_data['player_details'].append(player_info)
            
            # Export to JSON
            json_file = self.export_to_json(report_data, f"team_analysis_report_{timestamp}")
            
            # Also create Excel version
            excel_data = {
                'Team Summary': pd.DataFrame([report_data['team_summary']]),
                'Squad Analysis': pd.DataFrame([report_data['squad_analysis']]),
                'Player Details': pd.DataFrame(report_data['player_details'])
            }
            
            if recommendations:
                excel_data['Recommendations'] = pd.DataFrame([recommendations])
            
            excel_file = self.export_to_excel(excel_data, f"team_analysis_{timestamp}")
            
            return json_file, excel_file
            
        except Exception as e:
            self.logger.error(f"Error creating team analysis report: {e}")
            return "", ""
    
    def export_player_comparison(self, players_df: pd.DataFrame, selected_players: List[int]) -> str:
        """Export detailed player comparison analysis"""
        try:
            if not selected_players:
                return ""
            
            # Filter for selected players
            comparison_df = players_df[players_df['id'].isin(selected_players)].copy()
            
            if comparison_df.empty:
                return ""
            
            # Select key comparison columns
            comparison_columns = [
                'web_name', 'team_name', 'position_name', 'now_cost', 'total_points',
                'form', 'selected_by_percent', 'minutes', 'goals_scored', 'assists',
                'clean_sheets', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat'
            ]
            
            # Filter available columns
            available_cols = [col for col in comparison_columns if col in comparison_df.columns]
            export_df = comparison_df[available_cols]
            
            # Add calculated metrics
            export_df['points_per_million'] = export_df['total_points'] / (export_df['now_cost'] / 10)
            export_df['points_per_game'] = export_df['total_points'] / export_df.get('starts', 1).replace(0, 1)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return self.export_to_csv(export_df, f"player_comparison_{timestamp}")
            
        except Exception as e:
            self.logger.error(f"Error exporting player comparison: {e}")
            return ""
    
    def create_comprehensive_archive(self, team_data: Dict, players_df: pd.DataFrame,
                                   fixtures_df: Optional[pd.DataFrame] = None,
                                   recommendations: Optional[Dict] = None) -> str:
        """Create comprehensive zip archive with all data"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_name = f"fpl_comprehensive_export_{timestamp}.zip"
            archive_path = self.export_dir / archive_name
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add team analysis JSON
                team_json = self.export_to_json(team_data, "team_data")
                if team_json:
                    zipf.write(team_json, "team_analysis.json")
                
                # Add players CSV
                players_csv = self.export_to_csv(players_df, "all_players")
                if players_csv:
                    zipf.write(players_csv, "all_players.csv")
                
                # Add fixtures if available
                if fixtures_df is not None and not fixtures_df.empty:
                    fixtures_csv = self.export_to_csv(fixtures_df, "fixtures")
                    if fixtures_csv:
                        zipf.write(fixtures_csv, "fixtures.csv")
                
                # Add recommendations if available
                if recommendations:
                    rec_json = self.export_to_json(recommendations, "recommendations")
                    if rec_json:
                        zipf.write(rec_json, "recommendations.json")
                
                # Add metadata
                metadata = {
                    'export_timestamp': datetime.now().isoformat(),
                    'fpl_agent_version': '2.0',
                    'included_files': [
                        'team_analysis.json',
                        'all_players.csv',
                        'fixtures.csv' if fixtures_df is not None else None,
                        'recommendations.json' if recommendations else None
                    ]
                }
                metadata_json = self.export_to_json(metadata, "metadata")
                if metadata_json:
                    zipf.write(metadata_json, "export_metadata.json")
            
            return str(archive_path)
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive archive: {e}")
            return ""


class FPLDataImporter:
    """Advanced data import system for FPL analytics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def import_from_csv(self, file_content: bytes, filename: str) -> Optional[pd.DataFrame]:
        """Import data from CSV file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    content_str = file_content.decode(encoding)
                    df = pd.read_csv(io.StringIO(content_str))
                    self.logger.info(f"Successfully imported {filename} with {encoding} encoding")
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    self.logger.warning(f"Error reading CSV with {encoding}: {e}")
                    continue
            
            # If all encodings fail, try with error handling
            content_str = file_content.decode('utf-8', errors='replace')
            df = pd.read_csv(io.StringIO(content_str))
            return df
            
        except Exception as e:
            self.logger.error(f"Error importing CSV {filename}: {e}")
            return None
    
    def import_from_excel(self, file_content: bytes, filename: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Import data from Excel file with multiple sheets"""
        try:
            excel_file = pd.ExcelFile(io.BytesIO(file_content))
            sheets_data = {}
            
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    sheets_data[sheet_name] = df
                except Exception as e:
                    self.logger.warning(f"Error reading sheet {sheet_name}: {e}")
            
            return sheets_data if sheets_data else None
            
        except Exception as e:
            self.logger.error(f"Error importing Excel {filename}: {e}")
            return None
    
    def import_from_json(self, file_content: bytes, filename: str) -> Optional[Dict]:
        """Import data from JSON file"""
        try:
            content_str = file_content.decode('utf-8')
            data = json.loads(content_str)
            return data
        except Exception as e:
            self.logger.error(f"Error importing JSON {filename}: {e}")
            return None
    
    def import_team_backup(self, file_content: bytes, filename: str) -> Optional[Dict]:
        """Import team backup data"""
        try:
            if filename.endswith('.json'):
                return self.import_from_json(file_content, filename)
            elif filename.endswith(('.xlsx', '.xls')):
                sheets_data = self.import_from_excel(file_content, filename)
                if sheets_data:
                    # Convert Excel data back to team format
                    return self._reconstruct_team_data(sheets_data)
            elif filename.endswith('.zip'):
                return self._import_from_archive(file_content, filename)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error importing team backup: {e}")
            return None
    
    def _reconstruct_team_data(self, sheets_data: Dict[str, pd.DataFrame]) -> Dict:
        """Reconstruct team data from Excel sheets"""
        try:
            team_data = {}
            
            if 'Team Summary' in sheets_data:
                team_summary_df = sheets_data['Team Summary']
                if not team_summary_df.empty:
                    team_data['team_data'] = team_summary_df.iloc[0].to_dict()
            
            if 'Player Details' in sheets_data:
                players_df = sheets_data['Player Details']
                team_data['player_details'] = players_df.to_dict('records')
            
            if 'Squad Analysis' in sheets_data:
                squad_df = sheets_data['Squad Analysis']
                if not squad_df.empty:
                    team_data['statistics'] = squad_df.iloc[0].to_dict()
            
            return team_data
            
        except Exception as e:
            self.logger.error(f"Error reconstructing team data: {e}")
            return {}
    
    def _import_from_archive(self, file_content: bytes, filename: str) -> Optional[Dict]:
        """Import data from zip archive"""
        try:
            with zipfile.ZipFile(io.BytesIO(file_content)) as zipf:
                archive_data = {}
                
                # Import team analysis
                if 'team_analysis.json' in zipf.namelist():
                    team_content = zipf.read('team_analysis.json')
                    team_data = self.import_from_json(team_content, 'team_analysis.json')
                    if team_data:
                        archive_data['team_data'] = team_data
                
                # Import players data
                if 'all_players.csv' in zipf.namelist():
                    players_content = zipf.read('all_players.csv')
                    players_df = self.import_from_csv(players_content, 'all_players.csv')
                    if players_df is not None:
                        archive_data['players_data'] = players_df
                
                # Import recommendations
                if 'recommendations.json' in zipf.namelist():
                    rec_content = zipf.read('recommendations.json')
                    rec_data = self.import_from_json(rec_content, 'recommendations.json')
                    if rec_data:
                        archive_data['recommendations'] = rec_data
                
                return archive_data if archive_data else None
                
        except Exception as e:
            self.logger.error(f"Error importing from archive: {e}")
            return None


class UserPreferenceManager:
    """User preference storage and management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prefs_dir = Path("user_preferences")
        self.prefs_dir.mkdir(exist_ok=True)
        self.prefs_file = self.prefs_dir / "preferences.json"
        self._load_preferences()
    
    def _load_preferences(self):
        """Load user preferences from file"""
        try:
            if self.prefs_file.exists():
                with open(self.prefs_file, 'r', encoding='utf-8') as f:
                    self.preferences = json.load(f)
            else:
                self.preferences = self._get_default_preferences()
        except Exception as e:
            self.logger.error(f"Error loading preferences: {e}")
            self.preferences = self._get_default_preferences()
    
    def _get_default_preferences(self) -> Dict:
        """Get default user preferences"""
        return {
            'team_id': None,
            'favorite_players': [],
            'preferred_formation': '3-5-2',
            'risk_tolerance': 'medium',  # low, medium, high
            'budget_strategy': 'balanced',  # conservative, balanced, aggressive
            'analysis_depth': 'comprehensive',  # basic, detailed, comprehensive
            'export_format': 'excel',  # csv, excel, json
            'auto_refresh': True,
            'cache_duration': 300,
            'notification_preferences': {
                'price_changes': True,
                'injury_updates': True,
                'form_alerts': True
            },
            'display_preferences': {
                'show_advanced_metrics': True,
                'highlight_differentials': True,
                'show_expected_stats': True,
                'compact_tables': False
            },
            'recommendation_weights': {
                'value_efficiency': 0.25,
                'form_trend': 0.20,
                'fixture_difficulty': 0.15,
                'ownership_factor': 0.05,
                'expected_performance': 0.35
            }
        }
    
    def save_preferences(self):
        """Save preferences to file"""
        try:
            with open(self.prefs_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving preferences: {e}")
    
    def get_preference(self, key: str, default=None):
        """Get a specific preference value"""
        keys = key.split('.')
        value = self.preferences
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_preference(self, key: str, value):
        """Set a specific preference value"""
        keys = key.split('.')
        current = self.preferences
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        self.save_preferences()
    
    def update_preferences(self, updates: Dict):
        """Update multiple preferences at once"""
        def deep_update(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    deep_update(target[key], value)
                else:
                    target[key] = value
        
        deep_update(self.preferences, updates)
        self.save_preferences()
    
    def export_preferences(self) -> str:
        """Export preferences to JSON file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            export_file = self.prefs_dir / f"preferences_backup_{timestamp}.json"
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, indent=2, ensure_ascii=False)
            
            return str(export_file)
        except Exception as e:
            self.logger.error(f"Error exporting preferences: {e}")
            return ""
    
    def import_preferences(self, file_content: bytes) -> bool:
        """Import preferences from JSON file"""
        try:
            content_str = file_content.decode('utf-8')
            imported_prefs = json.loads(content_str)
            
            # Validate and merge with defaults
            default_prefs = self._get_default_preferences()
            
            def safe_merge(default, imported):
                result = default.copy()
                for key, value in imported.items():
                    if key in default:
                        if isinstance(value, dict) and isinstance(default[key], dict):
                            result[key] = safe_merge(default[key], value)
                        else:
                            result[key] = value
                return result
            
            self.preferences = safe_merge(default_prefs, imported_prefs)
            self.save_preferences()
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing preferences: {e}")
            return False


# Global instances
data_exporter = FPLDataExporter()
data_importer = FPLDataImporter()
preference_manager = UserPreferenceManager()


def create_download_link(file_path: str, link_text: str) -> str:
    """Create a download link for a file"""
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        b64_data = base64.b64encode(file_data).decode()
        filename = Path(file_path).name
        
        href = f'<a href="data:application/octet-stream;base64,{b64_data}" download="{filename}">{link_text}</a>'
        return href
    except Exception as e:
        return f"Error creating download link: {e}"


def display_export_options(team_data: Dict, players_df: pd.DataFrame, 
                          recommendations: Optional[Dict] = None):
    """Display export options in Streamlit interface"""
    st.subheader("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Team Analysis"):
            with st.spinner("Creating team analysis report..."):
                json_file, excel_file = data_exporter.create_team_analysis_report(
                    team_data, players_df, recommendations
                )
                if json_file and excel_file:
                    st.success("✅ Team analysis exported!")
                    st.markdown(create_download_link(json_file, "📄 Download JSON Report"), 
                              unsafe_allow_html=True)
                    st.markdown(create_download_link(excel_file, "📊 Download Excel Report"), 
                              unsafe_allow_html=True)
                else:
                    st.error("❌ Export failed")
    
    with col2:
        if st.button("📦 Create Full Archive"):
            with st.spinner("Creating comprehensive archive..."):
                archive_path = data_exporter.create_comprehensive_archive(
                    team_data, players_df, None, recommendations
                )
                if archive_path:
                    st.success("✅ Archive created!")
                    st.markdown(create_download_link(archive_path, "📦 Download Archive"), 
                              unsafe_allow_html=True)
                else:
                    st.error("❌ Archive creation failed")
    
    with col3:
        if st.button("⚙️ Export Preferences"):
            with st.spinner("Exporting preferences..."):
                prefs_file = preference_manager.export_preferences()
                if prefs_file:
                    st.success("✅ Preferences exported!")
                    st.markdown(create_download_link(prefs_file, "⚙️ Download Preferences"), 
                              unsafe_allow_html=True)
                else:
                    st.error("❌ Preferences export failed")


def display_import_options():
    """Display import options in Streamlit interface"""
    st.subheader("📤 Import Data")
    
    uploaded_file = st.file_uploader(
        "Choose a file to import",
        type=['csv', 'xlsx', 'xls', 'json', 'zip'],
        help="Import team backups, player data, or preferences"
    )
    
    if uploaded_file is not None:
        file_content = uploaded_file.read()
        filename = uploaded_file.name
        
        if filename.endswith('.json') and 'preferences' in filename.lower():
            if st.button("Import Preferences"):
                if preference_manager.import_preferences(file_content):
                    st.success("✅ Preferences imported successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to import preferences")
        
        elif filename.endswith(('.csv', '.xlsx', '.xls', '.json', '.zip')):
            if st.button("Import Data"):
                with st.spinner("Importing data..."):
                    imported_data = data_importer.import_team_backup(file_content, filename)
                    if imported_data:
                        st.success("✅ Data imported successfully!")
                        st.json(imported_data.get('metadata', {}))
                    else:
                        st.error("❌ Failed to import data")


def display_preferences_manager():
    """Display user preferences management interface"""
    st.subheader("⚙️ User Preferences")
    
    # Team Settings
    st.write("**🏠 Team Settings**")
    col1, col2 = st.columns(2)
    
    with col1:
        team_id = st.number_input(
            "FPL Team ID",
            value=preference_manager.get_preference('team_id') or 0,
            min_value=0,
            help="Your FPL team ID from the official website"
        )
        if team_id != preference_manager.get_preference('team_id'):
            preference_manager.set_preference('team_id', team_id)
    
    with col2:
        formation = st.selectbox(
            "Preferred Formation",
            ['3-4-3', '3-5-2', '4-3-3', '4-4-2', '4-5-1', '5-3-2', '5-4-1'],
            index=['3-4-3', '3-5-2', '4-3-3', '4-4-2', '4-5-1', '5-3-2', '5-4-1'].index(
                preference_manager.get_preference('preferred_formation', '3-5-2')
            )
        )
        if formation != preference_manager.get_preference('preferred_formation'):
            preference_manager.set_preference('preferred_formation', formation)
    
    # Strategy Settings
    st.write("**📈 Strategy Settings**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_tolerance = st.selectbox(
            "Risk Tolerance",
            ['low', 'medium', 'high'],
            index=['low', 'medium', 'high'].index(
                preference_manager.get_preference('risk_tolerance', 'medium')
            )
        )
        if risk_tolerance != preference_manager.get_preference('risk_tolerance'):
            preference_manager.set_preference('risk_tolerance', risk_tolerance)
    
    with col2:
        budget_strategy = st.selectbox(
            "Budget Strategy",
            ['conservative', 'balanced', 'aggressive'],
            index=['conservative', 'balanced', 'aggressive'].index(
                preference_manager.get_preference('budget_strategy', 'balanced')
            )
        )
        if budget_strategy != preference_manager.get_preference('budget_strategy'):
            preference_manager.set_preference('budget_strategy', budget_strategy)
    
    with col3:
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ['basic', 'detailed', 'comprehensive'],
            index=['basic', 'detailed', 'comprehensive'].index(
                preference_manager.get_preference('analysis_depth', 'comprehensive')
            )
        )
        if analysis_depth != preference_manager.get_preference('analysis_depth'):
            preference_manager.set_preference('analysis_depth', analysis_depth)
    
    # Display Settings
    st.write("**🎨 Display Settings**")
    display_prefs = preference_manager.get_preference('display_preferences', {})
    
    col1, col2 = st.columns(2)
    with col1:
        show_advanced = st.checkbox(
            "Show Advanced Metrics",
            value=display_prefs.get('show_advanced_metrics', True)
        )
        highlight_diffs = st.checkbox(
            "Highlight Differentials",
            value=display_prefs.get('highlight_differentials', True)
        )
    
    with col2:
        show_expected = st.checkbox(
            "Show Expected Stats",
            value=display_prefs.get('show_expected_stats', True)
        )
        compact_tables = st.checkbox(
            "Compact Tables",
            value=display_prefs.get('compact_tables', False)
        )
    
    # Update display preferences
    new_display_prefs = {
        'show_advanced_metrics': show_advanced,
        'highlight_differentials': highlight_diffs,
        'show_expected_stats': show_expected,
        'compact_tables': compact_tables
    }
    
    if new_display_prefs != display_prefs:
        preference_manager.set_preference('display_preferences', new_display_prefs)
    
    # Recommendation Weights
    st.write("**🎯 Recommendation Weights**")
    st.write("Adjust how different factors influence player recommendations:")
    
    weights = preference_manager.get_preference('recommendation_weights', {})
    
    col1, col2 = st.columns(2)
    with col1:
        value_weight = st.slider(
            "Value Efficiency",
            0.0, 1.0, weights.get('value_efficiency', 0.25), 0.05
        )
        form_weight = st.slider(
            "Form Trend",
            0.0, 1.0, weights.get('form_trend', 0.20), 0.05
        )
    
    with col2:
        fixture_weight = st.slider(
            "Fixture Difficulty",
            0.0, 1.0, weights.get('fixture_difficulty', 0.15), 0.05
        )
        expected_weight = st.slider(
            "Expected Performance",
            0.0, 1.0, weights.get('expected_performance', 0.35), 0.05
        )
    
    new_weights = {
        'value_efficiency': value_weight,
        'form_trend': form_weight,
        'fixture_difficulty': fixture_weight,
        'expected_performance': expected_weight,
        'ownership_factor': 1.0 - (value_weight + form_weight + fixture_weight + expected_weight)
    }
    
    if new_weights != weights:
        preference_manager.set_preference('recommendation_weights', new_weights)
        st.success("✅ Recommendation weights updated!")
    
    # Show total
    total_weight = sum(new_weights.values())
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"⚠️ Weights total: {total_weight:.2f} (should be close to 1.0)")


# Helper functions for getting user preferences
def get_user_team_id() -> Optional[int]:
    """Get user's FPL team ID from preferences"""
    return preference_manager.get_preference('team_id')

def get_user_risk_tolerance() -> str:
    """Get user's risk tolerance preference"""
    return preference_manager.get_preference('risk_tolerance', 'medium')

def get_recommendation_weights() -> Dict[str, float]:
    """Get user's recommendation weights"""
    return preference_manager.get_preference('recommendation_weights', {
        'value_efficiency': 0.25,
        'form_trend': 0.20,
        'fixture_difficulty': 0.15,
        'ownership_factor': 0.05,
        'expected_performance': 0.35
    })

def should_show_advanced_metrics() -> bool:
    """Check if user wants to see advanced metrics"""
    return preference_manager.get_preference('display_preferences.show_advanced_metrics', True)