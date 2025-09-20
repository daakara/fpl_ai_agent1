"""
Automated Recommendation Implementation Service
Handles the complete automation of recommendation generation, validation, and execution
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
import logging
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Optional imports with fallbacks
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    schedule = None

from services.ai_recommendation_engine import PlayerRecommendation, get_player_recommendations
from services.automated_iteration_service import iteration_engine, get_automated_recommendations
from services.fpl_data_service import FPLDataService
from services.transfer_planning_service import TransferPlanningService
from services.fixture_service import FixtureService
from services.personalization_service import get_personalized_suggestions
from utils.error_handling import handle_errors, logger


@dataclass
class AutomatedRecommendationBatch:
    """Represents a batch of automated recommendations"""
    batch_id: str
    recommendations: List[PlayerRecommendation]
    generation_time: datetime
    user_context: Dict[str, Any]
    confidence_score: float
    status: str  # 'pending', 'approved', 'rejected', 'executed'
    execution_time: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None


@dataclass
class AutomationRule:
    """Defines rules for automated recommendation execution"""
    rule_id: str
    name: str
    trigger_condition: str  # 'weekly', 'gameweek_start', 'price_change', 'injury_news'
    min_confidence: float
    max_transfers: int
    auto_execute: bool
    notification_enabled: bool
    filters: Dict[str, Any]


class AutomatedRecommendationImplementor:
    """Handles automated implementation of recommendations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_service = FPLDataService()
        self.transfer_service = TransferPlanningService()
        self.fixture_service = FixtureService()
        
        # Storage
        self.automation_dir = Path("automation_data")
        self.automation_dir.mkdir(exist_ok=True)
        
        self.batches_file = self.automation_dir / "recommendation_batches.json"
        self.rules_file = self.automation_dir / "automation_rules.json"
        self.execution_log_file = self.automation_dir / "execution_log.json"
        
        # Runtime data
        self.recommendation_batches: List[AutomatedRecommendationBatch] = []
        self.automation_rules: List[AutomationRule] = []
        self.execution_log: List[Dict] = []
        
        # Automation settings
        self.automation_enabled = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self._load_automation_data()
        self._setup_scheduling()
    
    def _load_automation_data(self):
        """Load automation data from files"""
        try:
            if self.batches_file.exists():
                with open(self.batches_file, 'r') as f:
                    batches_data = json.load(f)
                    self.recommendation_batches = [
                        AutomatedRecommendationBatch(**batch) for batch in batches_data
                    ]
            
            if self.rules_file.exists():
                with open(self.rules_file, 'r') as f:
                    rules_data = json.load(f)
                    self.automation_rules = [
                        AutomationRule(**rule) for rule in rules_data
                    ]
            
            if self.execution_log_file.exists():
                with open(self.execution_log_file, 'r') as f:
                    self.execution_log = json.load(f)
        
        except Exception as e:
            self.logger.error(f"Error loading automation data: {e}")
    
    def _save_automation_data(self):
        """Save automation data to files"""
        try:
            # Save batches
            batches_data = [asdict(batch) for batch in self.recommendation_batches]
            for batch in batches_data:
                if isinstance(batch.get('generation_time'), datetime):
                    batch['generation_time'] = batch['generation_time'].isoformat()
                if isinstance(batch.get('execution_time'), datetime):
                    batch['execution_time'] = batch['execution_time'].isoformat()
            
            with open(self.batches_file, 'w') as f:
                json.dump(batches_data, f, indent=2, default=str)
            
            # Save rules
            rules_data = [asdict(rule) for rule in self.automation_rules]
            with open(self.rules_file, 'w') as f:
                json.dump(rules_data, f, indent=2)
            
            # Save execution log
            with open(self.execution_log_file, 'w') as f:
                json.dump(self.execution_log, f, indent=2, default=str)
        
        except Exception as e:
            self.logger.error(f"Error saving automation data: {e}")
    
    def _setup_scheduling(self):
        """Setup automated scheduling"""
        if not SCHEDULE_AVAILABLE:
            self.logger.warning("Schedule module not available. Automated scheduling disabled.")
            return
        
        try:
            # Schedule weekly recommendation generation
            schedule.every().monday.at("08:00").do(self._generate_weekly_recommendations)
            
            # Schedule gameweek start checks
            schedule.every().tuesday.at("18:00").do(self._check_gameweek_recommendations)
            
            # Schedule price change monitoring
            schedule.every().day.at("01:30").do(self._monitor_price_changes)
            
            # Start scheduler in background thread
            scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            scheduler_thread.start()
            
            self.logger.info("Automated scheduling initialized successfully")
        except Exception as e:
            self.logger.error(f"Error setting up scheduling: {e}")
    
    def _run_scheduler(self):
        """Run the scheduler in background"""
        if not SCHEDULE_AVAILABLE:
            return
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except Exception as e:
            self.logger.error(f"Error in scheduler loop: {e}")
    
    @handle_errors("Error generating automated recommendations")
    def generate_automated_batch(self, user_context: Dict = None, 
                                rule_id: str = None) -> AutomatedRecommendationBatch:
        """Generate a new batch of automated recommendations"""
        try:
            # Load fresh data
            players_df, teams_df = self.data_service.load_fpl_data()
            
            if players_df.empty:
                raise ValueError("No player data available")
            
            # Get user context or use defaults
            if not user_context:
                user_context = self._get_default_context()
            
            # Apply rule filters if specified
            if rule_id:
                rule = next((r for r in self.automation_rules if r.rule_id == rule_id), None)
                if rule:
                    user_context.update(rule.filters)
            
            # Generate recommendations using the automated iteration service
            recommendations = get_automated_recommendations(players_df, user_context)
            
            if not recommendations:
                raise ValueError("No recommendations generated")
            
            # Calculate batch confidence score
            confidence_score = np.mean([rec.confidence_score for rec in recommendations])
            
            # Create batch
            batch = AutomatedRecommendationBatch(
                batch_id=f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                recommendations=recommendations,
                generation_time=datetime.now(),
                user_context=user_context,
                confidence_score=confidence_score,
                status='pending'
            )
            
            # Add to batches
            self.recommendation_batches.append(batch)
            self._save_automation_data()
            
            # Log generation
            self._log_execution(
                action='batch_generated',
                batch_id=batch.batch_id,
                details={
                    'recommendation_count': len(recommendations),
                    'confidence_score': confidence_score,
                    'context': user_context
                }
            )
            
            self.logger.info(f"Generated automated batch {batch.batch_id} with {len(recommendations)} recommendations")
            return batch
        
        except Exception as e:
            self.logger.error(f"Error generating automated batch: {e}")
            raise
    
    def validate_batch(self, batch_id: str) -> Dict[str, Any]:
        """Validate a recommendation batch before execution"""
        try:
            batch = self._get_batch(batch_id)
            if not batch:
                return {'valid': False, 'errors': ['Batch not found']}
            
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'recommendations_validated': 0
            }
            
            # Load current data for validation
            players_df, _ = self.data_service.load_fpl_data()
            
            for rec in batch.recommendations:
                # Check if player is still available
                player_data = players_df[players_df['id'] == rec.player_id]
                if player_data.empty:
                    validation_results['errors'].append(f"Player {rec.web_name} not found")
                    continue
                
                player = player_data.iloc[0]
                
                # Check price changes
                current_price = player.get('now_cost', 0) / 10.0
                if abs(current_price - rec.current_price) > 0.1:
                    validation_results['warnings'].append(
                        f"{rec.web_name} price changed: £{rec.current_price:.1f}m → £{current_price:.1f}m"
                    )
                
                # Check availability status
                if player.get('status') != 'a':
                    validation_results['errors'].append(f"{rec.web_name} is not available")
                
                # Check if player is suspended/injured
                news = player.get('news', '').lower()
                if any(word in news for word in ['injured', 'suspended', 'doubt']):
                    validation_results['warnings'].append(f"{rec.web_name} has injury/suspension news")
                
                validation_results['recommendations_validated'] += 1
            
            # Overall validation
            if validation_results['errors']:
                validation_results['valid'] = False
            
            return validation_results
        
        except Exception as e:
            self.logger.error(f"Error validating batch {batch_id}: {e}")
            return {'valid': False, 'errors': [f"Validation error: {str(e)}"]}
    
    def execute_batch(self, batch_id: str, force: bool = False) -> Dict[str, Any]:
        """Execute a validated recommendation batch"""
        try:
            batch = self._get_batch(batch_id)
            if not batch:
                return {'success': False, 'error': 'Batch not found'}
            
            # Validate first unless forced
            if not force:
                validation = self.validate_batch(batch_id)
                if not validation['valid']:
                    return {
                        'success': False, 
                        'error': 'Batch validation failed',
                        'validation_errors': validation['errors']
                    }
            
            # Execute recommendations
            execution_results = {
                'success': True,
                'executed_count': 0,
                'failed_count': 0,
                'recommendations': [],
                'errors': []
            }
            
            for rec in batch.recommendations:
                try:
                    # Simulate transfer execution (in real implementation, this would call FPL API)
                    execution_result = self._execute_recommendation(rec)
                    
                    execution_results['recommendations'].append({
                        'player': rec.web_name,
                        'action': 'transfer_in',
                        'success': execution_result['success'],
                        'message': execution_result.get('message', '')
                    })
                    
                    if execution_result['success']:
                        execution_results['executed_count'] += 1
                    else:
                        execution_results['failed_count'] += 1
                        execution_results['errors'].append(
                            f"Failed to execute {rec.web_name}: {execution_result.get('error', 'Unknown error')}"
                        )
                
                except Exception as e:
                    execution_results['failed_count'] += 1
                    execution_results['errors'].append(f"Error executing {rec.web_name}: {str(e)}")
            
            # Update batch status
            batch.status = 'executed' if execution_results['executed_count'] > 0 else 'failed'
            batch.execution_time = datetime.now()
            batch.results = execution_results
            
            self._save_automation_data()
            
            # Log execution
            self._log_execution(
                action='batch_executed',
                batch_id=batch_id,
                details=execution_results
            )
            
            return execution_results
        
        except Exception as e:
            self.logger.error(f"Error executing batch {batch_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_recommendation(self, recommendation: PlayerRecommendation) -> Dict[str, Any]:
        """Execute a single recommendation (placeholder for actual FPL API integration)"""
        # This is a placeholder - in real implementation, this would:
        # 1. Check team composition
        # 2. Make necessary transfers
        # 3. Handle transfer costs
        # 4. Update team
        
        # For now, just simulate success
        return {
            'success': True,
            'message': f"Successfully added {recommendation.web_name} to watchlist",
            'action_type': 'watchlist_add'  # In real app: 'transfer_in', 'transfer_out', etc.
        }
    
    def create_automation_rule(self, name: str, trigger_condition: str, 
                              min_confidence: float, max_transfers: int,
                              auto_execute: bool = False, filters: Dict = None) -> str:
        """Create a new automation rule"""
        rule_id = f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        rule = AutomationRule(
            rule_id=rule_id,
            name=name,
            trigger_condition=trigger_condition,
            min_confidence=min_confidence,
            max_transfers=max_transfers,
            auto_execute=auto_execute,
            notification_enabled=True,
            filters=filters or {}
        )
        
        self.automation_rules.append(rule)
        self._save_automation_data()
        
        self.logger.info(f"Created automation rule: {name}")
        return rule_id
    
    def _generate_weekly_recommendations(self):
        """Scheduled weekly recommendation generation"""
        if not self.automation_enabled:
            return
        
        try:
            self.logger.info("Generating weekly automated recommendations")
            
            # Check for weekly automation rules
            weekly_rules = [r for r in self.automation_rules if r.trigger_condition == 'weekly']
            
            for rule in weekly_rules:
                batch = self.generate_automated_batch(rule_id=rule.rule_id)
                
                # Auto-execute if rule allows and confidence is high enough
                if rule.auto_execute and batch.confidence_score >= rule.min_confidence:
                    self.execute_batch(batch.batch_id)
        
        except Exception as e:
            self.logger.error(f"Error in weekly recommendation generation: {e}")
    
    def _check_gameweek_recommendations(self):
        """Check for gameweek-specific recommendations"""
        if not self.automation_enabled:
            return
        
        try:
            # Check for pending batches that should be executed
            pending_batches = [b for b in self.recommendation_batches if b.status == 'pending']
            
            for batch in pending_batches:
                # Check if batch meets execution criteria
                if batch.confidence_score >= 0.8:  # High confidence threshold
                    validation = self.validate_batch(batch.batch_id)
                    if validation['valid']:
                        self.execute_batch(batch.batch_id)
        
        except Exception as e:
            self.logger.error(f"Error in gameweek recommendation check: {e}")
    
    def _monitor_price_changes(self):
        """Monitor for significant price changes"""
        if not self.automation_enabled:
            return
        
        try:
            # This would monitor for price rises/falls and trigger recommendations
            # Implementation would check price change data and generate reactive recommendations
            self.logger.info("Monitoring price changes for reactive recommendations")
        
        except Exception as e:
            self.logger.error(f"Error in price change monitoring: {e}")
    
    def _get_batch(self, batch_id: str) -> Optional[AutomatedRecommendationBatch]:
        """Get batch by ID"""
        return next((b for b in self.recommendation_batches if b.batch_id == batch_id), None)
    
    def _get_default_context(self) -> Dict[str, Any]:
        """Get default user context for recommendations"""
        return {
            'position_filter': None,
            'budget_max': 12.0,
            'recommendation_count': 10,
            'risk_preference': 'medium',
            'formation_preference': '3-4-3'
        }
    
    def _log_execution(self, action: str, batch_id: str, details: Dict = None):
        """Log automation execution"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'batch_id': batch_id,
            'details': details or {}
        }
        
        self.execution_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.execution_log) > 1000:
            self.execution_log = self.execution_log[-1000:]
    
    def get_automation_status(self) -> Dict[str, Any]:
        """Get current automation status"""
        recent_batches = [b for b in self.recommendation_batches 
                         if b.generation_time > datetime.now() - timedelta(days=7)]
        
        return {
            'automation_enabled': self.automation_enabled,
            'total_batches': len(self.recommendation_batches),
            'recent_batches': len(recent_batches),
            'active_rules': len([r for r in self.automation_rules if r.auto_execute]),
            'pending_batches': len([b for b in self.recommendation_batches if b.status == 'pending']),
            'last_execution': max([b.execution_time for b in self.recommendation_batches 
                                 if b.execution_time], default=None)
        }
    
    def enable_automation(self):
        """Enable automated recommendation system"""
        self.automation_enabled = True
        self.logger.info("Automated recommendation system enabled")
    
    def disable_automation(self):
        """Disable automated recommendation system"""
        self.automation_enabled = False
        self.logger.info("Automated recommendation system disabled")


# Global instance
automated_implementor = AutomatedRecommendationImplementor()