"""
UI Components for Automated Recommendation Implementation
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any

from services.automated_recommendation_implementation import (
    automated_implementor, 
    AutomatedRecommendationBatch, 
    AutomationRule
)
from services.ai_recommendation_engine import PlayerRecommendation


def render_automated_implementation_dashboard():
    """Render the main automated implementation dashboard"""
    st.header("🤖 Automated Recommendation Implementation")
    
    # Status overview
    status = automated_implementor.get_automation_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Total Batches", 
            status['total_batches'],
            delta=f"{status['recent_batches']} this week"
        )
    
    with col2:
        st.metric(
            "🔄 Active Rules", 
            status['active_rules']
        )
    
    with col3:
        st.metric(
            "⏳ Pending Batches", 
            status['pending_batches']
        )
    
    with col4:
        automation_status = "🟢 Enabled" if status['automation_enabled'] else "🔴 Disabled"
        st.metric("🤖 Automation", automation_status)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Generate Batch",
        "📋 Manage Batches", 
        "⚙️ Automation Rules",
        "📊 Performance Analytics",
        "🔧 Settings"
    ])
    
    with tab1:
        render_batch_generation_tab()
    
    with tab2:
        render_batch_management_tab()
    
    with tab3:
        render_automation_rules_tab()
    
    with tab4:
        render_performance_analytics_tab()
    
    with tab5:
        render_automation_settings_tab()


def render_batch_generation_tab():
    """Render batch generation interface"""
    st.subheader("🚀 Generate New Recommendation Batch")
    
    # Generation options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Batch Configuration")
        
        position_filter = st.selectbox(
            "Position Focus",
            ["All", "Goalkeeper", "Defender", "Midfielder", "Forward"],
            key="batch_position_filter"
        )
        
        budget_max = st.number_input(
            "Max Budget per Player (£m)",
            min_value=4.0,
            max_value=15.0,
            value=10.0,
            step=0.5,
            key="batch_budget_max"
        )
        
        recommendation_count = st.selectbox(
            "Number of Recommendations",
            [5, 10, 15, 20],
            index=1,
            key="batch_rec_count"
        )
    
    with col2:
        st.markdown("#### 🎯 Advanced Options")
        
        risk_preference = st.selectbox(
            "Risk Preference",
            ["Conservative", "Balanced", "Aggressive"],
            index=1,
            key="batch_risk_pref"
        )
        
        formation_preference = st.selectbox(
            "Preferred Formation",
            ["3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"],
            key="batch_formation"
        )
        
        auto_execute = st.checkbox(
            "Auto-execute if confidence > 80%",
            key="batch_auto_execute"
        )
    
    # Generate button
    if st.button("🚀 Generate Automated Batch", type="primary", key="generate_batch_btn"):
        with st.spinner("🤖 Generating AI-enhanced recommendations..."):
            try:
                user_context = {
                    'position_filter': None if position_filter == "All" else position_filter,
                    'budget_max': budget_max,
                    'recommendation_count': recommendation_count,
                    'risk_preference': risk_preference.lower(),
                    'formation_preference': formation_preference
                }
                
                batch = automated_implementor.generate_automated_batch(user_context)
                
                st.success(f"✅ Generated batch {batch.batch_id} with {len(batch.recommendations)} recommendations!")
                st.info(f"📊 Batch confidence score: {batch.confidence_score:.1%}")
                
                # Auto-execute if requested and confidence is high
                if auto_execute and batch.confidence_score >= 0.8:
                    with st.spinner("🔄 Auto-executing high-confidence batch..."):
                        execution_result = automated_implementor.execute_batch(batch.batch_id)
                        
                        if execution_result['success']:
                            st.success(f"🎯 Auto-executed {execution_result['executed_count']} recommendations!")
                        else:
                            st.error(f"❌ Auto-execution failed: {execution_result.get('error', 'Unknown error')}")
                
                # Display batch preview
                render_batch_preview(batch)
                
            except Exception as e:
                st.error(f"❌ Error generating batch: {str(e)}")
                st.info("💡 Try loading fresh data or adjusting your parameters.")


def render_batch_management_tab():
    """Render batch management interface"""
    st.subheader("📋 Recommendation Batch Management")
    
    batches = automated_implementor.recommendation_batches
    
    if not batches:
        st.info("📝 No recommendation batches found. Generate your first batch in the 'Generate Batch' tab.")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "pending", "executed", "failed"],
            key="batch_status_filter"
        )
    
    with col2:
        days_back = st.selectbox(
            "Show batches from",
            [7, 14, 30, 90],
            index=1,
            key="batch_days_filter"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Generation Time", "Confidence Score", "Recommendation Count"],
            key="batch_sort"
        )
    
    # Filter batches
    filtered_batches = batches
    
    if status_filter != "All":
        filtered_batches = [b for b in filtered_batches if b.status == status_filter]
    
    cutoff_date = datetime.now() - timedelta(days=days_back)
    filtered_batches = [b for b in filtered_batches if b.generation_time > cutoff_date]
    
    # Sort batches
    if sort_by == "Generation Time":
        filtered_batches.sort(key=lambda x: x.generation_time, reverse=True)
    elif sort_by == "Confidence Score":
        filtered_batches.sort(key=lambda x: x.confidence_score, reverse=True)
    else:  # Recommendation Count
        filtered_batches.sort(key=lambda x: len(x.recommendations), reverse=True)
    
    # Display batches
    for batch in filtered_batches:
        render_batch_card(batch)


def render_batch_card(batch: AutomatedRecommendationBatch):
    """Render a card for a single batch"""
    status_colors = {
        'pending': '🟡',
        'executed': '🟢', 
        'failed': '🔴',
        'approved': '🔵'
    }
    
    status_emoji = status_colors.get(batch.status, '⚪')
    
    with st.expander(f"{status_emoji} Batch {batch.batch_id} - {batch.status.title()} ({len(batch.recommendations)} recs)"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Batch Details:**")
            st.write(f"• **Generated:** {batch.generation_time.strftime('%Y-%m-%d %H:%M')}")
            st.write(f"• **Confidence:** {batch.confidence_score:.1%}")
            st.write(f"• **Status:** {batch.status.title()}")
            if batch.execution_time:
                st.write(f"• **Executed:** {batch.execution_time.strftime('%Y-%m-%d %H:%M')}")
        
        with col2:
            st.markdown("**🎯 Context:**")
            context = batch.user_context
            st.write(f"• **Position:** {context.get('position_filter', 'All')}")
            st.write(f"• **Budget:** £{context.get('budget_max', 0):.1f}m max")
            st.write(f"• **Risk:** {context.get('risk_preference', 'balanced').title()}")
        
        # Batch actions
        action_col1, action_col2, action_col3, action_col4 = st.columns(4)
        
        with action_col1:
            if st.button(f"👁️ View Details", key=f"view_{batch.batch_id}"):
                render_batch_details_modal(batch)
        
        with action_col2:
            if batch.status == 'pending' and st.button(f"✅ Validate", key=f"validate_{batch.batch_id}"):
                validation = automated_implementor.validate_batch(batch.batch_id)
                if validation['valid']:
                    st.success("✅ Batch validation passed!")
                else:
                    st.error("❌ Validation failed:")
                    for error in validation['errors']:
                        st.write(f"• {error}")
        
        with action_col3:
            if batch.status == 'pending' and st.button(f"🚀 Execute", key=f"execute_{batch.batch_id}"):
                with st.spinner("Executing batch..."):
                    result = automated_implementor.execute_batch(batch.batch_id)
                    if result['success']:
                        st.success(f"✅ Executed {result['executed_count']} recommendations!")
                        st.rerun()
                    else:
                        st.error(f"❌ Execution failed: {result.get('error', 'Unknown error')}")
        
        with action_col4:
            if st.button(f"📊 Analytics", key=f"analytics_{batch.batch_id}"):
                render_batch_analytics(batch)


def render_batch_preview(batch: AutomatedRecommendationBatch):
    """Render a preview of batch recommendations"""
    st.markdown("### 👁️ Batch Preview")
    
    # Top recommendations
    top_recs = batch.recommendations[:5]
    
    for i, rec in enumerate(top_recs, 1):
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
            
            with col1:
                st.write(f"**{i}. {rec.web_name}** ({rec.position})")
                st.write(f"🏠 {rec.team_name}")
            
            with col2:
                st.metric("Price", f"£{rec.current_price:.1f}m")
            
            with col3:
                st.metric("Predicted Points", f"{rec.predicted_points:.1f}")
            
            with col4:
                confidence_color = "🟢" if rec.confidence_score > 0.8 else "🟡" if rec.confidence_score > 0.6 else "🔴"
                st.metric("Confidence", f"{confidence_color} {rec.confidence_score:.1%}")
            
            # Reasoning
            if rec.reasoning:
                with st.expander("🧠 AI Reasoning"):
                    for reason in rec.reasoning:
                        st.write(f"• {reason}")


def render_automation_rules_tab():
    """Render automation rules management"""
    st.subheader("⚙️ Automation Rules")
    
    # Create new rule section
    with st.expander("➕ Create New Automation Rule"):
        render_rule_creation_form()
    
    # Existing rules
    rules = automated_implementor.automation_rules
    
    if not rules:
        st.info("📝 No automation rules configured. Create your first rule above.")
        return
    
    st.markdown("### 📋 Existing Rules")
    
    for rule in rules:
        render_rule_card(rule)


def render_rule_creation_form():
    """Render form for creating automation rules"""
    col1, col2 = st.columns(2)
    
    with col1:
        rule_name = st.text_input("Rule Name", key="new_rule_name")
        trigger_condition = st.selectbox(
            "Trigger Condition",
            ["weekly", "gameweek_start", "price_change", "injury_news"],
            key="new_rule_trigger"
        )
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.1,
            key="new_rule_confidence"
        )
    
    with col2:
        max_transfers = st.number_input(
            "Max Transfers per Execution",
            min_value=1,
            max_value=10,
            value=3,
            key="new_rule_transfers"
        )
        auto_execute = st.checkbox(
            "Enable Auto-execution",
            key="new_rule_auto_execute"
        )
        
        # Rule filters
        st.markdown("**Filters:**")
        filter_position = st.selectbox(
            "Position Filter",
            ["All", "Goalkeeper", "Defender", "Midfielder", "Forward"],
            key="new_rule_position"
        )
    
    if st.button("➕ Create Rule", type="primary", key="create_rule_btn"):
        if rule_name:
            filters = {
                'position_filter': None if filter_position == "All" else filter_position
            }
            
            rule_id = automated_implementor.create_automation_rule(
                name=rule_name,
                trigger_condition=trigger_condition,
                min_confidence=min_confidence,
                max_transfers=max_transfers,
                auto_execute=auto_execute,
                filters=filters
            )
            
            st.success(f"✅ Created automation rule: {rule_name}")
            st.rerun()
        else:
            st.error("❌ Please provide a rule name")


def render_rule_card(rule: AutomationRule):
    """Render a card for an automation rule"""
    status_emoji = "🟢" if rule.auto_execute else "🟡"
    
    with st.expander(f"{status_emoji} {rule.name} - {rule.trigger_condition}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Trigger:** {rule.trigger_condition}")
            st.write(f"**Min Confidence:** {rule.min_confidence:.1%}")
            st.write(f"**Max Transfers:** {rule.max_transfers}")
        
        with col2:
            auto_status = "✅ Enabled" if rule.auto_execute else "⏸️ Manual"
            st.write(f"**Auto-execute:** {auto_status}")
            st.write(f"**Notifications:** {'🔔 On' if rule.notification_enabled else '🔕 Off'}")
            
            if rule.filters:
                st.write(f"**Filters:** {rule.filters}")
        
        # Rule actions
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button(f"🚀 Execute Now", key=f"execute_rule_{rule.rule_id}"):
                with st.spinner("Executing rule..."):
                    try:
                        batch = automated_implementor.generate_automated_batch(rule_id=rule.rule_id)
                        st.success(f"✅ Generated batch with {len(batch.recommendations)} recommendations")
                    except Exception as e:
                        st.error(f"❌ Error executing rule: {str(e)}")
        
        with action_col2:
            if st.button(f"🗑️ Delete Rule", key=f"delete_rule_{rule.rule_id}"):
                # In real implementation, would add confirmation dialog
                st.warning("⚠️ Rule deletion not implemented in demo")


def render_performance_analytics_tab():
    """Render performance analytics dashboard"""
    st.subheader("📊 Performance Analytics")
    
    batches = automated_implementor.recommendation_batches
    execution_log = automated_implementor.execution_log
    
    if not batches:
        st.info("📈 No data available for analytics. Generate some batches first.")
        return
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    executed_batches = [b for b in batches if b.status == 'executed']
    pending_batches = [b for b in batches if b.status == 'pending']
    
    with col1:
        success_rate = len(executed_batches) / len(batches) * 100 if batches else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col2:
        avg_confidence = np.mean([b.confidence_score for b in batches]) if batches else 0
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col3:
        total_recommendations = sum(len(b.recommendations) for b in executed_batches)
        st.metric("Total Executed", total_recommendations)
    
    with col4:
        st.metric("Pending Batches", len(pending_batches))
    
    # Charts
    if executed_batches:
        render_performance_charts(executed_batches, execution_log)


def render_performance_charts(batches: List[AutomatedRecommendationBatch], 
                            execution_log: List[Dict]):
    """Render performance visualization charts"""
    
    # Confidence score distribution
    st.markdown("### 📈 Confidence Score Distribution")
    
    confidence_scores = [b.confidence_score for b in batches]
    
    fig_hist = px.histogram(
        x=confidence_scores,
        nbins=20,
        title="Batch Confidence Score Distribution",
        labels={'x': 'Confidence Score', 'y': 'Number of Batches'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Execution timeline
    st.markdown("### ⏱️ Execution Timeline")
    
    timeline_data = []
    for batch in batches:
        if batch.execution_time:
            timeline_data.append({
                'date': batch.execution_time.date(),
                'recommendations': len(batch.recommendations),
                'confidence': batch.confidence_score,
                'batch_id': batch.batch_id
            })
    
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        
        fig_timeline = px.scatter(
            timeline_df,
            x='date',
            y='recommendations',
            size='confidence',
            hover_data=['batch_id'],
            title="Recommendation Execution Timeline"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)


def render_automation_settings_tab():
    """Render automation settings"""
    st.subheader("🔧 Automation Settings")
    
    status = automated_implementor.get_automation_status()
    
    # Automation toggle
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🤖 Automation Control")
        
        current_status = status['automation_enabled']
        
        if st.button("🟢 Enable Automation" if not current_status else "🔴 Disable Automation", 
                    type="primary"):
            if current_status:
                automated_implementor.disable_automation()
                st.success("🔴 Automation disabled")
            else:
                automated_implementor.enable_automation()
                st.success("🟢 Automation enabled")
            st.rerun()
        
        st.write(f"**Current Status:** {'🟢 Enabled' if current_status else '🔴 Disabled'}")
    
    with col2:
        st.markdown("#### 📊 System Statistics")
        st.write(f"**Total Batches:** {status['total_batches']}")
        st.write(f"**Recent Batches:** {status['recent_batches']}")
        st.write(f"**Active Rules:** {status['active_rules']}")
        st.write(f"**Pending Batches:** {status['pending_batches']}")
    
    # Export/Import data
    st.markdown("---")
    st.markdown("#### 💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Learning Data"):
            export_file = automated_implementor.export_learning_data()
            if export_file:
                st.success(f"✅ Data exported to: {export_file}")
            else:
                st.error("❌ Export failed")
    
    with col2:
        if st.button("🗑️ Clear All Data"):
            st.warning("⚠️ This would clear all automation data (not implemented in demo)")


def render_batch_details_modal(batch: AutomatedRecommendationBatch):
    """Render detailed view of a batch"""
    st.markdown(f"### 📋 Batch Details: {batch.batch_id}")
    
    # Batch info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Batch Information:**")
        st.write(f"• **ID:** {batch.batch_id}")
        st.write(f"• **Generated:** {batch.generation_time}")
        st.write(f"• **Status:** {batch.status}")
        st.write(f"• **Confidence:** {batch.confidence_score:.1%}")
    
    with col2:
        st.markdown("**🎯 Context:**")
        for key, value in batch.user_context.items():
            st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
    
    # Recommendations table
    st.markdown("**📋 Recommendations:**")
    
    rec_data = []
    for rec in batch.recommendations:
        rec_data.append({
            'Player': rec.web_name,
            'Team': rec.team_name,
            'Position': rec.position,
            'Price': f"£{rec.current_price:.1f}m",
            'Predicted Points': f"{rec.predicted_points:.1f}",
            'Confidence': f"{rec.confidence_score:.1%}",
            'Risk Level': rec.risk_level
        })
    
    rec_df = pd.DataFrame(rec_data)
    st.dataframe(rec_df, use_container_width=True)
    
    # Results if executed
    if batch.results:
        st.markdown("**📈 Execution Results:**")
        results = batch.results
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Executed", results['executed_count'])
        with col2:
            st.metric("Failed", results['failed_count'])
        with col3:
            success_rate = results['executed_count'] / (results['executed_count'] + results['failed_count']) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")


def render_batch_analytics(batch: AutomatedRecommendationBatch):
    """Render analytics for a specific batch"""
    st.markdown(f"### 📊 Analytics: {batch.batch_id}")
    
    # Recommendation distribution
    positions = [rec.position for rec in batch.recommendations]
    position_counts = pd.Series(positions).value_counts()
    
    fig_pie = px.pie(
        values=position_counts.values,
        names=position_counts.index,
        title="Recommendations by Position"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Price distribution
    prices = [rec.current_price for rec in batch.recommendations]
    
    fig_box = px.box(
        y=prices,
        title="Price Distribution of Recommendations"
    )
    st.plotly_chart(fig_box, use_container_width=True)