"""
UI Components for Automated Iteration Recommendation System
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any

from services.automated_iteration_service import (
    get_automated_recommendations,
    collect_recommendation_feedback,
    track_prediction_accuracy,
    get_learning_insights,
    iteration_engine
)
from services.ai_recommendation_engine import PlayerRecommendation


def render_automated_recommendations_tab(players_df: pd.DataFrame):
    """Render the automated iteration recommendations tab"""
    st.header("🤖 AI-Enhanced Recommendations")
    st.markdown("*Continuously improving recommendations based on your feedback and performance*")
    
    # Learning status indicator
    insights = get_learning_insights()
    if insights.get('status') == 'No data available yet':
        st.info("🌱 **Learning Mode**: The AI is in initial learning phase. Your feedback will help improve recommendations!")
    elif insights.get('learning_status') == 'Active':
        st.success(f"🧠 **Active Learning**: AI has processed {insights.get('total_feedback_entries', 0)} feedback entries")
    else:
        st.warning("📊 **Collecting Data**: Building your preference profile...")
    
    # User context inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        position_filter = st.selectbox(
            "Position Focus",
            ["All", "Goalkeeper", "Defender", "Midfielder", "Forward"],
            key="auto_position_filter"
        )
    
    with col2:
        budget_max = st.number_input(
            "Max Budget (£m)",
            min_value=4.0,
            max_value=15.0,
            value=10.0,
            step=0.5,
            key="auto_budget_max"
        )
    
    with col3:
        recommendation_count = st.selectbox(
            "Number of Recommendations",
            [5, 10, 15],
            index=1,
            key="auto_rec_count"
        )
    
    # Generate recommendations button
    if st.button("🚀 Generate AI-Enhanced Recommendations", type="primary"):
        with st.spinner("🤖 AI is analyzing patterns and generating personalized recommendations..."):
            user_context = {
                'position_filter': None if position_filter == "All" else position_filter,
                'budget_max': budget_max,
                'recommendation_count': recommendation_count,
                'timestamp': datetime.now().isoformat()
            }
            
            recommendations = get_automated_recommendations(players_df, user_context)
            
            if recommendations:
                st.session_state['auto_recommendations'] = recommendations
                st.session_state['auto_context'] = user_context
                st.success(f"✨ Generated {len(recommendations)} AI-enhanced recommendations!")
            else:
                st.error("❌ Unable to generate recommendations. Please check your filters and try again.")
    
    # Display recommendations if available
    if 'auto_recommendations' in st.session_state:
        render_recommendation_results(st.session_state['auto_recommendations'])
    
    # Learning insights section
    st.markdown("---")
    render_learning_insights()


def render_recommendation_results(recommendations: List[PlayerRecommendation]):
    """Render the recommendation results with feedback collection"""
    st.subheader("🎯 AI-Enhanced Recommendations")
    
    for i, rec in enumerate(recommendations, 1):
        with st.expander(f"#{i} {rec.web_name} ({rec.team_name}) - £{rec.current_price:.1f}m", expanded=i <= 3):
            # Main recommendation info
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Position:** {rec.position}")
                st.markdown(f"**Predicted Points:** {rec.predicted_points:.1f}")
                st.markdown(f"**Confidence:** {rec.confidence_score:.1%}")
                
                # AI reasoning
                if rec.reasoning:
                    st.markdown("**🧠 AI Reasoning:**")
                    for reason in rec.reasoning:
                        st.markdown(f"• {reason}")
            
            with col2:
                st.metric("Value Score", f"{rec.value_score:.1f}")
                st.metric("Form Score", f"{rec.form_score:.1f}")
                st.metric("Risk Level", rec.risk_level)
            
            with col3:
                st.metric("Fixture Score", f"{rec.fixture_score:.1f}")
                st.metric("Ownership Score", f"{rec.ownership_score:.1f}")
                st.metric("Expected ROI", f"{rec.expected_roi:.1f}%")
            
            # Feedback collection
            st.markdown("---")
            render_feedback_widget(rec, i)


def render_feedback_widget(recommendation: PlayerRecommendation, index: int):
    """Render feedback collection widget for a recommendation"""
    st.markdown("**📝 Help Improve AI Recommendations:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feedback_score = st.slider(
            "Rate this recommendation",
            min_value=1,
            max_value=5,
            value=3,
            help="1=Poor, 5=Excellent",
            key=f"feedback_score_{index}_{recommendation.player_id}"
        )
    
    with col2:
        user_action = st.selectbox(
            "Your action",
            ["Not sure yet", "Will consider", "Planning to transfer in", "Already own", "Not interested"],
            key=f"user_action_{index}_{recommendation.player_id}"
        )
    
    with col3:
        if st.button(f"💬 Submit Feedback", key=f"submit_feedback_{index}_{recommendation.player_id}"):
            # Determine feedback type based on score
            if feedback_score >= 4:
                feedback_type = "positive"
            elif feedback_score <= 2:
                feedback_type = "negative"
            else:
                feedback_type = "neutral"
            
            # Collect feedback
            feedback_id = collect_recommendation_feedback(
                recommendation=recommendation,
                feedback_type=feedback_type,
                feedback_score=feedback_score,
                user_action=user_action,
                context={
                    'recommendation_index': index,
                    'total_recommendations': len(st.session_state.get('auto_recommendations', [])),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            if feedback_id:
                st.success("✅ Thank you! Your feedback helps improve future recommendations.")
            else:
                st.error("❌ Failed to save feedback. Please try again.")


def render_learning_insights():
    """Render AI learning insights and performance metrics"""
    st.subheader("📊 AI Learning Insights")
    
    insights = get_learning_insights()
    
    if insights.get('status') in ['No data available yet', 'Error generating insights']:
        st.info(f"📈 {insights.get('status', 'No learning data available yet')}")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = insights.get('recommendation_accuracy', 0)
        accuracy_trend = insights.get('accuracy_trend', 0)
        delta_color = "normal" if accuracy_trend >= 0 else "inverse"
        st.metric(
            "Prediction Accuracy",
            f"{accuracy:.1%}",
            delta=f"{accuracy_trend:+.1%}" if accuracy_trend != 0 else None,
            delta_color=delta_color
        )
    
    with col2:
        conversion = insights.get('user_conversion_rate', 0)
        st.metric("User Adoption Rate", f"{conversion:.1%}")
    
    with col3:
        satisfaction = insights.get('user_satisfaction_rate', 0)
        st.metric("User Satisfaction", f"{satisfaction:.1%}")
    
    with col4:
        total_feedback = insights.get('total_feedback_entries', 0)
        st.metric("Total Feedback", total_feedback)
    
    # Learning status and weights
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Current AI Weights:**")
        weights = insights.get('current_weights', {})
        for factor, weight in weights.items():
            st.progress(weight, text=f"{factor.replace('_', ' ').title()}: {weight:.1%}")
    
    with col2:
        st.markdown(f"**🔄 Learning Status:** {insights.get('learning_status', 'Unknown')}")
        st.markdown(f"**📈 Recommendations Generated:** {insights.get('recommendations_count', 0)}")
        
        if insights.get('learning_status') == 'Active':
            st.success("🎯 AI is actively learning from your feedback!")
        elif total_feedback < 5:
            st.info("🌱 Provide more feedback to activate advanced learning features")
    
    # Performance visualization
    if iteration_engine.performance_metrics['recommendation_accuracy']:
        render_performance_charts()
    
    # Export learning data
    st.markdown("---")
    if st.button("📊 Export Learning Data"):
        export_file = iteration_engine.export_learning_data()
        if export_file:
            st.success(f"✅ Learning data exported to: {export_file}")
            with open(export_file, 'r') as f:
                st.download_button(
                    label="⬇️ Download Learning Data",
                    data=f.read(),
                    file_name=f"fpl_ai_learning_data_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        else:
            st.error("❌ Failed to export learning data")


def render_performance_charts():
    """Render performance visualization charts"""
    st.markdown("**📈 AI Performance Trends:**")
    
    metrics = iteration_engine.performance_metrics
    
    if not metrics['recommendation_accuracy']:
        st.info("No performance data available yet")
        return
    
    # Create performance trend chart
    fig = go.Figure()
    
    # Add accuracy trend
    if len(metrics['recommendation_accuracy']) > 1:
        fig.add_trace(go.Scatter(
            y=metrics['recommendation_accuracy'],
            mode='lines+markers',
            name='Prediction Accuracy',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
    
    # Add conversion rate if available
    if len(metrics['conversion_rate']) > 1:
        fig.add_trace(go.Scatter(
            y=metrics['conversion_rate'],
            mode='lines+markers',
            name='User Adoption Rate',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=6),
            yaxis='y2'
        ))
    
    fig.update_layout(
        title="AI Performance Over Time",
        xaxis_title="Recommendation Batch",
        yaxis=dict(title="Accuracy", tickformat='.1%'),
        yaxis2=dict(title="Adoption Rate", overlaying='y', side='right', tickformat='.1%'),
        height=400,
        showlegend=True,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show recent prediction errors
    if metrics['prediction_error']:
        col1, col2 = st.columns(2)
        
        with col1:
            avg_error = sum(metrics['prediction_error'][-10:]) / min(len(metrics['prediction_error']), 10)
            st.metric("Recent Avg. Prediction Error", f"{avg_error:.1f} points")
        
        with col2:
            error_trend = "Improving" if len(metrics['prediction_error']) > 1 and metrics['prediction_error'][-1] < metrics['prediction_error'][-2] else "Stable"
            st.metric("Error Trend", error_trend)


def render_feedback_summary():
    """Render feedback summary for analysis"""
    st.subheader("📋 Feedback Summary")
    
    if not iteration_engine.feedback_history:
        st.info("No feedback data available yet")
        return
    
    # Create feedback DataFrame
    feedback_data = []
    for feedback in iteration_engine.feedback_history[-50:]:  # Last 50 entries
        feedback_data.append({
            'Player': feedback.player_name,
            'Feedback Type': feedback.feedback_type,
            'Score': feedback.feedback_score,
            'Action': feedback.user_action,
            'Date': feedback.timestamp.strftime('%Y-%m-%d') if isinstance(feedback.timestamp, datetime) else feedback.timestamp[:10]
        })
    
    feedback_df = pd.DataFrame(feedback_data)
    
    if not feedback_df.empty:
        # Feedback distribution chart
        fig = px.histogram(
            feedback_df, 
            x='Feedback Type', 
            color='Feedback Type',
            title="Feedback Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Average scores by action
        avg_scores = feedback_df.groupby('Action')['Score'].mean().sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Average Scores by User Action:**")
            for action, score in avg_scores.items():
                st.markdown(f"• {action}: {score:.1f}/5")
        
        with col2:
            st.markdown("**Recent Feedback:**")
            recent_feedback = feedback_df.tail(5)[['Player', 'Score', 'Action']].copy()
            st.dataframe(recent_feedback, hide_index=True)


def display_automated_iteration_help():
    """Display help information about the automated iteration system"""
    st.markdown("""
    ### 🤖 How AI-Enhanced Recommendations Work
    
    **Continuous Learning System:**
    - The AI learns from your feedback on every recommendation
    - Prediction accuracy improves over time based on actual player performance
    - Your preferences are automatically detected and incorporated
    
    **Feedback Impact:**
    - ⭐ **Positive feedback** (4-5 stars): Increases weight of similar recommendation factors
    - 👎 **Negative feedback** (1-2 stars): Reduces weight of factors that led to poor recommendations
    - 🔄 **Neutral feedback** (3 stars): Maintains current weighting balance
    
    **Personalization Features:**
    - Risk appetite learning (conservative vs aggressive transfers)
    - Price range preferences by position
    - Formation and playing style preferences
    - Team diversity preferences
    
    **Performance Tracking:**
    - Actual vs predicted points comparison
    - User adoption rate of recommendations
    - Satisfaction scores and trends
    - Continuous model improvement
    
    **Privacy Note:** All learning data is stored locally and used only to improve your personal recommendations.
    """)