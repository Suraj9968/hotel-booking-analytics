import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the app
st.set_page_config(
    page_title="Hotel Booking Analytics Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set API URL
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .insight-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .health-healthy {
        color: green;
        font-weight: bold;
    }
    .health-degraded {
        color: orange;
        font-weight: bold;
    }
    .health-unhealthy {
        color: red;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Define header
st.markdown("<h1 class='main-header'>Hotel Booking Analytics Dashboard</h1>", unsafe_allow_html=True)

# Define functions to interact with the API
def get_analytics(insight_types=None):
    """Get analytics insights from the API"""
    if insight_types is None:
        insight_types = ["revenue_trends", "cancellation_rate", "geographical_distribution", 
                        "lead_time_distribution", "additional_insights"]
    try:
        response = requests.post(
            f"{API_URL}/analytics", 
            json={"insight_types": insight_types, "format_type": "json"}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching analytics: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def ask_question(question, max_results=5):
    """Send a question to the API"""
    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/ask", 
                json={"text": question, "max_results": max_results}
            )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error asking question: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_health():
    """Check system health"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error checking health: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

def get_query_history(limit=10):
    """Get query history"""
    try:
        response = requests.get(f"{API_URL}/query_history?limit={limit}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching query history: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# Create sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page", [
    "📊 Analytics Dashboard", 
    "❓ Question Answering", 
    "🔍 Query History",
    "🩺 System Health"
])

# Display API connection status in sidebar
with st.sidebar.expander("API Connection", expanded=False):
    if st.button("Check Connection"):
        try:
            health = get_health()
            if health:
                st.success(f"Connected to API at {API_URL}")
                st.write(f"System status: {health['status']}")
            else:
                st.error(f"Cannot connect to API at {API_URL}")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

# Add sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Hotel Booking Analytics")
st.sidebar.markdown("Powered by FastAPI, Streamlit, and Google Gemini")

# Main content based on selected page
if page == "📊 Analytics Dashboard":
    st.title("Analytics Dashboard")
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Revenue Trends", 
        "Cancellation Rates", 
        "Geographical Distribution", 
        "Lead Time Analysis",
        "Additional Insights"
    ])
    
    # Fetch analytics data
    with st.spinner("Loading analytics data..."):
        analytics_data = get_analytics()
    
    if analytics_data and "insights" in analytics_data:
        insights = analytics_data["insights"]
        
        # Tab 1: Revenue Trends
        with tab1:
            if "revenue_trends" in insights and insights["revenue_trends"]:
                revenue_data = insights["revenue_trends"]
                
                st.subheader("Revenue Trends Over Time")
                
                # Create a dataframe for plotting
                df_revenue = pd.DataFrame({
                    "Date": revenue_data["dates"],
                    "Revenue": revenue_data["revenues"],
                    "Bookings": revenue_data["booking_counts"]
                })
                
                # Plot revenue trend
                fig = px.line(
                    df_revenue, x="Date", y="Revenue", 
                    title="Monthly Revenue Trends",
                    markers=True
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show booking counts as a bar chart
                fig2 = px.bar(
                    df_revenue, x="Date", y="Bookings",
                    title="Monthly Booking Counts"
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Display summary stats
                total_revenue = sum(revenue_data["revenues"])
                avg_monthly_revenue = total_revenue / len(revenue_data["revenues"])
                max_revenue_month = df_revenue.loc[df_revenue["Revenue"].idxmax()]["Date"]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Revenue", f"${total_revenue:,.2f}")
                col2.metric("Avg. Monthly Revenue", f"${avg_monthly_revenue:,.2f}")
                col3.metric("Highest Revenue Month", max_revenue_month)
            else:
                st.info("No revenue trend data available.")
        
        # Tab 2: Cancellation Rates
        with tab2:
            if "cancellation_rate" in insights and insights["cancellation_rate"]:
                cancellation_data = insights["cancellation_rate"]
                
                st.subheader("Booking Cancellation Analysis")
                
                # Overall cancellation rate
                overall_rate = cancellation_data["overall"]["cancellation_rate"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Overall Cancellation Rate", f"{overall_rate:.2f}%")
                col2.metric("Total Bookings", f"{cancellation_data['overall']['total_bookings']:,}")
                col3.metric("Canceled Bookings", f"{cancellation_data['overall']['canceled_bookings']:,}")
                
                # Hotel-specific cancellation rates
                st.subheader("Cancellation Rates by Hotel Type")
                df_cancel = pd.DataFrame({
                    "Hotel": cancellation_data["by_hotel"]["hotels"],
                    "Cancellation Rate": cancellation_data["by_hotel"]["cancellation_rates"],
                    "Total Bookings": cancellation_data["by_hotel"]["total_bookings"],
                    "Canceled Bookings": cancellation_data["by_hotel"]["canceled_bookings"]
                })
                
                # Create bar chart
                fig = px.bar(
                    df_cancel, x="Hotel", y="Cancellation Rate",
                    title="Cancellation Rate by Hotel Type",
                    text_auto='.2f',
                    color="Cancellation Rate"
                )
                fig.update_layout(height=400, yaxis_title="Cancellation Rate (%)")
                fig.add_hline(y=overall_rate, line_dash="dash", line_color="red", 
                              annotation_text=f"Overall Rate: {overall_rate:.2f}%")
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data table
                st.dataframe(df_cancel, use_container_width=True)
            else:
                st.info("No cancellation rate data available.")
        
        # Tab 3: Geographical Distribution
        with tab3:
            if "geographical_distribution" in insights and insights["geographical_distribution"]:
                geo_data = insights["geographical_distribution"]
                
                st.subheader("Geographical Distribution of Bookings")
                
                # Create dataframe
                if "top_20" in geo_data:
                    df_geo = pd.DataFrame({
                        "Country": geo_data["top_20"]["countries"],
                        "Booking Count": geo_data["top_20"]["booking_counts"]
                    })
                    
                    # Create bar chart of top countries
                    fig = px.bar(
                        df_geo, x="Country", y="Booking Count",
                        title="Top 20 Countries by Booking Count",
                        color="Booking Count"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Create full dataframe
                df_geo_full = pd.DataFrame({
                    "Country": geo_data["countries"],
                    "Total Bookings": geo_data["booking_counts"],
                    "Confirmed Bookings": geo_data["confirmed_bookings"],
                    "Canceled Bookings": geo_data["canceled_bookings"],
                    "Total Revenue": geo_data["total_revenue"]
                })
                
                # Add cancellation rate
                df_geo_full["Cancellation Rate (%)"] = (df_geo_full["Canceled Bookings"] / df_geo_full["Total Bookings"]) * 100
                
                # Sort and get top 10 by revenue
                df_top_revenue = df_geo_full.sort_values("Total Revenue", ascending=False).head(10)
                
                # Show top countries by revenue
                st.subheader("Top 10 Countries by Revenue")
                fig2 = px.bar(
                    df_top_revenue, x="Country", y="Total Revenue",
                    title="Top 10 Countries by Revenue",
                    color="Total Revenue"
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Allow users to search/filter countries
                st.subheader("Country Details")
                search_country = st.text_input("Search for a country:")
                if search_country:
                    filtered_df = df_geo_full[df_geo_full["Country"].str.contains(search_country, case=False)]
                    st.dataframe(filtered_df, use_container_width=True)
                else:
                    st.dataframe(df_geo_full.head(20), use_container_width=True)
            else:
                st.info("No geographical distribution data available.")
        
        # Tab 4: Lead Time Analysis
        with tab4:
            if "lead_time_distribution" in insights and insights["lead_time_distribution"]:
                lead_time_data = insights["lead_time_distribution"]
                
                st.subheader("Booking Lead Time Analysis")
                
                # Display lead time statistics
                stats = lead_time_data["statistics"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Min Lead Time", f"{stats['min']} days")
                col2.metric("Max Lead Time", f"{stats['max']} days")
                col3.metric("Mean Lead Time", f"{stats['mean']:.1f} days")
                col4.metric("Median Lead Time", f"{stats['median']:.1f} days")
                
                # Create dataframe for distribution
                if "distribution" in lead_time_data:
                    df_dist = pd.DataFrame({
                        "Lead Time Range": lead_time_data["distribution"]["bins"],
                        "Number of Bookings": lead_time_data["distribution"]["counts"]
                    })
                    
                    # Create bar chart of distribution
                    fig = px.bar(
                        df_dist, x="Lead Time Range", y="Number of Bookings",
                        title="Lead Time Distribution",
                        color="Number of Bookings"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display percentile information
                    st.subheader("Lead Time Percentiles")
                    st.write(f"25th Percentile: {stats['percentiles']['25']:.1f} days")
                    st.write(f"50th Percentile (Median): {stats['percentiles']['50']:.1f} days")
                    st.write(f"75th Percentile: {stats['percentiles']['75']:.1f} days")
                    st.write(f"90th Percentile: {stats['percentiles']['90']:.1f} days")
            else:
                st.info("No lead time distribution data available.")
        
        # Tab 5: Additional Insights
        with tab5:
            if "additional_insights" in insights and insights["additional_insights"]:
                add_insights = insights["additional_insights"]
                
                st.subheader("Additional Booking Insights")
                
                # Create subtabs for different insights
                subtab1, subtab2, subtab3, subtab4 = st.tabs([
                    "ADR by Room Type", 
                    "Market Segment Analysis", 
                    "Booking Patterns",
                    "Room Type Distribution"
                ])
                
                # Subtab 1: ADR by Room Type
                with subtab1:
                    if "adr_by_room_type" in add_insights:
                        adr_data = add_insights["adr_by_room_type"]
                        
                        # Create dataframe
                        df_adr = pd.DataFrame({
                            "Hotel": adr_data["hotel_types"],
                            "Room Type": adr_data["room_types"],
                            "Average Daily Rate": adr_data["avg_rates"],
                            "Booking Count": adr_data["booking_counts"]
                        })
                        
                        # Group by hotel and room type
                        hotel_types = df_adr["Hotel"].unique()
                        
                        for hotel in hotel_types:
                            st.subheader(f"{hotel} - Room Rates")
                            hotel_df = df_adr[df_adr["Hotel"] == hotel].sort_values("Average Daily Rate", ascending=False)
                            
                            # Create bar chart
                            fig = px.bar(
                                hotel_df, x="Room Type", y="Average Daily Rate",
                                title=f"Average Daily Rate by Room Type ({hotel})",
                                color="Average Daily Rate",
                                text_auto='.2f'
                            )
                            fig.update_layout(height=400, yaxis_title="Average Daily Rate ($)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show data table
                            st.dataframe(hotel_df, use_container_width=True)
                    else:
                        st.info("No ADR by room type data available.")
                
                # Subtab 2: Market Segment Analysis
                with subtab2:
                    if "market_segment_analysis" in add_insights:
                        segment_data = add_insights["market_segment_analysis"]
                        
                        # Create dataframe
                        df_segment = pd.DataFrame({
                            "Segment": segment_data["segments"],
                            "Booking Count": segment_data["booking_counts"],
                            "Cancellation Rate": segment_data["cancellation_rates"],
                            "Average Daily Rate": segment_data["avg_rates"],
                            "Average Stay Duration": segment_data["avg_stay_durations"]
                        })
                        
                        # Sort by booking count
                        df_segment = df_segment.sort_values("Booking Count", ascending=False)
                        
                        # Create bar chart of booking counts
                        fig1 = px.bar(
                            df_segment, x="Segment", y="Booking Count",
                            title="Booking Count by Market Segment",
                            color="Booking Count"
                        )
                        fig1.update_layout(height=400)
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Create a scatter plot of rate vs cancellation
                        fig2 = px.scatter(
                            df_segment, x="Average Daily Rate", y="Cancellation Rate",
                            size="Booking Count", color="Segment", 
                            title="ADR vs Cancellation Rate by Market Segment",
                            hover_data=["Average Stay Duration"]
                        )
                        fig2.update_layout(height=500)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Show data table
                        st.dataframe(df_segment, use_container_width=True)
                    else:
                        st.info("No market segment analysis data available.")
                
                # Subtab 3: Booking Patterns
                with subtab3:
                    if "day_of_month_distribution" in add_insights:
                        day_data = add_insights["day_of_month_distribution"]
                        
                        # Create dataframe
                        df_day = pd.DataFrame({
                            "Day of Month": day_data["days"],
                            "Booking Count": day_data["booking_counts"]
                        })
                        
                        st.subheader("Booking Distribution by Day of Month")
                        
                        # Create line chart
                        fig = px.line(
                            df_day, x="Day of Month", y="Booking Count",
                            title="Booking Count by Day of Month",
                            markers=True
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No booking pattern data available.")
                        
                # Subtab 4: Room Type Distribution
                with subtab4:
                    if "room_type_distribution" in add_insights:
                        room_data = add_insights["room_type_distribution"]
                        
                        # Create dataframe
                        df_room = pd.DataFrame({
                            "Room Type": room_data["room_types"],
                            "Total Bookings": room_data["booking_counts"],
                            "Confirmed Bookings": room_data["confirmed_bookings"],
                            "Canceled Bookings": room_data["canceled_bookings"],
                            "Average Daily Rate": room_data["avg_rates"],
                            "Average Stay Duration": room_data["avg_stay_durations"]
                        })
                        
                        # Calculate cancellation rate
                        df_room["Cancellation Rate (%)"] = (df_room["Canceled Bookings"] / df_room["Total Bookings"]) * 100
                        
                        # Sort by booking count
                        df_room = df_room.sort_values("Total Bookings", ascending=False)
                        
                        # Display key metrics - most common room type
                        most_common_room = df_room.iloc[0]
                        
                        st.subheader("Most Common Room Type")
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Room Type", most_common_room["Room Type"])
                        col2.metric("Total Bookings", f"{most_common_room['Total Bookings']:,}")
                        col3.metric("Percentage of All Bookings", f"{(most_common_room['Total Bookings'] / df_room['Total Bookings'].sum()) * 100:.1f}%")
                        
                        # Create bar chart of room types
                        fig1 = px.bar(
                            df_room, x="Room Type", y="Total Bookings",
                            title="Bookings by Room Type",
                            color="Total Bookings"
                        )
                        fig1.update_layout(height=400)
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Create pie chart of room type distribution
                        fig2 = px.pie(
                            df_room, values="Total Bookings", names="Room Type",
                            title="Room Type Distribution"
                        )
                        fig2.update_layout(height=500)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Create scatter plot of price vs cancellation rate by room type
                        fig3 = px.scatter(
                            df_room, x="Average Daily Rate", y="Cancellation Rate (%)",
                            size="Total Bookings", color="Room Type",
                            title="Room Type: Price vs Cancellation Rate",
                            hover_data=["Average Stay Duration"]
                        )
                        fig3.update_layout(height=500)
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        # Show data table
                        st.dataframe(df_room, use_container_width=True)
                    else:
                        st.info("No room type distribution data available.")
            else:
                st.info("No additional insights available.")
    else:
        st.error("Failed to fetch analytics data. Please check API connection.")

elif page == "❓ Question Answering":
    st.title("Question Answering")
    
    st.write("""
    Ask questions about the hotel booking data in natural language.
    The system will retrieve relevant information and provide an answer.
    """)
    
    # Create a container for sample questions
    with st.expander("Sample Questions", expanded=True):
        sample_questions = [
            "Show me total revenue for July 2017.",
            "Which locations had the highest booking cancellations?",
            "What is the average price of a hotel booking?",
            "What is the average lead time for Resort Hotel bookings?",
            "Which market segment has the highest cancellation rate?",
            "What is the distribution of bookings across different countries?",
            "How long do guests typically stay in City Hotels vs Resort Hotels?",
            "What is the most common room type booked?"
        ]
        
        # Display sample questions as clickable buttons
        col1, col2 = st.columns(2)
        sample_question = None
        
        for i, question in enumerate(sample_questions):
            if i % 2 == 0:
                if col1.button(question, key=f"sample_{i}"):
                    sample_question = question
            else:
                if col2.button(question, key=f"sample_{i}"):
                    sample_question = question
    
    # Input for custom question
    user_question = st.text_input("Enter your question:", value=sample_question if sample_question else "")
    
    if st.button("Ask", key="ask_button") or sample_question:
        if user_question:
            # Get answer from API
            response = ask_question(user_question)
            
            if response:
                st.markdown("### Answer")
                st.markdown(response["answer"])
                
                # Display metadata
                with st.expander("Response Metadata", expanded=False):
                    st.write(f"Execution Time: {response['execution_time']:.2f} seconds")
                    st.write(f"Timestamp: {response['timestamp']}")
        else:
            st.warning("Please enter a question.")

elif page == "🔍 Query History":
    st.title("Query History")
    
    st.write("View the history of questions asked to the system.")
    
    # Input for number of queries to retrieve
    limit = st.slider("Number of queries to display:", min_value=5, max_value=50, value=10, step=5)
    
    if st.button("Refresh History"):
        # Get query history from API
        history_data = get_query_history(limit=limit)
        
        if history_data and "history" in history_data:
            history = history_data["history"]
            
            if not history:
                st.info("No query history found.")
            else:
                # Display each query in an expander
                for item in history:
                    with st.expander(f"Query: {item['query']}", expanded=False):
                        st.write(f"**Timestamp:** {item['timestamp']}")
                        st.write(f"**Execution Time:** {item['execution_time']:.2f} seconds")
                        st.write(f"**ID:** {item['id']}")
        else:
            st.error("Failed to fetch query history. Please check API connection.")

elif page == "🩺 System Health":
    st.title("System Health")
    
    st.write("Check the status of the system and its components.")
    
    if st.button("Check Health"):
        # Get health status from API
        health_data = get_health()
        
        if health_data:
            # Display overall status
            status = health_data["status"]
            status_color = {
                "healthy": "health-healthy",
                "degraded": "health-degraded",
                "unhealthy": "health-unhealthy"
            }.get(status, "")
            
            st.markdown(f"### Overall Status: <span class='{status_color}'>{status.upper()}</span>", unsafe_allow_html=True)
            
            # Display detailed status
            col1, col2, col3 = st.columns(3)
            col1.metric("Database Connection", "✅ Connected" if health_data["database_connected"] else "❌ Disconnected")
            col2.metric("Vector Database", "✅ Available" if health_data["vector_db_status"] else "❌ Unavailable")
            col3.metric("LLM Status", "✅ Operational" if health_data["llm_status"] else "❌ Non-operational")
            
            # Display timestamp and execution time
            st.write(f"Checked at: {health_data['checked_at']}")
            st.write(f"Check execution time: {health_data['execution_time']:.4f} seconds")
            
            # Provide recommendations based on status
            if status == "degraded":
                st.warning("System is operating in degraded mode. Some features may not be available.")
                
                if not health_data["llm_status"]:
                    st.info("LLM service is not available. Question answering will not work correctly. " +
                            "Please check your Google API key in the .env file.")
            
            elif status == "unhealthy":
                st.error("System is unhealthy. Most features will not work correctly.")
                
                if not health_data["database_connected"]:
                    st.info("Database connection failed. Please check if the database file exists and is accessible.")
                
                if not health_data["vector_db_status"]:
                    st.info("Vector database is not available. Please run 'python RAG_QA.py --setup_only' to set it up.")
        else:
            st.error("Failed to fetch health status. API might be down.")

# Footer
st.markdown("---")
st.markdown("Hotel Booking Analytics System") 