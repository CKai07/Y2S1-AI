import streamlit as st

# Set page configuration first before any other Streamlit commands
st.set_page_config(
    page_title="Stock Market Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
    }
    .info-text {
        font-size: 1rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 15px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Only import other modules after setting page config
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os
import altair as alt

# Import StockPredictor and other functions from stock_market_prediction_system
try:
    from stock_market_prediction_system import StockPredictor, format_prediction, test_prediction_accuracy
except ImportError:
    st.error("Could not import stock_market_prediction_system. Make sure the file is in the same directory.")
    st.stop()

# Import the functions from our new functions file
from streamlit_app_functions import display_stock_prediction, display_stock_comparison, display_prediction_results

# Define our own main function without the page_config call
def main():
    """Main function to run the Streamlit app."""
    # Page config is already set above, so we don't need it here
    
    st.title("Stock Market Prediction Dashboard")
    
    # Initialize the predictor
    predictor = StockPredictor()
    
    # Create a navigation menu
    nav_options = ["Single Stock Prediction", "Stock Comparison", "About"]
    nav_selection = st.sidebar.radio("Navigation", nav_options)
    
    # Display appropriate content based on navigation
    if nav_selection == "Single Stock Prediction":
        display_stock_prediction(predictor)
    elif nav_selection == "Stock Comparison":
        display_stock_comparison(predictor)
    else:  # About
        st.header("About the Stock Market Prediction System")
        st.write("""
        This dashboard uses machine learning algorithms to predict stock prices based on 
        historical data and technical indicators. The system employs LSTM neural networks 
        and Random Forest models to generate predictions for day trading and long-term investment.
        
        ### Features:
        - **Single Stock Prediction**: Analyze individual stocks with detailed insights
        - **Stock Comparison**: Compare multiple stocks to identify the best investment opportunities
        - **Technical Indicators**: RSI, MACD, Bollinger Bands, and more
        - **Market Status Detection**: Adjusts predictions based on whether the market is open or closed
        - **Confidence Levels**: Provides confidence metrics for predictions
        
        ### Usage Tips:
        - For day trading predictions, focus on the short-term price movements and technical indicators
        - For long-term investments, consider the overall trend and fundamentals
        - Always check if the market is open, as predictions during market closure have higher uncertainty
        - Use the comparison feature to diversify your portfolio effectively
        """)
        
        st.sidebar.info("Created by AI Stock Prediction Team")
    
    # Add a footer
    st.markdown("---")
    st.caption("Stock Market Prediction Dashboard - Data refreshed on demand")

if __name__ == "__main__":
    main() 