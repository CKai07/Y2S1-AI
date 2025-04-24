import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os
import altair as alt

# Handle potentially missing dependencies
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import the StockPredictor class 
from stock_market_prediction_system import StockPredictor, format_prediction, test_prediction_accuracy

def get_predictor(ticker, day_trading=False):
    """Cache the StockPredictor to avoid rebuilding it every time"""
    with st.spinner(f"Building prediction model for {ticker}..."):
        try:
            return StockPredictor(ticker, day_trading=day_trading)
        except Exception as e:
            st.error(f"Error creating predictor: {str(e)}")
            return None

def display_prediction_results(prediction):
    """Display the prediction results in a nice format"""
    ticker = prediction["ticker"]
    current_price = prediction["current_price"]
    predicted_price = prediction["predicted_price"]
    prediction_type = prediction["prediction_type"]
    market_status = prediction["market_status"]
    confidence_levels = prediction["confidence_levels"]
    action = prediction["action"]
    reason = prediction["reason"]
    expected_change = prediction["expected_change_percent"]
    last_updated = prediction.get("last_updated", "Unknown")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price", 
            value=f"${current_price:.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Predicted Price", 
            value=f"${predicted_price:.2f}", 
            delta=f"{expected_change:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Market Status", 
            value=market_status
        )
    
    with col4:
        st.metric(
            label="Recommended Action", 
            value=action
        )
    
    # Display confidence levels
    st.markdown("### Confidence Levels")
    conf_col1, conf_col2, conf_col3 = st.columns(3)
    
    with conf_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>80% Confidence</h4>
                <h2>${confidence_levels['80%']:.2f}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with conf_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>70% Confidence</h4>
                <h2>${confidence_levels['70%']:.2f}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with conf_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>65% Confidence</h4>
                <h2>${confidence_levels['65%']:.2f}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # More details
    st.markdown("### Prediction Details")
    st.markdown(f"**Reason:** {reason}")
    st.markdown(f"**Last Updated:** {last_updated}")
    
    if market_status == "Closed":
        st.warning("Market is currently closed. Predictions may have higher uncertainty. Consider waiting for market open for more accurate predictions.")

def plot_prediction_streamlit(predictor):
    """Plot prediction with Plotly in Streamlit"""
    prediction = predictor.predict_with_confidence(force_update=True)
    
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly is not installed. Using a simplified visualization.")
        # Create a simple visualization with Streamlit's built-in charts
        
        # Prepare data for the chart
        hist_data = predictor.data.copy()
        hist_data['Date'] = hist_data.index
        
        # Get prediction points
        current_price = prediction["current_price"]
        last_date = predictor.data.index[-1]
        next_date = last_date + timedelta(days=1 if predictor.day_trading else 30)
        confidence_levels = prediction["confidence_levels"]
        
        # Plot historical prices
        st.subheader("Historical Price Data")
        st.line_chart(hist_data[['Close']], x='Date')
        
        # Plot volume
        st.subheader("Trading Volume")
        st.bar_chart(hist_data[['Volume']], x='Date')
        
        # Create a table with predictions
        st.subheader("Price Predictions")
        
        pred_data = {
            "Price Point": ["Current Price", "Raw Prediction", "80% Confidence", "70% Confidence", "65% Confidence"],
            "Date": [last_date, next_date, next_date, next_date, next_date],
            "Price": [
                f"${current_price:.2f}",
                f"${prediction['predicted_price']:.2f}",
                f"${confidence_levels['80%']:.2f}",
                f"${confidence_levels['70%']:.2f}",
                f"${confidence_levels['65%']:.2f}",
            ]
        }
        
        st.table(pd.DataFrame(pred_data))
        
        return prediction
    
    # If Plotly is available, use it for better visualizations
    # Create figure
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True, 
                       vertical_spacing=0.1,
                       subplot_titles=('Price', 'Volume'),
                       row_heights=[0.7, 0.3])
    
    # Add price data
    fig.add_trace(
        go.Candlestick(
            x=predictor.data.index,
            open=predictor.data['Open'],
            high=predictor.data['High'],
            low=predictor.data['Low'],
            close=predictor.data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add volume data
    fig.add_trace(
        go.Bar(
            x=predictor.data.index,
            y=predictor.data['Volume'],
            name='Volume'
        ),
        row=2, col=1
    )
    
    # Add prediction points
    current_price = prediction["current_price"]
    last_date = predictor.data.index[-1]
    next_date = last_date + timedelta(days=1 if predictor.day_trading else 30)
    
    # Add current price point
    fig.add_trace(
        go.Scatter(
            x=[last_date],
            y=[current_price],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name='Current Price'
        ),
        row=1, col=1
    )
    
    # Add prediction with confidence levels
    confidence_levels = prediction["confidence_levels"]
    
    fig.add_trace(
        go.Scatter(
            x=[next_date],
            y=[confidence_levels["80%"]],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='80% Confidence'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[next_date],
            y=[confidence_levels["70%"]],
            mode='markers',
            marker=dict(size=10, color='orange'),
            name='70% Confidence'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[next_date],
            y=[confidence_levels["65%"]],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='65% Confidence'
        ),
        row=1, col=1
    )
    
    # Add raw prediction point
    fig.add_trace(
        go.Scatter(
            x=[next_date],
            y=[prediction["predicted_price"]],
            mode='markers',
            marker=dict(size=10, color='purple', symbol='star'),
            name='Raw Prediction'
        ),
        row=1, col=1
    )
    
    # Update layout
    title_text = f"{predictor.ticker} - {prediction['prediction_type']} Prediction"
    if prediction["market_status"] == "Closed":
        title_text += " (Market Closed)"
        
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=600,
        showlegend=True
    )
    
    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
    return prediction

def display_stock_prediction(predictor):
    """Display the stock prediction interface"""
    st.header("Single Stock Prediction")
    
    # Get user inputs
    col1, col2 = st.columns(2)
    
    with col1:
        ticker = st.text_input("Enter Stock Ticker Symbol", "AAPL").strip().upper()
        
    with col2:
        prediction_type = st.radio(
            "Prediction Type",
            ["Day Trading", "Long-term Investment"]
        )
    
    # Additional options
    force_update = st.checkbox("Force Data Refresh", True, 
                              help="When checked, the system will fetch fresh data even if cached data is available")
    
    day_trading = prediction_type == "Day Trading"
    
    if st.button("Generate Prediction"):
        if not ticker:
            st.warning("Please enter a valid ticker symbol.")
        else:
            try:
                with st.spinner(f"Analyzing {ticker}..."):
                    # If we have a PlaceholderPredictor, create a real one
                    if not hasattr(predictor, 'data') or not hasattr(predictor, 'predict_with_confidence'):
                        try:
                            predictor = StockPredictor(ticker, day_trading=day_trading)
                        except Exception as e:
                            st.error(f"Error initializing predictor: {str(e)}")
                            st.info("Please try a different ticker symbol.")
                            return
                    else:
                        # Set the ticker and prediction type on an existing predictor
                        predictor.set_ticker(ticker)
                        predictor.day_trading = day_trading
                    
                    # Make the prediction
                    prediction = predictor.predict_with_confidence(force_update=force_update)
                    
                    # Display results in a nice format
                    st.subheader("Prediction Results")
                    
                    # Create columns for metrics
                    col1, col2, col3 = st.columns(3)
                    
                    market_status = "OPEN" if prediction["market_open"] else "CLOSED"
                    last_updated = prediction.get("last_updated", "Unknown")
                    
                    col1.metric(
                        "Current Price", 
                        f"${prediction['current_price']:.2f}",
                        delta=None,
                        help=f"Last updated: {last_updated}"
                    )
                    
                    price_change = prediction['predicted_price'] - prediction['current_price']
                    price_change_pct = (price_change / prediction['current_price']) * 100
                    
                    col2.metric(
                        "Predicted Price", 
                        f"${prediction['predicted_price']:.2f}",
                        delta=f"{price_change_pct:.2f}%",
                        delta_color="normal"
                    )
                    
                    col3.metric(
                        "Market Status",
                        market_status,
                        delta=None
                    )
                    
                    # Display action recommendation
                    action_color = {
                        "BUY": "success",
                        "SELL": "danger",
                        "HOLD": "warning"
                    }.get(prediction['action'].upper(), "info")
                    
                    st.markdown(f"<div style='background-color: {'#d4edda' if action_color == 'success' else '#f8d7da' if action_color == 'danger' else '#fff3cd'}; padding: 10px; border-radius: 5px; margin: 10px 0;'><strong>Recommendation:</strong> {prediction['action'].upper()}</div>", unsafe_allow_html=True)
                    
                    # Display confidence levels
                    st.subheader("Prediction Confidence")
                    
                    # Create a dataframe for confidence levels
                    confidence_data = {
                        "Level": list(prediction['confidence_levels'].keys()),
                        "Confidence": [value * 100 for value in prediction['confidence_levels'].values()]
                    }
                    
                    # Display as a bar chart
                    confidence_df = pd.DataFrame(confidence_data)
                    chart = alt.Chart(confidence_df).mark_bar().encode(
                        x=alt.X('Confidence:Q', title='Confidence (%)'),
                        y=alt.Y('Level:N', title=None),
                        color=alt.Color('Level:N', legend=None),
                        tooltip=['Level', 'Confidence']
                    ).properties(
                        height=200
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Market closure warning
                    if not prediction["market_open"]:
                        st.warning("⚠️ Market is currently closed. Predictions have increased uncertainty.")
                    
                    # Tabs for additional information
                    tab1, tab2, tab3 = st.tabs(["Historical Data", "Technical Indicators", "Model Details"])
                    
                    with tab1:
                        st.subheader("Recent Historical Data")
                        st.dataframe(predictor.data.tail(10), use_container_width=True)
                        
                    with tab2:
                        st.subheader("Technical Indicators")
                        if hasattr(predictor, 'features') and predictor.features is not None:
                            st.dataframe(predictor.features.tail(10), use_container_width=True)
                        else:
                            st.info("Technical indicators not available")
                            
                    with tab3:
                        st.subheader("Model Information")
                        st.markdown(f"""
                        * **Prediction Type**: {prediction_type}
                        * **Models Used**: LSTM Neural Network and Random Forest Ensemble
                        * **Historical Data**: {len(predictor.data) if hasattr(predictor, 'data') else 'Unknown'} days
                        * **Technical Indicators**: RSI, MACD, Bollinger Bands, OBV, ATR, and more
                        * **Last Data Update**: {last_updated}
                        """)
                    
                    # Option to plot the prediction
                    if st.button("Visualize Prediction"):
                        st.session_state.plot_requested = True
                        st.session_state.plot_ticker = ticker
                        st.session_state.plot_day_trading = day_trading
                        st.rerun()
                        
                    # Check if we need to show the plot
                    if hasattr(st.session_state, 'plot_requested') and st.session_state.plot_requested:
                        if st.session_state.plot_ticker == ticker and st.session_state.plot_day_trading == day_trading:
                            st.subheader(f"Price Prediction Visualization for {ticker}")
                            # Use the plot_prediction method to create the chart
                            try:
                                fig = predictor.plot_prediction()
                                st.plotly_chart(fig, use_container_width=True)
                                st.session_state.plot_requested = False
                            except Exception as e:
                                st.error(f"Error creating plot: {str(e)}")
                                try:
                                    # Fallback to matplotlib
                                    fig, ax = predictor._plot_with_matplotlib()
                                    st.pyplot(fig)
                                    st.session_state.plot_requested = False
                                except:
                                    st.error("Could not create visualization. Try refreshing the page.")
            
            except Exception as e:
                st.error(f"Error generating prediction: {str(e)}")
                st.info("Please try a different ticker symbol or check if the symbol is valid.")

def display_stock_comparison(predictor):
    """Display the stock comparison interface"""
    st.header("Stock Comparison Analysis")
    st.write("Compare predictions and performance for multiple stocks.")
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        tickers_input = st.text_input(
            "Enter Stock Ticker Symbols (comma-separated)", 
            "AAPL, MSFT, GOOGL"
        )
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
    
    with col2:
        prediction_type = st.radio(
            "Prediction Type",
            ["Day Trading", "Long-term Investment"]
        )
    
    # Force data refresh option
    force_update = st.checkbox("Force Data Refresh", True)
    
    day_trading = prediction_type == "Day Trading"
    
    if st.button("Compare Stocks"):
        if not tickers:
            st.warning("Please enter at least one valid ticker symbol.")
        else:
            # Check if we have too many tickers
            if len(tickers) > 10:
                st.warning("Please limit your comparison to 10 stocks for optimal performance.")
                tickers = tickers[:10]
            
            # Table to store results
            results = []
            errors = []
            
            # Process each ticker
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # If we have a placeholder predictor, create a real one with the first ticker
            real_predictor = predictor
            if not hasattr(predictor, 'data') or not hasattr(predictor, 'predict_with_confidence'):
                try:
                    real_predictor = StockPredictor(tickers[0], day_trading=day_trading)
                except Exception as e:
                    st.error(f"Error initializing predictor with {tickers[0]}: {str(e)}")
                    st.info("Please try different ticker symbols.")
                    return
            
            for i, ticker in enumerate(tickers):
                status_text.text(f"Analyzing {ticker}...")
                try:
                    # Set the ticker and prediction type
                    real_predictor.set_ticker(ticker)
                    real_predictor.day_trading = day_trading
                    
                    # Make the prediction
                    prediction = real_predictor.predict_with_confidence(force_update=force_update)
                    
                    # Process the prediction data
                    price_change = prediction['predicted_price'] - prediction['current_price']
                    price_change_pct = (price_change / prediction['current_price']) * 100
                    
                    results.append({
                        'Ticker': ticker,
                        'Current Price': f"${prediction['current_price']:.2f}",
                        'Predicted Price': f"${prediction['predicted_price']:.2f}",
                        'Change (%)': f"{price_change_pct:.2f}%",
                        'Action': prediction['action'].upper(),
                        'Market Open': "Yes" if prediction["market_open"] else "No"
                    })
                except Exception as e:
                    errors.append(f"Error with {ticker}: {str(e)}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(tickers))
            
            # Clear the status
            status_text.empty()
            progress_bar.empty()
            
            # Display results
            if results:
                st.subheader("Comparison Results")
                
                # Convert to DataFrame for display
                results_df = pd.DataFrame(results)
                
                # Style the dataframe
                def highlight_action(val):
                    if val == 'BUY':
                        return 'background-color: #d4edda'
                    elif val == 'SELL':
                        return 'background-color: #f8d7da'
                    elif val == 'HOLD':
                        return 'background-color: #fff3cd'
                    return ''
                
                # Apply styling
                styled_df = results_df.style.applymap(
                    highlight_action, subset=['Action']
                )
                
                # Display the table
                st.dataframe(styled_df, use_container_width=True)
                
                # Create visualization for price changes
                st.subheader("Predicted Price Changes")
                
                # Prepare data for chart
                chart_data = pd.DataFrame({
                    'Ticker': results_df['Ticker'],
                    'Change (%)': results_df['Change (%)'].str.rstrip('%').astype(float)
                })
                
                # Create bar chart
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Ticker:N', title='Stock Ticker'),
                    y=alt.Y('Change (%):Q', title='Predicted Change (%)'),
                    color=alt.condition(
                        alt.datum['Change (%)'] > 0,
                        alt.value('#4CAF50'),  # positive in green
                        alt.value('#F44336')   # negative in red
                    ),
                    tooltip=['Ticker', 'Change (%)']
                ).properties(
                    height=300
                )
                
                st.altair_chart(chart, use_container_width=True)
                
                # Recommendation summary
                buy_tickers = [row['Ticker'] for row in results if row['Action'] == 'BUY']
                sell_tickers = [row['Ticker'] for row in results if row['Action'] == 'SELL']
                hold_tickers = [row['Ticker'] for row in results if row['Action'] == 'HOLD']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### Buy")
                    if buy_tickers:
                        st.write(", ".join(buy_tickers))
                    else:
                        st.write("None")
                
                with col2:
                    st.markdown("### Sell")
                    if sell_tickers:
                        st.write(", ".join(sell_tickers))
                    else:
                        st.write("None")
                
                with col3:
                    st.markdown("### Hold")
                    if hold_tickers:
                        st.write(", ".join(hold_tickers))
                    else:
                        st.write("None")
                
                # Download button for CSV
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Comparison as CSV",
                    data=csv,
                    file_name=f"stock_comparison_{'-'.join(tickers)}.csv",
                    mime="text/csv",
                )
            
            # Display errors if any
            if errors:
                with st.expander("Errors"):
                    for error in errors:
                        st.error(error) 