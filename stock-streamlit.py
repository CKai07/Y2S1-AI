import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Import the StockPredictor class from your existing file
from stock_market_prediction_system import StockPredictor, format_prediction, test_prediction_accuracy

# Set page configuration
st.set_page_config(
    page_title="Stock Market Prediction AI",
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

# Cache the StockPredictor instances
@st.cache_resource
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

def display_historical_test(ticker, test_date=None, prediction_window=30):
    """Display historical prediction accuracy test"""
    
    with st.spinner("Testing prediction accuracy on historical data..."):
        try:
            # Get historical data up to cutoff date for training the model
            # If no test date provided, use 30 days ago + prediction window as default
            if test_date is None:
                cutoff_date = (datetime.now() - timedelta(days=prediction_window + 30)).strftime('%Y-%m-%d')
            else:
                cutoff_date = test_date
            
            # Calculate the end date for validation (cutoff date + prediction window)
            cutoff_dt = datetime.strptime(cutoff_date, '%Y-%m-%d')
            validation_end_date = (cutoff_dt + timedelta(days=prediction_window)).strftime('%Y-%m-%d')
            
            st.info(f"Using data up to: {cutoff_date}")
            st.info(f"Testing predictions against actual prices until: {validation_end_date}")
            
            # Get historical data
            stock = yf.Ticker(ticker)
            training_data = stock.history(start=(datetime.strptime(cutoff_date, '%Y-%m-%d') - timedelta(days=1095)).strftime('%Y-%m-%d'), 
                                         end=cutoff_date)
            
            if len(training_data) < 60:
                st.error(f"Not enough historical data for {ticker} before {cutoff_date}.")
                return
            
            # Get actual data for the validation period
            validation_data = stock.history(start=cutoff_date, end=validation_end_date)
            
            if len(validation_data) < 5:
                st.error(f"Not enough validation data after {cutoff_date}.")
                return
                
            st.success(f"Successfully retrieved {len(training_data)} historical data points and {len(validation_data)} validation points.")
            
            # Create a custom predictor with only the training data
            predictor = StockPredictor(ticker, day_trading=False)
            
            # Override the data with our cutoff data
            predictor.data = training_data
            predictor.calculate_technical_indicators()
            
            # Build and train models
            predictor.prepare_features()
            predictor.build_and_train_models()
            
            # Make a prediction for the end of the validation period
            prediction = predictor.predict_with_confidence()
            predicted_price = prediction["predicted_price"]
            
            # Get the actual price at the end of the validation period
            actual_price = validation_data['Close'].iloc[-1]
            
            # Calculate accuracy metrics
            price_diff = actual_price - predicted_price
            price_diff_percent = (price_diff / actual_price) * 100
            accuracy = 100 - abs(price_diff_percent)
            
            # Display Results
            st.markdown("### Prediction Accuracy Results", unsafe_allow_html=True)
            
            # Create columns for key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Predicted Price", 
                    value=f"${predicted_price:.2f}"
                )
            
            with col2:
                st.metric(
                    label="Actual Price", 
                    value=f"${actual_price:.2f}"
                )
                
            with col3:
                st.metric(
                    label="Accuracy", 
                    value=f"{accuracy:.2f}%",
                    delta=f"{price_diff_percent:.2f}%" if price_diff_percent > 0 else f"{price_diff_percent:.2f}%",
                    delta_color="normal"
                )
            
            # Check if prediction was directionally correct
            initial_price = training_data['Close'].iloc[-1]
            actual_direction = "UP" if actual_price > initial_price else "DOWN"
            predicted_direction = "UP" if predicted_price > initial_price else "DOWN"
            direction_correct = actual_direction == predicted_direction
            
            st.markdown("### Price Direction Analysis", unsafe_allow_html=True)
            dir_col1, dir_col2, dir_col3, dir_col4 = st.columns(4)
            
            with dir_col1:
                st.metric(
                    label="Initial Price", 
                    value=f"${initial_price:.2f}"
                )
                
            with dir_col2:
                st.metric(
                    label="Actual Direction", 
                    value=actual_direction
                )
                
            with dir_col3:
                st.metric(
                    label="Predicted Direction", 
                    value=predicted_direction
                )
                
            with dir_col4:
                st.metric(
                    label="Direction Correct", 
                    value="✓" if direction_correct else "✗"
                )
            
            # Calculate accuracy for every point in the validation period
            if len(validation_data) > 1:
                st.markdown("### Day-by-Day Prediction Tracking", unsafe_allow_html=True)
                
                # Calculate the range of predictions
                confidence_low = prediction["confidence_levels"]["80%"]
                confidence_mid = prediction["confidence_levels"]["70%"]
                
                # Linear interpolation between start and predicted end
                days = len(validation_data)
                daily_change = (predicted_price - initial_price) / days
                daily_errors = []
                daily_direction_correct = 0
                
                # Create a dataframe for the daily tracking
                tracking_data = []
                
                for i, (date, row) in enumerate(validation_data.iterrows()):
                    day_pred = initial_price + (daily_change * (i + 1))
                    day_actual = row['Close']
                    day_error = ((day_actual - day_pred) / day_actual) * 100
                    daily_errors.append(abs(day_error))
                    
                    # Check daily direction
                    prev_actual = validation_data['Close'].iloc[i-1] if i > 0 else initial_price
                    actual_daily_direction = "UP" if day_actual > prev_actual else "DOWN"
                    prev_pred = initial_price + (daily_change * i) if i > 0 else initial_price
                    pred_daily_direction = "UP" if day_pred > prev_pred else "DOWN"
                    daily_dir_correct = actual_daily_direction == pred_daily_direction
                    
                    if daily_dir_correct:
                        daily_direction_correct += 1
                    
                    tracking_data.append({
                        "Date": date,
                        "Predicted": day_pred,
                        "Actual": day_actual,
                        "Error %": day_error,
                        "Direction": "✓" if daily_dir_correct else "✗"
                    })
                
                # Convert to DataFrame and display
                tracking_df = pd.DataFrame(tracking_data)
                st.dataframe(tracking_df, use_container_width=True)
                
                # Calculate average error
                avg_error = sum(daily_errors) / len(daily_errors)
                avg_accuracy = 100 - avg_error
                direction_accuracy = (daily_direction_correct / len(validation_data)) * 100
                
                # Show summary metrics
                st.markdown("### Summary Statistics", unsafe_allow_html=True)
                sum_col1, sum_col2, sum_col3 = st.columns(3)
                
                with sum_col1:
                    st.metric(
                        label="Average Error", 
                        value=f"{avg_error:.2f}%"
                    )
                
                with sum_col2:
                    st.metric(
                        label="Average Accuracy", 
                        value=f"{avg_accuracy:.2f}%"
                    )
                    
                with sum_col3:
                    st.metric(
                        label="Direction Accuracy", 
                        value=f"{direction_accuracy:.2f}%"
                    )
                
                # Check if prediction was within confidence intervals
                within_80_conf = min(confidence_low, confidence_mid) <= actual_price <= max(confidence_low, confidence_mid)
                
                if within_80_conf:
                    st.success("✓ Actual final price was WITHIN the predicted confidence intervals")
                else:
                    st.error("✗ Actual final price was OUTSIDE the predicted confidence intervals")
                
                # Plot the predictions vs actual
                st.markdown("### Prediction vs Actual Prices", unsafe_allow_html=True)
                
                fig = go.Figure()
                
                # Add the actual prices
                fig.add_trace(
                    go.Scatter(
                        x=validation_data.index,
                        y=validation_data['Close'],
                        mode='lines+markers',
                        name='Actual Price',
                        line=dict(color='blue', width=2)
                    )
                )
                
                # Add the predicted trend line
                pred_dates = [cutoff_dt + timedelta(days=i) for i in range(days + 1)]
                pred_prices = [initial_price + (daily_change * i) for i in range(days + 1)]
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=pred_prices,
                        mode='lines',
                        line=dict(color='red', width=2, dash='dash'),
                        name='Predicted Trend'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{ticker} - Prediction vs Actual Prices",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during accuracy testing: {str(e)}")

def main():
    """Main Streamlit application"""
    st.markdown('<h1 class="main-header">Stock Market Prediction AI</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## Configuration")
    
    # Add app selection
    app_mode = st.sidebar.selectbox(
        "Choose the App Mode",
        ["📊 Make Predictions", "📈 Plot Prediction", "🧪 Test Historical Accuracy"]
    )
    
    # Common ticker input
    ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").strip().upper()
    
    if app_mode == "📊 Make Predictions":
        st.markdown('<h2 class="sub-header">Stock Price Predictions</h2>', unsafe_allow_html=True)
        
        # Prediction type selection
        prediction_type = st.sidebar.radio(
            "Prediction Type",
            ["Day Trading", "Long-term Investment"]
        )
        
        day_trading = prediction_type == "Day Trading"
        
        # Force update option
        force_update = st.sidebar.checkbox("Force Data Refresh", True)
        
        if st.sidebar.button("Generate Prediction"):
            if not ticker:
                st.warning("Please enter a valid ticker symbol.")
            else:
                try:
                    # Create the predictor
                    predictor = get_predictor(ticker, day_trading)
                    
                    if predictor:
                        with st.spinner("Generating prediction..."):
                            # Make the prediction
                            prediction = predictor.predict_with_confidence(force_update=force_update)
                            # Display results
                            display_prediction_results(prediction)
                            
                            # Display some of the historical data
                            st.markdown("### Historical Data")
                            st.dataframe(predictor.data.tail(10), use_container_width=True)
                            
                            # Option to view technical indicators
                            if st.checkbox("Show Technical Indicators"):
                                st.markdown("### Technical Indicators")
                                st.dataframe(predictor.features.tail(10), use_container_width=True)
                            
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
    
    elif app_mode == "📈 Plot Prediction":
        st.markdown('<h2 class="sub-header">Prediction Visualization</h2>', unsafe_allow_html=True)
        
        # Prediction type selection
        prediction_type = st.sidebar.radio(
            "Prediction Type",
            ["Day Trading", "Long-term Investment"]
        )
        
        day_trading = prediction_type == "Day Trading"
        
        if st.sidebar.button("Generate Plot"):
            if not ticker:
                st.warning("Please enter a valid ticker symbol.")
            else:
                try:
                    # Create the predictor
                    predictor = get_predictor(ticker, day_trading)
                    
                    if predictor:
                        # Plot the prediction
                        prediction = plot_prediction_streamlit(predictor)
                        
                        # Display a summary of the prediction below the plot
                        st.markdown("### Prediction Summary")
                        display_prediction_results(prediction)
                        
                except ValueError as e:
                    if "hourly data" in str(e) and day_trading:
                        st.error(f"Error: {str(e)}")
                        st.info("Day trading prediction requires sufficient hourly data. This is often limited for some stocks.")
                        
                        if st.button("Try Long-term prediction instead"):
                            predictor = get_predictor(ticker, day_trading=False)
                            if predictor:
                                prediction = plot_prediction_streamlit(predictor)
                                display_prediction_results(prediction)
                    else:
                        st.error(f"Error: {str(e)}")
                
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
    
    elif app_mode == "🧪 Test Historical Accuracy":
        st.markdown('<h2 class="sub-header">Historical Prediction Accuracy Test</h2>', unsafe_allow_html=True)
        
        # Test configuration in sidebar
        st.sidebar.markdown("### Test Configuration")
        
        use_custom_date = st.sidebar.checkbox("Use Specific Test Date", False)
        
        test_date = None
        if use_custom_date:
            test_date = st.sidebar.date_input(
                "Test Date",
                value=datetime.now() - timedelta(days=60),
                max_value=datetime.now() - timedelta(days=1)
            ).strftime('%Y-%m-%d')
        
        prediction_window = st.sidebar.slider(
            "Prediction Window (days)",
            min_value=7,
            max_value=90,
            value=30,
            step=1
        )
        
        if st.sidebar.button("Run Accuracy Test"):
            if not ticker:
                st.warning("Please enter a valid ticker symbol.")
            else:
                display_historical_test(ticker, test_date, prediction_window)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="info-text">
        This application uses machine learning models to predict stock prices.
        The predictions are based on historical data and are not financial advice.
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 
