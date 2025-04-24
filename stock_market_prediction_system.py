import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
import warnings
import pytz
import time
from pandas.tseries.holiday import USFederalHolidayCalendar
warnings.filterwarnings('ignore')

# Try to import plotly but make it optional
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not installed. Visualizations will be limited to matplotlib.")

class StockPredictor:
    def __init__(self, ticker_symbol, day_trading=False):
        """
        Initialize the Stock Prediction system with both day trading and long-term capabilities
        
        Parameters:
        -----------
        ticker_symbol : str
            The stock ticker symbol (e.g., 'AAPL' for Apple)
        day_trading : bool
            If True, use hourly data for day trading predictions
            If False, use daily data for long-term investment predictions
        """
        self.ticker = ticker_symbol
        self.day_trading = day_trading
        self.interval = '1h' if day_trading else '1d'
        self.period = '7d' if day_trading else '3y'  # Increased data period
        self.prediction_days = 24 if day_trading else 60  # 24 hours or 60 days
        
        # Model containers
        self.lstm_model = None
        self.rf_model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Data containers
        self.data = None
        self.features = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        # Data cache timestamp
        self.last_data_update = None
        self.data_cache_expiry = 600  # 10 minutes for hourly data, will be adjusted for daily
        
        # Check market status
        self.market_open = self.is_market_open()
        
        # Load the stock data
        self.load_data(force_update=True)
        
    def is_market_open(self):
        """Check if the US stock market is currently open"""
        now = datetime.now(pytz.timezone('US/Eastern'))
        
        # Check if it's a weekend
        if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            return False
        
        # Check if it's a holiday
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=now.date(), end=now.date())
        if now.date() in holidays:
            return False
        
        # Check if it's within trading hours (9:30 AM to 4:00 PM ET)
        market_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def is_data_fresh(self):
        """Check if the cached data is still fresh or needs updating"""
        if self.last_data_update is None:
            return False
            
        current_time = time.time()
        # If market is closed, data stays fresh longer
        if not self.market_open:
            # For day trading data during closed market, cache for 1 hour
            if self.day_trading:
                expiry = 3600  # 1 hour in seconds
            else:
                # For daily data during closed market, cache until next day
                now = datetime.now(pytz.timezone('US/Eastern'))
                if now.hour < 9:  # Before market opens
                    expiry = (9 * 3600) - (now.hour * 3600 + now.minute * 60 + now.second)  # Time until 9am
                else:
                    # If after market close, cache until next day 9am
                    expiry = (24 - now.hour + 9) * 3600 - (now.minute * 60 + now.second)
        else:
            # If market is open, use standard cache time
            expiry = self.data_cache_expiry
            
        return (current_time - self.last_data_update) < expiry
        
    def load_data(self, force_update=False):
        """
        Load stock data from Yahoo Finance
        
        Parameters:
        -----------
        force_update : bool
            If True, force reload data even if cache is fresh
        """
        # Check if we have fresh data already cached
        if not force_update and self.is_data_fresh() and self.data is not None:
            print(f"Using cached data for {self.ticker} (last updated {datetime.fromtimestamp(self.last_data_update).strftime('%Y-%m-%d %H:%M:%S')})")
            return
            
        try:
            print(f"Fetching fresh data for {self.ticker}...")
            stock = yf.Ticker(self.ticker)
            
            if self.day_trading:
                # For day trading, get hourly data with increased history
                self.data = stock.history(interval=self.interval, period=self.period)
                
                if len(self.data) < 50:  # Increased minimum data points requirement
                    # Try with more days
                    print(f"Not enough hourly data (only {len(self.data)} points). Trying with more days...")
                    self.data = stock.history(interval=self.interval, period='10d')
                    
                    if len(self.data) < 50:
                        raise ValueError(f"Not enough hourly data for {self.ticker}. Only found {len(self.data)} data points, need at least 50.")
            else:
                # For long-term investment, get 3 years of daily data
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')  # 3 years
                self.data = stock.history(start=start_date, end=end_date)
                
                if len(self.data) < 60:
                    raise ValueError(f"Not enough daily data for {self.ticker}. Only found {len(self.data)} data points, need at least 60.")
            
            print(f"Successfully retrieved {len(self.data)} data points.")
            
            # Update the timestamp for the data cache
            self.last_data_update = time.time()
            
            # Check if market is closed and adjust current price if needed
            if not self.market_open and len(self.data) > 0:
                print("Market is currently closed. Using latest available data.")
                
                # Try to get the latest real-time price quote to update the last row
                try:
                    real_time_data = stock.history(period='1d', interval='1m')
                    if len(real_time_data) > 0:
                        # Update the last row's Close price with the most recent real-time price
                        latest_price = real_time_data['Close'].iloc[-1]
                        print(f"Using real-time quote: ${latest_price:.2f} (Last close: ${self.data['Close'].iloc[-1]:.2f})")
                        # Create a copy of the last row and update its Close price
                        last_row = self.data.iloc[-1].copy()
                        last_row['Close'] = latest_price
                        # Adjust High/Low if necessary
                        if latest_price > last_row['High']:
                            last_row['High'] = latest_price
                        if latest_price < last_row['Low']:
                            last_row['Low'] = latest_price
                        # Replace the last row with the updated one
                        self.data.iloc[-1] = last_row
                except Exception as e:
                    print(f"Could not get real-time quote: {str(e)}")
            
            # Calculate technical indicators
            self.calculate_technical_indicators()
            
        except Exception as e:
            raise ValueError(f"Error fetching data for {self.ticker}: {str(e)}")
            
    def calculate_technical_indicators(self):
        """Calculate technical indicators for feature engineering"""
        df = self.data.copy()
        
        # Simple Moving Averages
        if self.day_trading:
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
        else:
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        if self.day_trading:
            df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        else:
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Modified RSI calculation with fewer NaN values
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        # Avoid division by zero
        avg_loss = avg_loss.replace(0, 0.001)
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        fast_ema = df['Close'].ewm(span=12, adjust=False).mean()
        slow_ema = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = fast_ema - slow_ema
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']  # Added MACD Histogram
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_StdDev'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']  # Added BB Width
        df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])  # Added BB %B
        
        # Volume indicators
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Price momentum
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Momentum_5'] = df['Close'].pct_change(periods=5)
        df['Price_Momentum_10'] = df['Close'].pct_change(periods=10)
        
        # Volatility
        df['Volatility'] = df['Close'].rolling(window=20).std() / df['Close']
        
        # Average True Range (ATR) for volatility
        df['High-Low'] = df['High'] - df['Low']
        df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
        df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # Store the processed data
        self.data = df.dropna()
        
    def prepare_features(self):
        """Prepare features for model training"""
        # Select relevant features
        feature_columns = [
            'Close', 'Volume', 'RSI', 'MACD', 'MACD_Histogram', 'Volume_Change', 
            'Price_Change', 'Volatility', 'ATR', 'OBV', 'BB_Width', 'BB_Percent'
        ]
        
        # Add specific features for day trading or long-term investment
        if self.day_trading:
            feature_columns.extend(['SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'Price_Momentum_5'])
        else:
            feature_columns.extend(['SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'Price_Momentum_10'])
        
        # Make sure all columns exist in the dataset
        available_columns = [col for col in feature_columns if col in self.data.columns]
        if len(available_columns) < len(feature_columns):
            missing = set(feature_columns) - set(available_columns)
            print(f"Warning: Missing columns: {missing}. Using only available columns.")
            feature_columns = available_columns
            
        # Create feature dataset
        self.features = self.data[feature_columns].copy()
        
        # Check for NaN values and fill them if needed
        if self.features.isna().any().any():
            print("Warning: NaN values found in features. Filling with forward/backward fill.")
            self.features = self.features.fillna(method='ffill').fillna(method='bfill')
            
        # Adjust prediction_days if needed
        if len(self.features) <= self.prediction_days:
            old_days = self.prediction_days
            self.prediction_days = max(5, len(self.features) // 3)  # Use at most 1/3 of available data
            print(f"Warning: Not enough data points for prediction window. Reducing from {old_days} to {self.prediction_days} days.")
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(self.features)
        
        # Create sequences for LSTM
        X = []
        y = []
        
        for i in range(self.prediction_days, len(scaled_data)):
            X.append(scaled_data[i-self.prediction_days:i])
            y.append(scaled_data[i, 0])  # Target is the Close price (first column)
        
        # Check that we have enough sequences
        if len(X) < 10:
            raise ValueError(f"Not enough sequences for training. Only {len(X)} sequences generated. Try a different stock or use long-term mode.")
            
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets (80% train, 20% test)
        split = int(0.8 * len(X))
        if split == 0:
            split = 1  # Ensure at least one sample for training
            
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]
        
        # Also prepare data for Random Forest (use the last value of each sequence)
        self.X_train_rf = np.array([seq[-1] for seq in self.X_train])
        self.X_test_rf = np.array([seq[-1] for seq in self.X_test])
        
    def build_and_train_models(self, epochs=50, batch_size=32):
        """Build and train both LSTM and Random Forest models"""
        print("Preparing features...")
        try:
            self.prepare_features()
            
            # Adjust batch size if needed
            if len(self.X_train) < batch_size:
                old_batch = batch_size
                batch_size = max(1, len(self.X_train) // 2)  # Use at most half of training samples
                print(f"Warning: Reducing batch size from {old_batch} to {batch_size} due to small dataset.")
                
            # Ensure we have enough data for validation
            validation_split = 0.2
            if len(self.X_train) < 5:  # Need at least 5 samples to have a validation split
                validation_split = 0.0
                print("Warning: Not enough data for validation split. Training without validation.")
            
            # Build and train LSTM model with improved architecture
            print("Building and training LSTM model...")
            self.lstm_model = Sequential()
            
            # First LSTM layer with more units
            self.lstm_model.add(LSTM(units=100, return_sequences=True, 
                            input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
            self.lstm_model.add(Dropout(0.4))  # Increased dropout
            
            # Additional LSTM layer
            self.lstm_model.add(LSTM(units=50, return_sequences=True))
            self.lstm_model.add(Dropout(0.3))
            
            # Final LSTM layer
            self.lstm_model.add(LSTM(units=50, return_sequences=False))
            self.lstm_model.add(Dropout(0.3))
            
            # Output layer
            self.lstm_model.add(Dense(units=1))
            
            # Use RMSprop optimizer which often works better for time series
            self.lstm_model.compile(optimizer='rmsprop', loss='mean_squared_error')
            
            # Only use early stopping if we have validation data
            callbacks = []
            if validation_split > 0:
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=15,  # Increased patience
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
            
            # Adjust epochs for small datasets
            if len(self.X_train) < 20:
                old_epochs = epochs
                epochs = min(20, epochs)  # Reduce to at most 20 epochs for small datasets
                print(f"Warning: Small dataset detected. Reducing epochs from {old_epochs} to {epochs}.")
                
            self.lstm_model.fit(
                self.X_train, 
                self.y_train, 
                epochs=epochs, 
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            # Build and train Random Forest model with improved parameters
            print("Building and training Random Forest model...")
            # Adjust n_estimators for small datasets
            n_estimators = 200  # Increased from 100
            if len(self.X_train_rf) < 20:
                n_estimators = max(10, len(self.X_train_rf))
                print(f"Warning: Small dataset. Reducing Random Forest estimators to {n_estimators}.")
                
            self.rf_model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=min(15, len(self.X_train_rf)),  # Increased max_depth
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42
            )
            
            self.rf_model.fit(self.X_train_rf, self.y_train)
            
            # Evaluate models if we have test data
            if len(self.X_test) > 0:
                self.evaluate_models()
            else:
                print("Warning: No test data available for evaluation.")
                
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise ValueError(f"Failed to build and train models: {str(e)}")
        
    def evaluate_models(self):
        """Evaluate both models on test data"""
        # LSTM predictions
        lstm_predictions = self.lstm_model.predict(self.X_test)
        
        # RF predictions
        rf_predictions = self.rf_model.predict(self.X_test_rf)
        
        # Prepare to convert predictions back to original scale
        lstm_pred_full = np.zeros((len(lstm_predictions), self.features.shape[1]))
        lstm_pred_full[:, 0] = lstm_predictions.flatten()
        
        rf_pred_full = np.zeros((len(rf_predictions), self.features.shape[1]))
        rf_pred_full[:, 0] = rf_predictions
        
        # Inverse transform
        lstm_predictions = self.scaler.inverse_transform(lstm_pred_full)[:, 0]
        rf_predictions = self.scaler.inverse_transform(rf_pred_full)[:, 0]
        
        # Get actual values
        y_test_full = np.zeros((len(self.y_test), self.features.shape[1]))
        y_test_full[:, 0] = self.y_test
        y_test_actual = self.scaler.inverse_transform(y_test_full)[:, 0]
        
        # Calculate metrics
        lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_predictions))
        rf_rmse = np.sqrt(mean_squared_error(y_test_actual, rf_predictions))
        lstm_mae = mean_absolute_error(y_test_actual, lstm_predictions)
        rf_mae = mean_absolute_error(y_test_actual, rf_predictions)
        
        # Calculate R² score
        lstm_r2 = r2_score(y_test_actual, lstm_predictions)
        rf_r2 = r2_score(y_test_actual, rf_predictions)
        
        print(f"\nModel Evaluation:")
        print(f"LSTM RMSE: ${lstm_rmse:.2f}")
        print(f"Random Forest RMSE: ${rf_rmse:.2f}")
        print(f"LSTM MAE: ${lstm_mae:.2f}")
        print(f"Random Forest MAE: ${rf_mae:.2f}")
        print(f"LSTM R² Score: {lstm_r2:.4f}")
        print(f"Random Forest R² Score: {rf_r2:.4f}")
        
        # Calculate combined model accuracy
        combined_predictions = 0.7 * lstm_predictions + 0.3 * rf_predictions
        combined_rmse = np.sqrt(mean_squared_error(y_test_actual, combined_predictions))
        combined_mae = mean_absolute_error(y_test_actual, combined_predictions)
        combined_r2 = r2_score(y_test_actual, combined_predictions)
        
        print(f"Combined Model RMSE: ${combined_rmse:.2f}")
        print(f"Combined Model MAE: ${combined_mae:.2f}")
        print(f"Combined Model R² Score: {combined_r2:.4f}")
        
    def predict_with_confidence(self, ticker=None, force_update=True):
        """
        Make predictions with confidence levels
        
        Parameters:
        -----------
        ticker : str, optional
            New ticker symbol to predict for
        force_update : bool, default=True
            Whether to force refresh data, even if cache is fresh
            
        Returns a dict with predictions at different confidence levels
        """
        # If a new ticker is provided, load data for that ticker
        if ticker is not None and ticker != self.ticker:
            self.ticker = ticker
            # Check market status for the new ticker
            self.market_open = self.is_market_open()
            self.load_data(force_update=True)  # Always force update for new ticker
            self.prepare_features()
            self.build_and_train_models()
        else:
            # For same ticker, check if we need to refresh the data
            # Store the previous close price to compare with new data
            previous_close = None
            if self.data is not None and len(self.data) > 0:
                previous_close = self.data['Close'].iloc[-1]
                
            # Reload data if forced or cache expired
            if force_update or not self.is_data_fresh():
                self.load_data(force_update=True)
                self.prepare_features()
                # Only rebuild models if data actually changed
                if previous_close is not None and previous_close != self.data['Close'].iloc[-1]:
                    print(f"Price changed from ${previous_close:.2f} to ${self.data['Close'].iloc[-1]:.2f}. Rebuilding models...")
                    self.build_and_train_models()
        
        # Make sure models are trained
        if self.lstm_model is None or self.rf_model is None:
            self.build_and_train_models()
        
        # Get the latest data sequence for prediction
        latest_data = self.features.values[-self.prediction_days:]
        latest_scaled = self.scaler.transform(latest_data)
        
        # Reshape for LSTM
        lstm_input = np.array([latest_scaled])
        
        # Get last day features for RF
        rf_input = np.array([latest_scaled[-1]])
        
        # Make predictions
        lstm_pred = self.lstm_model.predict(lstm_input, verbose=0)  # Silent prediction
        rf_pred = self.rf_model.predict(rf_input)
        
        # Adjust weights based on model performance (measured in evaluate_models)
        # Default weights: 70% LSTM, 30% RF
        lstm_weight = 0.7
        rf_weight = 0.3
        
        # Combine predictions with weights
        combined_pred = (lstm_weight * lstm_pred + rf_weight * rf_pred)[0][0]
        
        # Get standard deviation from test predictions for confidence intervals
        lstm_test_preds = self.lstm_model.predict(self.X_test, verbose=0)  # Silent prediction
        rf_test_preds = self.rf_model.predict(self.X_test_rf)
        combined_test_preds = lstm_weight * lstm_test_preds.flatten() + rf_weight * rf_test_preds
        
        # Calculate error on test data
        y_test_full = np.zeros((len(self.y_test), self.features.shape[1]))
        y_test_full[:, 0] = self.y_test
        y_test_actual = self.scaler.inverse_transform(y_test_full)[:, 0]
        
        combined_test_full = np.zeros((len(combined_test_preds), self.features.shape[1]))
        combined_test_full[:, 0] = combined_test_preds
        combined_test_preds_orig = self.scaler.inverse_transform(combined_test_full)[:, 0]
        
        # Calculate percentage errors
        perc_errors = np.abs((combined_test_preds_orig - y_test_actual) / y_test_actual)
        
        # If market is closed, adjust confidence levels with increased uncertainty
        uncertainty_factor = 1.0
        if not self.market_open:
            uncertainty_factor = 1.2  # 20% more uncertainty when market is closed
            print("Market is closed. Adjusting prediction confidence levels to account for overnight uncertainty.")
        
        # Calculate error margins for confidence levels (with uncertainty factor)
        error_80 = np.percentile(perc_errors, 80) * uncertainty_factor
        error_70 = np.percentile(perc_errors, 70) * uncertainty_factor
        error_65 = np.percentile(perc_errors, 65) * uncertainty_factor
        
        # Transform prediction back to original scale
        pred_full = np.zeros((1, self.features.shape[1]))
        pred_full[0, 0] = combined_pred
        predicted_price = self.scaler.inverse_transform(pred_full)[0, 0]
        
        # Get current price
        current_price = self.data['Close'].iloc[-1]
        
        # Calculate target prices with confidence margins
        if predicted_price > current_price:
            target_80 = predicted_price * (1 + error_80)
            target_70 = predicted_price * (1 + error_70)
            target_65 = predicted_price * (1 + error_65)
        else:
            target_80 = predicted_price * (1 - error_80)
            target_70 = predicted_price * (1 - error_70)
            target_65 = predicted_price * (1 - error_65)
        
        # Generate trading signal
        price_change = ((predicted_price - current_price) / current_price) * 100
        
        # Market closure affects trading decision thresholds
        if self.day_trading:
            # For day trading, more sensitive thresholds
            buy_threshold = 1.5 if self.market_open else 2.0  # Higher threshold when market is closed
            sell_threshold = -1.0 if self.market_open else -1.5  # Lower threshold when market is closed
            
            if price_change > buy_threshold:
                action = "BUY"
                reason = self.generate_buy_reason()
            elif price_change < sell_threshold:
                action = "SELL"
                reason = self.generate_sell_reason()
            else:
                action = "HOLD"
                reason = "Price expected to stay within normal trading range"
        else:
            # For long-term investment
            buy_threshold = 3.0 if self.market_open else 4.0  # Higher threshold when market is closed
            sell_threshold = -2.5 if self.market_open else -3.5  # Lower threshold when market is closed
            
            if price_change > buy_threshold:
                action = "BUY"
                reason = self.generate_buy_reason()
            elif price_change < sell_threshold:
                action = "SELL"
                reason = self.generate_sell_reason()
            else:
                action = "HOLD"
                reason = "Price expected to remain relatively stable"
        
        # Get the data timestamp for when the current price was last updated
        last_price_timestamp = None
        if self.last_data_update is not None:
            last_price_timestamp = datetime.fromtimestamp(self.last_data_update).strftime('%Y-%m-%d %H:%M:%S')
        
        # Create prediction results
        result = {
            "ticker": self.ticker,
            "current_price": current_price,
            "predicted_price": predicted_price,  # Added raw predicted price
            "prediction_type": "Day Trading" if self.day_trading else "Long-term Investment",
            "market_status": "Open" if self.market_open else "Closed",
            "last_updated": last_price_timestamp,
            "confidence_levels": {
                "80%": target_80,
                "70%": target_70,
                "65%": target_65
            },
            "action": action,
            "reason": reason,
            "expected_change_percent": price_change
        }
        
        return result
    
    def generate_buy_reason(self):
        """Generate a reason for buy recommendation based on indicators"""
        reasons = []
        
        # Check for uptrend in price
        if self.data['Close'].iloc[-1] > self.data['Close'].iloc[-5]:
            reasons.append("uptrend in price")
        
        # Check volume
        if self.data['Volume'].iloc[-1] > self.data['Volume_MA'].iloc[-1]:
            reasons.append("high volume")
        
        # Check RSI
        if 40 < self.data['RSI'].iloc[-1] < 60:
            reasons.append("neutral RSI")
        elif self.data['RSI'].iloc[-1] < 40:
            reasons.append("oversold conditions (low RSI)")
        
        # Check MACD
        if self.data['MACD'].iloc[-1] > self.data['Signal_Line'].iloc[-1]:
            reasons.append("MACD above signal line")
        
        # Check moving averages
        if self.day_trading:
            if self.data['Close'].iloc[-1] > self.data['SMA_5'].iloc[-1]:
                reasons.append("price above 5-period SMA")
        else:
            if self.data['Close'].iloc[-1] > self.data['SMA_20'].iloc[-1]:
                reasons.append("price above 20-day SMA")
        
        if not reasons:
            reasons.append("positive price forecast")
        
        return ", ".join(reasons)
    
    def generate_sell_reason(self):
        """Generate a reason for sell recommendation based on indicators"""
        reasons = []
        
        # Check for downtrend in price
        if self.data['Close'].iloc[-1] < self.data['Close'].iloc[-5]:
            reasons.append("downtrend in price")
        
        # Check RSI
        if self.data['RSI'].iloc[-1] > 70:
            reasons.append("overbought conditions (high RSI)")
        
        # Check MACD
        if self.data['MACD'].iloc[-1] < self.data['Signal_Line'].iloc[-1]:
            reasons.append("MACD below signal line")
        
        # Check moving averages
        if self.day_trading:
            if self.data['Close'].iloc[-1] < self.data['SMA_5'].iloc[-1]:
                reasons.append("price below 5-period SMA")
        else:
            if self.data['Close'].iloc[-1] < self.data['SMA_20'].iloc[-1]:
                reasons.append("price below 20-day SMA")
        
        if not reasons:
            reasons.append("negative price forecast")
            
        return ", ".join(reasons)
    
    def plot_prediction(self):
        """Plot the historical data and predictions"""
        if not PLOTLY_AVAILABLE:
            self._plot_with_matplotlib()
            return
            
        # Get prediction with confidence
        prediction = self.predict_with_confidence(force_update=True)
        
        # Create figure
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.1,
                           subplot_titles=('Price', 'Volume'),
                           row_heights=[0.7, 0.3])
        
        # Add price data
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add volume data
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Add prediction points
        current_price = prediction["current_price"]
        last_date = self.data.index[-1]
        next_date = last_date + timedelta(days=1 if self.day_trading else 30)
        
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
        title_text = f"{self.ticker} - {prediction['prediction_type']} Prediction"
        if prediction["market_status"] == "Closed":
            title_text += " (Market Closed)"
            
        fig.update_layout(
            title=title_text,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=800,
            width=1200,
            showlegend=True
        )
        
        # Display the prediction summary first
        print("\n=== Prediction Summary ===")
        format_prediction(prediction)
        
        # Then display the chart
        print("\nDisplaying chart...")
        
        # Use a separate thread to show the plot so it doesn't block the menu
        import threading
        
        def show_plot():
            try:
                # Display the plot
                fig.show()
            except Exception as e:
                print(f"Error displaying plot: {str(e)}")
        
        # Start the plot in a separate thread
        plot_thread = threading.Thread(target=show_plot)
        plot_thread.daemon = True  # Set as daemon so it doesn't prevent program exit
        plot_thread.start()
        
        # Let the user know they can continue
        print("\nPlot displayed in separate window.")
        print("You can continue using the program while viewing the plot.")
        print("When you're done viewing the plot, you can close it and return to this window.")
        
        return prediction
    
    def _plot_with_matplotlib(self):
        """Fallback plotting with matplotlib if plotly is not available"""
        prediction = self.predict_with_confidence(force_update=True)
        
        # Display the prediction summary first
        print("\n=== Prediction Summary ===")
        format_prediction(prediction)
        
        # Use matplotlib in non-blocking mode
        import matplotlib
        # Use a non-blocking backend if possible
        try:
            matplotlib.use('TkAgg')  # Try to use TkAgg which is better for interactive plots
        except Exception:
            pass  # If it fails, use the default backend
            
        plt.figure(figsize=(12, 8))
        
        # Plot historical prices
        plt.subplot(2, 1, 1)
        plt.plot(self.data.index, self.data['Close'], label='Close Price')
        
        # Add prediction points
        current_price = prediction["current_price"]
        last_date = self.data.index[-1]
        next_date = last_date + timedelta(days=1 if self.day_trading else 30)
        
        confidence_levels = prediction["confidence_levels"]
        
        plt.scatter([last_date], [current_price], color='blue', s=100, label='Current Price')
        plt.scatter([next_date], [confidence_levels["80%"]], color='green', s=100, label='80% Confidence')
        plt.scatter([next_date], [confidence_levels["70%"]], color='orange', s=100, label='70% Confidence')
        plt.scatter([next_date], [confidence_levels["65%"]], color='red', s=100, label='65% Confidence')
        plt.scatter([next_date], [prediction["predicted_price"]], color='purple', s=150, marker='*', label='Raw Prediction')
        
        title_text = f"{self.ticker} - {prediction['prediction_type']} Prediction"
        if prediction["market_status"] == "Closed":
            title_text += " (Market Closed)"
        plt.title(title_text)
        plt.ylabel('Price ($)')
        plt.legend()
        
        # Plot volume
        plt.subplot(2, 1, 2)
        plt.bar(self.data.index, self.data['Volume'], color='gray', alpha=0.5)
        plt.ylabel('Volume')
        plt.xlabel('Date')
        
        plt.tight_layout()
        
        # Then display the chart in a non-blocking way
        print("\nDisplaying chart...")
        
        # Use a separate thread to show the plot
        import threading
        
        def show_plot():
            try:
                # Show the plot in non-blocking mode
                plt.ion()  # Turn on interactive mode
                plt.show()
                
                # Keep the plot window open without blocking the main thread
                import time
                start_time = time.time()
                # Keep the plot alive for a few seconds to make sure it's displayed
                while time.time() - start_time < 2 and plt.get_fignums():
                    plt.pause(0.1)
                    
                # Note: The plot window will stay open until the user closes it
            except Exception as e:
                print(f"Error displaying plot: {str(e)}")
        
        # Start the plot in a separate thread
        plot_thread = threading.Thread(target=show_plot)
        plot_thread.daemon = True  # Set as daemon so it doesn't prevent program exit
        plot_thread.start()
        
        # Let the user know they can continue
        print("\nPlot displayed in separate window.")
        print("You can continue using the program while viewing the plot.")
        print("When you're done viewing the plot, you can close it and return to this window.")
        
        return prediction

    def test_historical_accuracy(self, days=30):
        """
        Test the accuracy of the prediction system on historical data
        
        Args:
            days (int): Number of days to test
            
        Returns:
            dict: Dictionary with test results including RMSE, MAE, R² score,
                  actual prices, predicted prices, dates, and errors
        """
        try:
            # Load the most recent data
            self.load_data(force_update=True)
            
            if len(self.data) < days + 60:
                raise ValueError(f"Not enough historical data for {self.ticker}. Need at least {days + 60} days.")
            
            # Prepare results storage
            actual_prices = []
            predicted_prices = []
            dates = []
            errors = []
            
            # For each day in the test range
            for i in range(days):
                # Use data up to (today - days + i) to predict the price for (today - days + i + 1)
                test_day_idx = len(self.data) - days + i
                
                # Create a copy of the data up to the test day
                test_data = self.data.iloc[:test_day_idx].copy()
                
                # Store actual price for the next day
                next_day = self.data.iloc[test_day_idx]
                actual_price = next_day['Close']
                actual_date = self.data.index[test_day_idx]
                
                # Use the data up to test_day to predict the next day
                predictor_test = StockPredictor(self.ticker)
                predictor_test.data = test_data
                predictor_test.calculate_technical_indicators()
                predictor_test.prepare_features()
                predictor_test.build_and_train_models()
                
                # Get prediction
                prediction = predictor_test.predict_with_confidence()
                predicted_price = prediction["predicted_price"]
                
                # Calculate error
                error = actual_price - predicted_price
                
                # Store results
                actual_prices.append(actual_price)
                predicted_prices.append(predicted_price)
                dates.append(actual_date)
                errors.append(error)
            
            # Calculate metrics
            rmse = np.sqrt(np.mean(np.square(errors)))
            mae = np.mean(np.abs(errors))
            
            # Calculate R² score if we have enough data points
            if len(actual_prices) > 1:
                r2 = 1 - (np.sum(np.square(errors)) / np.sum(np.square(actual_prices - np.mean(actual_prices))))
            else:
                r2 = 0
            
            # Return results
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'actual_prices': actual_prices,
                'predicted_prices': predicted_prices,
                'dates': dates,
                'errors': errors
            }
            
        except Exception as e:
            print(f"Error in historical accuracy test: {str(e)}")
            raise e

def format_prediction(prediction):
    """Format the prediction results for display"""
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
    
    print(f"\nStock: {ticker} (Current Price: ${current_price:.2f})")
    print(f"Last Updated: {last_updated}")
    print(f"Market Status: {market_status}")
    print(f"Prediction Type: {prediction_type}")
    print(f"Raw Predicted Price: ${predicted_price:.2f} (Expected Change: {expected_change:.2f}%)")
    print(f"- 80% confidence → Target Price: ${confidence_levels['80%']:.2f}")
    print(f"- 70% confidence → Target Price: ${confidence_levels['70%']:.2f}")
    print(f"- 65% confidence → Target Price: ${confidence_levels['65%']:.2f}")
    print(f"\nAction: {action}")
    print(f"Reason: {reason}")
    
    if market_status == "Closed":
        print("\nNote: Market is currently closed. Predictions may have higher uncertainty.")
        print("Consider waiting for market open for more accurate predictions.")

def test_prediction(ticker="AAPL"):
    """Test the stock prediction functionality with a sample ticker"""
    print(f"Testing prediction for {ticker}...")
    
    try:
        # Test long-term investment prediction
        print("\nGenerating Long-term Investment prediction...")
        long_term = StockPredictor(ticker, day_trading=False)
        long_term_prediction = long_term.predict_with_confidence(force_update=True)
        format_prediction(long_term_prediction)
        
        # Test day trading prediction
        print("\nGenerating Day Trading prediction...")
        day_trading = StockPredictor(ticker, day_trading=True)
        day_trading_prediction = day_trading.predict_with_confidence(force_update=True)
        format_prediction(day_trading_prediction)
        
        return True
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False

def test_prediction_accuracy(ticker, test_date=None, prediction_window=30):
    """
    Test the accuracy of the prediction system using historical data
    
    Parameters:
    -----------
    ticker : str
        The stock ticker symbol to test
    test_date : str, optional
        The cutoff date (YYYY-MM-DD) to test predictions from. If None, uses 30 days ago.
    prediction_window : int, optional
        Number of days to check accuracy for (default 30 days)
    """
    print(f"\nTesting prediction accuracy for {ticker}")
    
    # If no test date provided, use 30 days ago + prediction window as default
    if test_date is None:
        cutoff_date = (datetime.now() - timedelta(days=prediction_window + 30)).strftime('%Y-%m-%d')
    else:
        cutoff_date = test_date
    
    # Calculate the end date for validation (cutoff date + prediction window)
    cutoff_dt = datetime.strptime(cutoff_date, '%Y-%m-%d')
    validation_end_date = (cutoff_dt + timedelta(days=prediction_window)).strftime('%Y-%m-%d')
    
    print(f"Using data up to: {cutoff_date}")
    print(f"Testing predictions against actual prices until: {validation_end_date}")
    
    try:
        # Get historical data up to cutoff date for training the model
        stock = yf.Ticker(ticker)
        training_data = stock.history(start=(datetime.strptime(cutoff_date, '%Y-%m-%d') - timedelta(days=1095)).strftime('%Y-%m-%d'), 
                                     end=cutoff_date)
        
        if len(training_data) < 60:
            print(f"Not enough historical data for {ticker} before {cutoff_date}.")
            return
        
        # Get actual data for the validation period
        validation_data = stock.history(start=cutoff_date, end=validation_end_date)
        
        if len(validation_data) < 5:
            print(f"Not enough validation data after {cutoff_date}.")
            return
            
        print(f"Successfully retrieved {len(training_data)} historical data points and {len(validation_data)} validation points.")
        
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
        
        print("\n=== Prediction Accuracy Results ===")
        print(f"Prediction Date: {cutoff_date}")
        print(f"Validation End Date: {validation_end_date}")
        print(f"Predicted Price: ${predicted_price:.2f}")
        print(f"Actual Price: ${actual_price:.2f}")
        print(f"Difference: ${price_diff:.2f} ({price_diff_percent:.2f}%)")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Check if prediction was directionally correct
        initial_price = training_data['Close'].iloc[-1]
        actual_direction = "UP" if actual_price > initial_price else "DOWN"
        predicted_direction = "UP" if predicted_price > initial_price else "DOWN"
        direction_correct = actual_direction == predicted_direction
        
        print(f"\nInitial Price on {cutoff_date}: ${initial_price:.2f}")
        print(f"Actual Direction: {actual_direction}")
        print(f"Predicted Direction: {predicted_direction}")
        print(f"Direction Prediction: {'CORRECT' if direction_correct else 'INCORRECT'}")
        
        # Calculate accuracy for every point in the validation period
        if len(validation_data) > 1:
            print("\n=== Day-by-Day Prediction Tracking ===")
            print("Date\t\tPredicted\tActual\t\tError %\tDirection")
            print("-" * 70)
            
            # Calculate the range of predictions
            confidence_low = prediction["confidence_levels"]["80%"]
            confidence_mid = prediction["confidence_levels"]["70%"]
            
            # Linear interpolation between start and predicted end
            days = len(validation_data)
            daily_change = (predicted_price - initial_price) / days
            daily_errors = []
            daily_direction_correct = 0
            
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
                
                print(f"{date.strftime('%Y-%m-%d')}\t${day_pred:.2f}\t${day_actual:.2f}\t{day_error:.2f}%\t{'✓' if daily_dir_correct else '✗'}")
            
            # Calculate average error
            avg_error = sum(daily_errors) / len(daily_errors)
            avg_accuracy = 100 - avg_error
            direction_accuracy = (daily_direction_correct / len(validation_data)) * 100
            
            print("-" * 70)
            print(f"Average Error: {avg_error:.2f}%")
            print(f"Average Accuracy: {avg_accuracy:.2f}%")
            print(f"Direction Accuracy: {direction_accuracy:.2f}%")
            
            # Check if prediction was within confidence intervals
            within_80_conf = min(confidence_low, confidence_mid) <= actual_price <= max(confidence_low, confidence_mid)
            print(f"\nActual final price was {'WITHIN' if within_80_conf else 'OUTSIDE'} the predicted confidence intervals")
            
    except Exception as e:
        print(f"\nError during accuracy testing: {str(e)}")

def main():
    """Main function to run the stock prediction system"""
    print("\nStock Market Prediction AI System")
    print("================================")
    print("\nThis system predicts stock prices using machine learning for both:")
    print("- Short-term (day trading) strategy")
    print("- Long-term (investment) strategy")
    
    # Configure matplotlib for non-blocking interactive plots
    import matplotlib
    try:
        matplotlib.use('TkAgg')  # Try to use TkAgg which is better for interactive plots
        import matplotlib.pyplot as plt
        plt.ion()  # Turn on interactive mode for matplotlib
    except:
        pass  # If it fails, use the default backend
    
    # Store predictor instances for reuse
    predictors = {}
    
    while True:
        print("\nOptions:")
        print("1. Short-term (Day Trading) Prediction")
        print("2. Long-term (Investment) Prediction")
        print("3. Plot Prediction")
        print("4. Test Historical Prediction Accuracy")
        print("5. Exit")
        
        try:
            choice = input("\nEnter your choice (1-5): ")
            
            if choice == '5':
                print("Exiting. Thank you!")
                break
            
            if choice in ['1', '2']:
                ticker = input("Enter stock ticker (e.g., AAPL, TSLA): ").strip().upper()
                
                try:
                    if choice == '1':
                        # Day Trading prediction
                        try:
                            predictor = StockPredictor(ticker, day_trading=True)
                            prediction = predictor.predict_with_confidence()
                            format_prediction(prediction)
                            # Store for later refresh
                            predictors[(ticker, True)] = predictor
                        except ValueError as e:
                            if "hourly data" in str(e):
                                print(f"\nError: {str(e)}")
                                print("\nNote: Day trading prediction requires sufficient hourly data.")
                                print("This is often limited for some stocks.")
                                use_long_term = input("\nWould you like to try long-term prediction instead? (y/n): ").lower() == 'y'
                                if use_long_term:
                                    print("\nSwitching to long-term prediction model...")
                                    predictor = StockPredictor(ticker, day_trading=False)
                                    prediction = predictor.predict_with_confidence()
                                    format_prediction(prediction)
                                    # Store for later refresh
                                    predictors[(ticker, False)] = predictor
                            else:
                                raise e
                    
                    elif choice == '2':
                        # Long-term Investment prediction
                        predictor = StockPredictor(ticker, day_trading=False)
                        prediction = predictor.predict_with_confidence()
                        format_prediction(prediction)
                        # Store for later refresh
                        predictors[(ticker, False)] = predictor
                    
                except ValueError as e:
                    print(f"\nError: {str(e)}")
                    
                except Exception as e:
                    print(f"\nUnexpected error: {str(e)}")
                    print("Please try another stock or option.")
            
            elif choice == '3':
                # Plot prediction
                ticker = input("Enter stock ticker (e.g., AAPL, TSLA): ").strip().upper()
                print("\nChoose prediction type for visualization:")
                print("1. Day Trading (hourly data)")
                print("2. Long-term Investment (daily data)")
                viz_choice = input("\nEnter choice (1 or 2): ")
                
                try:
                    # Create a prediction and display summary before attempting to plot
                    day_trading = viz_choice == '1'
                    
                    try:
                        # Try to create the predictor
                        predictor = StockPredictor(ticker, day_trading=day_trading)
                        
                        # Generate prediction and show summary
                        prediction = predictor.predict_with_confidence(force_update=True)
                        print("\n=== Prediction Summary ===")
                        format_prediction(prediction)
                        
                        # Store for later refresh
                        predictors[(ticker, day_trading)] = predictor
                        
                        # Prepare to display the graph
                        print("\nPreparing graph...")
                        
                        import threading
                        import time
                        
                        def show_plot():
                            """Function to display plot in a separate thread"""
                            try:
                                if PLOTLY_AVAILABLE:
                                    # Use Plotly for better interactive graphs
                                    import plotly.graph_objects as go
                                    from plotly.subplots import make_subplots
                                    
                                    # Create the plot figure
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
                                    next_date = last_date + timedelta(days=1 if day_trading else 30)
                                    
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
                                    title_text = f"{ticker} - {prediction['prediction_type']} Prediction"
                                    if prediction["market_status"] == "Closed":
                                        title_text += " (Market Closed)"
                                        
                                    fig.update_layout(
                                        title=title_text,
                                        xaxis_title="Date",
                                        yaxis_title="Price ($)",
                                        height=800,
                                        width=1200,
                                        showlegend=True
                                    )
                                    
                                    # Show the plot directly without saving
                                    fig.show()
                                    
                                else:
                                    # Use Matplotlib as fallback
                                    import matplotlib.pyplot as plt
                                    
                                    # Turn on interactive mode
                                    plt.ion()
                                    
                                    plt.figure(figsize=(12, 8))
                                    
                                    # Plot historical prices
                                    plt.subplot(2, 1, 1)
                                    plt.plot(predictor.data.index, predictor.data['Close'], label='Close Price')
                                    
                                    # Add prediction points
                                    current_price = prediction["current_price"]
                                    last_date = predictor.data.index[-1]
                                    next_date = last_date + timedelta(days=1 if day_trading else 30)
                                    
                                    confidence_levels = prediction["confidence_levels"]
                                    
                                    plt.scatter([last_date], [current_price], color='blue', s=100, label='Current Price')
                                    plt.scatter([next_date], [confidence_levels["80%"]], color='green', s=100, label='80% Confidence')
                                    plt.scatter([next_date], [confidence_levels["70%"]], color='orange', s=100, label='70% Confidence')
                                    plt.scatter([next_date], [confidence_levels["65%"]], color='red', s=100, label='65% Confidence')
                                    plt.scatter([next_date], [prediction["predicted_price"]], color='purple', s=150, marker='*', label='Raw Prediction')
                                    
                                    title_text = f"{ticker} - {prediction['prediction_type']} Prediction"
                                    if prediction["market_status"] == "Closed":
                                        title_text += " (Market Closed)"
                                    plt.title(title_text)
                                    plt.ylabel('Price ($)')
                                    plt.legend()
                                    
                                    # Plot volume
                                    plt.subplot(2, 1, 2)
                                    plt.bar(predictor.data.index, predictor.data['Volume'], color='gray', alpha=0.5)
                                    plt.ylabel('Volume')
                                    plt.xlabel('Date')
                                    
                                    plt.tight_layout()
                                    plt.show()
                                    
                                    # Keep the plot window open without blocking
                                    plt.draw()
                                    plt.pause(0.001)  # Small pause to update the UI
                            
                            except Exception as e:
                                print(f"Error displaying plot: {str(e)}")
                        
                        # Start the plot in a separate thread
                        plot_thread = threading.Thread(target=show_plot)
                        plot_thread.daemon = True  # Set as daemon so it doesn't prevent program exit
                        plot_thread.start()
                        
                        # Let the user know they can continue
                        print("\nDisplaying graph in a separate window...")
                        print("You can continue using the program while viewing the graph.")
                        print("The graph window will stay open even as you continue using the menu.")
                        
                        # Give the plotting thread a moment to start
                        time.sleep(1)
                    
                    except ValueError as e:
                        if "hourly data" in str(e) and day_trading:
                            print(f"\nError: {str(e)}")
                            print("\nNote: Day trading prediction requires sufficient hourly data.")
                            print("This is often limited for some stocks.")
                            print("\nWould you like to try plotting with long-term data instead?")
                            try_long_term = input("(y/n): ").lower() == 'y'
                            
                            if try_long_term:
                                # Try again with long-term prediction
                                try:
                                    predictor = StockPredictor(ticker, day_trading=False)
                                    prediction = predictor.predict_with_confidence(force_update=True)
                                    
                                    # Display prediction summary
                                    print("\n=== Long-term Prediction Summary ===")
                                    format_prediction(prediction)
                                    
                                    # Store for later refresh
                                    predictors[(ticker, False)] = predictor
                                    
                                    # Continue to the next menu iteration
                                    # The user can select option 3 again to plot if desired
                                    print("\nPrediction generated successfully. You can now plot it if desired.")
                                    
                                except Exception as e:
                                    print(f"Error generating long-term prediction: {str(e)}")
                        else:
                            raise e
                    
                except Exception as e:
                    print(f"\nError: {str(e)}")
            
            elif choice == '4':
                # Test historical prediction accuracy
                ticker = input("Enter stock ticker (e.g., AAPL, TSLA): ").strip().upper()
                use_custom_date = input("Use specific test date? (y/n): ").lower() == 'y'
                
                test_date = None
                if use_custom_date:
                    test_date = input("Enter test date (YYYY-MM-DD): ").strip()
                    try:
                        # Validate date format
                        datetime.strptime(test_date, '%Y-%m-%d')
                    except ValueError:
                        print("Invalid date format. Using default (30 days ago).")
                        test_date = None
                
                prediction_window = 30
                custom_window = input("Use custom prediction window? (default: 30 days) (y/n): ").lower() == 'y'
                if custom_window:
                    try:
                        prediction_window = int(input("Enter prediction window in days: "))
                        if prediction_window < 1:
                            print("Invalid window. Using default (30 days).")
                            prediction_window = 30
                    except ValueError:
                        print("Invalid input. Using default (30 days).")
                
                test_prediction_accuracy(ticker, test_date, prediction_window)
            
            else:
                print("Invalid choice. Please enter a number between 1 and 5.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            choice = input("\nReturn to menu? (y/n): ").lower()
            if choice != 'y':
                print("Exiting. Thank you!")
                break
                
        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main() 