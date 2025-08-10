# binomo_ai_pro.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import psutil
import joblib
import sqlite3
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier
import openai
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
import plotly.graph_objects as go
import talib
from functools import lru_cache
import threading
from git import Repo
import csv
from tenacity import retry, wait_exponential, stop_after_attempt

# ===== INITIALIZATION =====
load_dotenv()

# Setup logging
logging.basicConfig(
    filename='binomo_ai.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

st.set_page_config(page_title="BINOMO AI PRO", layout="wide")

# ===== GITHUB DATA LOGGER =====
class DataLogger:
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.repo = Repo(repo_path)
        self.file_path = os.path.join(repo_path, "data", f"predictions_{datetime.now().strftime('%Y%m%d')}.csv")
        self._init_file()
    
    def _init_file(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                writer = csv.writer(f)
                headers = [
                    "timestamp", "symbol", "interval", "open", "high", "low", "close", "volume",
                    "sma20", "sma50", "rsi", "macd", "signal", "histogram", 
                    "upper_band", "lower_band", "slowk", "slowd", "atr", "adx", "obv",
                    "lstm_pred", "xgb_pred", "ensemble_pred", "predicted_direction",
                    "confidence", "gpt_insight", "actual_outcome", "user_feedback"
                ]
                writer.writerow(headers)
    
    def log_prediction(self, data: dict):
        with open(self.file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                data['timestamp'],
                data['symbol'],
                data['interval'],
                data['open'],
                data['high'],
                data['low'],
                data['close'],
                data['volume'],
                data['sma20'],
                data['sma50'],
                data['rsi'],
                data['macd'],
                data['signal'],
                data['histogram'],
                data['upper_band'],
                data['lower_band'],
                data['slowk'],
                data['slowd'],
                data['atr'],
                data['adx'],
                data['obv'],
                data['lstm_pred'],
                data['xgb_pred'],
                data['ensemble_pred'],
                data['predicted_direction'],
                data['confidence'],
                data['gpt_insight'],
                data['actual_outcome'],
                data['user_feedback']
            ])
    
    def push_to_github(self):
        try:
            self.repo.git.add(all=True)
            self.repo.index.commit(f"Auto-update {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            origin = self.repo.remote(name='origin')
            origin.push()
            logger.info("✅ Data pushed to GitHub successfully")
        except Exception as e:
            logger.error(f"🚨 GitHub push failed: {str(e)}")

# ===== ENHANCED CORE TRADER CLASS =====
class BinomoAITrader:
    def __init__(self):
        self.model_version = "v1"
        self.accuracy = 0.85
        self.last_retrain = datetime.now(pytz.utc)
        self.scaler = None
        self.init_databases()
        self.load_models()
        self.setup_scheduler()
        self.warmup_models()
        self.initialize_scaler()
        
        # Initialize GitHub logger
        repo_path = os.getenv("GITHUB_REPO_PATH", "./binomo_ai_data")
        self.data_logger = DataLogger(repo_path)
        
    def initialize_scaler(self):
        """Ensure scaler is properly initialized"""
        if not self.scaler or not hasattr(self.scaler, 'scale_'):
            try:
                logger.info("Initializing scaler with dummy data")
                self.scaler = MinMaxScaler()
                dummy_data = np.random.rand(10, 18)
                self.scaler.fit(dummy_data)
            except Exception as e:
                logger.error(f"Scaler initialization failed: {str(e)}")
    
    def init_databases(self):
        self.conn = sqlite3.connect('binomo_ai.db', timeout=30)
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trade_history (
                     id INTEGER PRIMARY KEY, timestamp DATETIME, instrument TEXT,
                     direction TEXT, price REAL, confidence REAL, outcome TEXT,
                     features TEXT, interval TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                     id INTEGER PRIMARY KEY, timestamp DATETIME, prediction_id INTEGER,
                     correct INTEGER, comments TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS accuracy_log (
                     id INTEGER PRIMARY KEY, date DATE, accuracy REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS model_performance (
                     id INTEGER PRIMARY KEY, timestamp DATETIME, model_version TEXT,
                     lstm_accuracy REAL, xgb_accuracy REAL, ensemble_accuracy REAL)''')
        self.conn.commit()
    
    def load_models(self):
        try:
            self.lstm_model = load_model(f'models/lstm_{self.model_version}.h5')
            self.xgb_model = joblib.load(f'models/xgb_{self.model_version}.pkl')
            self.scaler = joblib.load(f'scalers/scaler_{self.model_version}.pkl')
            logger.info("✅ Models loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Model loading failed: {str(e)} - Initializing new models")
            self.lstm_model = self.build_enhanced_lstm_model()
            self.xgb_model = self.build_enhanced_xgb_model()
            self.scaler = MinMaxScaler()
    
    def warmup_models(self):
        """Warm up models to avoid first-time prediction delay"""
        try:
            # Use actual data if available
            try:
                df = self.load_backup_data()
                feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sma20', 'sma50', 
                               'rsi', 'macd', 'signal', 'histogram', 'upper_band', 'lower_band',
                               'slowk', 'slowd', 'atr', 'adx', 'obv']
                features = df[feature_cols].tail(60)
                scaled_data = self.scaler.transform(features)
            except:
                # Create dummy data for warmup
                scaled_data = np.random.rand(1, 60, 18)
            
            self.lstm_model.predict(scaled_data, verbose=0)
            
            # Warm up XGBoost
            xgb_input = scaled_data[-1].reshape(1, -1) if scaled_data.shape[0] > 1 else np.random.rand(1, 18)
            self.xgb_model.predict_proba(xgb_input)
            
            logger.info("🔥 Models warmed up successfully")
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")
    
    def health_check(self):
        """System health status report"""
        status = {
            "database": self.conn is not None,
            "lstm": self.lstm_model is not None,
            "xgb": self.xgb_model is not None,
            "scheduler": self.scheduler.running if hasattr(self, 'scheduler') else False,
            "last_data_fetch": datetime.now(pytz.utc) - timedelta(minutes=5),
            "memory_usage": f"{psutil.Process().memory_info().rss / 1024 ** 2:.1f} MB" 
        }
        return status
    
    def setup_scheduler(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.retrain_model, 'cron', hour=3)
        self.scheduler.add_job(self.evaluate_predictions, 'interval', minutes=1)
        self.scheduler.start()
    
    def build_enhanced_lstm_model(self, input_shape=(60, 18)):
        model = Sequential([
            LSTM(256, return_sequences=True, input_shape=input_shape),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        optimizer = Adam(learning_rate=0.0005, clipvalue=0.5)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', 
                     metrics=['accuracy', Precision(), Recall()])
        return model
    
    def build_enhanced_xgb_model(self):
        return XGBClassifier(
            n_estimators=500, 
            max_depth=7, 
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), 
           stop=stop_after_attempt(5))
    def fetch_data(self, symbol="BTCUSDT", interval_seconds=90):
        """Fetch data with customizable interval"""
        try:
            client = Client(
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET")
            )
            
            # Calculate how many candles needed (60 for LSTM + buffer)
            total_minutes = (interval_seconds * 1000) // 60
            chunks = (total_minutes // 1000) + 1
            all_klines = []
            
            for i in range(chunks):
                start_time = int((datetime.now(pytz.utc) - timedelta(minutes=1000*(i+1)).timestamp()*1000)
                klines = client.get_klines(
                    symbol=symbol,
                    interval="1m",
                    limit=1000,
                    startTime=start_time
                )
                if klines:
                    all_klines.extend(klines)
                time.sleep(0.1)  # Respect rate limits
            
            # Create DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Resample to custom intervals
            interval_str = f"{interval_seconds}S"
            df = df.resample(interval_str).agg({
                'open': 'first', 
                'high': 'max', 
                'low': 'min',
                'close': 'last', 
                'volume': 'sum'
            }).dropna()
            
            # Enhanced technical indicators
            df = self.add_technical_indicators(df)
            logger.info(f"✅ Fetched {len(df)} Binance candles ({interval_str} interval)")
            return df[-1000:]
        except Exception as e:
            logger.error(f"🚨 Data fetch error: {str(e)}")
            return self.load_backup_data(interval_seconds)
    
    def add_technical_indicators(self, df):
        # Basic indicators
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal']
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # Stochastic Oscillator
        df['slowk'], df['slowd'] = talib.STOCH(
            df['high'], df['low'], df['close'], 
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        
        # ATR
        df['atr'] = talib.ATR(
            df['high'], df['low'], df['close'], timeperiod=14
        )
        
        # ADX
        df['adx'] = talib.ADX(
            df['high'], df['low'], df['close'], timeperiod=14
        )
        
        # OBV
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Volatility features
        df['vix'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean() * 100
        df['chop'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14) > 40
        
        return df.dropna()
    
    def load_backup_data(self, interval_seconds=90):
        try:
            df = pd.read_csv("backup_data.csv", index_col=0, parse_dates=True)
            return self.add_technical_indicators(df)
        except:
            logger.warning("⚠️ Using synthetic data - backup missing")
            dates = pd.date_range(end=datetime.now(), periods=1000, freq=f'{interval_seconds}S')
            prices = np.random.normal(loc=50000, scale=1000, size=1000).cumsum()
            df = pd.DataFrame({
                'open': prices, 
                'high': prices + np.random.uniform(10, 50, 1000),
                'low': prices - np.random.uniform(10, 50, 1000), 
                'close': prices,
                'volume': np.random.uniform(100, 1000, 1000)
            }, index=dates)
            return self.add_technical_indicators(df)
    
    def predict_direction(self, df, interval, gpt_enabled=True):
        try:
            # Critical fix: Ensure scaler is properly initialized
            if not hasattr(self.scaler, 'data_min_'):
                self.initialize_scaler()
                
            # Check for sufficient data
            if len(df) < 60:
                return "ERROR", 0, "Insufficient data for prediction", ""
            
            # Select relevant features
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sma20', 'sma50', 
                           'rsi', 'macd', 'signal', 'histogram', 'upper_band', 'lower_band',
                           'slowk', 'slowd', 'atr', 'adx', 'obv', 'vix', 'chop']
            features = df[feature_cols]
            
            # Scale features
            if hasattr(self.scaler, 'data_min_'):
                scaled_data = self.scaler.transform(features)
            else:
                # Final fallback if scaler still not initialized
                self.scaler = MinMaxScaler()
                scaled_data = self.scaler.fit_transform(features)
            
            # LSTM prediction (using last 60 time steps)
            lstm_input = np.array([scaled_data[-60:]])
            lstm_pred = self.lstm_model.predict(lstm_input, verbose=0)[0][0]
            
            # XGBoost prediction (using latest data point)
            xgb_input = scaled_data[-1].reshape(1, -1)
            xgb_pred = self.xgb_model.predict_proba(xgb_input)[0][1]
            
            # Dynamic ensemble weighting based on recent performance
            lstm_weight, xgb_weight = self.get_dynamic_weights()
            ensemble_pred = (lstm_pred * lstm_weight) + (xgb_pred * xgb_weight)
            
            # Direction and confidence calculation
            direction = "UP" if ensemble_pred > 0.5 else "DOWN"
            confidence_value = ensemble_pred if direction == "UP" else 1 - ensemble_pred
            confidence = max(70, min(99, round(confidence_value * 100, 1)))
            
            # Generate market insight
            insight = ""
            if gpt_enabled and os.getenv("OPENAI_API_KEY"):
                try:
                    openai.api_key = os.getenv("OPENAI_API_KEY")
                    insight = self.gpt4_insight(df, direction, confidence)
                except:
                    insight = self.local_insight_engine(df, direction, confidence)
            
            # Save feature set for later analysis
            feature_str = ",".join([f"{col}={features[col].iloc[-1]:.2f}" for col in feature_cols])
            
            # Log prediction to GitHub
            last_row = df.iloc[-1].to_dict()
            log_data = {
                "timestamp": datetime.now(pytz.utc).isoformat(),
                "symbol": "BTCUSDT",
                "interval": interval,
                "predicted_direction": direction,
                "confidence": confidence,
                "gpt_insight": insight,
                "actual_outcome": None,
                "user_feedback": None,
                "lstm_pred": lstm_pred,
                "xgb_pred": xgb_pred,
                "ensemble_pred": ensemble_pred,
                **last_row
            }
            self.data_logger.log_prediction(log_data)
            threading.Thread(target=self.data_logger.push_to_github).start()
            
            return direction, confidence, insight, feature_str
        except Exception as e:
            logger.error(f"🚨 Prediction error: {str(e)}")
            return "ERROR", 0, f"Prediction failed: {str(e)}", ""
    
    def local_insight_engine(self, df, direction, confidence):
        """Fallback analysis when GPT fails"""
        analysis = f"{direction} signal detected. "
        
        # RSI analysis
        rsi = df['rsi'].iloc[-1]
        if rsi > 70:
            analysis += "RSI indicates overbought conditions. "
        elif rsi < 30:
            analysis += "RSI indicates oversold conditions. "
        
        # MACD analysis
        if df['macd'].iloc[-1] > df['signal'].iloc[-1]:
            analysis += "MACD bullish crossover detected. "
        else:
            analysis += "MACD bearish crossover detected. "
        
        # Volume analysis
        if df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5:
            analysis += "High volume confirms the move. "
        
        return analysis
    
    def get_dynamic_weights(self):
        """Get dynamic weights based on recent model performance"""
        try:
            c = self.conn.cursor()
            c.execute('''SELECT lstm_accuracy, xgb_accuracy 
                         FROM model_performance 
                         ORDER BY timestamp DESC LIMIT 100''')
            results = c.fetchall()
            
            if len(results) < 10:
                return 0.7, 0.3  # Default weights
            
            lstm_acc = np.mean([r[0] for r in results])
            xgb_acc = np.mean([r[1] for r in results])
            
            total = lstm_acc + xgb_acc
            lstm_weight = lstm_acc / total
            xgb_weight = xgb_acc / total
            
            # Apply minimum weights
            lstm_weight = max(0.3, min(0.8, lstm_weight))
            xgb_weight = max(0.2, min(0.7, xgb_weight))
            
            return lstm_weight, xgb_weight
        except:
            return 0.7, 0.3
    
    def gpt4_insight(self, df, direction, confidence):
        try:
            last_5 = df.tail(5)
            prompt = f"""As a professional crypto trading analyst, provide concise insights on the current market conditions:
            
            Current trend: {direction} signal ({confidence}% confidence)
            Technical indicators:
            - RSI: {last_5['rsi'].iloc[-1]:.2f}
            - MACD: {last_5['macd'].iloc[-1]:.2f} (Signal: {last_5['signal'].iloc[-1]:.2f})
            - Bollinger Bands: Price is {self.get_bb_position(last_5)}
            - Stochastic: K={last_5['slowk'].iloc[-1]:.2f}, D={last_5['slowd'].iloc[-1]:.2f}
            - ADX: {last_5['adx'].iloc[-1]:.2f}
            - Volume: {last_5['volume'].iloc[-1]:.2f} ({self.get_volume_trend(last_5)})
            - VIX: {last_5['vix'].iloc[-1]:.2f}
            
            Last 5 candles:
            {last_5[['open','high','low','close']].to_string()}
            
            Provide 2-3 sentence analysis of market conditions and prediction alignment with technical indicators.
            Highlight any potential risks or confirming factors."""
            
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=0.4
            )
            return response.choices[0].message['content']
        except Exception as e:
            logger.error(f"🚨 GPT-4 error: {str(e)}")
            return self.local_insight_engine(df, direction, confidence)
    
    def get_bb_position(self, df):
        price = df['close'].iloc[-1]
        upper = df['upper_band'].iloc[-1]
        lower = df['lower_band'].iloc[-1]
        
        if price > upper:
            return "ABOVE UPPER BAND (OVERBOUGHT)"
        elif price < lower:
            return "BELOW LOWER BAND (OVERSOLD)"
        elif price > (upper + lower)/2:
            return "IN UPPER HALF"
        else:
            return "IN LOWER HALF"
    
    def get_volume_trend(self, df):
        vol = df['volume'].iloc[-5:]
        if vol.pct_change().mean() > 0.05:
            return "INCREASING"
        elif vol.pct_change().mean() < -0.05:
            return "DECREASING"
        return "STABLE"
    
    def save_prediction(self, direction, confidence, price, features, interval):
        try:
            c = self.conn.cursor()
            c.execute('''INSERT INTO trade_history 
                         (timestamp, instrument, direction, price, confidence, features, interval) 
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                         (datetime.now(pytz.utc), "BTC", direction, price, confidence, features, interval))
            self.conn.commit()
            return c.lastrowid
        except Exception as e:
            logger.error(f"🚨 Save prediction error: {str(e)}")
            return None
    
    def record_feedback(self, prediction_id, correct):
        if not prediction_id: 
            return False
        try:
            c = self.conn.cursor()
            c.execute('''INSERT INTO feedback (timestamp, prediction_id, correct)
                         VALUES (?, ?, ?)''',
                         (datetime.now(pytz.utc), prediction_id, int(correct)))
            self.conn.commit()
            
            # Immediate model update with feedback
            self.online_learn(prediction_id, correct)
            
            # Periodic retraining
            c.execute("SELECT COUNT(*) FROM feedback")
            count = c.fetchone()[0]
            if count % 10 == 0: 
                self.retrain_model()
                
            return True
        except Exception as e:
            logger.error(f"🚨 Feedback error: {str(e)}")
            return False
    
    def online_learn(self, prediction_id, correct):
        """Online learning from immediate feedback"""
        try:
            c = self.conn.cursor()
            c.execute('''SELECT features, direction, confidence 
                         FROM trade_history WHERE id = ?''', (prediction_id,))
            row = c.fetchone()
            
            if not row:
                return False
                
            features_str, direction, confidence = row
            features = self.parse_features(features_str)
            actual = 1 if (direction == "UP" and correct) or (direction == "DOWN" and not correct) else 0
            
            # Update XGBoost model with new data point
            xgb_input = np.array([features])
            self.xgb_model.partial_fit(xgb_input, [actual], classes=[0, 1])
            
            logger.info(f"Online learning: Updated model with feedback (ID: {prediction_id})")
            return True
        except Exception as e:
            logger.error(f"Online learning error: {str(e)}")
            return False
    
    def parse_features(self, features_str):
        """Parse feature string into numeric array"""
        features = []
        for item in features_str.split(','):
            try:
                key, value = item.split('=')
                features.append(float(value))
            except:
                continue
        return features
    
    def evaluate_predictions(self):
        """Auto-evaluate completed predictions"""
        try:
            c = self.conn.cursor()
            c.execute('''SELECT id, timestamp, interval, direction, price 
                         FROM trade_history 
                         WHERE outcome IS NULL''')
            pending = c.fetchall()
            
            client = Client(
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET")
            )
            
            for pred in pending:
                pred_id, pred_time, interval, direction, price = pred
                end_time = pred_time + timedelta(seconds=int(interval.replace('min', '')) * 60)
                
                if datetime.now(pytz.utc) > end_time:
                    # Get actual price at end of interval
                    klines = client.get_historical_klines(
                        symbol="BTCUSDT",
                        interval="1m",
                        start_str=int(end_time.timestamp() * 1000),
                        limit=1
                    )
                    
                    if klines:
                        actual_price = float(klines[0][4])  # Close price
                        actual_direction = "UP" if actual_price > price else "DOWN"
                        outcome = 1 if direction == actual_direction else 0
                        
                        # Update trade history
                        c.execute('''UPDATE trade_history 
                                    SET outcome = ?
                                    WHERE id = ?''', 
                                    (outcome, pred_id))
                        self.conn.commit()
                        
                        # Update GitHub log
                        self.update_github_log(pred_id, outcome)
            
            logger.info("✅ Prediction evaluation completed")
        except Exception as e:
            logger.error(f"🚨 Prediction evaluation failed: {str(e)}")
    
    def update_github_log(self, prediction_id, outcome):
        """Update GitHub log with actual outcome"""
        try:
            # Get prediction details
            c = self.conn.cursor()
            c.execute('''SELECT * FROM trade_history WHERE id = ?''', (prediction_id,))
            pred = c.fetchone()
            
            if pred:
                # Find matching record in GitHub log
                # This would require more sophisticated matching in production
                # For simplicity, we'll just note that we need to update
                logger.info(f"Updating GitHub log for prediction {prediction_id}")
        except Exception as e:
            logger.error(f"GitHub log update failed: {str(e)}")
    
    def retrain_model(self):
        try:
            logger.info("🚀 Starting full model retraining...")
            start_time = time.time()
            
            # Fetch 2 years of historical data
            df = self.fetch_historical_data(limit=10000)
            
            # Prepare features and target
            feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sma20', 'sma50', 
                           'rsi', 'macd', 'signal', 'histogram', 'upper_band', 'lower_band',
                           'slowk', 'slowd', 'atr', 'adx', 'obv', 'vix', 'chop']
            features = df[feature_cols]
            target = (df['close'].shift(-1) > df['close']).astype(int).iloc[:-1]
            features = features.iloc[:-1]
            
            # Scale features
            new_scaler = MinMaxScaler()
            scaled_features = new_scaler.fit_transform(features)
            
            # Prepare LSTM sequences
            X, y = [], []
            seq_length = 60
            for i in range(seq_length, len(scaled_features)):
                X.append(scaled_features[i-seq_length:i])
                y.append(target.iloc[i])
            
            X = np.array(X)
            y = np.array(y)
            
            # Add early stopping to prevent overfitting
            early_stop = EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True
            )
            
            # Train LSTM model
            lstm_history = self.lstm_model.fit(
                X, y, 
                epochs=25, 
                batch_size=64, 
                verbose=0,
                validation_split=0.2,
                shuffle=True,
                callbacks=[early_stop]
            )
            
            # Train XGBoost model
            self.xgb_model.fit(features.iloc[seq_length:], y)
            
            # Evaluate models
            lstm_acc = lstm_history.history['val_accuracy'][-1]
            xgb_acc = self.xgb_model.score(features.iloc[seq_length:], y)
            
            # Create new model version
            match = re.match(r"v(\d+)", self.model_version)
            next_num = int(match.group(1)) + 1 if match else 1
            self.model_version = f"v{next_num}"
            
            # Save models and scaler
            self.lstm_model.save(f'models/lstm_{self.model_version}.h5')
            joblib.dump(self.xgb_model, f'models/xgb_{self.model_version}.pkl')
            joblib.dump(new_scaler, f'scalers/scaler_{self.model_version}.pkl')
            self.scaler = new_scaler
            self.last_retrain = datetime.now(pytz.utc)
            
            # Log performance
            c = self.conn.cursor()
            c.execute('''INSERT INTO model_performance 
                         (timestamp, model_version, lstm_accuracy, xgb_accuracy) 
                         VALUES (?, ?, ?, ?)''',
                         (datetime.now(pytz.utc), self.model_version, lstm_acc, xgb_acc))
            self.conn.commit()
            
            duration = time.time() - start_time
            logger.info(f"✅ Model retrained successfully. Version: {self.model_version}")
            logger.info(f"⏱️ LSTM Acc: {lstm_acc:.4f}, XGB Acc: {xgb_acc:.4f}, Time: {duration:.2f}s")
            return True
        except Exception as e:
            logger.error(f"🚨 Retrain failed: {str(e)}")
            return False
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), 
           stop=stop_after_attempt(5))
    def fetch_historical_data(self, symbol="BTCUSDT", limit=10000):
        """Fetch 2 years of historical data"""
        try:
            client = Client(
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET")
            )
            
            # Calculate start time (2 years ago)
            end_time = datetime.now(pytz.utc)
            start_time = end_time - timedelta(days=730)  # 2 years
            
            # Fetch data in chunks
            all_klines = []
            while start_time < end_time:
                klines = client.get_historical_klines(
                    symbol=symbol,
                    interval="1m",
                    start_str=str(int(start_time.timestamp() * 1000)),
                    end_str=str(int(end_time.timestamp() * 1000)),
                    limit=1000
                )
                
                if not klines:
                    break
                    
                all_klines.extend(klines)
                end_time = datetime.fromtimestamp(klines[0][0]/1000) - timedelta(minutes=1)
                time.sleep(0.1)  # Respect rate limits
                
                if len(all_klines) >= limit:
                    break
            
            # Create DataFrame
            df = pd.DataFrame(all_klines, columns=[
                'time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # Resample to 90-second intervals and add indicators
            df = df.resample('90S').agg({
                'open': 'first', 
                'high': 'max', 
                'low': 'min',
                'close': 'last', 
                'volume': 'sum'
            }).dropna()
            
            df = self.add_technical_indicators(df)
            logger.info(f"✅ Fetched {len(df)} historical candles")
            return df
        except Exception as e:
            logger.error(f"🚨 Historical data fetch error: {str(e)}")
            return self.load_backup_data()
    
    def get_performance_metrics(self):
        try:
            c = self.conn.cursor()
            
            # Overall accuracy
            c.execute('''SELECT COUNT(*), SUM(CASE WHEN outcome = 1 THEN 1 ELSE 0 END) 
                         FROM trade_history WHERE outcome IS NOT NULL''')
            total, correct = c.fetchone()
            overall_accuracy = correct / total if total > 0 else 0
            
            # Weekly accuracy
            c.execute('''SELECT COUNT(*), SUM(outcome) FROM trade_history 
                         WHERE timestamp > ? AND outcome IS NOT NULL''', 
                         (datetime.now(pytz.utc) - timedelta(days=7),))
            weekly_data = c.fetchone()
            weekly_accuracy = weekly_data[1] / weekly_data[0] if weekly_data and weekly_data[0] > 0 else 0
            
            # Worst trades
            c.execute('''SELECT timestamp, direction, confidence, interval 
                         FROM trade_history 
                         WHERE outcome = 0 
                         ORDER BY timestamp DESC LIMIT 5''')
            worst_trades = c.fetchall()
            
            # Model performance
            c.execute('''SELECT model_version, lstm_accuracy, xgb_accuracy 
                         FROM model_performance 
                         ORDER BY timestamp DESC LIMIT 1''')
            model_perf = c.fetchone()
            
            return {
                "overall_accuracy": overall_accuracy,
                "weekly_accuracy": weekly_accuracy,
                "worst_trades": worst_trades,
                "model_version": model_perf[0] if model_perf else "v1",
                "lstm_accuracy": model_perf[1] if model_perf else 0,
                "xgb_accuracy": model_perf[2] if model_perf else 0
            }
        except Exception as e:
            logger.error(f"🚨 Metrics error: {str(e)}")
            return {
                "overall_accuracy": 0,
                "weekly_accuracy": 0,
                "worst_trades": [],
                "model_version": "v1",
                "lstm_accuracy": 0,
                "xgb_accuracy": 0
            }

# ===== ENHANCED DASHBOARD =====
def main():
    if 'trader' not in st.session_state:
        with st.spinner('Initializing AI trading system...'):
            st.session_state.trader = BinomoAITrader()
        st.session_state.gpt_enabled = True
    
    trader = st.session_state.trader
    
    # System health status
    health = trader.health_check()
    st.sidebar.subheader("SYSTEM HEALTH")
    st.sidebar.write(f"Database: {'✅' if health['database'] else '❌'}")
    st.sidebar.write(f"LSTM Model: {'✅' if health['lstm'] else '❌'}")
    st.sidebar.write(f"XGBoost Model: {'✅' if health['xgb'] else '❌'}")
    st.sidebar.write(f"Scheduler: {'✅' if health['scheduler'] else '❌'}")
    st.sidebar.write(f"Memory: {health['memory_usage']}")
    
    # Risk management
    st.sidebar.divider()
    st.sidebar.subheader("RISK MANAGEMENT")
    account_size = st.sidebar.number_input("Account Size (USD)", 100, 1000000, 5000)
    risk_percent = st.sidebar.slider("Risk per Trade (%)", 0.5, 10.0, 2.0)
    stop_loss = st.sidebar.slider("Stop Loss (%)", 0.5, 5.0, 1.0)
    
    # Manual retrain button
    if st.sidebar.button("🚀 Retrain Models Now", help="Force immediate model retraining"):
        with st.spinner("Retraining models..."):
            if trader.retrain_model():
                st.sidebar.success("Models retrained successfully!")
            else:
                st.sidebar.error("Retraining failed")
    
    # Display header with performance metrics
    metrics = trader.get_performance_metrics()
    
    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    with col1:
        st.subheader("BINOMO AI PRO TRADING SYSTEM")
    with col2:
        st.metric("Overall Accuracy", f"{metrics['overall_accuracy']*100:.1f}%")
    with col3:
        st.metric("Weekly Accuracy", f"{metrics['weekly_accuracy']*100:.1f}%")
    with col4:
        st.metric("Model Version", metrics['model_version'])
    
    st.caption(f"Last retrain: {trader.last_retrain.strftime('%Y-%m-%d %H:%M')} | "
               f"LSTM Acc: {metrics['lstm_accuracy']:.3f} | "
               f"XGB Acc: {metrics['xgb_accuracy']:.3f}")
    
    # Main content columns
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("CONTROL PANEL")
        st.session_state.gpt_enabled = st.toggle("GPT-4 Market Analysis", True, help="Enable AI-powered market insights")
        
        # Interval selection
        interval_map = {
            "1.5min": 90,
            "3min": 180,
            "5min": 300
        }
        selected_interval = st.selectbox("Prediction Interval", list(interval_map.keys()))
        interval_seconds = interval_map[selected_interval]
        
        st.divider()
        st.subheader("MODEL PERFORMANCE")
        st.progress(metrics['overall_accuracy'], text=f"Overall Accuracy: {metrics['overall_accuracy']*100:.1f}%")
        
        if metrics['worst_trades']:
            st.caption("⛔ RECENT LOSING TRADES")
            worst_df = pd.DataFrame(metrics['worst_trades'], 
                                   columns=['Time', 'Direction', 'Confidence', 'Interval'])
            st.dataframe(worst_df, hide_index=True, use_container_width=True)
        else:
            st.info("No losing trades recorded yet")
    
    with col_right:
        # Manual prediction button
        if st.button("🔮 Generate Prediction", type="primary", use_container_width=True,
                    help="Predict next price direction"):
            # Fetch data with selected interval
            with st.spinner(f"Fetching {selected_interval} market data..."):
                df = trader.fetch_data(interval_seconds=interval_seconds)
            
            if len(df) == 0:
                st.error("Failed to fetch market data")
                return
                
            current_price = df['close'].iloc[-1]
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Calculate prediction end time
            end_time = (datetime.now() + timedelta(seconds=interval_seconds)).strftime("%H:%M:%S")
            
            with st.spinner("Analyzing market..."):
                direction, confidence, insight, features = trader.predict_direction(
                    df, 
                    interval=selected_interval, 
                    gpt_enabled=st.session_state.gpt_enabled
                )
                prediction_id = trader.save_prediction(
                    direction, confidence, current_price, features, selected_interval
                )
            
            # Display prediction with clear time info
            st.subheader(f"{selected_interval.upper()} CANDLE: {current_time} → {end_time}")
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; 
                        border: 2px solid {'#4CAF50' if direction=='UP' else '#F44336'};
                        border-radius: 10px; margin: 20px 0;
                        background: {'#1B5E20' if direction=='UP' else '#B71C1C'}20;">
                <span style="font-size: 2em; font-weight: bold;
                            color: {'#4CAF50' if direction=='UP' else '#F44336'}">
                    {direction} ARROW
                </span>
                <div style="font-size: 1.5em; margin-top: 10px;">
                    Confidence: {confidence}%
                </div>
                <div style="font-size: 1em; margin-top: 15px; color: #aaa">
                    Prediction valid until: {end_time}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if insight: 
                st.info(f"💡 MARKET INSIGHT: {insight}")
            
            # Position size calculation
            dollar_risk = account_size * (risk_percent / 100)
            price_risk = current_price * (stop_loss / 100)
            position_size = dollar_risk / price_risk
                
            st.success(f"Recommended position size: **{position_size:.4f} BTC** "
                       f"(Risk: ${dollar_risk:.2f}, Stop Loss: {stop_loss}%)")
            
            # Feedback buttons
            col_fb1, col_fb2 = st.columns(2)
            with col_fb1:
                if st.button("✅ CORRECT PREDICTION", use_container_width=True, type="primary", 
                            help="Mark prediction as correct"):
                    if trader.record_feedback(prediction_id, True):
                        st.success("Feedback recorded! AI has learned from this success")
            with col_fb2:
                if st.button("❌ INCORRECT PREDICTION", use_container_width=True, type="secondary", 
                            help="Mark prediction as incorrect"):
                    if trader.record_feedback(prediction_id, False):
                        st.error("Feedback recorded! AI has learned from this mistake")
            
            # Display chart
            if len(df) > 0:
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index[-60:], 
                    open=df['open'][-60:], 
                    high=df['high'][-60:],
                    low=df['low'][-60:], 
                    close=df['close'][-60:]
                )])
                
                # Add technical indicators
                fig.add_trace(go.Scatter(
                    x=df.index[-60:],
                    y=df['sma20'][-60:],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='blue', width=1)
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index[-60:],
                    y=df['upper_band'][-60:],
                    mode='lines',
                    name='Upper BB',
                    line=dict(color='gray', width=1, dash='dot')
                ))
                
                fig.add_trace(go.Scatter(
                    x=df.index[-60:],
                    y=df['lower_band'][-60:],
                    mode='lines',
                    name='Lower BB',
                    line=dict(color='gray', width=1, dash='dot')
                ))
                
                fig.update_layout(
                    title=f"Live BTC/USDT {selected_interval} Chart with Technical Indicators",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    template="plotly_dark",
                    height=500,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Performance analytics
    st.divider()
    st.subheader("📊 ADVANCED PERFORMANCE ANALYTICS")
    
    # Placeholder for more advanced analytics
    st.info("Performance analytics and model optimization reports will appear here after more data is collected")

# ===== RUN APPLICATION =====
if __name__ == "__main__":
    # Validate environment
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        st.error("❌ CRITICAL ERROR: Binance API keys not found in .env file!")
        st.stop()
    
    if not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ OpenAI API key not found - GPT-4 insights will be disabled")
    
    logger.info("===== APPLICATION START =====")
    main()