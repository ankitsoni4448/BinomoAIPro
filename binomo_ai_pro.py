# binomo_ai_pro.py
import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import joblib
import sqlite3
import hashlib
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBClassifier
import openai
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
import plotly.graph_objects as go

# ===== INITIALIZATION =====
load_dotenv()

# Setup logging
logging.basicConfig(
    filename='binomo_ai.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())  # Print logs to console

st.set_page_config(page_title="BINOMO AI PRO", layout="wide")

# ===== CORE TRADER CLASS =====
class BinomoAITrader:
    def __init__(self):
        self.model_version = "v1"
        self.accuracy = 0.85
        self.last_retrain = datetime.now(pytz.utc)
        self.scaler = None
        self.init_databases()
        self.load_models()
        self.setup_scheduler()
        
    def init_databases(self):
        self.conn = sqlite3.connect('binomo_ai.db')
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS trade_history (
                     id INTEGER PRIMARY KEY, timestamp DATETIME, instrument TEXT,
                     direction TEXT, price REAL, confidence REAL, outcome TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                     id INTEGER PRIMARY KEY, timestamp DATETIME, prediction_id INTEGER,
                     correct INTEGER, comments TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS accuracy_log (
                     id INTEGER PRIMARY KEY, date DATE, accuracy REAL)''')
        self.conn.commit()
    
    def load_models(self):
        try:
            self.lstm_model = load_model(f'models/lstm_{self.model_version}.h5')
            self.xgb_model = joblib.load(f'models/xgb_{self.model_version}.pkl')
            self.scaler = joblib.load(f'scalers/scaler_{self.model_version}.pkl')
            logger.info("✅ Models loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Model loading failed: {str(e)} - Initializing new models")
            self.lstm_model = self.build_lstm_model()
            self.xgb_model = self.build_xgb_model()
            self.scaler = MinMaxScaler()
    
    def setup_scheduler(self):
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.retrain_model, 'cron', hour=3)
        self.scheduler.start()
    
    def build_lstm_model(self, input_shape=(60, 8)):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def build_xgb_model(self):
        return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
    
    def fetch_data(self, symbol="BTCUSDT", limit=1500):
        try:
            client = Client(
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET")
            )
            klines = client.get_klines(symbol=symbol, interval="1m", limit=limit)
            df = pd.DataFrame(klines, columns=[
                'time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True)
            df = df.resample('90S').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum'
            }).dropna()
            df['sma20'] = df['close'].rolling(20).mean()
            df['rsi'] = self.compute_rsi(df['close'], 14)
            df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            for col in df.columns: df[col] = df[col].astype(float)
            logger.info(f"✅ Fetched {len(df)} Binance candles")
            return df[-1000:]
        except Exception as e:
            logger.error(f"🚨 Data fetch error: {str(e)}")
            return self.load_backup_data()
    
    def compute_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def load_backup_data(self):
        try:
            return pd.read_csv("backup_data.csv", index_col=0, parse_dates=True)
        except:
            logger.warning("⚠️ Using synthetic data - backup missing")
            dates = pd.date_range(end=datetime.now(), periods=1000, freq='90S')
            prices = np.random.normal(loc=50000, scale=1000, size=1000).cumsum()
            return pd.DataFrame({
                'open': prices, 'high': prices + np.random.uniform(10, 50, 1000),
                'low': prices - np.random.uniform(10, 50, 1000), 'close': prices,
                'volume': np.random.uniform(100, 1000, 1000)
            }, index=dates)
    
    def predict_direction(self, df, gpt_enabled=True):
        try:
            features = df[['open', 'high', 'low', 'close', 'volume', 'sma20', 'rsi', 'macd']]
            
            if self.scaler:
                scaled_data = self.scaler.transform(features)
            else:
                scaled_data = self.scaler.fit_transform(features)
            
            lstm_pred = self.lstm_model.predict(np.array([scaled_data[-60:]]), verbose=0)[0][0]
            xgb_pred = self.xgb_model.predict_proba(scaled_data[-1].reshape(1, -1))[0][1]
            ensemble_pred = (lstm_pred * 0.7) + (xgb_pred * 0.3)
            
            direction = "UP" if ensemble_pred > 0.5 else "DOWN"
            confidence_value = ensemble_pred if direction == "UP" else 1 - ensemble_pred
            confidence = max(70, min(99, round(confidence_value * 100, 1)))
            
            insight = ""
            if gpt_enabled and openai.api_key:
                insight = self.gpt4_insight(df, direction, confidence)
            
            return direction, confidence, insight
        except Exception as e:
            logger.error(f"🚨 Prediction error: {str(e)}")
            return "ERROR", 0, f"Prediction failed: {str(e)}"
    
    def gpt4_insight(self, df, direction, confidence):
        try:
            last_5 = df.tail(5)
            prompt = f"Crypto trading context:\n- Current trend: {direction} signal ({confidence}% confidence)\n"
            prompt += f"Last 5 candles:\n{last_5[['open','high','low','close']].to_string()}\n\n"
            prompt += "Provide 2-sentence analysis of market conditions and prediction alignment with technical indicators."
            
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            return response.choices[0].message['content']
        except Exception as e:
            logger.error(f"🚨 GPT-4 error: {str(e)}")
            return ""
    
    def save_prediction(self, direction, confidence, price):
        try:
            c = self.conn.cursor()
            c.execute('''INSERT INTO trade_history (timestamp, instrument, direction, price, confidence)
                         VALUES (?, ?, ?, ?, ?)''',
                         (datetime.now(pytz.utc), "BTC", direction, price, confidence))
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
            
            c.execute("SELECT COUNT(*) FROM feedback")
            count = c.fetchone()[0]
            if count % 10 == 0: 
                self.retrain_model()
                
            return True
        except Exception as e:
            logger.error(f"🚨 Feedback error: {str(e)}")
            return False
    
    def retrain_model(self):
        try:
            logger.info("🚀 Starting model retraining...")
            df = self.fetch_data(limit=5000)
            features = df[['open', 'high', 'low', 'close', 'volume', 'sma20', 'rsi', 'macd']]
            target = (df['close'].shift(-1) > df['close']).astype(int).iloc[:-1]
            features = features.iloc[:-1]
            
            new_scaler = MinMaxScaler()
            scaled_features = new_scaler.fit_transform(features)
            
            X, y = [], []
            for i in range(60, len(scaled_features)):
                X.append(scaled_features[i-60:i])
                y.append(target.iloc[i])
            
            X = np.array(X)
            y = np.array(y)
            
            self.lstm_model.fit(X, y, epochs=15, batch_size=32, verbose=0)
            self.xgb_model.fit(features.iloc[60:], y)
            
            match = re.match(r"v(\d+)", self.model_version)
            next_num = int(match.group(1)) + 1 if match else 1
            self.model_version = f"v{next_num}"
            
            self.lstm_model.save(f'models/lstm_{self.model_version}.h5')
            joblib.dump(self.xgb_model, f'models/xgb_{self.model_version}.pkl')
            joblib.dump(new_scaler, f'scalers/scaler_{self.model_version}.pkl')
            self.scaler = new_scaler
            self.last_retrain = datetime.now(pytz.utc)
            logger.info(f"✅ Model retrained successfully. New version: {self.model_version}")
            return True
        except Exception as e:
            logger.error(f"🚨 Retrain failed: {str(e)}")
            return False
    
    def get_performance_metrics(self):
        try:
            c = self.conn.cursor()
            c.execute('SELECT accuracy FROM accuracy_log ORDER BY date DESC LIMIT 1')
            row = c.fetchone()
            accuracy = row[0] if row else 0.0
            
            c.execute('''SELECT COUNT(*), SUM(correct) FROM feedback 
                         WHERE timestamp > ?''', 
                         (datetime.now(pytz.utc) - timedelta(days=7),))
            weekly_data = c.fetchone()
            weekly_accuracy = weekly_data[1] / weekly_data[0] if weekly_data and weekly_data[0] > 0 else 0
            
            c.execute('''SELECT trade_history.timestamp, trade_history.direction, 
                         trade_history.confidence FROM trade_history
                         JOIN feedback ON trade_history.id = feedback.prediction_id
                         WHERE feedback.correct = 0 
                         ORDER BY trade_history.timestamp DESC LIMIT 5''')
            worst_trades = c.fetchall()
            
            return {
                "overall_accuracy": accuracy,
                "weekly_accuracy": weekly_accuracy,
                "worst_trades": worst_trades
            }
        except Exception as e:
            logger.error(f"🚨 Metrics error: {str(e)}")
            return {
                "overall_accuracy": 0,
                "weekly_accuracy": 0,
                "worst_trades": []
            }

# ===== DASHBOARD =====
def verify_password(input_password):
    try:
        stored_hash = os.getenv("APP_PASSWORD_HASH", "").strip()
        input_hash = hashlib.sha256(input_password.encode()).hexdigest()
        logger.info(f"Stored hash: {stored_hash}")
        logger.info(f"Input hash: {input_hash}")
        return input_hash == stored_hash
    except Exception as e:
        logger.error(f"Password verification failed: {str(e)}")
        return False

def main():
    if 'trader' not in st.session_state:
        st.session_state.trader = BinomoAITrader()
        st.session_state.gpt_enabled = True
    
    trader = st.session_state.trader
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("CONTROL PANEL")
        metrics = trader.get_performance_metrics()
        st.metric("Overall Accuracy", f"{metrics['overall_accuracy']*100:.1f}%")
        st.metric("Model Version", trader.model_version)
        st.caption(f"Last retrain: {trader.last_retrain.strftime('%Y-%m-%d %H:%M')}")
        st.session_state.gpt_enabled = st.toggle("GPT-4 Analysis", True)
    
    with col2:
        df = trader.fetch_data()
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
        current_time = datetime.now().strftime("%H:%M:%S")
        next_time = (datetime.now() + timedelta(seconds=90)).strftime("%H:%M:%S")
        
        direction, confidence, insight = trader.predict_direction(df, st.session_state.gpt_enabled)
        prediction_id = trader.save_prediction(direction, confidence, current_price)
        
        st.subheader(f"NEXT 1.5min CANDLE: {current_time} → {next_time}")
        st.markdown(f"""<div style="background:#1e1e1e; padding:20px; border-radius:10px; 
                     border-left:5px solid {'#4CAF50' if direction=='UP' else '#F44336'}">
                     <h1 style="text-align:center;color:{'#4CAF50' if direction=='UP' else '#F44336'}">
                     {direction} ({confidence}% CONFIDENCE)</h1></div>""", unsafe_allow_html=True)
        
        if insight: 
            st.info(f"💡 MARKET INSIGHT: {insight}")
        
        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            if st.button("✅ CORRECT", use_container_width=True, type="primary", key="correct_btn"):
                if trader.record_feedback(prediction_id, True):
                    st.success("Feedback recorded! AI will learn from this success")
        with col_fb2:
            if st.button("❌ INCORRECT", use_container_width=True, type="secondary", key="incorrect_btn"):
                if trader.record_feedback(prediction_id, False):
                    st.error("Feedback recorded! AI will learn from this mistake")
        
        if len(df) > 0:
            fig = go.Figure(data=[go.Candlestick(
                x=df.index[-60:], 
                open=df['open'][-60:], 
                high=df['high'][-60:],
                low=df['low'][-60:], 
                close=df['close'][-60:]
            )])
            fig.update_layout(
                title="Live BTC/USDT 1.5min Chart",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_dark",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Analytics section
    st.divider()
    st.subheader("📊 PERFORMANCE ANALYTICS")
    
    metrics = trader.get_performance_metrics()
    if metrics['worst_trades']:
        st.caption("⛔ WORST RECENT TRADES")
        worst_df = pd.DataFrame(metrics['worst_trades'], 
                               columns=['Time', 'Direction', 'Confidence'])
        st.dataframe(worst_df, hide_index=True)
    else:
        st.info("No losing trades recorded yet")

# ===== RUN =====
if __name__ == "__main__":
    # Validate environment first
    if not os.getenv("APP_PASSWORD_HASH"):
        st.error("❌ CRITICAL ERROR: APP_PASSWORD_HASH not found in .env file!")
        st.stop()
    
    logger.info("===== APPLICATION START =====")
    
    password = st.sidebar.text_input("🔒 ENTER PASSWORD:", type="password", key="pw_input")
    
    # Debug output
    if password:
        st.sidebar.write(f"Password entered: {password}")
        st.sidebar.write(f"Stored hash: {os.getenv('APP_PASSWORD_HASH')}")
    
    if password and verify_password(password):
        logger.info("Password verified successfully")
        main()
    elif password:
        logger.warning("Incorrect password attempt")
        st.error("❌ INCORRECT PASSWORD")