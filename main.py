#!/usr/bin/env python3
"""
RPD Alert Bot - Web Service for Render.com (Enhanced Version)
A cloud-based trading alert bot with improved Yahoo Finance handling
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import json
import logging
import os
import signal
import sys
import threading
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from flask import Flask, jsonify, request
import warnings

warnings.filterwarnings('ignore')

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Try to import talib, fallback to ta library if not available
try:
    import talib
    TALIB_AVAILABLE = True
    logger.info("✅ Using TA-Lib for technical indicators")
except ImportError:
    try:
        import ta
        TALIB_AVAILABLE = False
        logger.info("⚠️  TA-Lib not available, using 'ta' library as fallback")
    except ImportError:
        raise ImportError("Neither talib nor ta library is available. Please install one of them.")

class Config:
    """Enhanced Configuration class"""
    
    # Required environment variables
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Web service configuration
    PORT = int(os.getenv('PORT', '10000'))
    
    # Configuration methods - increased intervals to avoid rate limits
    CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '1800'))  # 30 minutes default
    MIN_PROBABILITY = int(os.getenv('MIN_PROBABILITY', '65'))
    
    @classmethod
    def get_trading_config(cls):
        """Get reliable trading configuration"""
        # Check for environment configuration first
        simple_symbols = os.getenv('SYMBOLS')
        simple_timeframe = os.getenv('TIMEFRAME', '1h')
        
        if simple_symbols:
            symbols = [s.strip() for s in simple_symbols.split(',')]
            config_dict = {symbol: simple_timeframe for symbol in symbols}
            logger.info(f"Using configured symbols: {config_dict}")
            return config_dict
        
        # Use reliable default symbols
        logger.info("Using default reliable symbols")
        reliable_config = {
            # Major stocks (most reliable on Yahoo Finance)
            'AAPL': '1h',    # Apple - very reliable
            'MSFT': '1h',    # Microsoft - very reliable  
            'SPY': '1h',     # S&P 500 ETF - extremely reliable
            'QQQ': '1h',     # NASDAQ ETF - very reliable
            
            # Forex (reliable during market hours)
            'EURUSD=X': '1h', # EUR/USD - reliable
            
            # Crypto (use 4h to reduce API load)
            'BTC-USD': '4h',  # Bitcoin - reliable but use longer timeframe
        }
        return reliable_config
    
    # Feature flags
    ENABLE_VOLUME_FILTER = os.getenv('ENABLE_VOLUME_FILTER', 'true').lower() == 'true'
    ENABLE_RSI_FILTER = os.getenv('ENABLE_RSI_FILTER', 'true').lower() == 'true'
    
    # Technical parameters
    ADAPTIVE_PERIOD = int(os.getenv('ADAPTIVE_PERIOD', '25'))
    FRACTAL_STRENGTH = int(os.getenv('FRACTAL_STRENGTH', '2'))
    ANALYSIS_LEVELS = int(os.getenv('ANALYSIS_LEVELS', '6'))
    EDGE_SENSITIVITY = int(os.getenv('EDGE_SENSITIVITY', '3'))
    ENTROPY_THRESHOLD = float(os.getenv('ENTROPY_THRESHOLD', '0.85'))
    RSI_LENGTH = int(os.getenv('RSI_LENGTH', '17'))
    RSI_TOP = int(os.getenv('RSI_TOP', '65'))
    RSI_BOTTOM = int(os.getenv('RSI_BOTTOM', '40'))
    VOL_LOOKBACK = int(os.getenv('VOL_LOOKBACK', '17'))
    VOL_MULTIPLIER = float(os.getenv('VOL_MULTIPLIER', '1.2'))
    
    # Health check
    HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '7200'))  # 2 hours
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN environment variable is required")
        
        if not cls.TELEGRAM_CHAT_ID:
            errors.append("TELEGRAM_CHAT_ID environment variable is required")
        
        if errors:
            for error in errors:
                logger.error(error)
            raise ValueError("Configuration validation failed")
        
        trading_config = cls.get_trading_config()
        
        logger.info(f"✅ Configuration validated")
        logger.info(f"📊 Monitoring {len(trading_config)} assets: {list(trading_config.keys())}")
        logger.info(f"⏰ Check interval: {cls.CHECK_INTERVAL}s ({cls.CHECK_INTERVAL//60} minutes)")
        
        return trading_config

class TechnicalIndicators:
    """Technical indicators with fallback support"""
    
    @staticmethod
    def atr(high, low, close, timeperiod=14):
        """Calculate Average True Range"""
        if TALIB_AVAILABLE:
            return talib.ATR(high, low, close, timeperiod=timeperiod)
        else:
            df = pd.DataFrame({'high': high, 'low': low, 'close': close})
            return ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=timeperiod).values
    
    @staticmethod
    def rsi(close, timeperiod=14):
        """Calculate Relative Strength Index"""
        if TALIB_AVAILABLE:
            return talib.RSI(close, timeperiod=timeperiod)
        else:
            close_series = pd.Series(close)
            return ta.momentum.rsi(close_series, window=timeperiod).values

class MarketData:
    """Enhanced Market data fetcher"""
    
    @staticmethod
    def configure_session():
        """Configure requests session with proper headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive'
        })
        return session
    
    @staticmethod
    def get_data(symbol, period="5d", interval="1h", max_retries=2):
        """Enhanced data fetching with better error handling"""
        session = MarketData.configure_session()
        
        # Use longer periods for better data availability
        period_map = {
            '1d': '5d', '2d': '5d', '5d': '1mo', 
            '10d': '1mo', '30d': '3mo', '60d': '6mo'
        }
        safe_period = period_map.get(period, period)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching {symbol} - attempt {attempt + 1}")
                
                ticker = yf.Ticker(symbol, session=session)
                df = ticker.history(period=safe_period, interval=interval, auto_adjust=True, prepost=True)
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    if attempt < max_retries - 1:
                        time.sleep(5 * (attempt + 1))  # Progressive delay
                        continue
                    return None
                
                if len(df) < 30:  # Need sufficient data for analysis
                    logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                    if attempt < max_retries - 1:
                        time.sleep(5 * (attempt + 1))
                        continue
                    return None
                
                logger.info(f"✅ Successfully fetched {len(df)} data points for {symbol}")
                return df
                
            except Exception as e:
                error_msg = str(e).lower()
                if "expecting value" in error_msg:
                    logger.warning(f"Yahoo Finance API issue for {symbol}: JSON parse error")
                elif "no data found" in error_msg:
                    logger.warning(f"No data available for {symbol}")
                else:
                    logger.warning(f"Error fetching {symbol}: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)  # 10, 20 seconds
                    logger.info(f"Retrying {symbol} in {wait_time}s...")
                    time.sleep(wait_time)
                
        logger.error(f"❌ Failed to fetch data for {symbol} after {max_retries} attempts")
        return None

# RPD Indicator class (keeping original logic)
class RPDIndicator:
    """RPD Indicator - Python version"""
    
    def __init__(self):
        self.last_signals = {}
    
    def calculate_entropy(self, series, length):
        """Calculate market entropy"""
        if len(series) < length:
            return 0.5
        
        price_changes = series.diff().dropna().tail(length)
        if len(price_changes) == 0:
            return 0.5
        
        up_changes = (price_changes > 0).sum()
        total = len(price_changes)
        
        if total == 0:
            return 0.5
        
        p = up_changes / total
        if p <= 0 or p >= 1:
            return 0.5
        
        return (-p * np.log2(p) - (1-p) * np.log2(1-p))
    
    def quantum_state_analysis(self, prices, states, period):
        """Price quantum state analysis"""
        if len(prices) < period:
            return 0, 0.0
        
        recent_prices = prices.tail(period)
        hi = recent_prices.max()
        lo = recent_prices.min()
        current_price = prices.iloc[-1]
        
        price_range = hi - lo
        if price_range <= 0:
            return 0, 0.0
        
        state = round((current_price - lo) / price_range * (states - 1))
        state_percent = (current_price - lo) / price_range
        
        return int(state), float(state_percent)
    
    def calculate_psr(self, series, period):
        """Price/State Reversal velocity and acceleration"""
        if len(series) < period:
            return 0.0, 0.0
        
        half_period = max(1, period // 2)
        if len(series) < half_period + 1:
            return 0.0, 0.0
        
        velocity = series.iloc[-1] - series.iloc[-half_period-1]
        
        if len(series) < half_period + 2:
            acceleration = 0.0
        else:
            prev_velocity = series.iloc[-2] - series.iloc[-half_period-2]
            acceleration = velocity - prev_velocity
        
        return float(velocity), float(acceleration)
    
    def get_supertrend(self, df, multiplier=1.1, length=16):
        """Calculate Supertrend"""
        try:
            atr = TechnicalIndicators.atr(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=length)
            hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
            
            upper = hlc3 - multiplier * atr
            lower = hlc3 + multiplier * atr
            
            trend = pd.Series([1] * len(df), index=df.index)
            
            for i in range(1, len(df)):
                if pd.notna(upper[i]) and pd.notna(lower[i]):
                    if df['Close'].iloc[i-1] > upper[i-1]:
                        upper[i] = max(upper[i], upper[i-1])
                    if df['Close'].iloc[i-1] < lower[i-1]:
                        lower[i] = min(lower[i], lower[i-1])
                    
                    if trend.iloc[i-1] == -1 and df['Close'].iloc[i] > lower[i-1]:
                        trend.iloc[i] = 1
                    elif trend.iloc[i-1] == 1 and df['Close'].iloc[i] < upper[i-1]:
                        trend.iloc[i] = -1
                    else:
                        trend.iloc[i] = trend.iloc[i-1]
            
            return trend
        except Exception as e:
            logger.error(f"Supertrend calculation error: {e}")
            return pd.Series([1] * len(df), index=df.index)
    
    def detect_fractals(self, df, strength=2):
        """Detect fractal patterns"""
        highs = df['High'].values
        lows = df['Low'].values
        
        fractal_highs = pd.Series([False] * len(df), index=df.index)
        fractal_lows = pd.Series([False] * len(df), index=df.index)
        
        for i in range(strength, len(df) - strength):
            # Fractal high
            is_high = True
            for j in range(1, strength + 1):
                if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                    is_high = False
                    break
            fractal_highs.iloc[i] = is_high
            
            # Fractal low
            is_low = True
            for j in range(1, strength + 1):
                if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                    is_low = False
                    break
            fractal_lows.iloc[i] = is_low
        
        return fractal_highs, fractal_lows
    
    def calculate_probability(self, entropy_val, is_divergence, rsi_bonus, vol_bonus, 
                            trend_strength, atr_value, psr_acceleration, volume_ratio):
        """Calculate signal probability"""
        base_score = 40 + (trend_strength / atr_value * 30) + (1 - entropy_val) * 10
        entropy_score = 10 + (1 - entropy_val) * 5 if entropy_val < Config.ENTROPY_THRESHOLD else -5
        divergence_bonus = 20 + abs(psr_acceleration) * 2 if is_divergence else 0
        rsi_bonus_val = 8 if rsi_bonus else 0
        vol_bonus_val = 5 + (volume_ratio - 1) * 3 if vol_bonus else 0
        adaptive_bonus = 10 + trend_strength * 0.5 if trend_strength > atr_value else 0
        
        raw_prob = base_score + entropy_score + divergence_bonus + rsi_bonus_val + vol_bonus_val + adaptive_bonus
        return max(40, min(99, raw_prob))
    
    def analyze(self, df):
        """Main analysis function"""
        try:
            if len(df) < Config.ADAPTIVE_PERIOD:
                return {'signal': None, 'probability': 0, 'data': {}}
            
            # Calculate indicators
            hlc3 = (df['High'] + df['Low'] + df['Close']) / 3
            hlc3_smooth = hlc3.rolling(3).mean()
            
            atr = TechnicalIndicators.atr(df['High'].values, df['Low'].values, df['Close'].values, timeperiod=14)
            atr_current = atr[-1] if len(atr) > 0 else 1.0
            
            rsi = TechnicalIndicators.rsi(df['Close'].values, timeperiod=Config.RSI_LENGTH)
            rsi_current = rsi[-1] if len(rsi) > 0 else 50.0
            
            # Volume analysis
            volume_ma = df['Volume'].rolling(Config.VOL_LOOKBACK).mean()
            vol_spike = df['Volume'].iloc[-1] > (volume_ma.iloc[-1] * Config.VOL_MULTIPLIER)
            volume_ratio = df['Volume'].iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1.0
            
            # Calculate metrics
            entropy = self.calculate_entropy(hlc3_smooth, Config.ADAPTIVE_PERIOD)
            state, state_percent = self.quantum_state_analysis(hlc3_smooth, Config.ANALYSIS_LEVELS, Config.ADAPTIVE_PERIOD)
            psr_velocity, psr_acceleration = self.calculate_psr(hlc3_smooth, Config.ADAPTIVE_PERIOD)
            trend = self.get_supertrend(df)
            trend_current = trend.iloc[-1]
            
            # Fractal detection
            fractal_highs, fractal_lows = self.detect_fractals(df, Config.FRACTAL_STRENGTH)
            
            # Trend strength
            trend_strength = abs(hlc3_smooth.diff().rolling(10).mean().iloc[-1]) if len(hlc3_smooth) > 10 else 0.1
            
            # Signal detection
            is_peak_event = len(fractal_highs) > 1 and fractal_highs.iloc[-2] and df['Close'].iloc[-1] < df['Low'].iloc[-2]
            is_valley_event = len(fractal_lows) > 1 and fractal_lows.iloc[-2] and df['Close'].iloc[-1] > df['High'].iloc[-2]
            
            # State requirements
            is_peak_state_met = is_peak_event and (state >= Config.ANALYSIS_LEVELS - 1 - Config.EDGE_SENSITIVITY) and (trend_current == -1)
            is_valley_state_met = is_valley_event and (state <= Config.EDGE_SENSITIVITY) and (trend_current == 1)
            
            # RSI conditions
            rsi_top_cond = rsi_current > Config.RSI_TOP
            rsi_bot_cond = rsi_current < Config.RSI_BOTTOM
            
            # Calculate probabilities
            peak_prob = valley_prob = 0
            
            if is_peak_state_met:
                rsi_bonus = rsi_top_cond and Config.ENABLE_RSI_FILTER
                peak_prob = self.calculate_probability(
                    entropy, False, rsi_bonus, vol_spike and Config.ENABLE_VOLUME_FILTER, 
                    trend_strength, atr_current, psr_acceleration, volume_ratio
                )
            
            if is_valley_state_met:
                rsi_bonus = rsi_bot_cond and Config.ENABLE_RSI_FILTER
                valley_prob = self.calculate_probability(
                    entropy, False, rsi_bonus, vol_spike and Config.ENABLE_VOLUME_FILTER, 
                    trend_strength, atr_current, psr_acceleration, volume_ratio
                )
            
            # Determine signal
            signal_type = None
            probability = 0
            
            if peak_prob >= Config.MIN_PROBABILITY:
                signal_type = 'peak'
                probability = peak_prob
            elif valley_prob >= Config.MIN_PROBABILITY:
                signal_type = 'valley'
                probability = valley_prob
            
            return {
                'signal': signal_type,
                'probability': probability,
                'data': {
                    'price': df['Close'].iloc[-1],
                    'rsi': rsi_current,
                    'entropy': entropy,
                    'volume_spike': vol_spike,
                    'trend': trend_current,
                    'state': state,
                    'atr': atr_current,
                    'volume_ratio': volume_ratio,
                    'peak_prob': peak_prob,
                    'valley_prob': valley_prob
                }
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {'signal': None, 'probability': 0, 'data': {}}

class TelegramBot:
    """Telegram notification bot"""
    
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_message(self, message):
        """Send message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=payload, timeout=30)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    def test_connection(self):
        """Test bot connection"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('ok'):
                logger.info(f"✅ Bot connected: {data['result']['username']}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False

class RPDAlertBot:
    """Main RPD Alert Bot with enhanced error handling"""
    
    def __init__(self, trading_config):
        self.rpd = RPDIndicator()
        self.telegram = TelegramBot(Config.TELEGRAM_BOT_TOKEN, Config.TELEGRAM_CHAT_ID)
        self.trading_config = trading_config
        self.last_alerts = {}
        self.running = False
        self.last_health_check = datetime.now()
        self.failed_symbols = set()  # Track symbols that consistently fail
        self.stats = {
            'total_scans': 0,
            'alerts_sent': 0,
            'api_errors': 0,
            'uptime_start': datetime.now(),
            'successful_fetches': 0,
            'failed_fetches': 0
        }
    
    def should_send_alert(self, symbol, signal_type):
        """Prevent spam alerts with cooldown"""
        key = f"{symbol}_{signal_type}"
        current_time = datetime.now()
        
        if key in self.last_alerts:
            time_diff = current_time - self.last_alerts[key]
            timeframe = self.trading_config.get(symbol, '1h')
            
            # Longer cooldown periods to reduce noise
            cooldown_hours = {
                '1m': 1, '5m': 2, '15m': 4, 
                '1h': 6, '4h': 12, '1d': 24
            }.get(timeframe, 6)
            
            if time_diff < timedelta(hours=cooldown_hours):
                return False
        
        self.last_alerts[key] = current_time
        return True
    
    def format_message(self, symbol, result, timeframe):
        """Format alert message"""
        signal = result['signal']
        prob = result['probability']
        data = result['data']
        
        emoji = "🔴" if signal == 'peak' else "🟢"
        direction = "BEARISH (SHORT)" if signal == 'peak' else "BULLISH (LONG)"
        
        quality = "🔥 EXCEPTIONAL" if prob >= 90 else "⭐ STRONG" if prob >= 75 else "📊 MODERATE"
        
        message = f"""
{emoji} <b>RPD {signal.upper()} SIGNAL</b>

📊 <b>Symbol:</b> {symbol}
⏰ <b>Timeframe:</b> {timeframe}
💰 <b>Price:</b> ${data['price']:.4f}
🎯 <b>Probability:</b> {prob:.1f}% ({quality})
📈 <b>Direction:</b> {direction}

📋 <b>Technical Details:</b>
• RSI: {data['rsi']:.1f}
• Entropy: {data['entropy']:.3f}
• Volume Spike: {'✅ Yes' if data['volume_spike'] else '❌ No'}
• Trend: {'🟢 Bullish' if data['trend'] == 1 else '🔴 Bearish'}
• State Level: {data['state']}/{Config.ANALYSIS_LEVELS-1}
• Volume Ratio: {data['volume_ratio']:.2f}x

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
        """.strip()
        
        return message
    
    def analyze_symbol(self, symbol, timeframe):
        """Analyze single symbol with enhanced error handling"""
        try:
            # Skip symbols that have consistently failed
            if symbol in self.failed_symbols:
                logger.debug(f"Skipping {symbol} - in failed symbols list")
                return None
            
            # Adjust data period based on timeframe
            period_map = {
                '1m': '2d', '5m': '5d', '15m': '1mo', 
                '1h': '1mo', '4h': '3mo', '1d': '6mo'
            }
            period = period_map.get(timeframe, '1mo')
            
            df = MarketData.get_data(symbol, period=period, interval=timeframe)
            
            if df is None:
                self.stats['failed_fetches'] += 1
                logger.warning(f"❌ Failed to fetch data for {symbol}")
                
                # Add to failed symbols if it fails multiple times
                if symbol not in self.failed_symbols:
                    self.failed_symbols.add(symbol)
                    logger.warning(f"🚫 Adding {symbol} to failed symbols list")
                
                return None
            
            if len(df) < 50:
                self.stats['failed_fetches'] += 1
                logger.warning(f"⚠️ Insufficient data for {symbol}: {len(df)} rows")
                return None
            
            self.stats['successful_fetches'] += 1
            
            # Remove from failed symbols if fetch succeeds
            if symbol in self.failed_symbols:
                self.failed_symbols.remove(symbol)
                logger.info(f"✅ Removed {symbol} from failed symbols list")
            
            result = self.rpd.analyze(df)
            
            if result['signal'] and self.should_send_alert(symbol, result['signal']):
                message = self.format_message(symbol, result, timeframe)
                
                if self.telegram.send_message(message):
                    logger.info(f"🚨 Alert sent: {symbol}({timeframe}) {result['signal']} ({result['probability']:.1f}%)")
                    self.stats['alerts_sent'] += 1
                    return result
                else:
                    logger.error(f"❌ Failed to send alert for {symbol}")
            
            return None
            
        except Exception as e:
            self.stats['api_errors'] += 1
            logger.error(f"❌ Error analyzing {symbol}({timeframe}): {e}")
            return None
    
    def run_scan(self):
        """Run market scan with enhanced error handling"""
        logger.info("🔄 Starting enhanced market scan...")
        alerts_sent = 0
        errors = 0
        successful_scans = 0
        
        active_symbols = {k: v for k, v in self.trading_config.items() if k not in self.failed_symbols}
        
        if not active_symbols:
            logger.warning("⚠️ No active symbols available for scanning")
            # Reset failed symbols list if all symbols failed
            if self.failed_symbols:
                logger.info("🔄 Resetting failed symbols list")
                self.failed_symbols.clear()
            return 0
        
        logger.info(f"📊 Scanning {len(active_symbols)} active symbols (skipping {len(self.failed_symbols)} failed)")
        
        for symbol, timeframe in active_symbols.items():
            try:
                result = self.analyze_symbol(symbol, timeframe)
                if result:
                    alerts_sent += 1
                successful_scans += 1
                
                # Add delay between symbols to avoid rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Scan error for {symbol}({timeframe}): {e}")
                errors += 1
                continue
        
        self.stats['total_scans'] += 1
        
        success_rate = (successful_scans / len(active_symbols)) * 100 if active_symbols else 0
        
        logger.info(f"✅ Scan complete: {alerts_sent} alerts, {errors} errors, {success_rate:.1f}% success rate")
        
        return alerts_sent
    
    def send_health_check(self):
        """Send enhanced health check message"""
        if datetime.now() - self.last_health_check >= timedelta(seconds=Config.HEALTH_CHECK_INTERVAL):
            
            uptime = datetime.now() - self.stats['uptime_start']
            uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m"
            
            # Calculate success rates
            total_fetches = self.stats['successful_fetches'] + self.stats['failed_fetches']
            success_rate = (self.stats['successful_fetches'] / total_fetches * 100) if total_fetches > 0 else 0
            
            active_symbols = len(self.trading_config) - len(self.failed_symbols)
            
            health_msg = f"""
🤖 <b>RPD Bot Health Check</b>

✅ <b>Status:</b> Running Smoothly
📊 <b>Assets:</b> {active_symbols}/{len(self.trading_config)} active
⏰ <b>Uptime:</b> {uptime_str}
🔄 <b>Total Scans:</b> {self.stats['total_scans']}
🚨 <b>Alerts Sent:</b> {self.stats['alerts_sent']}

📈 <b>Data Quality:</b>
• Success Rate: {success_rate:.1f}%
• Successful Fetches: {self.stats['successful_fetches']}
• Failed Fetches: {self.stats['failed_fetches']}
• API Errors: {self.stats['api_errors']}

🚫 <b>Failed Symbols:</b> {len(self.failed_symbols)}
{', '.join(self.failed_symbols) if self.failed_symbols else 'None'}

🔄 <b>Next Check:</b> {Config.HEALTH_CHECK_INTERVAL // 3600}h
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
            """.strip()
            
            self.telegram.send_message(health_msg)
            self.last_health_check = datetime.now()
    
    def start_monitoring(self):
        """Start enhanced monitoring"""
        logger.info("🚀 Starting Enhanced RPD Alert Bot...")
        
        # Test Telegram connection
        if not self.telegram.test_connection():
            logger.error("❌ Telegram connection failed!")
            return False
        
        # Send startup message
        startup_msg = f"""
🤖 <b>Enhanced RPD Bot Started</b>

📊 <b>Configuration:</b>
• {len(self.trading_config)} assets monitored
• Check interval: {Config.CHECK_INTERVAL//60} minutes
• Min probability: {Config.MIN_PROBABILITY}%

🔧 <b>Features:</b>
• Enhanced Yahoo Finance handling
• Automatic symbol failure detection
• Progressive retry logic
• Volume filter: {'✅' if Config.ENABLE_VOLUME_FILTER else '❌'}
• RSI filter: {'✅' if Config.ENABLE_RSI_FILTER else '❌'}

✅ <b>Bot is now active and monitoring!</b>
        """.strip()
        
        self.telegram.send_message(startup_msg)
        logger.info("✅ Enhanced bot started successfully")
        
        self.running = True
        
        while self.running:
            try:
                self.run_scan()
                self.send_health_check()
                
                # Sleep with periodic status updates
                for i in range(0, Config.CHECK_INTERVAL, 300):  # Every 5 minutes
                    if not self.running:
                        break
                    time.sleep(min(300, Config.CHECK_INTERVAL - i))
                    if i > 0:  # Don't log immediately after scan
                        logger.info(f"💤 Sleeping... next scan in {(Config.CHECK_INTERVAL - i)//60} minutes")
                
            except KeyboardInterrupt:
                logger.info("🛑 Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
    
    def stop(self):
        """Stop the bot gracefully"""
        self.running = False
        if self.telegram:
            self.telegram.send_message("🛑 <b>RPD Alert Bot Stopped</b>\n\nBot has been gracefully shut down.")
        logger.info("🛑 Bot stopped gracefully")

# Global bot instance
bot = None

# Flask web application
app = Flask(__name__)

@app.route('/')
def health_check():
    """Enhanced health check endpoint"""
    if bot and bot.running:
        uptime = datetime.now() - bot.stats['uptime_start']
        uptime_str = f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m"
        
        total_fetches = bot.stats['successful_fetches'] + bot.stats['failed_fetches']
        success_rate = (bot.stats['successful_fetches'] / total_fetches * 100) if total_fetches > 0 else 0
        
        return jsonify({
            'status': 'running',
            'uptime': uptime_str,
            'total_scans': bot.stats['total_scans'],
            'alerts_sent': bot.stats['alerts_sent'],
            'assets_monitored': len(bot.trading_config),
            'active_assets': len(bot.trading_config) - len(bot.failed_symbols),
            'failed_symbols': list(bot.failed_symbols),
            'success_rate': f"{success_rate:.1f}%",
            'api_errors': bot.stats['api_errors'],
            'last_check': datetime.now().isoformat()
        })
    return jsonify({'status': 'stopped'})

@app.route('/status')
def status():
    """Detailed status endpoint"""
    if bot and bot.running:
        return jsonify({
            'bot_status': 'running',
            'configuration': bot.trading_config,
            'stats': {
                'total_scans': bot.stats['total_scans'],
                'alerts_sent': bot.stats['alerts_sent'],
                'successful_fetches': bot.stats['successful_fetches'],
                'failed_fetches': bot.stats['failed_fetches'],
                'api_errors': bot.stats['api_errors'],
                'uptime_start': bot.stats['uptime_start'].isoformat()
            },
            'failed_symbols': list(bot.failed_symbols),
            'last_alerts': {k: v.isoformat() for k, v in bot.last_alerts.items()},
            'config': {
                'check_interval': Config.CHECK_INTERVAL,
                'min_probability': Config.MIN_PROBABILITY,
                'volume_filter': Config.ENABLE_VOLUME_FILTER,
                'rsi_filter': Config.ENABLE_RSI_FILTER
            }
        })
    return jsonify({'bot_status': 'stopped'})

@app.route('/scan', methods=['POST'])
def manual_scan():
    """Manual scan endpoint"""
    if not bot or not bot.running:
        return jsonify({'error': 'Bot not running'}), 503
    
    try:
        alerts_sent = bot.run_scan()
        return jsonify({
            'status': 'completed',
            'alerts_sent': alerts_sent,
            'active_symbols': len(bot.trading_config) - len(bot.failed_symbols),
            'failed_symbols': list(bot.failed_symbols),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Manual scan error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/reset-failed', methods=['POST'])
def reset_failed_symbols():
    """Reset failed symbols list"""
    if not bot or not bot.running:
        return jsonify({'error': 'Bot not running'}), 503
    
    failed_count = len(bot.failed_symbols)
    bot.failed_symbols.clear()
    
    return jsonify({
        'status': 'success',
        'message': f'Reset {failed_count} failed symbols',
        'timestamp': datetime.now().isoformat()
    })

def run_monitoring():
    """Run monitoring in background thread"""
    global bot
    try:
        trading_config = Config.validate()
        bot = RPDAlertBot(trading_config)
        bot.start_monitoring()
    except Exception as e:
        logger.error(f"❌ Monitoring setup error: {e}")

def main():
    """Main entry point"""
    try:
        # Start monitoring in background thread
        monitoring_thread = threading.Thread(target=run_monitoring, daemon=True)
        monitoring_thread.start()
        
        # Give the bot time to initialize
        time.sleep(10)
        
        # Start Flask web server
        port = Config.PORT
        logger.info(f"🌐 Starting web service on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt, shutting down...")
        if bot:
            bot.stop()
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
