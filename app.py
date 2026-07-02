import time
import threading
import pytz
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_socketio import SocketIO
from moomoo import *
import yfinance as yf
import argparse
import os
import json
import sqlite3
import secrets
from collections import defaultdict
from functools import wraps
from authlib.integrations.flask_client import OAuth
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
import pricing
import pandas_ta as ta

# --- CONFIGURATION ---
config_path = os.path.join(os.path.dirname(__file__), 'config.json')
CONFIG = {}
if os.path.exists(config_path):
    with open(config_path) as f:
        CONFIG = json.load(f)

ENV = CONFIG.get('env', 'test')
FLASK_PORT = 3000 if ENV == 'prod' else 3001
OPEND_ADDR = CONFIG.get('opend_addr', 'opend.garylu.com')
OPEND_PORT = CONFIG.get('opend_port', 11111)

parser = argparse.ArgumentParser(description="XSP Options Monitor")
parser.add_argument("--floor", type=float, default=0.95, help="Floor percentage (default: 0.95)")
parser.add_argument("--ceiling", type=float, default=1.05, help="Ceiling percentage (default: 1.05)")
parser.add_argument("--refresh", type=int, default=5, help="Refresh frequency in seconds (default: 5)")
parser.add_argument("--watchlist-size", type=int, default=20, help="Watchlist size (default: 20)")
parser.add_argument("--option-days", type=int, default=14, help="Option days (default: 14)")
args, unknown = parser.parse_known_args()

FLOOR_PERCENT = args.floor
CEILING_PERCENT = args.ceiling
REF_SYMBOL = 'US.SPY'
MES_SYMBOL = 'US.MESmain'  
REFRESH_INTERVAL = args.refresh
WATCHLIST_SIZE = args.watchlist_size
OPTION_DAYS = args.option_days
WATCHLIST_FILE = 'watchlist.json'

# --- PREMIUM LOGGER SETTINGS ---
DB_PATH      = 'premium_log.db'
LOG_INTERVAL = 600          # seconds between DB snapshots (10 min)
ET_TZ        = pytz.timezone('America/New_York')
last_log_ts  = 0.0

app = Flask(__name__)
app.secret_key = CONFIG.get('secret_key') or secrets.token_hex(32)

# Proxy fix so url_for(_external=True) generates correct https:// behind nginx
from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- AUTHENTICATION ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, email, name):
        self.id = email
        self.email = email
        self.name = name

_users = {}

@login_manager.user_loader
def load_user(user_id):
    return _users.get(user_id)

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=CONFIG.get('google_client_id', ''),
    client_secret=CONFIG.get('google_client_secret', ''),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

# API routes that should return 401 JSON instead of redirect
def api_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# 全局数据缓存
latest_data = {
    "index": {"price": 0, "floor": 0, "ceiling": 0},
    "options": {}
}

_toast_throttle = {"msg": "", "ts": 0}

def emit_toast(sio, msg):
    global _toast_throttle
    now = time.time()
    if msg != _toast_throttle["msg"] or now - _toast_throttle["ts"] > 60:
        _toast_throttle["msg"] = msg
        _toast_throttle["ts"] = now
        print(msg)
        sio.emit('toast_error', msg)

user_watchlist = []
if os.path.exists(WATCHLIST_FILE):
    try:
        with open(WATCHLIST_FILE, 'r') as f:
            user_watchlist = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load watchlist: {e}")


# 历史统计缓存 (用于 VIX 和 ATR 计算)
historical_stats = {
    "vix": 15.0,  # 默认值
    "vix_rank": 0.0,
    "vix_percentile": 0.0,
    "atr_14": 5.0,  # 默认值
    "ema_20": 0.0,  # 默认值
    "skew": 0.0,    # 默认值
    "adx": 20.0,    # 趋势指标默认
    "er": 0.5,
    "bbw": 25.0,
    "dev": 0.0,
    "vr": 1.0,
    "last_updated": 0
}

def init_db():
    """Create the SQLite premium_log table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS premium_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          INTEGER NOT NULL,
            trade_date  TEXT    NOT NULL,
            session     TEXT    NOT NULL,
            xsp_price   REAL,
            vix         REAL,
            group_idx   INTEGER,
            opt_symbol  TEXT,
            strike      REAL,
            expiry      TEXT,
            opt_type    TEXT,
            role        TEXT,
            bid         REAL,
            ask         REAL,
            mid         REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pl_date ON premium_log(trade_date, group_idx, role)")
    conn.commit()
    conn.close()
    print(f"✅ Premium Logger DB ready: {DB_PATH}")


def get_trade_date_and_session(ts_unix):
    """
    Map a UTC Unix timestamp to a US trade date (ET calendar date) and session label.

    Sessions (all in ET):
      asia_pm    22:00 prev-day – 04:00  (user's AEST afternoon / evening)
      pre_market 04:00 – 09:30
      open_30    09:30 – 10:00  (first 30 min; IV-release window)
      regular    10:00 – 22:00

    Recordings at 22:00-23:59 ET are attributed to the NEXT calendar day so
    that every 'asia_pm' snapshot for a given US trading day shares the same
    trade_date key.  Down/up classification is then done by comparing XSP
    close vs prev_close on that same ET date.
    """
    dt_utc = datetime.fromtimestamp(ts_unix, tz=pytz.utc)
    dt_et  = dt_utc.astimezone(ET_TZ)
    t      = dt_et.hour * 60 + dt_et.minute   # minutes since ET midnight

    if t >= 22 * 60:                      # 22:00-23:59 → next day's asia_pm
        trade_date = (dt_et + timedelta(days=1)).strftime('%Y-%m-%d')
        session    = 'asia_pm'
    elif t < 4 * 60:                      # 00:00-03:59 → same day asia_pm
        trade_date = dt_et.strftime('%Y-%m-%d')
        session    = 'asia_pm'
    elif t < 9 * 60 + 30:                 # 04:00-09:29
        trade_date = dt_et.strftime('%Y-%m-%d')
        session    = 'pre_market'
    elif t < 10 * 60:                     # 09:30-09:59
        trade_date = dt_et.strftime('%Y-%m-%d')
        session    = 'open_30'
    else:                                  # 10:00+
        trade_date = dt_et.strftime('%Y-%m-%d')
        session    = 'regular'

    return trade_date, session


def log_premium_snapshot():
    """
    Self-throttled logger: persists a premium snapshot to SQLite at most once
    every LOG_INTERVAL seconds.  Records short leg, long leg, and spread
    (bid / ask / mid) for every watchlist group, plus XSP price and VIX.
    Older-than-90-day rows are pruned on every write.
    """
    global last_log_ts, latest_data, user_watchlist

    now_ts = time.time()
    if now_ts - last_log_ts < LOG_INTERVAL:
        return
    last_log_ts = now_ts

    trade_date, session = get_trade_date_and_session(now_ts)
    xsp_price = latest_data["index"].get("price", 0)
    vix       = latest_data["index"].get("vix", 0)

    if xsp_price <= 0:
        return  # anchor price not yet available

    rows = []
    for idx, group in enumerate(user_watchlist):
        ds       = group.get('date')
        s_strike = group.get('short')
        l_strike = group.get('long')
        m_strike = group.get('mid')
        strategy = group.get('strategy', 'xmas')
        if not (ds and s_strike and l_strike):
            continue
        try:
            s_val    = float(s_strike)
            l_val    = float(l_strike)
            opt_type = 'P' if s_val > l_val else 'C'
            s_sym    = f"US.XSP{ds}{opt_type}{int(s_val * 1000)}"
            l_sym    = f"US.XSP{ds}{opt_type}{int(l_val * 1000)}"
            expiry   = f"20{ds[0:2]}-{ds[2:4]}-{ds[4:6]}"

            if s_sym in latest_data["options"]:
                o = latest_data["options"][s_sym]
                rows.append((int(now_ts), trade_date, session, xsp_price, vix,
                             idx + 1, s_sym, s_val, expiry, opt_type, 'short',
                             o['bid'], o['ask'], o['mid']))

            if l_sym in latest_data["options"]:
                o = latest_data["options"][l_sym]
                rows.append((int(now_ts), trade_date, session, xsp_price, vix,
                             idx + 1, l_sym, l_val, expiry, opt_type, 'long',
                             o['bid'], o['ask'], o['mid']))

            has_mid = m_strike and str(m_strike).strip()
            if s_sym in latest_data["options"] and l_sym in latest_data["options"]:
                so = latest_data["options"][s_sym]
                lo = latest_data["options"][l_sym]
                if not has_mid:
                    rows.append((int(now_ts), trade_date, session, xsp_price, vix,
                                 idx + 1, f"{s_sym}|{l_sym}", None, expiry, opt_type, 'spread',
                                 round(so['bid'] - lo['ask'], 4),
                                 round(so['ask'] - lo['bid'], 4),
                                 round(so['mid'] - lo['mid'], 4)))

            if has_mid:
                try:
                    m_val = float(m_strike)
                    lower = min(s_val, l_val)
                    upper = max(s_val, l_val)
                    if lower < m_val < upper:
                        m_sym = f"US.XSP{ds}{opt_type}{int(m_val * 1000)}"
                        if m_sym in latest_data["options"]:
                            mo = latest_data["options"][m_sym]
                            rows.append((int(now_ts), trade_date, session, xsp_price, vix,
                                         idx + 1, m_sym, m_val, expiry, opt_type, 'mid',
                                         mo['bid'], mo['ask'], mo['mid']))

                        low_sym = f"US.XSP{ds}{opt_type}{int(lower * 1000)}"
                        up_sym  = f"US.XSP{ds}{opt_type}{int(upper * 1000)}"
                        if low_sym in latest_data["options"] and m_sym in latest_data["options"] and up_sym in latest_data["options"]:
                            lo = latest_data["options"][low_sym]
                            mo = latest_data["options"][m_sym]
                            uo = latest_data["options"][up_sym]
                            if strategy == 'bfly':
                                rows.append((int(now_ts), trade_date, session, xsp_price, vix,
                                             idx + 1, f"{low_sym}|{m_sym}|{up_sym}", None, expiry, opt_type, 'bfly',
                                             round(lo['bid'] + uo['bid'] - 2 * mo['ask'], 4),
                                             round(lo['ask'] + uo['ask'] - 2 * mo['bid'], 4),
                                             round(lo['mid'] + uo['mid'] - 2 * mo['mid'], 4)))
                            else:
                                short_x = latest_data["options"][s_sym]
                                long_x = latest_data["options"][l_sym]
                                rows.append((int(now_ts), trade_date, session, xsp_price, vix,
                                             idx + 1, f"{low_sym}|{m_sym}|{up_sym}", None, expiry, opt_type, 'xmas',
                                                 round(short_x['bid'] + 2*long_x['bid'] - 3*mo['ask'], 4),
                                                 round(short_x['ask'] + 2*long_x['ask'] - 3*mo['bid'], 4),
                                                 round(short_x['mid'] + 2*long_x['mid'] - 3*mo['mid'], 4)))
                except Exception as bf_e:
                    print(f"⚠️ log_premium_snapshot group {idx + 1} three-leg error: {bf_e}")

        except Exception as e:
            print(f"⚠️ log_premium_snapshot group {idx + 1} error: {e}")

    if not rows:
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.executemany("""
            INSERT INTO premium_log
                (ts, trade_date, session, xsp_price, vix,
                 group_idx, opt_symbol, strike, expiry, opt_type, role,
                 bid, ask, mid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        cutoff = int(now_ts - 90 * 86400)
        conn.execute("DELETE FROM premium_log WHERE ts < ?", (cutoff,))
        conn.commit()
        conn.close()
        print(f"📝 Logged {len(rows)} premium rows | {trade_date} | {session} | XSP={xsp_price:.2f}")
    except Exception as e:
        print(f"⚠️ DB write error: {e}")


def update_historical_data():
    global historical_stats
    now_ts = time.time()
    # 2小时更新一次
    if now_ts - historical_stats["last_updated"] < 7200:
        return
    
    try:
        print("🔄 Fetching historical data for VIX and ATR from YFinance...")
        # VIX 1 year history
        vix_ticker = yf.Ticker("^VIX")
        vix_hist = vix_ticker.history(period="1y")
        if not vix_hist.empty:
            current_vix = vix_hist['Close'].iloc[-1]
            vix_min = vix_hist['Close'].min()
            vix_max = vix_hist['Close'].max()
            vix_rank = (current_vix - vix_min) / (vix_max - vix_min) if (vix_max - vix_min) > 0 else 0.0
            vix_percentile = (vix_hist['Close'] < current_vix).mean()
            
            historical_stats["vix"] = float(current_vix)
            historical_stats["vix_rank"] = float(vix_rank) * 100
            historical_stats["vix_percentile"] = float(vix_percentile) * 100
            
        # XSP ATR 14 & EMA 20
        xsp_ticker = yf.Ticker("^XSP")
        xsp_hist = xsp_ticker.history(period="2mo")
        if len(xsp_hist) >= 15:
            highs = xsp_hist['High']
            lows = xsp_hist['Low']
            closes = xsp_hist['Close'].shift(1)
            
            tr1 = highs - lows
            tr2 = (highs - closes).abs()
            tr3 = (lows - closes).abs()
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_14 = tr.iloc[-14:].mean()
            historical_stats["atr_14"] = float(atr_14)

        if len(xsp_hist) >= 20:
            ema_20 = xsp_hist['Close'].ewm(span=20, adjust=False).mean().iloc[-1]
            historical_stats["ema_20"] = float(ema_20)

        # SPY 日线趋势指标 (ADX, ER, BBW, Deviation, Vol Ratio)
        try:
            spy_ticker = yf.Ticker("SPY")
            spy_daily = spy_ticker.history(period="2mo")
            if len(spy_daily) >= 25:
                close = spy_daily['Close']
                high  = spy_daily['High']
                low   = spy_daily['Low']
                vol   = spy_daily['Volume']

                # 1. ADX(14)
                adx_df = ta.adx(high, low, close, length=14)
                historical_stats["adx"] = float(adx_df['ADX_14'].iloc[-1])

                # 2. Efficiency Ratio(10)
                changes = close.diff().abs()
                er_val = abs(close.iloc[-1] - close.iloc[-11]) / changes.tail(10).sum()
                historical_stats["er"] = float(er_val) if er_val > 0 else 0.0

                # 3. Bollinger Bands Width%(20,2)
                bb_df = ta.bbands(close, length=20, std=2)
                upper = bb_df.iloc[:, 2]  # BBU column
                lower = bb_df.iloc[:, 0]  # BBL column
                mid   = bb_df.iloc[:, 1]  # BBM column
                bbw_val = (upper - lower) / mid * 100
                historical_stats["bbw"] = float(bbw_val.iloc[-1])

                # 4. Price Deviation from SMA20(%)
                sma20 = close.rolling(20).mean()
                dev_val = (close.iloc[-1] - sma20.iloc[-1]) / sma20.iloc[-1] * 100
                historical_stats["dev"] = float(dev_val)

                # 5. Volume Ratio (当前量 / 20日均量)
                avg_vol = vol.rolling(20).mean()
                vr_val = vol.iloc[-1] / avg_vol.iloc[-1]
                historical_stats["vr"] = float(vr_val)
        except Exception as spy_err:
            emit_toast(socketio, f"⚠️ SPY 趋势数据获取失败: {spy_err}")

        historical_stats["last_updated"] = now_ts
        print(f"✅ Historical data updated: VIX={historical_stats['vix']:.2f} (Rank={historical_stats['vix_rank']:.1f}%, Percentile={historical_stats['vix_percentile']:.1f}%), ATR_14={historical_stats['atr_14']:.2f}, EMA_20={historical_stats['ema_20']:.2f}, SKEW={historical_stats['skew']:.2f}, ADX={historical_stats['adx']:.1f}, ER={historical_stats['er']:.2f}, BBW={historical_stats['bbw']:.1f}%, Dev={historical_stats['dev']:.2f}%, VR={historical_stats['vr']:.2f}x")
    except Exception as e:
        emit_toast(socketio, f"⚠️ 历史数据更新失败: {e}")


def get_xsp_anchor_price():
    try:
        # ^XSP is the Yahoo Finance symbol for the Mini-SPX Index
        ticker = yf.Ticker("^XSP")
        # fast_info provides the most recent price without a full download
        current_price = ticker.fast_info['last_price']
        
        # Fallback to previous close if current is 0 or NaN
        if not current_price or current_price <= 0:
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
        return float(current_price)
    except Exception as e:
        emit_toast(socketio, f"⚠️ 行情数据获取失败: {e}")
        return 0
        
def format_row(row):
    symbol = row['code']
    try:
        # XSP 格式固定: US.XSP (6位) + YYMMDD (6位) + Type (1位) + Strike
        # 例子: US.XSP260528C760000
        # 索引 0-5: US.XSP
        # 索引 6-11: 260528 (日期)
        # 索引 12: C 或 P (类型)
        # 索引 13+: 760000 (行权价)
        
        d_str = symbol[6:12]
        expiry = f"20{d_str[0:2]}-{d_str[2:4]}-{d_str[4:6]}"
        
        opt_type = symbol[12] # 直接取第 13 个字符
        strike_raw = symbol[13:] # 取第 14 个字符往后的所有内容
        strike = float(strike_raw) / 1000
    except Exception as e:
        print(f"❌ 解析错误 {symbol}: {e}")
        return None

    bid = float(row.get('bid_price') or 0.0)
    ask = float(row.get('ask_price') or 0.0)
    last = float(row.get('last_price') or 0.0)
    mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last
    delta = float(row.get('option_delta') or 0.0)
    gamma = float(row.get('option_gamma') or 0.0)
    theta = float(row.get('option_theta') or 0.0)
    vega = float(row.get('option_vega') or 0.0)
    iv = float(row.get('option_implied_volatility') or 0.0)
    open_interest = int(row.get('option_open_interest') or 0)

    # 过滤逻辑 (Put 看负 Delta, Call 看正 Delta)
    # 如果刚开盘 Delta 还没算出来，可以先注释掉这两行
    #if opt_type == 'P' and delta > -0.15: return None
    #if opt_type == 'C' and delta < 0.15: return None

    # 星标逻辑
    is_watched = False
    for group in user_watchlist:
        if group.get('date') == d_str:
            try:
                short_val = group.get('short')
                long_val = group.get('long')
                mid_val = group.get('mid')
                s_strike = int(float(short_val)) if short_val else None
                l_strike = int(float(long_val)) if long_val else None
                m_strike = int(float(mid_val)) if mid_val else None
                if s_strike is not None and l_strike is not None:
                    group_type = 'P' if s_strike > l_strike else 'C'
                    if opt_type == group_type and strike in (s_strike, l_strike, m_strike):
                        is_watched = True
                        break
            except Exception:
                continue
    result = {
        'symbol': symbol, 'strike': strike, 'expiry': expiry, 'opt_type': opt_type,
        'bid': bid, 'ask': ask, 'mid': mid, 'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'iv': iv, 'is_watched': is_watched, 'open_interest': open_interest
    }
    return result
    
def generate_xsp_symbols(current_price, floor_price, ceiling_price):
    if current_price <= 0:
        return []
    
    symbols = []
    tz = pytz.timezone('Australia/Sydney')
    now = datetime.now(tz)

    # 2026 US Market Holidays (YYMMDD format)
    # Source: NYSE/CBOE 2026 Holiday Calendar
    holidays = [
        # 2026
        '260101', '260119', '260216', '260403', '260525', 
        '260619', '260703', '260907', '261126', '261225',
        # 2027
        '270101', '270118', '270215', '270326', '270531', 
        '270618', '270705', '270906', '271125', '271224'
    ]
    
    dates = []
    check_date = now
    while len(dates) < OPTION_DAYS:
        # Step 1: Define date_str FIRST
        date_str = check_date.strftime('%y%m%d')
        
        # Step 2: Check if it's a weekday AND not a holiday
        if check_date.weekday() < 5 and date_str not in holidays: 
            dates.append(date_str)
            
        # Step 3: Always move to the next day
        check_date += timedelta(days=1)

    # --- Robust Strike Logic ---
    # Put Range: From Floor up to Current Price + 2 ticks
    p_start = int((floor_price // 5) * 5) + 5
    p_end = int((current_price // 5) * 5) + 25
    
    # Call Range: From Current -25 up to Ceiling (mirrors Put's +25 ITM coverage)
    c_start = int((current_price // 5) * 5) - 25
    c_end = int((ceiling_price // 5) * 5)

    ticker = yf.Ticker("^XSP")
    for ds in dates:
        # get option chain from yfinance
        opt_chain = ticker.option_chain(datetime.strptime(ds, "%y%m%d").strftime("%Y-%m-%d"))
        # Generate Puts (ensure start < end)
        if p_start <= p_end:
            for strike in range(p_start, p_end + 5, 5):
                if strike in opt_chain.puts['strike'].values:
                    symbols.append(f"US.XSP{ds}P{int(strike * 1000)}")
        
        # Generate Calls (ensure start < end)
        if c_start <= c_end:
            for strike in range(c_start, c_end + 5, 5):
                if strike in opt_chain.calls['strike'].values:
                    symbols.append(f"US.XSP{ds}C{int(strike * 1000)}")
        
        # Force-add ATM option(s) for SKEW calculation
        atm_strike = round(current_price / 5) * 5
        for t in ('P', 'C'):
            sym = f"US.XSP{ds}{t}{int(atm_strike * 1000)}"
            if sym not in symbols:
                symbols.append(sym)
        
        if len(symbols) >= 399: break
    
    print(f"📊 Generated {len(symbols)} total contracts (Puts & Calls)")
    return symbols[:399]


def calc_skew_from_options():
    """Calculate SKEW = 4% deep Put IV - ATM IV using cached Moomoo option data.
    Finds the expiry closest to 5 trading days out.
    """
    global latest_data, historical_stats
    price = latest_data["index"].get("price", 0)
    if price <= 0:
        return

    today = datetime.now(ET_TZ).date()
    options = latest_data["options"]

    # Gather unique expiry dates from cached options, pick the 5th
    expiries = sorted(set(
        datetime.strptime(opt['expiry'], '%Y-%m-%d').date()
        for opt in options.values()
    ))
    future_expiries = [e for e in expiries if e > today]
    if len(future_expiries) < 5:
        return
    expiry_str = future_expiries[4].strftime('%Y-%m-%d')  # 5th = index 4

    # Filter options for this expiry
    exp_opts = [opt for opt in options.values() if opt['expiry'] == expiry_str]

    # Find ATM strike (closest to current price)
    strikes = sorted(set(opt['strike'] for opt in exp_opts))
    if not strikes:
        return
    atm_strike = min(strikes, key=lambda s: abs(s - price))
    atm_put = next((opt for opt in exp_opts if opt['strike'] == atm_strike and opt['opt_type'] == 'P'), None)

    if not atm_put or atm_put['iv'] <= 0:
        return

    # Find 4% deep OTM put
    deep_target = price * 0.96
    puts = [opt for opt in exp_opts if opt['opt_type'] == 'P']
    if not puts:
        return
    deep_put = min(puts, key=lambda p: abs(p['strike'] - deep_target))
    if deep_put['iv'] <= 0:
        return

    skew_val = (deep_put['iv'] - atm_put['iv'])  # in decimal (0.15 = 15 points)

    historical_stats["skew"] = round(skew_val, 1)
    latest_data["index"]["skew"] = historical_stats["skew"]

def start_moomoo():
    global latest_data
    print(f"🚀 Unified Snapshot Loop Active (Interval: {REFRESH_INTERVAL}s)...")
    
    with OpenQuoteContext(host=OPEND_ADDR, port=OPEND_PORT) as quote_ctx:
        while True:
            try:
                # 检查并更新历史统计数据 (VIX, ATR)
                update_historical_data()
                
                price = get_xsp_anchor_price()
                if price > 0:
                    mes_price = latest_data["index"].get("mes_price")
                    mes_change = latest_data["index"].get("mes_change")
                    mes_change_pct = latest_data["index"].get("mes_change_pct")
                    latest_data["index"] = {
                        "price": price, 
                        "floor": price * FLOOR_PERCENT,
                        "ceiling": price * CEILING_PERCENT,
                        "vix": historical_stats["vix"],
                        "vix_rank": historical_stats["vix_rank"],
                        "vix_percentile": historical_stats["vix_percentile"],
                        "atr_14": historical_stats["atr_14"],
                        "ema_20": historical_stats["ema_20"],
                        "skew": historical_stats["skew"],
                        "adx": historical_stats["adx"],
                        "er": historical_stats["er"],
                        "bbw": historical_stats["bbw"],
                        "dev": historical_stats["dev"],
                        "vr": historical_stats["vr"]
                    }
                    if mes_price is not None:
                        latest_data["index"]["mes_price"] = mes_price
                        latest_data["index"]["mes_change"] = mes_change
                        latest_data["index"]["mes_change_pct"] = mes_change_pct
                socketio.emit('index_update', latest_data["index"])

                # 通过 yfinance 获取 MES 期货当前行情（延迟约 15-20 分钟，免费版局限）
                try:
                    mes_ticker = yf.Ticker("MES=F")
                    curr_price = None
                    prev_close = None
                    try:
                        info = mes_ticker.info
                        curr_price = info.get('regularMarketPrice')
                        prev_close = info.get('regularMarketPreviousClose')
                    except Exception:
                        pass
                    if not curr_price or curr_price <= 0 or not prev_close or prev_close <= 0:
                        mes_hist = mes_ticker.history(period="1mo")
                        if not mes_hist.empty and len(mes_hist) >= 2:
                            curr_price = curr_price or float(mes_hist['Close'].iloc[-1])
                            prev_close = prev_close or float(mes_hist['Close'].iloc[-2])
                    if curr_price and prev_close and curr_price > 0 and prev_close > 0:
                        mes_chg = curr_price - prev_close
                        mes_chg_pct = (mes_chg / prev_close * 100) if prev_close > 0 else 0
                        latest_data["index"]["mes_price"] = round(float(curr_price), 2)
                        latest_data["index"]["mes_change"] = round(float(mes_chg), 2)
                        latest_data["index"]["mes_change_pct"] = round(float(mes_chg_pct), 2)
                        socketio.emit('index_update', latest_data["index"])
                except Exception as e:
                    print(f"⚠️ Failed to fetch MES from yfinance: {e}")

                # 1. 动态确定本次需要拉取的代码列表
                current_price = latest_data["index"].get("price", 0)
                valid_dates = []
                if current_price <= 0:
                    # 第一次运行或没拿到价格：只请求 SPY
                    all_symbols = [REF_SYMBOL]
                else:
                    # 已有价格：请求 SPY + 生成的期权列表
                    floor = latest_data["index"]["floor"]
                    ceiling = latest_data["index"]["ceiling"]
                    opt_symbols = generate_xsp_symbols(current_price, floor, ceiling)
                    
                    # 提取当前所有有效的到期日 (YYYY-MM-DD 格式)
                    valid_dates = sorted(list(set([
                        f"20{s[6:8]}-{s[8:10]}-{s[10:12]}" for s in opt_symbols
                    ])))
                    all_symbols = opt_symbols

                # 注入 Watchlist 中的三腿，确保一定能请求到快照数据
                for group in user_watchlist:
                    ds = group.get('date')
                    s_strike = group.get('short')
                    l_strike = group.get('long')
                    if ds and s_strike and l_strike:
                        try:
                            s_val = float(s_strike)
                            l_val = float(l_strike)
                            opt_type = 'P' if s_val > l_val else 'C'
                            
                            s_sym = f"US.XSP{ds}{opt_type}{int(s_val * 1000)}"
                            l_sym = f"US.XSP{ds}{opt_type}{int(l_val * 1000)}"
                            
                            if s_sym not in all_symbols:
                                all_symbols.append(s_sym)
                            if l_sym not in all_symbols:
                                all_symbols.append(l_sym)
                                
                            m_strike = group.get('mid')
                            if m_strike and str(m_strike).strip():
                                try:
                                    m_val = float(m_strike)
                                    m_sym = f"US.XSP{ds}{opt_type}{int(m_val * 1000)}"
                                    if m_sym not in all_symbols:
                                        all_symbols.append(m_sym)
                                except:
                                    pass
                                
                            # 同时也把对应的到期日加入 valid_dates，防止被前端过滤掉卡片
                            expiry_date = f"20{ds[0:2]}-{ds[2:4]}-{ds[4:6]}"
                            if expiry_date not in valid_dates:
                                valid_dates.append(expiry_date)
                        except Exception as e:
                            print(f"⚠️ 解析 Watchlist 代码异常: {e}")

                if current_price > 0:
                    socketio.emit('active_dates', sorted(valid_dates))

                print(f"🔍 Requesting snapshot for {len(all_symbols)} symbols (Index Price: {current_price:.2f})...")
                # 2. 发起合并快照请求
                ret, data = quote_ctx.get_market_snapshot(all_symbols)
                
                if ret == RET_OK and not data.empty:
                    updated_symbols = []
                    for _, row in data.iterrows():
                        # A. 处理指数/基准
                        if row['code'] == REF_SYMBOL:
                            p = row['last_price'] if row['last_price'] > 0 else row['prev_close_price']
                            mes_price = latest_data["index"].get("mes_price")
                            mes_change = latest_data["index"].get("mes_change")
                            mes_change_pct = latest_data["index"].get("mes_change_pct")
                            latest_data["index"] = {
                                "price": p, 
                                "floor": p * FLOOR_PERCENT,
                                "ceiling": p * CEILING_PERCENT,
                                "vix": historical_stats["vix"],
                                "vix_rank": historical_stats["vix_rank"],
                                "vix_percentile": historical_stats["vix_percentile"],
                                "atr_14": historical_stats["atr_14"],
                                "ema_20": historical_stats["ema_20"],
                                "skew": historical_stats["skew"]
                            }
                            if mes_price is not None:
                                latest_data["index"]["mes_price"] = mes_price
                                latest_data["index"]["mes_change"] = mes_change
                                latest_data["index"]["mes_change_pct"] = mes_change_pct
                            socketio.emit('index_update', latest_data["index"])

                        # B. 处理期权数据
                        else:
                            item = format_row(row)
                            if not item:
                                continue
                            latest_data["options"][row['code']] = item
                            updated_symbols.append(row['code'])
                    
                    # Manage Order Book Subscriptions for watched options
                    watched_symbols = {sym for sym, item in latest_data["options"].items() if item['is_watched']}
                    global current_subscribed_ob
                    if 'current_subscribed_ob' not in globals():
                        current_subscribed_ob = set()
                    
                    new_to_subscribe = watched_symbols - current_subscribed_ob
                    to_unsubscribe = current_subscribed_ob - watched_symbols
                    
                    if new_to_subscribe:
                        quote_ctx.subscribe(list(new_to_subscribe), [SubType.ORDER_BOOK], subscribe_push=False)
                    if to_unsubscribe:
                        quote_ctx.unsubscribe(list(to_unsubscribe), [SubType.ORDER_BOOK])
                        
                    current_subscribed_ob = watched_symbols
                    
                    # Fetch order book for watched symbols
                    for sym in watched_symbols:
                        ret_ob, ob_data = quote_ctx.get_order_book(sym, num=3)
                        if ret_ob == RET_OK:
                            latest_data["options"][sym]['ob_ask'] = ob_data['Ask']
                            latest_data["options"][sym]['ob_bid'] = ob_data['Bid']
                    
                    # Emit updates to frontend
                    for sym in updated_symbols:
                        socketio.emit('option_update', latest_data["options"][sym])

                    # 计算各到期日 Gamma 敞口峰值 (Gamma × OI 最大的行权价)
                    put_walls = {}
                    for opt in latest_data["options"].values():
                        exp = opt['expiry']
                        oi = opt.get('open_interest', 0)
                        gamma = abs(opt.get('gamma', 0))
                        gex = gamma * oi
                        if opt['opt_type'] == 'P' and oi > 0 and gamma > 0:
                            if exp not in put_walls or gex > put_walls[exp]['gex']:
                                put_walls[exp] = {'strike': opt['strike'], 'oi': oi, 'gex': gex}
                    # 3. 计算和广播 Watchlist 中的组合数据 (Spreads / Butterflies / Christmas Trees)
                    for idx, group in enumerate(user_watchlist):
                        ds = group.get('date')
                        s_strike = group.get('short')
                        l_strike = group.get('long')
                        entry_val = group.get('entry')
                        m_strike = group.get('mid')
                        strategy = group.get('strategy', 'xmas')
                        
                        if ds and s_strike and l_strike:
                            try:
                                s_val = float(s_strike)
                                l_val = float(l_strike)
                                opt_type = 'P' if s_val > l_val else 'C'
                                
                                s_sym = f"US.XSP{ds}{opt_type}{int(s_val * 1000)}"
                                l_sym = f"US.XSP{ds}{opt_type}{int(l_val * 1000)}"
                                
                                if s_sym in latest_data["options"] and l_sym in latest_data["options"]:
                                    short_opt = latest_data["options"][s_sym]
                                    long_opt = latest_data["options"][l_sym]
                                    
                                    # --- Common: Expected Move & DTE ---
                                    expected_move = None
                                    expiry_date_str = f"20{ds[0:2]}-{ds[2:4]}-{ds[4:6]}"
                                    expiry_options = [opt for opt in latest_data["options"].values() if opt['expiry'] == expiry_date_str]
                                    strikes = sorted(list(set([opt['strike'] for opt in expiry_options])))
                                    if strikes and current_price > 0:
                                        atm_strike = min(strikes, key=lambda x: abs(x - current_price))
                                        atm_call = next((opt for opt in expiry_options if opt['strike'] == atm_strike and opt['opt_type'] == 'C'), None)
                                        atm_put = next((opt for opt in expiry_options if opt['strike'] == atm_strike and opt['opt_type'] == 'P'), None)
                                        if atm_call and atm_put:
                                            expected_move = 0.85 * (atm_call['mid'] + atm_put['mid'])
                                    if expected_move is None and current_price > 0 and historical_stats["vix"] > 0:
                                        try:
                                            expiry_dt = datetime.strptime(ds, "%y%m%d")
                                            today_date = datetime.now().date()
                                            days_to_expiry = max((expiry_dt.date() - today_date).days, 0.5)
                                            expected_move = current_price * (historical_stats["vix"] / 100.0) * ((days_to_expiry / 365.0) ** 0.5)
                                        except Exception as e:
                                            print(f"⚠️ Fallback Expected Move error: {e}")
                                    
                                    dte_val = None
                                    try:
                                        expiry_dt = datetime.strptime(ds, "%y%m%d")
                                        today_date = datetime.now().date()
                                        dte_val = max((expiry_dt.date() - today_date).days, 0.5)
                                    except:
                                        pass
                                    
                                    has_mid = m_strike and str(m_strike).strip()
                                    
                                    if has_mid:
                                        # --- Three-Leg Strategy (Butterfly or Christmas Tree) ---
                                        try:
                                            m_val = float(m_strike)
                                            lower = min(s_val, l_val)
                                            upper = max(s_val, l_val)
                                            if lower < m_val < upper:
                                                sym_type = opt_type
                                                lower_sym = f"US.XSP{ds}{sym_type}{int(lower * 1000)}"
                                                mid_sym = f"US.XSP{ds}{sym_type}{int(m_val * 1000)}"
                                                upper_sym = f"US.XSP{ds}{sym_type}{int(upper * 1000)}"
                                                
                                                if (lower_sym in latest_data["options"] and
                                                    mid_sym in latest_data["options"] and
                                                    upper_sym in latest_data["options"]):
                                                    
                                                    low_opt = latest_data["options"][lower_sym]
                                                    mid_opt = latest_data["options"][mid_sym]
                                                    up_opt = latest_data["options"][upper_sym]
                                                    
                                                    pnl_val = None
                                                    pnl_pct = None
                                                    entry_premium = None
                                                    if entry_val and str(entry_val).strip() != '':
                                                        entry_premium = float(entry_val)
                                                    
                                                    if strategy == 'bfly':
                                                        bfly_bid = low_opt['bid'] + up_opt['bid'] - 2 * mid_opt['ask']
                                                        bfly_ask = low_opt['ask'] + up_opt['ask'] - 2 * mid_opt['bid']
                                                        bfly_mid = low_opt['mid'] + up_opt['mid'] - 2 * mid_opt['mid']
                                                        bfly_delta = low_opt['delta'] + up_opt['delta'] - 2 * mid_opt['delta']
                                                        bfly_gamma = low_opt.get('gamma',0.0) + up_opt.get('gamma',0.0) - 2 * mid_opt.get('gamma',0.0)
                                                        bfly_theta = low_opt.get('theta',0.0) + up_opt.get('theta',0.0) - 2 * mid_opt.get('theta',0.0)
                                                        bfly_vega = low_opt.get('vega',0.0) + up_opt.get('vega',0.0) - 2 * mid_opt.get('vega',0.0)
                                                        
                                                        if entry_premium is not None:
                                                            pnl_val = bfly_bid - entry_premium
                                                            pnl_pct = (pnl_val / entry_premium) * 100 if entry_premium > 0 else 0.0
                                                        
                                                        bfly_info = {
                                                            "group_index": idx + 1,
                                                            "date": ds,
                                                            "expiry": expiry_date_str,
                                                            "opt_type": sym_type,
                                                            "lower": lower,
                                                            "mid_strike": m_val,
                                                            "upper": upper,
                                                            "short_strike": s_val,
                                                            "long_strike": l_val,
                                                            "width": upper - lower,
                                                            "entry": entry_premium,
                                                            "bid": round(bfly_bid, 2),
                                                            "ask": round(bfly_ask, 2),
                                                            "mid": round(bfly_mid, 2),
                                                            "delta": round(bfly_delta, 3),
                                                            "gamma": round(bfly_gamma, 4),
                                                            "theta": round(bfly_theta, 3),
                                                            "vega": round(bfly_vega, 3),
                                                            "pnl": pnl_val,
                                                            "pnl_percent": pnl_pct,
                                                            "expected_move": expected_move,
                                                            "index_price": current_price,
                                                            "dte": dte_val,
                                                            "combo_symbol": f"{lower_sym}|{mid_sym}|{upper_sym}"
                                                        }
                                                        socketio.emit('butterfly_update', bfly_info)
                                                    
                                                    else:
                                                        # Christmas Tree +1S/-3M/+2L (field order: short, mid, long)
                                                        mo = latest_data["options"][mid_sym]
                                                        xmas_bid = short_opt['bid'] + 2*long_opt['bid'] - 3*mo['ask']
                                                        xmas_ask = short_opt['ask'] + 2*long_opt['ask'] - 3*mo['bid']
                                                        xmas_mid = short_opt['mid'] + 2*long_opt['mid'] - 3*mo['mid']
                                                        xmas_delta = short_opt['delta'] + 2*long_opt['delta'] - 3*mo['delta']
                                                        xmas_gamma = short_opt.get('gamma',0.0) + 2*long_opt.get('gamma',0.0) - 3*mo.get('gamma',0.0)
                                                        xmas_theta = short_opt.get('theta',0.0) + 2*long_opt.get('theta',0.0) - 3*mo.get('theta',0.0)
                                                        xmas_vega = short_opt.get('vega',0.0) + 2*long_opt.get('vega',0.0) - 3*mo.get('vega',0.0)
                                                        
                                                        if entry_premium is not None:
                                                            pnl_val = xmas_bid - entry_premium
                                                            pnl_pct = (pnl_val / entry_premium) * 100 if entry_premium > 0 else 0.0
                                                        
                                                        # --- Christmas Tree Risk / Score / Theory / Scenarios ---
                                                        xmas_risk = None
                                                        xmas_score = None
                                                        xmas_theory = None
                                                        xmas_edge = None
                                                        xmas_scenarios = None
                                                        leg_details = []
                                                        entry_used = entry_premium if entry_premium is not None else xmas_mid
                                                        try:
                                                            atr_val = historical_stats.get("atr_14", 5.0)
                                                            xmas_risk = pricing.xmas_payoff_extrema(
                                                                entry_used, (s_val, m_val, l_val), sym_type,
                                                                upper - lower, atr_val
                                                            )
                                                        except Exception as e:
                                                            print(f"⚠️ xmas risk error: {e}")
                                                        try:
                                                            abs_delta = abs(xmas_delta)
                                                            if current_price > 0 and abs_delta > 0 and (upper - lower) > 0 and abs(entry_used) > 0:
                                                                credit = max(entry_used, 0.0)
                                                                pe = credit / (abs_delta * (upper - lower))
                                                                score_pe = min(max(pe * 150.0, 0.0), 100.0)
                                                                atr_v = historical_stats.get("atr_14", 5.0)
                                                                atr_m = atr_v * (dte_val ** 0.5) if dte_val else atr_v
                                                                dist_short = abs(current_price - s_val)
                                                                dist_long = abs(current_price - l_val)
                                                                dist_mid = abs(current_price - m_val)
                                                                min_dist = min(dist_short, dist_long, dist_mid)
                                                                atr_buf = min_dist / atr_m if atr_m > 0 else 1.0
                                                                score_atr = min(atr_buf * 50.0, 100.0)
                                                                score_vix = historical_stats.get("vix_rank", 0.0)
                                                                xmas_score = round((score_vix * 0.20) + (score_pe * 0.50) + (score_atr * 0.30), 1)
                                                        except Exception as e:
                                                            print(f"⚠️ xmas score error: {e}")
                                                        try:
                                                            tte = dte_val / 365.0 if dte_val and dte_val > 0 else 0.001
                                                            iv_s = short_opt.get('iv', 0.0)
                                                            iv_m = mo.get('iv', 0.0)
                                                            iv_l = long_opt.get('iv', 0.0)
                                                            atm_iv = iv_m if iv_m > 0 else iv_s if iv_s > 0 else 0.18
                                                            iv_s_f = iv_s if iv_s > 0 else atm_iv
                                                            iv_m_f = iv_m if iv_m > 0 else atm_iv
                                                            iv_l_f = iv_l if iv_l > 0 else atm_iv
                                                            xmas_theory = pricing.xmas_theory_price(
                                                                current_price, s_val, m_val, l_val,
                                                                tte, pricing.RISK_FREE_RATE,
                                                                iv_s_f, iv_m_f, iv_l_f, sym_type
                                                            )
                                                            xmas_theory = round(xmas_theory, 2)
                                                            xmas_edge = round(xmas_mid - xmas_theory, 2)
                                                            xmas_scenarios = pricing.xmas_scenarios(
                                                                current_price, s_val, m_val, l_val,
                                                                tte, pricing.RISK_FREE_RATE,
                                                                iv_s_f, iv_m_f, iv_l_f, sym_type, xmas_mid
                                                            )
                                                        except Exception as e:
                                                            print(f"⚠️ xmas theory/scenarios error: {e}")
                                                        try:
                                                            leg_details = [
                                                                {'leg': 'S', 'strike': s_val, 'type': sym_type,
                                                                 'bid': short_opt['bid'], 'ask': short_opt['ask'],
                                                                 'mid': short_opt['mid'], 'delta': short_opt.get('delta',0),
                                                                 'iv': short_opt.get('iv',0)},
                                                                {'leg': 'M×3', 'strike': m_val, 'type': sym_type,
                                                                 'bid': mo['bid'], 'ask': mo['ask'],
                                                                 'mid': mo['mid'], 'delta': mo.get('delta',0),
                                                                 'iv': mo.get('iv',0)},
                                                                {'leg': 'L×2', 'strike': l_val, 'type': sym_type,
                                                                 'bid': long_opt['bid'], 'ask': long_opt['ask'],
                                                                 'mid': long_opt['mid'], 'delta': long_opt.get('delta',0),
                                                                 'iv': long_opt.get('iv',0)},
                                                            ]
                                                        except Exception as e:
                                                            print(f"⚠️ leg details error: {e}")
                                                        
                                                        xmas_info = {
                                                            "group_index": idx + 1,
                                                            "date": ds,
                                                            "expiry": expiry_date_str,
                                                            "opt_type": sym_type,
                                                            "lower": lower,
                                                            "mid_strike": m_val,
                                                            "upper": upper,
                                                            "short_strike": s_val,
                                                            "long_strike": l_val,
                                                            "width": upper - lower,
                                                            "entry": entry_premium,
                                                            "bid": round(xmas_bid, 2),
                                                            "ask": round(xmas_ask, 2),
                                                            "mid": round(xmas_mid, 2),
                                                            "delta": round(xmas_delta, 3),
                                                            "gamma": round(xmas_gamma, 4),
                                                            "theta": round(xmas_theta, 3),
                                                            "vega": round(xmas_vega, 3),
                                                            "pnl": pnl_val,
                                                            "pnl_percent": pnl_pct,
                                                            "expected_move": expected_move,
                                                            "index_price": current_price,
                                                            "dte": dte_val,
                                                            "max_profit": xmas_risk['max_profit'] if xmas_risk else None,
                                                            "max_loss": xmas_risk['max_loss'] if xmas_risk else None,
                                                            "be_lower": xmas_risk['be_lower'] if xmas_risk else None,
                                                            "be_upper": xmas_risk['be_upper'] if xmas_risk else None,
                                                            "risk_reward": xmas_risk['risk_reward'] if xmas_risk else None,
                                                            "entry_score": xmas_score,
                                                            "theory": xmas_theory,
                                                            "edge": xmas_edge,
                                                            "scenarios": xmas_scenarios,
                                                            "legs": leg_details,
                                                            "combo_symbol": f"{lower_sym}|{mid_sym}|{upper_sym}"
                                                        }
                                                        socketio.emit('xmas_update', xmas_info)
                                                        
                                        except Exception as bf_ex:
                                            print(f"⚠️ 计算 Group {idx+1} 三腿策略异常: {bf_ex}")
                                    
                                    else:
                                        # --- Vertical Spread ---
                                        spread_bid = short_opt['bid'] - long_opt['ask']
                                        spread_ask = short_opt['ask'] - long_opt['bid']
                                        spread_mid = short_opt['mid'] - long_opt['mid']
                                        
                                        spread_delta = short_opt['delta'] - long_opt['delta']
                                        spread_gamma = short_opt.get('gamma', 0.0) - long_opt.get('gamma', 0.0)
                                        spread_theta = short_opt.get('theta', 0.0) - long_opt.get('theta', 0.0)
                                        spread_vega = short_opt.get('vega', 0.0) - long_opt.get('vega', 0.0)
                                        
                                        pnl = None
                                        pnl_percent = None
                                        entry_credit = None
                                        if entry_val and str(entry_val).strip() != '':
                                            entry_credit = float(entry_val)
                                            pnl = entry_credit - spread_ask
                                            pnl_percent = (pnl / entry_credit) * 100 if entry_credit > 0 else 0.0
                                        
                                        dist_to_short = 0.0
                                        dist_to_be = 0.0
                                        if current_price > 0:
                                            if opt_type == 'P':
                                                dist_to_short = (current_price - s_val) / current_price * 100
                                                if entry_credit is not None:
                                                    be_price = s_val - entry_credit
                                                    dist_to_be = (current_price - be_price) / current_price * 100
                                            else:
                                                dist_to_short = (s_val - current_price) / current_price * 100
                                                if entry_credit is not None:
                                                    be_price = s_val + entry_credit
                                                    dist_to_be = (be_price - current_price) / current_price * 100
                                        
                                        entry_score = None
                                        score_pe = None
                                        score_atr = None
                                        score_vix = None
                                        pe_ratio = None
                                        atr_buffers = None
                                        try:
                                            if current_price > 0 and abs(short_opt['delta']) > 0 and abs(s_val - l_val) > 0 and spread_bid > 0:
                                                pe_ratio = spread_bid / (abs(short_opt['delta']) * abs(s_val - l_val))
                                                score_pe = min(max(pe_ratio * 150.0, 0.0), 100.0)
                                                
                                                atr_val = historical_stats.get("atr_14", 5.0)
                                                atr_move = atr_val * (dte_val ** 0.5) if dte_val else atr_val
                                                dist_pts = abs(current_price - s_val)
                                                atr_buffers = dist_pts / atr_move if atr_move > 0 else 1.0
                                                score_atr = min(atr_buffers * 50.0, 100.0)
                                                
                                                score_vix = historical_stats.get("vix_rank", 0.0)
                                                entry_score = (score_vix * 0.20) + (score_pe * 0.50) + (score_atr * 0.30)
                                        except Exception as es_err:
                                            print(f"⚠️ 计算 Entry Score 异常: {es_err}")
                                        
                                        spread_info = {
                                            "group_index": idx + 1,
                                            "date": ds,
                                            "expiry": expiry_date_str,
                                            "opt_type": opt_type,
                                            "short": s_val,
                                            "long": l_val,
                                            "width": abs(s_val - l_val),
                                            "entry": entry_credit,
                                            "bid": spread_bid,
                                            "ask": spread_ask,
                                            "mid": spread_mid,
                                            "delta": spread_delta,
                                            "gamma": spread_gamma,
                                            "theta": spread_theta,
                                            "vega": spread_vega,
                                            "short_delta": short_opt['delta'],
                                            "short_gamma": short_opt.get('gamma', 0.0),
                                            "short_theta": short_opt.get('theta', 0.0),
                                            "short_vega": short_opt.get('vega', 0.0),
                                            "pnl": pnl,
                                            "pnl_percent": pnl_percent,
                                            "dist_to_short": dist_to_short,
                                            "dist_to_be": dist_to_be,
                                            "expected_move": expected_move,
                                            "index_price": current_price,
                                            "entry_score": entry_score,
                                            "score_pe": score_pe,
                                            "score_atr": score_atr,
                                            "score_vix": score_vix,
                                            "pe_ratio": pe_ratio,
                                            "atr_buffers": atr_buffers,
                                            "dte": dte_val,
                                            "put_wall": put_walls.get(expiry_date_str, {}).get('strike'),
                                            "combo_symbol": f"{s_sym}|{l_sym}"
                                        }
                                        socketio.emit('spread_update', spread_info)
                                        
                            except Exception as ex:
                                print(f"⚠️ 计算 Group {idx+1} 差价异常: {ex}")
                        
                else:
                    print(f"⚠️ API 请求未返回数据: {data}")

                # 4. 从现有 Moomoo 期权链计算 SKEW 并更新显示
                calc_skew_from_options()
                socketio.emit('index_update', latest_data["index"])

                # 5. 控制频率
                log_premium_snapshot()
                time.sleep(REFRESH_INTERVAL)
                
            except Exception as e:
                emit_toast(socketio, f"❌ 循环异常: {e}")
                time.sleep(5)

@app.route('/')
@login_required
def index():
    return render_template('index.html', watchlist_size=WATCHLIST_SIZE)


@app.route('/api/xsp/ta')
@api_login_required
def api_xsp_ta():
    return jsonify(compute_xsp_ta() or {})


@app.route('/history')
@login_required
def history_page():
    return render_template('history.html')


@app.route('/api/history/snapshots')
@api_login_required
def api_history_snapshots():
    """
    Return all recorded premium snapshots for a given watchlist group and role.
    Each data point includes offset_min = minutes from ET midnight of trade_date,
    so the frontend can align multiple calendar dates on the same X axis.
    """
    group_idx = request.args.get('group', 1, type=int)
    role      = request.args.get('role', 'spread')
    combo_sym = request.args.get('combo_symbol', None)
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        params = [group_idx, role]
        sql_extra = ""
        if combo_sym:
            sql_extra = " AND opt_symbol = ?"
            params.append(combo_sym)
        rows = conn.execute(f"""
            SELECT trade_date, ts, session, xsp_price, vix, bid, ask, mid
            FROM premium_log
            WHERE group_idx = ? AND role = ?{sql_extra}
            ORDER BY ts ASC
        """, params).fetchall()
        conn.close()

        days = defaultdict(list)
        for r in rows:
            dt_et = datetime.fromtimestamp(r['ts'], tz=pytz.utc).astimezone(ET_TZ)
            midnight_et = ET_TZ.localize(
                datetime(int(r['trade_date'][:4]),
                         int(r['trade_date'][5:7]),
                         int(r['trade_date'][8:10]), 0, 0, 0)
            )
            offset_min = round((dt_et.timestamp() - midnight_et.timestamp()) / 60)
            days[r['trade_date']].append({
                'ts':         r['ts'],
                'offset_min': offset_min,
                'session':    r['session'],
                'xsp':        r['xsp_price'],
                'vix':        r['vix'],
                'bid':        r['bid'],
                'ask':        r['ask'],
                'mid':        r['mid'],
            })
        return jsonify({'status': 'ok', 'data': dict(days)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/history/daily_returns')
@api_login_required
def api_history_daily_returns():
    """
    For each trade date in the DB, fetch XSP OHLC from yfinance and return
    the daily change %.  Down day = change_pct < 0 (close < prev_close).
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT trade_date FROM premium_log ORDER BY trade_date"
        ).fetchall()]
        conn.close()

        if not dates:
            return jsonify({'status': 'ok', 'data': {}})

        ticker    = yf.Ticker("^XSP")
        hist      = ticker.history(period="3mo")
        if hist.empty:
            return jsonify({'status': 'ok', 'data': {}})

        date_strs = hist.index.strftime('%Y-%m-%d').tolist()
        result    = {}
        for d in dates:
            try:
                if d in date_strs:
                    i      = date_strs.index(d)
                    close  = float(hist['Close'].iloc[i])
                    prev_c = float(hist['Close'].iloc[i - 1]) if i > 0 else close
                    chg    = (close - prev_c) / prev_c * 100
                    result[d] = {
                        'close':      round(close, 2),
                        'prev_close': round(prev_c, 2),
                        'change_pct': round(chg, 3),
                        'is_down':    chg < 0
                    }
            except Exception:
                pass
        return jsonify({'status': 'ok', 'data': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


@app.route('/api/history/groups')
@api_login_required
def api_history_groups():
    """Return watchlist groups from logged data, one entry per unique combo config."""
    try:
        conn = sqlite3.connect(DB_PATH)
        combos = conn.execute("""
            SELECT DISTINCT group_idx, expiry, opt_type, opt_symbol, role
            FROM premium_log
            WHERE role IN ('bfly', 'xmas') AND expiry >= date('now')
            ORDER BY expiry, group_idx
        """).fetchall()
        conn.close()
        result = []
        for (group_idx, expiry, opt_type, opt_symbol, role) in combos:
            syms = opt_symbol.split('|')
            if len(syms) < 3:
                continue
            strikes = []
            for s in syms:
                try:
                    strike = float(s[13:]) / 1000
                    strikes.append(strike)
                except:
                    continue
            if len(strikes) < 3:
                continue
            strikes.sort()
            if opt_type == 'P':
                short_strike, mid_strike, long_strike = strikes[2], strikes[1], strikes[0]
            else:
                short_strike, mid_strike, long_strike = strikes[0], strikes[1], strikes[2]
            result.append({
                'group_idx': group_idx,
                'expiry': expiry,
                'opt_type': opt_type,
                'short_strike': short_strike,
                'long_strike': long_strike,
                'mid_strike': mid_strike,
                'has_bfly': role == 'bfly',
                'has_xmas': role == 'xmas',
                'combo_symbol': opt_symbol,
            })
        # Deduplicate: same (group, expiry, type, strikes) could have both bfly and xmas rows
        seen = set()
        unique = []
        for r in result:
            key = (r['group_idx'], r['expiry'], r['opt_type'],
                   r['short_strike'], r['mid_strike'], r['long_strike'])
            if key in seen:
                # Merge: existing entry gets both flags
                for u in unique:
                    ukey = (u['group_idx'], u['expiry'], u['opt_type'],
                            u['short_strike'], u['mid_strike'], u['long_strike'])
                    if ukey == key:
                        if r['has_bfly']: u['has_bfly'] = True
                        if r['has_xmas']: u['has_xmas'] = True
                        break
            else:
                seen.add(key)
                unique.append(r)
        # Add day_count per combo config
        conn = sqlite3.connect(DB_PATH)
        for r in unique:
            cnt = conn.execute(
                "SELECT COUNT(DISTINCT trade_date) FROM premium_log WHERE group_idx=? AND expiry=? AND opt_type=? AND opt_symbol=?",
                (r['group_idx'], r['expiry'], r['opt_type'], r['combo_symbol'])
            ).fetchone()[0]
            r['day_count'] = cnt
        conn.close()
        return jsonify({'status': 'ok', 'data': unique})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/history/percentile_data')
@api_login_required
def api_percentile_data():
    """Return sorted list of historical mid prices for a given group+role."""
    group_idx = request.args.get('group', 1, type=int)
    role = request.args.get('role', 'xmas')
    combo_sym = request.args.get('combo_symbol', None)
    try:
        conn = sqlite3.connect(DB_PATH)
        params = [group_idx, role]
        sql_extra = ""
        if combo_sym:
            sql_extra = " AND opt_symbol = ?"
            params.append(combo_sym)
        mids = [r[0] for r in conn.execute(
            f"SELECT mid FROM premium_log WHERE group_idx=? AND role=? AND mid IS NOT NULL AND mid >= 0{sql_extra}",
            params
        ).fetchall()]
        conn.close()
        mids.sort()
        return jsonify({'status': 'ok', 'mids': mids, 'count': len(mids)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@socketio.on('update_watchlist')
def handle_watchlist(data):
    global user_watchlist
    user_watchlist = data
    try:
        with open(WATCHLIST_FILE, 'w') as f:
            json.dump(user_watchlist, f)
    except Exception as e:
        emit_toast(socketio, f"⚠️ Watchlist 保存失败: {e}")
    # 核心：广播给所有连接的客户端，触发它们的回写逻辑
    socketio.emit('sync_watchlist', user_watchlist)

@socketio.on('connect')
def handle_connect():
    # 当新设备连入时，立即同步当前的内存数据
    socketio.emit('index_update', latest_data["index"])
    socketio.emit('sync_watchlist', user_watchlist)
    # 按照 Expiry 和 Strike 排序后再推送给前端（可选，前端 JS 也有排序逻辑）
    for sym in latest_data["options"]:
        socketio.emit('option_update', latest_data["options"][sym])

# --- AUTH ROUTES ---
@app.route('/login')
def login():
    error = request.args.get('error')
    return render_template('login.html', error=error)

@app.route('/auth')
def auth_redirect():
    if not CONFIG.get('google_client_id') or not CONFIG.get('google_client_secret'):
        return redirect(url_for('login', error='Google OAuth not configured'))
    redirect_uri = url_for('auth_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/callback')
def auth_callback():
    try:
        token = google.authorize_access_token()
        userinfo = token.get('userinfo')
        if not userinfo:
            userinfo = google.parse_id_token(token)
        email = userinfo.get('email', '')
        name = userinfo.get('name', email)
        allowed = CONFIG.get('allowed_emails', [])
        if not allowed or email not in allowed:
            return redirect(url_for('login', error='Unauthorized email'))
        user = User(email, name)
        _users[email] = user
        login_user(user)
        return redirect(url_for('index'))
    except Exception as e:
        print(f"⚠️ Auth error: {e}")
        return redirect(url_for('login', error='Authentication failed'))

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    t = threading.Thread(target=start_moomoo, daemon=True)
    t.start()
    print(f"🌍 Dashboard: http://127.0.0.1:{FLASK_PORT}")
    socketio.run(app, host='0.0.0.0', port=FLASK_PORT, debug=False, allow_unsafe_werkzeug=True)
