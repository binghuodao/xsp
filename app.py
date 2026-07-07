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
import requests

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
parser.add_argument("--floor", type=float, default=0.93, help="Floor percentage (default: 0.93)")
parser.add_argument("--ceiling", type=float, default=1.03, help="Ceiling percentage (default: 1.03)")
parser.add_argument("--refresh", type=int, default=5, help="Refresh frequency in seconds (default: 5)")
parser.add_argument("--option-days", type=int, default=15, help="Option days (default: 15)")
args, unknown = parser.parse_known_args()

FLOOR_PERCENT = args.floor
CEILING_PERCENT = args.ceiling
REF_SYMBOL = 'US.SPY'
MES_SYMBOL = 'US.MESmain'  
REFRESH_INTERVAL = args.refresh
OPTION_DAYS = args.option_days
WATCHLIST_FILE = 'watchlist.json'

# --- PREMIUM LOGGER SETTINGS ---
DB_PATH      = 'premium_log.db'
LOG_INTERVAL = 600          # seconds between DB snapshots (10 min)
ET_TZ        = pytz.timezone('America/New_York')
last_log_ts  = 0.0

# --- TELEGRAM NOTIFICATION ---
TELEGRAM_TOKEN   = CONFIG.get('telegram_token', '')
TELEGRAM_CHAT_ID = CONFIG.get('telegram_chat_id', '')
_telegram_throttle = {"msg": "", "ts": 0}

def send_telegram(msg):
    global _telegram_throttle
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    now = time.time()
    if msg != _telegram_throttle["msg"] or now - _telegram_throttle["ts"] > 120:
        _telegram_throttle["msg"] = msg
        _telegram_throttle["ts"] = now
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": msg[:4000]},
                timeout=10
            )
        except:
            pass

# --- MORNING REPORT ---
_morning_report_date = ""
S_TZ = pytz.timezone('Australia/Sydney')

def _s5(v):
    return round(v / 5) * 5

def _score_ts(v, th):
    for t, s in zip(th, [100, 75, 50, 25, 0]):
        if v >= t:
            return s
    return 0

def _opt_mid(sym):
    o = latest_data["options"].get(sym)
    return o['mid'] if o else None

def _opt_delta(sym):
    o = latest_data["options"].get(sym)
    d = o.get('delta') if o else None
    return d if d is not None else 0

def _get_hist_mid(sym):
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT mid FROM premium_log WHERE opt_symbol=? AND role='option' AND mid>=0 ORDER BY trade_date DESC LIMIT 5",
            (sym,)
        ).fetchall()
        conn.close()
        if rows:
            return round(sum(r[0] for r in rows) / len(rows), 2)
    except:
        pass
    return None

def _find_14dte_expiry():
    now_et = datetime.now(ET_TZ)
    target = now_et + timedelta(days=14)
    exps = set()
    for o in latest_data["options"].values():
        exps.add(o['expiry'])
    best, best_diff = None, 999
    for e in exps:
        try:
            d = abs((datetime.strptime(e, '%Y-%m-%d') - target).days)
            if d < best_diff:
                best_diff, best = d, e
        except:
            continue
    return best

def _find_delta_strike(expiry, target_delta):
    best_s, best_d, best_diff = None, 0, 999
    for o in latest_data["options"].values():
        if o['expiry'] != expiry or o['opt_type'] != 'P':
            continue
        d = o.get('delta', 0)
        if d >= 0:
            continue
        diff = abs(d - target_delta)
        if diff < best_diff:
            best_diff, best_s, best_d = diff, o['strike'], d
    return best_s, best_d

def generate_morning_report():
    global _morning_report_date
    now_syd = datetime.now(S_TZ)
    today = now_syd.strftime('%y%m%d')
    if now_syd.weekday() >= 5 or now_syd.hour != 21 or now_syd.minute < 25 or now_syd.minute > 35:
        return
    if _morning_report_date == today:
        return

    idx = latest_data.get("index", {})
    price = idx.get("price", 0)
    if price <= 0:
        return

    hs = historical_stats
    ema20 = hs.get("ema_20", 0)
    bbl = hs.get("support", 0)
    bbu = hs.get("resistance", 0)
    bw = bbu - bbl if (bbl and bbu and bbu > bbl) else 1

    # Trend composite
    W = {'adx': .3, 'er': .2, 'bbw': .15, 'dev': .15, 'vr': .1}
    T = {'adx': [30, 25, 20, 15, 0], 'er': [.7, .55, .35, .2, 0],
         'bbw': [45, 30, 18, 10, 0], 'vr': [2.0, 1.3, .8, .5, 0]}
    total = 5  # 50*0.1 base
    for k, w in W.items():
        v = hs.get(k)
        if v is None:
            total += 5
            continue
        val = abs(v) if k == 'dev' else v
        total += _score_ts(val, T.get(k, [0])) * w
    score = round(total)
    icon = '🟢' if score >= 65 else '🟡' if score >= 35 else '🔴'
    slbl = 'Trending' if score >= 65 else 'Mixed' if score >= 35 else 'Ranging'
    is_trend = score >= 65

    # Direction
    dup = (bbu - price) / bw * 100
    dlow = (price - bbl) / bw * 100
    skew = hs.get('skew', 0)

    if dup < 5:
        direction, reason = 'PUT', f'贴BB上轨({dup:.0f}%)'
    elif dlow < 5:
        direction, reason = 'CALL', f'贴BB下轨({dlow:.0f}%)'
    else:
        direction, reason = ('PUT', f'Skew {skew:.1f}') if skew < 0 else ('CALL', f'Skew {skew:.1f}')

    # Mid leg
    off = -5 if (is_trend and direction == 'PUT') else 5 if (is_trend and direction == 'CALL') else 0
    m = _s5(ema20 + off)

    # Tree strikes
    s = m + 10 if direction == 'PUT' else m - 10
    l = m - 5 if direction == 'PUT' else m + 5

    expiry14 = _find_14dte_expiry()
    ds14 = expiry14[2:4] + expiry14[5:7] + expiry14[8:10] if expiry14 else None

    def sym_str(strike, ot):
        return f"US.XSP{ds14}{ot}{int(strike * 1000)}" if ds14 else None

    lines = [f"📊 XSP 早报 — {datetime.now(S_TZ).strftime('%a %Y-%m-%d %H:%M SGT')}",
             "━━━━━━━━━━━━━━━━━━━━━",
             f"{icon} 综合 {score} / {slbl}",
             f"ADX {hs.get('adx',0):.1f} | ER {hs.get('er',0):.2f} | BBW {hs.get('bbw',0):.1f}% | Dev {hs.get('dev',0):+.1f}% | VR {hs.get('vr',0):.1f}x",
             f"VIX {hs.get('vix',0):.1f} ({hs.get('vix_rank',0):.0f}%) | Skew {hs.get('skew',0):.1f}",
             f"EMA20 ${ema20:.2f} | 现价 ${price:.2f}",
             f"BBL ${bbl:.2f} | BBU ${bbu:.2f}",
             "", f"→ 方向: {direction} ({reason})", ""]

    if expiry14:
        ot = 'P' if direction == 'PUT' else 'C'
        sym_s = sym_str(s, ot)
        sym_m = sym_str(m, ot)
        sym_l = sym_str(l, ot)

        lines.append(f"═══ 14DTE {direction}树 ═══")
        lines.append(f"M={m} ({'EMA20' + ('%+d' % off) if is_trend else 'EMA20, Ranging'})")

        for label, strike, sym in [('S', s, sym_s), ('M', m, sym_m), ('L', l, sym_l)]:
            mid = _opt_mid(sym)
            hist = _get_hist_mid(sym)
            p = f"${mid:.2f}" if mid is not None else "--"
            if hist is not None:
                p += f" | 历均 ${hist:.2f}"
            lines.append(f"{label} {strike}  mid {p}")

        s_mid = _opt_mid(sym_s)
        m_mid = _opt_mid(sym_m)
        l_mid = _opt_mid(sym_l)
        if all(v is not None for v in (s_mid, m_mid, l_mid)):
            combo_val = s_mid + 2 * l_mid - 3 * m_mid
            tag = 'credit (收)' if combo_val >= 0 else 'debit (付)'
            lines.append(f"组合值: ${abs(combo_val):.2f} {tag} → 一手 max loss ${abs(combo_val)*100:.0f}")
        lines.append("")

        # PUT Spread
        lines.append("═══ 14DTE PUT价差 ═══")
        ps, pd = _find_delta_strike(expiry14, -0.075)
        if ps:
            pl = int(ps) - 5
            sym_ss = sym_str(int(ps), 'P')
            sym_sl = sym_str(pl, 'P')
            ss_mid = _opt_mid(sym_ss)
            sl_mid = _opt_mid(sym_sl)
            ss_hist = _get_hist_mid(sym_ss)
            sl_hist = _get_hist_mid(sym_sl)

            def fmt(m, h):
                p = f"${m:.2f}" if m is not None else "--"
                if h is not None:
                    p += f" | 历均 ${h:.2f}"
                return p

            lines.append(f"Short {int(ps)}P (Δ {pd:.3f})  mid {fmt(ss_mid, ss_hist)}")
            lines.append(f"Long  {pl}P                     mid {fmt(sl_mid, sl_hist)}")

            if ss_mid is not None and sl_mid is not None:
                credit = ss_mid - sl_mid
                max_loss = (5 - credit) * 100
                lines.append(f"Credit: ${credit:.2f} | 宽 5pt → 一手 ≈ ${max_loss:.0f} max loss")
    else:
        lines.append("⚠️ 无可用14DTE期权数据")

    msg = "\n".join(lines)
    send_telegram(msg)
    _morning_report_date = today

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
        send_telegram(msg)

user_watchlist = []
if os.path.exists(WATCHLIST_FILE):
    try:
        with open(WATCHLIST_FILE, 'r') as f:
            user_watchlist = [g for g in json.load(f) if g.get('date')]
    except Exception as e:
        print(f"⚠️ Failed to load watchlist: {e}")

_last_watchlist_clean_date = None

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
    "support": 0.0,
    "resistance": 0.0,
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pl_sym ON premium_log(opt_symbol, ts)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pl_exp ON premium_log(expiry, strike, ts)")
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
    Self-throttled logger: persists option premium data to SQLite at most once
    every LOG_INTERVAL seconds.  Records ALL individual option ticks within
    EMA20 ± 25 points and expiry within 21 calendar days.
    Older-than-90-day rows are pruned on every write.
    """
    global last_log_ts, latest_data

    now_ts = time.time()
    if now_ts - last_log_ts < LOG_INTERVAL:
        return
    last_log_ts = now_ts

    trade_date, session = get_trade_date_and_session(now_ts)
    xsp_price = latest_data["index"].get("price", 0)
    vix       = latest_data["index"].get("vix", 0)

    if xsp_price <= 0:
        return

    cutoff_dt = (datetime.now() + timedelta(days=21)).strftime('%Y-%m-%d')

    rows = []
    for sym, opt in latest_data["options"].items():
        try:
            if opt['expiry'] <= cutoff_dt:
                bid = float(opt.get('bid') or 0)
                ask = float(opt.get('ask') or 0)
                if not (bid > 0 and ask > 0 and ask > bid):
                    continue
                mid = round((bid + ask) / 2, 4)
                rows.append((int(now_ts), trade_date, session, xsp_price, vix,
                             None, sym, opt['strike'], opt['expiry'], opt['opt_type'],
                             'option', bid, ask, mid))
        except:
            continue

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
        cutoff = int(now_ts - 30 * 86400)
        conn.execute("DELETE FROM premium_log WHERE ts < ?", (cutoff,))
        conn.commit()
        conn.close()
        print(f"📝 Logged {len(rows)} option rows | {trade_date} | {session} | XSP={xsp_price:.2f}, expiry<={cutoff_dt}")
    except Exception as e:
        print(f"⚠️ DB write error: {e}")


def clean_expired_watchlist(sio):
    global user_watchlist, _last_watchlist_clean_date
    now_et = datetime.now(ET_TZ)
    today = now_et.strftime('%y%m%d')
    if now_et.hour < 17:
        return
    if _last_watchlist_clean_date == today:
        return
    before = len(user_watchlist)
    user_watchlist = [g for g in user_watchlist if g.get('date', '') > today]
    if len(user_watchlist) < before:
        try:
            with open(WATCHLIST_FILE, 'w') as f:
                json.dump(user_watchlist, f)
        except Exception as e:
            print(f"⚠️ Watchlist save failed: {e}")
        sio.emit('sync_watchlist', user_watchlist)
    _last_watchlist_clean_date = today


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
                historical_stats["support"] = float(lower.iloc[-1])
                historical_stats["resistance"] = float(upper.iloc[-1])

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
    iv = float(row.get('option_implied_volatility') or 0.0) / 100.0
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
                        "vr": historical_stats["vr"],
                        "support": historical_stats["support"],
                        "resistance": historical_stats["resistance"]
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
                                "skew": historical_stats["skew"],
                                "support": historical_stats["support"],
                                "resistance": historical_stats["resistance"]
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
                clean_expired_watchlist(socketio)
                generate_morning_report()
                time.sleep(REFRESH_INTERVAL)
                
            except Exception as e:
                emit_toast(socketio, f"❌ 循环异常: {e}")
                time.sleep(5)

@app.route('/')
@login_required
def index():
    return render_template('index.html', floor_pct=FLOOR_PERCENT, ceiling_pct=CEILING_PERCENT)


@app.route('/api/xsp/ta')
@api_login_required
def api_xsp_ta():
    return jsonify(compute_xsp_ta() or {})


@app.route('/history')
@login_required
def history_page():
    return render_template('history.html')


def compute_combo_price(legs, syms_ordered, strategy):
    """Compute combo bid/ask/mid from individual leg data at a single timestamp."""
    lo = legs[syms_ordered[0]]
    mi = legs[syms_ordered[1]]
    hi = legs[syms_ordered[2]]
    if strategy == 'bfly':
        bid = round(lo['bid'] + hi['bid'] - 2 * mi['ask'], 4)
        ask = round(lo['ask'] + hi['ask'] - 2 * mi['bid'], 4)
        mid = round(lo['mid'] + hi['mid'] - 2 * mi['mid'], 4)
    else:  # xmas: +1S / -3M / +2L
        bid = round(lo['bid'] + 2 * hi['bid'] - 3 * mi['ask'], 4)
        ask = round(lo['ask'] + 2 * hi['ask'] - 3 * mi['bid'], 4)
        mid = round(lo['mid'] + 2 * hi['mid'] - 3 * mi['mid'], 4)
    return bid, ask, mid


@app.route('/api/history/snapshots')
@api_login_required
def api_history_snapshots():
    """
    Return premium snapshots for a combo_symbol + role.
    Query BOTH new (role='option') and old (pre-computed) formats,
    merge by ts (new overlays old for same ts).
    """
    combo_sym = request.args.get('combo_symbol', '')
    role      = request.args.get('role', 'xmas')
    strategy  = request.args.get('strategy', 'xmas')
    syms = sorted(combo_sym.split('|'))
    if len(syms) < 2:
        return jsonify({'status': 'error', 'message': 'need at least 2 syms in combo_symbol'})

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row

        placeholders = ','.join('?' * len(syms))
        # New format: individual legs role='option'
        new_rows = conn.execute(f"""
            SELECT ts, trade_date, session, xsp_price, vix, opt_symbol, bid, ask, mid
            FROM premium_log
            WHERE role='option' AND opt_symbol IN ({placeholders})
            ORDER BY ts ASC
        """, syms).fetchall()

        # Old format: pre-computed combo or individual legs
        old_rows = []
        if role in ('bfly', 'xmas', 'spread'):
            old_roles = {'bfly': 'bfly', 'xmas': 'xmas', 'spread': 'spread'}
            old_rows = conn.execute(f"""
                SELECT ts, trade_date, session, xsp_price, vix, opt_symbol, bid, ask, mid
                FROM premium_log
                WHERE role = ? AND opt_symbol = ?
                ORDER BY ts ASC
            """, (old_roles.get(role, 'xmas'), combo_sym)).fetchall()
        elif role in ('short', 'mid', 'long'):
            opt_type = syms[0][12] if len(syms[0]) > 12 else 'P'
            if opt_type == 'P':
                sym_short, sym_mid, sym_long = syms[2], syms[1], syms[0]
            else:
                sym_short, sym_mid, sym_long = syms[0], syms[1], syms[2]
            target = {'short': sym_short, 'mid': sym_mid, 'long': sym_long}.get(role)
            if target:
                old_rows = conn.execute(f"""
                    SELECT ts, trade_date, session, xsp_price, vix, opt_symbol, bid, ask, mid
                    FROM premium_log
                    WHERE role = ? AND opt_symbol = ?
                    ORDER BY ts ASC
                """, (role, target)).fetchall()

        conn.close()

        # Merge: old format first, then new format overlays
        merged = {}  # ts -> dict
        for r in old_rows:
            if not (r['bid'] > 0 and r['ask'] > 0 and r['ask'] > r['bid']):
                continue
            mid = round((r['bid'] + r['ask']) / 2, 4)
            merged[r['ts']] = {
                'trade_date': r['trade_date'],
                'session':    r['session'],
                'xsp':        r['xsp_price'],
                'vix':        r['vix'],
                'bid':        r['bid'],
                'ask':        r['ask'],
                'mid':        mid,
            }

        # Group new rows by ts
        new_by_ts = {}
        for r in new_rows:
            ts = r['ts']
            if ts not in new_by_ts:
                new_by_ts[ts] = {}
            new_by_ts[ts][r['opt_symbol']] = r

        opt_type = syms[0][12] if len(syms[0]) > 12 else 'P'
        for ts, legs in sorted(new_by_ts.items()):
            if role in ('bfly', 'xmas'):
                if len(syms) < 3 or any(s not in legs for s in syms):
                    continue
                bid, ask, mid = compute_combo_price(legs, syms, strategy)
            elif role == 'spread':
                if len(syms) < 2 or any(s not in legs for s in syms):
                    continue
                if opt_type == 'P':
                    short_sym, long_sym = syms[1], syms[0]
                else:
                    short_sym, long_sym = syms[0], syms[1]
                bid = round(legs[short_sym]['bid'] - legs[long_sym]['ask'], 4)
                ask = round(legs[short_sym]['ask'] - legs[long_sym]['bid'], 4)
                mid = round(legs[short_sym]['mid'] - legs[long_sym]['mid'], 4)
            elif role in ('short', 'mid', 'long'):
                if len(syms) < 3:
                    continue
                if opt_type == 'P':
                    sym_short, sym_mid, sym_long = syms[2], syms[1], syms[0]
                else:
                    sym_short, sym_mid, sym_long = syms[0], syms[1], syms[2]
                target = {'short': sym_short, 'mid': sym_mid, 'long': sym_long}.get(role)
                if not target or target not in legs:
                    continue
                bid, ask, mid = legs[target]['bid'], legs[target]['ask'], legs[target]['mid']
            else:
                continue

            # Validate bid/ask on all legs used in the computation
            if role in ('bfly', 'xmas'):
                check_syms = syms
            elif role == 'spread':
                check_syms = syms[:2]
            else:
                check_syms = [target]
            if not all(
                legs[s]['bid'] > 0 and legs[s]['ask'] > 0 and legs[s]['ask'] > legs[s]['bid']
                for s in check_syms
            ):
                continue

            r0 = legs[syms[0]]
            merged[ts] = {
                'trade_date': r0['trade_date'],
                'session':    r0['session'],
                'xsp':        r0['xsp_price'],
                'vix':        r0['vix'],
                'bid':        bid,
                'ask':        ask,
                'mid':        mid,
            }

        days = defaultdict(list)
        for ts, snap in sorted(merged.items()):
            dt_et = datetime.fromtimestamp(ts, tz=pytz.utc).astimezone(ET_TZ)
            midnight_et = ET_TZ.localize(
                datetime(int(snap['trade_date'][:4]),
                         int(snap['trade_date'][5:7]),
                         int(snap['trade_date'][8:10]), 0, 0, 0)
            )
            offset_min = round((dt_et.timestamp() - midnight_et.timestamp()) / 60)
            days[snap['trade_date']].append({
                'ts':         ts,
                'offset_min': offset_min,
                'session':    snap['session'],
                'xsp':        snap['xsp'],
                'vix':        snap['vix'],
                'bid':        snap['bid'],
                'ask':        snap['ask'],
                'mid':        snap['mid'],
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
    """Return combo groups from current watchlist, with day_count from DB."""
    result = []
    for idx, group in enumerate(user_watchlist):
        ds = group.get('date')
        s  = group.get('short')
        m  = group.get('mid')
        l  = group.get('long')
        strategy = group.get('strategy', 'xmas')
        if not (ds and s and l and m):
            continue
        try:
            s_val = float(s)
            l_val = float(l)
            m_val = float(m)
        except:
            continue
        opt_type = 'P' if s_val > l_val else 'C'
        # Sort strikes: low, mid, high
        ordered = sorted([s_val, m_val, l_val])
        syms = [f"US.XSP{ds}{opt_type}{int(x * 1000)}" for x in ordered]
        combo_sym = '|'.join(syms)
        expiry = f"20{ds[0:2]}-{ds[2:4]}-{ds[4:6]}"

        if opt_type == 'P':
            short_strike, mid_strike, long_strike = ordered[2], ordered[1], ordered[0]
        else:
            short_strike, mid_strike, long_strike = ordered[0], ordered[1], ordered[2]

        result.append({
            'group_idx': idx + 1,
            'expiry': expiry,
            'opt_type': opt_type,
            'short_strike': short_strike,
            'mid_strike': mid_strike,
            'long_strike': long_strike,
            'has_bfly': strategy == 'bfly',
            'has_xmas': strategy == 'xmas',
            'strategy': strategy,
            'combo_symbol': combo_sym,
            'day_count': 0,
        })

    # Fill day_count from DB (new + old format)
    try:
        conn = sqlite3.connect(DB_PATH)
        for r in result:
            combo_sym = r['combo_symbol']
            syms = combo_sym.split('|')
            placeholders = ','.join('?' * len(syms))
            # New format: individual legs role='option'
            cnt_new = conn.execute(
                f"SELECT COUNT(DISTINCT trade_date) FROM premium_log WHERE role='option' AND opt_symbol IN ({placeholders})",
                syms
            ).fetchone()[0]
            # Old format: combined symbol role IN ('bfly','xmas')
            cnt_old = conn.execute(
                "SELECT COUNT(DISTINCT trade_date) FROM premium_log WHERE role IN ('bfly','xmas') AND opt_symbol = ?",
                (combo_sym,)
            ).fetchone()[0]
            r['day_count'] = max(cnt_new, cnt_old)
        conn.close()
    except Exception:
        pass
    return jsonify({'status': 'ok', 'data': result})

@app.route('/api/history/percentile_data')
@api_login_required
def api_percentile_data():
    """Return sorted list of historical mid prices for a given combo_symbol+role+strategy.
    Query BOTH new (role='option') and old (pre-computed) formats, merge by ts."""
    combo_sym = request.args.get('combo_symbol', '')
    role      = request.args.get('role', 'xmas')
    strategy  = request.args.get('strategy', 'xmas')
    syms = sorted(combo_sym.split('|'))
    if len(syms) < 2:
        return jsonify({'status': 'ok', 'mids': [], 'count': 0})

    try:
        conn = sqlite3.connect(DB_PATH)

        # Extract strikes for bound checking
        strike_vals = [int(s[13:]) / 1000 for s in syms]
        width = max(strike_vals) - min(strike_vals)

        placeholders = ','.join('?' * len(syms))
        # New format: individual legs role='option'
        new_rows = conn.execute(f"""
            SELECT ts, opt_symbol, bid, ask, mid
            FROM premium_log
            WHERE role='option' AND mid IS NOT NULL AND mid >= 0
              AND opt_symbol IN ({placeholders})
            ORDER BY ts ASC
        """, syms).fetchall()

        # Filter out rows with invalid quotes at row level
        new_rows = [
            (ts, sym, mid) for ts, sym, bid, ask, mid in new_rows
            if bid > 0 and ask > 0 and ask > bid
        ]

        # Old format: pre-computed rows
        old_rows = []
        if role in ('bfly', 'xmas', 'spread'):
            old_roles = {'bfly': 'bfly', 'xmas': 'xmas', 'spread': 'spread'}
            old_rows = conn.execute("""
                SELECT ts, opt_symbol, mid
                FROM premium_log
                WHERE role = ? AND opt_symbol = ? AND mid IS NOT NULL AND mid >= 0
                ORDER BY ts ASC
            """, (old_roles.get(role, 'xmas'), combo_sym)).fetchall()
        elif role in ('short', 'mid', 'long'):
            opt_type = syms[0][12] if len(syms[0]) > 12 else 'P'
            if opt_type == 'P':
                sym_short, sym_mid, sym_long = syms[2], syms[1], syms[0]
            else:
                sym_short, sym_mid, sym_long = syms[0], syms[1], syms[2]
            target = {'short': sym_short, 'mid': sym_mid, 'long': sym_long}.get(role)
            if target:
                old_rows = conn.execute("""
                    SELECT ts, opt_symbol, mid
                    FROM premium_log
                    WHERE role = ? AND opt_symbol = ? AND mid IS NOT NULL AND mid >= 0
                    ORDER BY ts ASC
                """, (role, target)).fetchall()
        conn.close()

        # Merge: old mids first (ts -> mid), then new computed mids overlay
        merged_mids = {}  # ts -> mid
        for r in old_rows:
            if width > 0 and abs(r[2]) <= width * 2:
                merged_mids[r[0]] = r[2]

        # Group new rows by ts
        new_by_ts = {}
        for ts, sym, mid in new_rows:
            if ts not in new_by_ts:
                new_by_ts[ts] = {}
            new_by_ts[ts][sym] = mid

        opt_type = syms[0][12] if len(syms[0]) > 12 else 'P'
        for ts, legs in new_by_ts.items():
            if role in ('bfly', 'xmas'):
                if len(syms) < 3 or any(s not in legs for s in syms):
                    continue
                lo, mi, hi = legs[syms[0]], legs[syms[1]], legs[syms[2]]
                if strategy == 'bfly':
                    combo_mid = lo + hi - 2 * mi
                else:
                    combo_mid = lo + 2 * hi - 3 * mi
            elif role == 'spread':
                if len(syms) < 2 or any(s not in legs for s in syms):
                    continue
                if opt_type == 'P':
                    combo_mid = legs[syms[1]] - legs[syms[0]]
                else:
                    combo_mid = legs[syms[0]] - legs[syms[1]]
            elif role in ('short', 'long', 'mid'):
                if opt_type == 'P':
                    sym_short, sym_mid, sym_long = syms[2], syms[1], syms[0]
                else:
                    sym_short, sym_mid, sym_long = syms[0], syms[1], syms[2]
                target = {'short': sym_short, 'mid': sym_mid, 'long': sym_long}.get(role)
                if not target or target not in legs:
                    continue
                combo_mid = legs[target]
            else:
                continue
            if width > 0 and abs(combo_mid) > width * 2:
                continue
            merged_mids[ts] = round(combo_mid, 4)

        mids = sorted(merged_mids.values())
        return jsonify({'status': 'ok', 'mids': mids, 'count': len(mids)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@socketio.on('update_watchlist')
def handle_watchlist(data):
    global user_watchlist
    # 只保存非空组（有 date 的）
    user_watchlist = [g for g in data if g.get('date')]
    try:
        with open(WATCHLIST_FILE, 'w') as f:
            json.dump(user_watchlist, f)
    except Exception as e:
        emit_toast(socketio, f"⚠️ Watchlist 保存失败: {e}")
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
