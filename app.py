import time
import threading
import pytz
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from moomoo import *
import yfinance as yf
import argparse
import os
import json
import sqlite3
from collections import defaultdict

# --- COMMAND LINE ARGS & CONFIGURATION ---
parser = argparse.ArgumentParser(description="XSP Options Monitor")
parser.add_argument("--floor", type=float, default=0.9, help="Floor percentage (default: 0.9)")
parser.add_argument("--ceiling", type=float, default=1.05, help="Ceiling percentage (default: 1.05)")
parser.add_argument("--refresh", type=int, default=5, help="Refresh frequency in seconds (default: 5)")
parser.add_argument("--watchlist-size", type=int, default=5, help="Watchlist size (default: 5)")
parser.add_argument("--option-days", type=int, default=7, help="Option days (default: 7)")
args, unknown = parser.parse_known_args()

OPEND_ADDR = '127.0.0.1'
OPEND_PORT = 11111
FLOOR_PERCENT = args.floor
CEILING_PERCENT = args.ceiling
REF_SYMBOL = 'US.SPY'  
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
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局数据缓存
latest_data = {
    "index": {"price": 0, "floor": 0, "ceiling": 0},
    "options": {}
}

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

            if s_sym in latest_data["options"] and l_sym in latest_data["options"]:
                so = latest_data["options"][s_sym]
                lo = latest_data["options"][l_sym]
                rows.append((int(now_ts), trade_date, session, xsp_price, vix,
                             idx + 1, f"{s_sym}|{l_sym}", None, expiry, opt_type, 'spread',
                             round(so['bid'] - lo['ask'], 4),
                             round(so['ask'] - lo['bid'], 4),
                             round(so['mid'] - lo['mid'], 4)))
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

        historical_stats["last_updated"] = now_ts
        print(f"✅ Historical data updated: VIX={historical_stats['vix']:.2f} (Rank={historical_stats['vix_rank']:.1f}%, Percentile={historical_stats['vix_percentile']:.1f}%), ATR_14={historical_stats['atr_14']:.2f}, EMA_20={historical_stats['ema_20']:.2f}, SKEW={historical_stats['skew']:.2f}")
    except Exception as e:
        print(f"⚠️ Failed to update historical data: {e}")


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
        print(f"⚠️ YFinance Error: {e}")
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

    # 过滤逻辑 (Put 看负 Delta, Call 看正 Delta)
    # 如果刚开盘 Delta 还没算出来，可以先注释掉这两行
    #if opt_type == 'P' and delta > -0.15: return None
    #if opt_type == 'C' and delta < 0.15: return None

    # 星标逻辑
    is_watched = False
    for group in user_watchlist:
        if group.get('date') == d_str:
            try:
                # 匹配短腿或长腿行权价
                short_val = group.get('short')
                long_val = group.get('long')
                s_price = str(int(float(short_val))) if short_val else ""
                l_price = str(int(float(long_val))) if long_val else ""
                if (s_price and s_price in strike_raw) or (l_price and l_price in strike_raw):
                    is_watched = True
                    break
            except Exception:
                continue
    result = {
        'symbol': symbol, 'strike': strike, 'expiry': expiry, 'opt_type': opt_type,
        'bid': bid, 'ask': ask, 'mid': mid, 'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'iv': iv, 'is_watched': is_watched
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
    # Put Range: From Floor up to Current Price
    p_start = int((floor_price // 5) * 5) + 5
    p_end = int((current_price // 5) * 5)
    
    # Call Range: From Current Price up to Ceiling
    c_start = int((current_price // 5) * 5) + 5
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
                    latest_data["index"] = {
                        "price": price, 
                        "floor": price * FLOOR_PERCENT,
                        "ceiling": price * CEILING_PERCENT,
                        "vix": historical_stats["vix"],
                        "vix_rank": historical_stats["vix_rank"],
                        "vix_percentile": historical_stats["vix_percentile"],
                        "atr_14": historical_stats["atr_14"],
                        "ema_20": historical_stats["ema_20"],
                        "skew": historical_stats["skew"]
                    }
                socketio.emit('index_update', latest_data["index"])

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

                # 注入 Watchlist 中的两腿，确保一定能请求到快照数据
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
                            socketio.emit('index_update', latest_data["index"])
                        
                        # B. 处理期权数据
                        else:
                            item = format_row(row)
                            if not item:
                                continue
                            delta = float(row.get('option_delta') or 0.0)
                            # Bypass delta filter if option is watched or ATM
                            is_atm = current_price > 0 and abs(item['strike'] - current_price) <= 2.5
                            if (delta >= -0.15 and delta <= 0.15) or item['is_watched'] or is_atm:
                                latest_data["options"][row['code']] = item
                                updated_symbols.append(row['code'])
                            else:
                                # Optional: If it was in our cache but no longer qualifies, remove it
                                if row['code'] in latest_data["options"]:
                                    del latest_data["options"][row['code']]
                                    # Tell frontend to remove the row
                                    socketio.emit('remove_row', {'symbol': row['code'], 'expiry': row.get('strike_time')})
                    
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

                    # 3. 计算和广播 Watchlist 中的差价组合 (Spreads) 数据
                    for idx, group in enumerate(user_watchlist):
                        ds = group.get('date')
                        s_strike = group.get('short')
                        l_strike = group.get('long')
                        entry_val = group.get('entry')
                        
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
                                    
                                    # Spread pricing (Bid / Ask / Mid)
                                    spread_bid = short_opt['bid'] - long_opt['ask']
                                    spread_ask = short_opt['ask'] - long_opt['bid']
                                    spread_mid = short_opt['mid'] - long_opt['mid']
                                    
                                    # Aggregated Greeks
                                    spread_delta = short_opt['delta'] - long_opt['delta']
                                    spread_gamma = short_opt.get('gamma', 0.0) - long_opt.get('gamma', 0.0)
                                    spread_theta = short_opt.get('theta', 0.0) - long_opt.get('theta', 0.0)
                                    spread_vega = short_opt.get('vega', 0.0) - long_opt.get('vega', 0.0)
                                    
                                    # Expected Move (EM) dynamically computed using ATM options
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
                                            
                                    # Fallback Expected Move using VIX
                                    if expected_move is None and current_price > 0 and historical_stats["vix"] > 0:
                                        try:
                                            expiry_dt = datetime.strptime(ds, "%y%m%d")
                                            today_date = datetime.now().date()
                                            days_to_expiry = max((expiry_dt.date() - today_date).days, 0.5)
                                            expected_move = current_price * (historical_stats["vix"] / 100.0) * ((days_to_expiry / 365.0) ** 0.5)
                                        except Exception as e:
                                            print(f"⚠️ Fallback Expected Move computation error: {e}")
                                    
                                    # P&L tracking
                                    pnl = None
                                    pnl_percent = None
                                    entry_credit = None
                                    if entry_val and str(entry_val).strip() != '':
                                        entry_credit = float(entry_val)
                                        # Realized P&L = Entry Credit - current cost to close (spread_ask)
                                        pnl = entry_credit - spread_ask
                                        pnl_percent = (pnl / entry_credit) * 100 if entry_credit > 0 else 0.0
                                    
                                    # Safety metrics (distance in % to short leg / breakeven)
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
                                    
                                    # Calculate Entry Quality Score (EQS)
                                    entry_score = None
                                    score_pe = None
                                    score_atr = None
                                    score_vix = None
                                    dte_val = None
                                    pe_ratio = None
                                    atr_buffers = None
                                    try:
                                        if current_price > 0 and abs(short_opt['delta']) > 0 and abs(s_val - l_val) > 0 and spread_bid > 0:
                                            # --- Component 1: Premium Efficiency (50% weight) ---
                                            # How much credit you collect relative to the risk you take.
                                            # Formula: Credit / (|Delta| * Width). Higher = more premium per unit of risk.
                                            # A ratio of 0.67 (e.g., $0.35 on a 5-wide with delta 0.10) would score 100.
                                            pe_ratio = spread_bid / (abs(short_opt['delta']) * abs(s_val - l_val))
                                            score_pe = min(max(pe_ratio * 150.0, 0.0), 100.0)

                                            # --- Component 2: ATR Safety Buffer (30% weight) ---
                                            # How many ATR multiples away is your short leg?
                                            # This scales by sqrt(DTE) to account for time: a strike 10pts away
                                            # is safer for a 4-day expiry than a 1-day expiry.
                                            expiry_dt = datetime.strptime(ds, "%y%m%d")
                                            today_date = datetime.now().date()
                                            dte_val = max((expiry_dt.date() - today_date).days, 0.5)
                                            atr_val = historical_stats.get("atr_14", 5.0)
                                            atr_move = atr_val * (dte_val ** 0.5)
                                            dist_pts = abs(current_price - s_val)
                                            atr_buffers = dist_pts / atr_move if atr_move > 0 else 1.0
                                            # 2+ ATR buffers = 100, 1 ATR = 50, 0 ATR = 0
                                            score_atr = min(atr_buffers * 50.0, 100.0)

                                            # --- Component 3: VIX Environment (20% weight) ---
                                            # High VIX Rank = elevated implied volatility = fatter premiums = better time to sell.
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
                                        "dte": dte_val
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
                print(f"❌ 循环异常: {e}")
                time.sleep(5)

@app.route('/')
def index():
    return render_template('index.html', watchlist_size=WATCHLIST_SIZE)


@app.route('/history')
def history_page():
    return render_template('history.html')


@app.route('/api/history/snapshots')
def api_history_snapshots():
    """
    Return all recorded premium snapshots for a given watchlist group and role.
    Each data point includes offset_min = minutes from ET midnight of trade_date,
    so the frontend can align multiple calendar dates on the same X axis.
    """
    group_idx = request.args.get('group', 1, type=int)
    role      = request.args.get('role', 'spread')
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT trade_date, ts, session, xsp_price, vix, bid, ask, mid
            FROM premium_log
            WHERE group_idx = ? AND role = ?
            ORDER BY ts ASC
        """, (group_idx, role)).fetchall()
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
def api_history_groups():
    """Return all watchlist groups that have logged data, with metadata."""
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT group_idx, expiry, opt_type,
                   MAX(CASE WHEN role='short' THEN strike END) as short_strike,
                   MIN(CASE WHEN role='long'  THEN strike END) as long_strike,
                   COUNT(DISTINCT trade_date) as day_count
            FROM premium_log
            GROUP BY group_idx, expiry, opt_type
            ORDER BY group_idx
        """).fetchall()
        conn.close()
        result = [{'group_idx': r[0], 'expiry': r[1], 'opt_type': r[2],
                   'short_strike': r[3], 'long_strike': r[4], 'day_count': r[5]}
                  for r in rows]
        return jsonify({'status': 'ok', 'data': result})
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
        print(f"⚠️ Failed to save watchlist: {e}")
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

if __name__ == '__main__':
    init_db()
    t = threading.Thread(target=start_moomoo, daemon=True)
    t.start()
    print(f"🌍 Dashboard: http://127.0.0.1:3000")
    socketio.run(app, host='0.0.0.0', port=3000, debug=False, allow_unsafe_werkzeug=True)
