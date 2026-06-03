import time
import threading
import pytz
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template
from flask_socketio import SocketIO
from moomoo import *
import yfinance as yf
import argparse
import os
import json

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
    "last_updated": 0
}

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
            
        # XSP ATR 14
        xsp_ticker = yf.Ticker("^XSP")
        xsp_hist = xsp_ticker.history(period="1mo")
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
            
        historical_stats["last_updated"] = now_ts
        print(f"✅ Historical data updated: VIX={historical_stats['vix']:.2f} (Rank={historical_stats['vix_rank']:.1f}%, Percentile={historical_stats['vix_percentile']:.1f}%), ATR_14={historical_stats['atr_14']:.2f}")
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
        'bid': bid, 'ask': ask, 'mid': mid, 'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'is_watched': is_watched
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
        
        if len(symbols) >= 399: break
    
    print(f"📊 Generated {len(symbols)} total contracts (Puts & Calls)")
    return symbols[:399]

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
                        "atr_14": historical_stats["atr_14"]
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
                                "atr_14": historical_stats["atr_14"]
                            }
                            socketio.emit('index_update', latest_data["index"])
                        
                        # B. 处理期权数据
                        else:
                            item = format_row(row)
                            if not item:
                                continue
                            delta = float(row.get('option_delta') or 0.0)
                            # Bypass delta filter if option is watched
                            if (delta >= -0.15 and delta <= 0.15) or item['is_watched']:
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

                # 3. 控制频率
                time.sleep(REFRESH_INTERVAL)
                
            except Exception as e:
                print(f"❌ 循环异常: {e}")
                time.sleep(5)

@app.route('/')
def index():
    return render_template('index.html', watchlist_size=WATCHLIST_SIZE)

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
    t = threading.Thread(target=start_moomoo, daemon=True)
    t.start()
    print(f"🌍 Dashboard: http://127.0.0.1:3000")
    socketio.run(app, host='0.0.0.0', port=3000, debug=False, allow_unsafe_werkzeug=True)
