import time
import threading
import pytz
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template
from flask_socketio import SocketIO
from moomoo import *
import yfinance as yf

# --- CONFIGURATION ---
OPEND_ADDR = '127.0.0.1'
OPEND_PORT = 11111
FLOOR_PERCENT = 0.90  
CEILING_PERCENT = 1.05
REF_SYMBOL = 'US.SPY'  
REFRESH_INTERVAL = 5

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局数据缓存
latest_data = {
    "index": {"price": 0, "floor": 0, "ceiling": 0},
    "options": {}
}
user_watchlist = [] # List of {'date': '260421', 'short': '600', 'long': '595'}


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

    # 过滤逻辑 (Put 看负 Delta, Call 看正 Delta)
    # 如果刚开盘 Delta 还没算出来，可以先注释掉这两行
    #if opt_type == 'P' and delta > -0.15: return None
    #if opt_type == 'C' and delta < 0.15: return None

    # 星标逻辑
    is_watched = False
    for group in user_watchlist:
        if group.get('date') == d_str:
            # 匹配短腿或长腿行权价
            s_price = str(int(float(group.get('short', 0))))
            l_price = str(int(float(group.get('long', 0))))
            if s_price in strike_raw or l_price in strike_raw:
                is_watched = True
                break
    result = {
        'symbol': symbol, 'strike': strike, 'expiry': expiry, 'opt_type': opt_type,
        'bid': bid, 'ask': ask, 'mid': mid, 'delta': delta, 'is_watched': is_watched
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
    while len(dates) < 14:
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
                price = get_xsp_anchor_price()
                if price > 0:
                    latest_data["index"] = {
                        "price": price, 
                        "floor": price * FLOOR_PERCENT,
                        "ceiling": price * CEILING_PERCENT
                    }
                socketio.emit('index_update', latest_data["index"])

                # 1. 动态确定本次需要拉取的代码列表
                current_price = latest_data["index"]["price"]
                if current_price <= 0:
                    # 第一次运行或没拿到价格：只请求 SPY
                    all_symbols = [REF_SYMBOL]
                else:
                    # 已有价格：请求 SPY + 生成的期权列表
                    floor = latest_data["index"]["floor"]
                    ceiling = latest_data["index"]["ceiling"]
                    opt_symbols = generate_xsp_symbols(current_price, floor, ceiling)
                    
                    # 提取当前所有有效的到期日 (YYYY-MM-DD 格式)
                    # 逻辑：从 symbols 列表推导
                    valid_dates = sorted(list(set([
                        f"20{s[6:8]}-{s[8:10]}-{s[10:12]}" for s in opt_symbols
                    ])))
                    # 通知前端当前的有效日期
                    socketio.emit('active_dates', valid_dates)

                    all_symbols = opt_symbols
                print(f"🔍 Requesting snapshot for {len(all_symbols)} symbols (Index Price: {current_price:.2f})...")
                # 2. 发起合并快照请求
                ret, data = quote_ctx.get_market_snapshot(all_symbols)
                
                if ret == RET_OK and not data.empty:
                    for _, row in data.iterrows():
                        # A. 处理指数/基准
                        if row['code'] == REF_SYMBOL:
                            p = row['last_price'] if row['last_price'] > 0 else row['prev_close_price']
                            latest_data["index"] = {
                                "price": p, 
                                "floor": p * FLOOR_PERCENT,
                                "ceiling": p * CEILING_PERCENT
                            }
                            socketio.emit('index_update', latest_data["index"])
                        
                        # B. 处理期权数据
                        else:
                            delta = float(row.get('option_delta') or 0.0)
                            # NEW FILTER: Only process and emit if Delta is -0.15 or more (e.g., -0.10)
                            if delta >= -0.15 and delta <= 0.15:
                                item = format_row(row)
                                latest_data["options"][row['code']] = item
                                socketio.emit('option_update', item)
                            else:
                                # Optional: If it was in our cache but no longer qualifies, remove it
                                if row['code'] in latest_data["options"]:
                                    del latest_data["options"][row['code']]
                                    # Tell frontend to remove the row
                                    socketio.emit('remove_row', {'symbol': row['code'], 'expiry': row.get('strike_time')})
                else:
                    print(f"⚠️ API 请求未返回数据: {data}")

                # 3. 控制频率
                time.sleep(REFRESH_INTERVAL)
                
            except Exception as e:
                print(f"❌ 循环异常: {e}")
                time.sleep(5)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('update_watchlist')
def handle_watchlist(data):
    global user_watchlist
    user_watchlist = data
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
