import time
import threading
import pytz
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template
from flask_socketio import SocketIO
from moomoo import *

# --- CONFIGURATION ---
OPEND_ADDR = '127.0.0.1'
OPEND_PORT = 11111
FLOOR_PERCENT = 0.9   
CEILING_PERCENT = 1 
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

def format_row(row):
    symbol = row['code']

    bid = float(row.get('bid_price') or 0.0)
    ask = float(row.get('ask_price') or 0.0)
    last = float(row.get('last_price') or 0.0)
    mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else last

    # Check if this symbol belongs to any of the 5 watched groups
    is_watched = False
    for group in user_watchlist:
        # Construct expected symbols for short and long legs
        # e.g., US.XSP260421P600000
        if group['date']:
            short_sym = f"US.XSP{group['date']}P{int(float(group['short'])*1000)}"
            long_sym = f"US.XSP{group['date']}P{int(float(group['long'])*1000)}"
            if symbol == short_sym or symbol == long_sym:
                is_watched = True
                break

    return {
        'symbol': symbol,
        'strike': float(row.get('option_strike_price', 0)),
        'expiry': row.get('strike_time'),
        'bid': bid,
        'ask': ask,
        'mid': mid,
        'delta': float(row.get('option_delta') or 0.0),
        'is_watched': is_watched  # New Flag
    }

def generate_xsp_symbols(current_price, floor_price, ceiling_price):
    if current_price <= 0:
        return []
    
    symbols = []
    tz = pytz.timezone('Australia/Sydney')
    now = datetime.now(tz)
    
    dates = []
    check_date = now
    while len(dates) < 14:
        if check_date.weekday() < 5: 
            dates.append(check_date.strftime('%y%m%d'))
        check_date += timedelta(days=1)

    start_strike = int((floor_price // 5) * 5) + 5
    end_strike = int((ceiling_price // 5) * 5)
    strikes = range(start_strike, end_strike + 5, 5)

    for date_str in dates:
        for strike in strikes:
            strike_str = str(int(strike * 1000))
            symbols.append(f"US.XSP{date_str}P{strike_str}")
            if len(symbols) >= 399: break
        if len(symbols) >= 399: break
    return symbols

def start_moomoo():
    global latest_data
    print(f"🚀 Unified Snapshot Loop Active (Interval: {REFRESH_INTERVAL}s)...")
    
    with OpenQuoteContext(host=OPEND_ADDR, port=OPEND_PORT) as quote_ctx:
        while True:
            try:
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

                    all_symbols = [REF_SYMBOL] + opt_symbols
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
                            if delta >= -0.15:
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
