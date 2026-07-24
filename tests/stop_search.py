"""
Stop-loss grid search over ^XSP historical data.

For each signal (v2_skew), simulate t+3 holding period:
- Check if daily High/Low hits stop% during t+1..t+3
- Hit → exit at -stop% loss
- Miss → exit at t+3 actual P&L

Group results by stop%, signal tier, and signal type.
"""
import os, json
import numpy as np
import pandas as pd
import yfinance as yf

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SRC_DIR

pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 150)

# ────────────────────────────────
# 1. Fetch data
# ────────────────────────────────
print("⏳ Fetching ^XSP (3y OHLC)...")
xsp = yf.download("^XSP", period="3y", interval="1d", auto_adjust=False)
xsp.columns = [c[0] for c in xsp.columns]
xsp.index = pd.to_datetime(xsp.index)
print(f"   ^XSP: {xsp.index[0].date()} → {xsp.index[-1].date()}  ({len(xsp)} days)")

print("⏳ Fetching ^VIX data...")
vix = yf.download("^VIX", period="3y", interval="1d", auto_adjust=False)
vix.columns = [c[0] for c in vix.columns]
vix.index = pd.to_datetime(vix.index)

print("⏳ Fetching ^SKEW data...")
skew_data = yf.download("^SKEW", period="3y", interval="1d", auto_adjust=False)
skew_data.columns = [c[0] for c in skew_data.columns]
skew_data.index = pd.to_datetime(skew_data.index)

close_col = 'Adj Close' if 'Adj Close' in xsp.columns else 'Close'
price_series = xsp[close_col]
vix_close = vix['Close'] if 'Close' in vix.columns else vix['Adj Close']
skew_close = skew_data['Close'] if 'Close' in skew_data.columns else skew_data['Adj Close']

xsp_h = xsp['High'] if 'High' in xsp.columns else price_series
xsp_l = xsp['Low'] if 'Low' in xsp.columns else price_series
xsp_c = price_series

# ────────────────────────────────
# 2. Indicators (identical to app.py / backtest_direction.py)
# ────────────────────────────────
df = pd.DataFrame(index=price_series.index)
df['price'] = price_series
df['high'] = xsp_h
df['low'] = xsp_l

# VIX rank & percentile
vix_aligned = vix_close.reindex(df.index, method='ffill')
vi = vix_aligned.values.astype(float)
n_vix = len(vi)
vix_pct_arr = np.full(n_vix, np.nan)
for i in range(n_vix):
    start = max(0, i - 504)
    window = vi[start:i+1]
    if len(window) >= 100:
        vix_pct_arr[i] = np.sum(window <= vi[i]) / len(window) * 100
    else:
        vix_pct_arr[i] = 50.0
df['vix_percentile'] = vix_pct_arr

# SKEW
skew_aligned = skew_close.reindex(df.index, method='ffill')
df['skew_index'] = skew_aligned.values.astype(float)

# EMA20
df['ema_20'] = df['price'].ewm(span=20, min_periods=20).mean()

# Bollinger Bands
def bb(df, col='price', period=20, std=2):
    sma = df[col].rolling(period).mean()
    sd = df[col].rolling(period).std()
    return sma, sma - std * sd, sma + std * sd

sma, df['bbl'], df['bbu'] = bb(df)
df['bw'] = (df['bbu'] - df['bbl']).fillna(0)
df['bbw'] = (df['bw'] / df['price'] * 100).fillna(0)
df['dev'] = ((df['price'] - sma) / sma * 100).fillna(0)

# ATR14
def atr(high, low, close, period=14):
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

df['atr_14'] = atr(xsp_h, xsp_l, xsp_c, 14)

# ADX
def adx_func(high, low, close, period=14):
    up = close.diff()
    plus_dm = pd.Series(np.where((up > -up) & (up > 0), up, 0), index=close.index)
    minus_dm = pd.Series(np.where((-up > up) & (-up > 0), -up, 0), index=close.index)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_tr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr_tr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean(), plus_di, minus_di

adx_series, plus_di_series, minus_di_series = adx_func(xsp_h, xsp_l, xsp_c, 14)
df['adx'] = adx_series
df['di_diff'] = (plus_di_series - minus_di_series) / 100

# Efficiency Ratio
def efficiency_ratio(close, period=10):
    change = close.diff(period).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (change / volatility).fillna(0)

df['er'] = efficiency_ratio(price_series, 10)

# Volatility Ratio
def vol_ratio(close, period=10):
    hi = close.rolling(period).max()
    lo = close.rolling(period).min()
    vr = ((close - lo) / (hi - lo).replace(0, np.nan)).fillna(0.5)
    return vr.clip(0, 1) * 2

df['vr'] = vol_ratio(price_series, 10)

df['price_ema20_pct'] = (df['price'] / df['ema_20'] - 1) * 100
df['support'] = df['bbl']
df['resistance'] = df['bbu']

# ────────────────────────────────
# 3. Score & direction (identical to app.py)
# ────────────────────────────────
def score_ts(v, th):
    for t, s in zip(th, [100, 75, 50, 25, 0]):
        if v >= t:
            return s
    return 0

W = {'adx': 0.3, 'er': 0.2, 'bbw': 0.15, 'dev': 0.15, 'vr': 0.1}
T = {'adx': [30, 25, 20, 15, 0], 'er': [0.7, 0.55, 0.35, 0.2, 0],
     'bbw': [45, 30, 18, 10, 0], 'dev': [3.0, 1.5, 0.8, 0.3, 0],
     'vr': [2.0, 1.3, 0.8, 0.5, 0]}

def compute_score(row):
    total = 5
    for k, w in W.items():
        v = row.get(k)
        if v is None or np.isnan(v):
            total += 5
            continue
        val = abs(v) if k == 'dev' else v
        total += score_ts(val, T.get(k, [0])) * w
    return round(total)

def get_direction_v2(row):
    price = row['price']; bbu = row['bbu']; bbl = row['bbl']
    bw = row['bw']; di_diff = row['di_diff']; atr14 = row['atr_14']
    score = row['score']; vix_pct = row.get('vix_percentile', 50)
    if bbu == bbl or bw <= 0 or np.isnan(bw):
        return None, '', None
    if atr14 and atr14 > 0 and not np.isnan(atr14):
        near_threshold = atr14 * 0.60
    else:
        near_threshold = bw * 0.10
    near_top = (bbu - price) < near_threshold
    near_bottom = (price - bbl) < near_threshold
    is_trend = score >= 50
    near_bb_overall = near_top or near_bottom
    # L1: trend
    if not near_bb_overall and is_trend:
        if di_diff > 0: return 'CALL', 'trend', 'trend'
        elif di_diff < 0: return 'PUT', 'trend', 'trend'
        else: return None, 'trend_neutral', None
    # L2: nearBB + VIX
    if near_top and score >= 50 and vix_pct > 75:
        return 'PUT', 'nearbb_vix', 'nearbb_vix'
    if near_bottom and score >= 50 and vix_pct > 75:
        return 'CALL', 'nearbb_vix', 'nearbb_vix'
    # L3: conflict
    if near_top and di_diff > 0: return None, 'filtered', 'filtered'
    if near_bottom and di_diff < 0: return None, 'filtered', 'filtered'
    # L4: nearBB
    if near_top and score >= 50: return 'PUT', 'nearbb', 'nearbb'
    if near_bottom and score >= 50: return 'CALL', 'nearbb', 'nearbb'
    if near_top or near_bottom: return None, 'BB_center', None
    return None, 'BB_center', None

def get_direction_v2_skew(row):
    d, r, t = get_direction_v2(row)
    if d is None:
        return None, r, None
    skew = row.get('skew_index', 146)
    if d == 'PUT' and skew < 140:
        return None, 'skew_filtered', 'skew_filtered'
    if d == 'CALL' and skew > 155:
        return None, 'skew_filtered', 'skew_filtered'
    return d, r + f'(S={skew:.0f})', t

def get_signal_tier(score, di_diff, skew_val):
    """Compute {weak,normal,strong} tier (matches app.py)."""
    di_strength = abs(di_diff)
    skew_confirm = (skew_val < 145) if di_diff > 0 else (skew_val > 145)
    if di_strength > 0 and score >= 72 and skew_confirm:
        return 'strong'
    elif score >= 65:
        return 'normal'
    else:
        return 'weak'

# ────────────────────────────────
# 4. Generate signals
# ────────────────────────────────
print("⏳ Computing signals...")
rows = df.to_dict('records')
dates = df.index.tolist()
signals = []

for i in range(len(rows)):
    row = rows[i].copy()
    row['date'] = dates[i]
    if i < 20:
        continue
    if np.isnan(row.get('ema_20', np.nan)) or np.isnan(row.get('bbl', np.nan)):
        continue
    if row['price'] <= 0:
        continue
    row['score'] = compute_score(row)
    if row['score'] is None:
        continue
    direction, reason, tier_v2 = get_direction_v2_skew(row)
    row['direction'] = direction
    row['reason'] = reason
    row['tier_v2'] = tier_v2
    if direction is None:
        continue

    # Store future prices + highs + lows for stop simulation
    futures = {}
    for offset in range(1, 6):
        j = i + offset
        if j < len(rows):
            futures[f't+{offset}'] = {
                'close': rows[j]['price'],
                'high': rows[j]['high'],
                'low': rows[j]['low'],
            }
        else:
            futures[f't+{offset}'] = None
    row['futures'] = futures
    row['signal_tier'] = get_signal_tier(row['score'], row['di_diff'], row['skew_index'])
    signals.append(row)

print(f"   Total signals: {len(signals)}")

# ────────────────────────────────
# 5. Stop-loss simulation
# ────────────────────────────────
STOP_PCTS = [0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, None]  # None = no stop (hold to t+3)
ETF_AMOUNTS = {'weak': 2000, 'normal': 4000, 'strong': 4000}
NAKED_BUY_COUNTS = {'weak': 0, 'normal': 1, 'strong': 2}
LEVERAGE = 3

def simulate_trade(signal, stop_pct):
    """Simulate a single trade. Returns dict with outcome."""
    direction = signal['direction']
    entry = signal['price']
    futures = signal['futures']
    tier = signal['signal_tier']
    reason = signal['tier_v2']  # trend, nearbb, nearbb_vix

    if stop_pct is None:
        # No stop — hold to t+3
        f3 = futures.get('t+3')
        if f3 is None:
            return None
        exit_price = f3['close']
        hit_stop = False
        exit_reason = 't+3'
        worst_move = 0
    else:
        stop_frac = stop_pct / 100.0
        hit_stop = False
        worst_move = 0
        exit_price = None

        # Check t+1..t+3 daily extremes
        for off in [1, 2, 3]:
            f = futures.get(f't+{off}')
            if f is None:
                break
            if direction == 'CALL':
                adverse = (entry - f['low']) / entry
            else:
                adverse = (f['high'] - entry) / entry

            if adverse > worst_move:
                worst_move = adverse

            if adverse >= stop_frac:
                hit_stop = True
                # Exit at stop% loss on the first adverse day
                exit_price = entry * (1 - stop_frac) if direction == 'CALL' else entry * (1 + stop_frac)
                break

        if not hit_stop:
            f3 = futures.get('t+3')
            if f3 is None:
                return None
            exit_price = f3['close']

    if exit_price is None:
        return None

    # P&L
    if direction == 'CALL':
        pnl_pct = (exit_price - entry) / entry
    else:
        pnl_pct = (entry - exit_price) / entry

    # ETF $ loss
    etf_amount = ETF_AMOUNTS.get(tier, 2000)
    etf_loss = etf_amount * LEVERAGE * abs(pnl_pct) if pnl_pct < 0 else etf_amount * LEVERAGE * pnl_pct * 0.3
    # Naked buy $ loss
    naked_count = NAKED_BUY_COUNTS.get(tier, 0)
    naked_loss = naked_count * 100 if pnl_pct < 0 else 0

    return {
        'date': str(signal['date'].date()),
        'direction': direction,
        'tier': tier,
        'reason': reason,
        'score': signal['score'],
        'price': entry,
        'exit_price': exit_price,
        'pnl_pct': round(pnl_pct * 100, 2),
        'hit_stop': hit_stop,
        'exit_reason': 'stop' if hit_stop else 't+3',
        'worst_move': round(worst_move * 100, 2),
        'etf_loss': round(etf_loss, 2),
        'naked_loss': round(naked_loss, 2),
        'total_loss': round(etf_loss + naked_loss, 2),
    }

print("⏳ Simulating trades for each stop%...")
all_results = {}

for sp in STOP_PCTS:
    label = f"stop_{sp}%" if sp is not None else "stop_none"
    trades = []
    for sig in signals:
        r = simulate_trade(sig, sp)
        if r is not None:
            trades.append(r)
    all_results[label] = trades

# ────────────────────────────────
# 6. Aggregate & print
# ────────────────────────────────
def agg(trades):
    if not trades:
        return {}
    n = len(trades)
    n_stopped = sum(1 for t in trades if t['hit_stop'])
    n_win = sum(1 for t in trades if t['pnl_pct'] > 0)
    pnl_vals = [t['pnl_pct'] for t in trades]
    win_vals = [t['pnl_pct'] for t in trades if t['pnl_pct'] > 0]
    loss_vals = [t['pnl_pct'] for t in trades if t['pnl_pct'] <= 0]
    total_loss_vals = [t['total_loss'] for t in trades if t['pnl_pct'] < 0]

    win_rate = n_win / n if n else 0
    avg_ret = np.mean(pnl_vals) if pnl_vals else 0
    avg_win = np.mean(win_vals) if win_vals else 0
    avg_loss = np.mean(loss_vals) if loss_vals else 0
    pf = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    max_loss_pct = min(pnl_vals) if pnl_vals else 0
    max_loss_dollar = min(total_loss_vals) if total_loss_vals else 0

    return {
        'total': n,
        'stopped': n_stopped,
        'stopped_pct': round(n_stopped / n * 100, 1) if n else 0,
        'wins': n_win,
        'win_rate': round(win_rate * 100, 1),
        'avg_ret': round(avg_ret, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(pf, 2),
        'max_loss_pct': round(max_loss_pct, 2),
        'max_loss_dollar': round(max_loss_dollar, 2),
    }

def pretty(d):
    if not d:
        return "  (none)"
    return (f"  signals={d['total']}  stopped={d['stopped']}({d['stopped_pct']}%)  "
            f"win_rate={d['win_rate']}%  avg_ret={d['avg_ret']:+.2f}%  "
            f"avg_win={d['avg_win']:+.2f}%  avg_loss={d['avg_loss']:+.2f}%  "
            f"PF={d['profit_factor']}  max_loss={d['max_loss_pct']:+.2f}%(${d['max_loss_dollar']})")

print("\n" + "=" * 100)
print("STOP-LOSS GRID SEARCH RESULTS")
print("=" * 100)

all_agg = {}
for sp in STOP_PCTS:
    label = f"stop_{sp}%" if sp is not None else "stop_none"
    trades = all_results[label]
    if not trades:
        continue

    # Overall
    print(f"\n{'─' * 100}")
    print(f"🛑 Stop: {sp}%" if sp is not None else "🛑 Stop: None (hold to t+3)")
    print(f"{'─' * 100}")
    overall = agg(trades)
    all_agg[label] = {'all': overall}
    print(pretty(overall))

    # By tier
    for tier in ['weak', 'normal', 'strong']:
        subset = [t for t in trades if t['tier'] == tier]
        if subset:
            r = agg(subset)
            all_agg[label][tier] = r
            print(f"\n  [{tier}]")
            print("  " + pretty(r))

    # By signal type
    for reason in ['trend', 'nearbb', 'nearbb_vix']:
        subset = [t for t in trades if t['reason'] == reason]
        if subset:
            r = agg(subset)
            all_agg[label][reason] = r
            print(f"\n  [{reason}]")
            print("  " + pretty(r))

# ────────────────────────────────
# 7. Summary table
# ────────────────────────────────
print("\n\n" + "=" * 100)
print("SUMMARY — Win rate & P&L by stop%")
print("=" * 100)
header = f"{'Stop%':>8} {'Signals':>7} {'Stopped%':>9} {'WinRate':>8} {'AvgRet':>8} {'AvgWin':>8} {'AvgLoss':>9} {'PF':>6} {'MaxLoss%':>9} {'MaxLoss$':>9}"
print(header)
print("-" * len(header))
for sp in STOP_PCTS:
    label = f"stop_{sp}%" if sp is not None else "stop_none"
    d = all_agg.get(label, {}).get('all', {})
    if d:
        print(f"{str(sp) + '%' if sp is not None else 'None':>8} {d['total']:>7} {d['stopped_pct']:>8.1f}% {d['win_rate']:>7.1f}% {d['avg_ret']:>+7.2f}% {d['avg_win']:>+7.2f}% {d['avg_loss']:>+8.2f}% {d['profit_factor']:>6.2f} {d['max_loss_pct']:>+8.2f}% ${d['max_loss_dollar']:>7.0f}")

# ────────────────────────────────
# 8. Best stop% picker
# ────────────────────────────────
print("\n\n🏆 BEST STOP% BY CRITERIA")
print("-" * 100)

# Metric: which stop% gives the best risk-adjusted return?
best_overall = None
best_pf = None
best_win = None

for sp in STOP_PCTS:
    label = f"stop_{sp}%" if sp is not None else "stop_none"
    d = all_agg.get(label, {}).get('all', {})
    if not d or d['total'] < 20:
        continue
    # Score = win_rate * profit_factor
    score = d['win_rate'] * d['profit_factor']
    cand = (sp, d['win_rate'], d['avg_ret'], d['profit_factor'], score, d['max_loss_dollar'])
    if best_overall is None or score > best_overall[4]:
        best_overall = cand
    if best_pf is None or d['profit_factor'] > best_pf[3]:
        best_pf = cand
    if best_win is None or d['win_rate'] > best_win[1]:
        best_win = cand

for name, best in [('Best overall (winRate×PF)', best_overall),
                   ('Best profit factor', best_pf),
                   ('Best win rate', best_win)]:
    if best:
        print(f"\n  {name}: stop={best[0]}%  win_rate={best[1]:.1f}%  "
              f"avg_ret={best[2]:+.2f}%  PF={best[3]:.2f}  max_loss=${best[5]:.0f}")
    else:
        print(f"\n  {name}: (none)")

# ────────────────────────────────
# 9. Save JSON
# ────────────────────────────────
out_path = os.path.join(OUT_DIR, 'stop_search_results.json')
with open(out_path, 'w') as f:
    json.dump(all_agg, f, indent=2)
print(f"\n📄 结果已保存: {out_path}")
print("✅ Stop-loss 搜索完成")
