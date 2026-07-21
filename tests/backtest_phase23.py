"""
Phase 2+3: Cross-asset features (DXY, TNX) + Real skew validation.

Part A — Full 3-year backtest: adds DXY and TNX as features, 
        tests if they improve direction accuracy.
Part B — Real skew analysis from premium_log.db (13 days):
        compute actual skew from put/call mid prices at ATM,
        compare with Price/EMA20 proxy, test directional signal.
"""
import os, sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SRC_DIR

pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 140)

# ────────────────────────────────────────────
# Shared helpers (same as backtest_direction.py)
# ────────────────────────────────────────────
def bb(df, col='price', period=20, std=2):
    sma = df[col].rolling(period).mean()
    sd = df[col].rolling(period).std()
    return sma, sma - std * sd, sma + std * sd

def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def adx_func(high, low, close, period=14):
    up = close.diff()
    down = -up
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=close.index)
    tr = pd.concat([
        high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_tr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr_tr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

def rsi_indicator(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    ag = gain.rolling(period).mean(); al = loss.rolling(period).mean()
    rs = ag / al.replace(0, float('nan')); return 100 - (100 / (1 + rs))

def efficiency_ratio(close, period=10):
    change = close.diff(period).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (change / volatility).fillna(0)

def vol_ratio(close, period=10):
    h = close.rolling(period).max(); l = close.rolling(period).min()
    vr = ((close - l) / (h - l).replace(0, np.nan)).fillna(0.5)
    return vr.clip(0, 1) * 2

def score_ts(v, th):
    for t, s in zip(th, [100, 75, 50, 25, 0]):
        if v >= t: return s
    return 0

W = {'adx': 0.3, 'er': 0.2, 'bbw': 0.15, 'dev': 0.15, 'vr': 0.1}
T = {'adx': [30, 25, 20, 15, 0], 'er': [.7, .55, .35, .2, 0],
     'bbw': [45, 30, 18, 10, 0], 'vr': [2.0, 1.3, .8, .5, 0]}

def compute_score(row):
    total = 5
    for k, w in W.items():
        v = row.get(k)
        if v is None or np.isnan(v):
            total += 5; continue
        total += score_ts(abs(v) if k == 'dev' else v, T.get(k, [0])) * w
    return round(total)

def get_direction_v2(row):
    price = row['price']; bbu = row['bbu']; bbl = row['bbl']
    bw = row['bw']; di_diff = row['di_diff']; atr14 = row['atr_14']
    score = row['score']; vix_pct = row.get('vix_percentile', 50)
    if bbu == bbl or bw <= 0 or np.isnan(bw):
        return None, 'insufficient_data'
    dup = (bbu - price) / bw * 100; dlow = (price - bbl) / bw * 100
    near_threshold = atr14 * 0.30 if (atr14 and atr14 > 0 and not np.isnan(atr14)) else bw * 0.10
    near_top = (bbu - price) < near_threshold
    near_bottom = (price - bbl) < near_threshold
    near_bb_overall = near_top or near_bottom
    is_trend = score >= 55
    # Level 1: trend (not near-BB)
    if not near_bb_overall and is_trend:
        if di_diff > 0: return 'CALL', 'trend'
        elif di_diff < 0: return 'PUT', 'trend'
    # Level 2: nearBB + VIX high
    if near_top and score >= 50 and vix_pct > 75: return 'PUT', 'nearbb_vix'
    if near_bottom and score >= 50 and vix_pct > 75: return 'CALL', 'nearbb_vix'
    # Level 3: conflict filter
    if near_top and di_diff > 0: return None, 'filtered'
    if near_bottom and di_diff < 0: return None, 'filtered'
    # Level 4: nearBB (original)
    if near_top and score >= 50: return 'PUT', 'nearbb'
    if near_bottom and score >= 50: return 'CALL', 'nearbb'
    if near_top or near_bottom: return None, 'BB_center'
    return None, 'BB_center'

# ════════════════════════════════════════════
# PART A: Full backtest with DXY/TNX features
# ════════════════════════════════════════════
print("=" * 70)
print("PART A: 3-Year Backtest with DXY/TNX")
print("=" * 70)

# 1. Fetch data
print("Fetching ^XSP...")
xsp = yf.download("^XSP", period="3y", interval="1d", auto_adjust=False)
xsp.columns = [c[0] for c in xsp.columns]
xsp.index = pd.to_datetime(xsp.index)

print("Fetching ^VIX...")
vix = yf.download("^VIX", period="3y", interval="1d", auto_adjust=False)
vix.columns = [c[0] for c in vix.columns]
vix.index = pd.to_datetime(vix.index)

print("Fetching DXY (DX-Y.NYB)...")
dxy = yf.download("DX-Y.NYB", period="3y", interval="1d", auto_adjust=False)
dxy.columns = [c[0] for c in dxy.columns]
dxy.index = pd.to_datetime(dxy.index)

print("Fetching TNX (^TNX)...")
tnx = yf.download("^TNX", period="3y", interval="1d", auto_adjust=False)
tnx.columns = [c[0] for c in tnx.columns]
tnx.index = pd.to_datetime(tnx.index)

close_col = 'Adj Close' if 'Adj Close' in xsp.columns else 'Close'
price_series = xsp[close_col]
vix_close = vix['Close'] if 'Close' in vix.columns else vix['Adj Close']
dxy_close = dxy['Close'] if 'Close' in dxy.columns else dxy['Adj Close']
tnx_close = tnx['Close'] if 'Close' in tnx.columns else tnx['Adj Close']

# 2. Build dataframe with indicators
df = pd.DataFrame(index=price_series.index)
df['price'] = price_series

# VIX percentile
vix_aligned = vix_close.reindex(df.index, method='ffill')
vi = vix_aligned.values.astype(float)
vix_pct_arr = np.full(len(vi), np.nan)
for i in range(len(vi)):
    start = max(0, i - 504)
    window = vi[start:i+1]
    vix_pct_arr[i] = np.sum(window <= vi[i]) / len(window) * 100 if len(window) >= 100 else 50.0
df['vix'] = vi
df['vix_percentile'] = vix_pct_arr

# EMA + BB
df['ema_20'] = df['price'].ewm(span=20, min_periods=20).mean()
sma, df['bbl'], df['bbu'] = bb(df)
df['bw'] = (df['bbu'] - df['bbl']).fillna(0)
df['bbw'] = (df['bw'] / df['price'] * 100).fillna(0)
df['dev'] = ((df['price'] - sma) / sma * 100).fillna(0)

# ATR
xsp_h = xsp['High'] if 'High' in xsp.columns else price_series
xsp_l = xsp['Low'] if 'Low' in xsp.columns else price_series
df['atr_14'] = atr(xsp_h, xsp_l, price_series, 14)

# ADX + DI
adx_series, plus_di, minus_di = adx_func(xsp_h, xsp_l, price_series, 14)
df['adx'] = adx_series
df['di_diff'] = (plus_di - minus_di) / 100

# ER, VR
df['er'] = efficiency_ratio(price_series, 10)
df['vr'] = vol_ratio(price_series, 10)

# RSI
df['rsi_14'] = rsi_indicator(price_series, 14)
df['price_ema20_pct'] = (df['price'] / df['ema_20'] - 1) * 100

# DXY, TNX
df['dxy'] = dxy_close.reindex(df.index, method='ffill')
df['tnx'] = tnx_close.reindex(df.index, method='ffill')
df['dxy_chg'] = df['dxy'].pct_change(1) * 100    # daily % change
df['tnx_chg'] = df['tnx'].diff(1)                 # daily bp change
df['dxy_chg_5'] = df['dxy'].pct_change(5) * 100  # 5d change
df['tnx_chg_5'] = df['tnx'].diff(5)               # 5d change

# 3. Backtest loop
rows = df.to_dict('records')
dates = df.index.tolist()
signals = []

for i in range(len(rows)):
    row = rows[i].copy()
    row['date'] = dates[i]
    if i < 20: continue
    if np.isnan(row.get('ema_20', np.nan)) or np.isnan(row.get('bbl', np.nan)): continue
    if row['price'] <= 0: continue
    row['score'] = compute_score(row)
    if row['score'] is None: continue

    direction, tier = get_direction_v2(row)
    row['direction'] = direction
    row['tier'] = tier

    for offset in [1, 2, 3]:
        j = i + offset
        row[f't+{offset}'] = rows[j]['price'] if j < len(rows) else None
    row['price_t'] = row['price']
    signals.append(row.copy())

df_sig = pd.DataFrame(signals).set_index('date')

# 4. Evaluate
def eval_col(df_sig, offset_key, dir_col='direction'):
    valid = df_sig[df_sig[dir_col].notna()].copy()
    if valid.empty: return {}
    valid = valid[valid[offset_key].notna()]
    if valid.empty: return {}
    actual_change = valid[offset_key].values - valid['price_t'].values
    actual_dir = np.where(actual_change > 0, 'CALL', 'PUT')
    correct = valid[dir_col].values == actual_dir
    n = len(correct); c = int(correct.sum())
    return {'accuracy': round(c/n, 4) if n>0 else 0, 'correct': c, 'total': n}

has_sig = df_sig[df_sig['direction'].notna()].copy()
print(f"\nV2 (baseline) — signals: {len(has_sig)}/{len(df_sig)} ({len(has_sig)/len(df_sig)*100:.1f}%)")
for offset in [1, 2, 3]:
    r = eval_col(df_sig, f't+{offset}')
    print(f"  {f't+{offset}':>8}: {r.get('accuracy',0)*100:>5.1f}% ({r.get('correct',0)}/{r.get('total',0)})")

# 5. Augmented direction: add DXY/TNX signal
print("\n--- Augmented: DXY/TNX enhancement ---")
# Simple rule: if DXY down (USD weak) + TNX up → risk-on → amplify CALL
# DXY up (USD strong) + TNX down → risk-off → amplify PUT
def direction_augmented(row):
    d = get_direction_v2(row)
    if d[0] is None:
        return None, 'filtered'
    if row.get('dxy_chg') is not None and not np.isnan(row['dxy_chg']) and \
       row.get('tnx_chg') is not None and not np.isnan(row['tnx_chg']):
        risk_on = row['dxy_chg'] < -0.2 and row['tnx_chg'] > 0.05
        risk_off = row['dxy_chg'] > 0.2 and row['tnx_chg'] < -0.05
        if risk_on and d[0] == 'CALL':
            return 'CALL', 'trend+risk_on'
        if risk_off and d[0] == 'PUT':
            return 'PUT', 'trend+risk_off'
        if risk_on and d[0] == 'PUT':
            return None, 'conflict_risk_on'  # filter out
        if risk_off and d[0] == 'CALL':
            return None, 'conflict_risk_off'  # filter out
    return d

# Collect augmented signals (only rows with DXY/TNX data)
aug_signals = []
for i in range(len(rows)):
    row = rows[i].copy()
    row['date'] = dates[i]
    if i < 20: continue
    if np.isnan(row.get('ema_20', np.nan)) or np.isnan(row.get('bbl', np.nan)): continue
    if row['price'] <= 0: continue
    if np.isnan(row.get('dxy_chg', np.nan)): continue  # skip rows without DXY
    row['score'] = compute_score(row)
    if row['score'] is None: continue
    direction, tier = direction_augmented(row)
    row['direction'] = direction; row['tier'] = tier
    for offset in [1, 2, 3]:
        j = i + offset
        row[f't+{offset}'] = rows[j]['price'] if j < len(rows) else None
    row['price_t'] = row['price']
    aug_signals.append(row.copy())

df_aug = pd.DataFrame(aug_signals).set_index('date')
has_aug = df_aug[df_aug['direction'].notna()].copy()

print(f"\nAugmented — signals: {len(has_aug)}/{len(df_aug)} ({len(has_aug)/len(df_aug)*100:.1f}%)")
for offset in [1, 2, 3]:
    r = eval_col(df_aug, f't+{offset}')
    print(f"  {f't+{offset}':>8}: {r.get('accuracy',0)*100:>5.1f}% ({r.get('correct',0)}/{r.get('total',0)})")

# Compare: V2 baseline vs Augmented
print("\n--- Accuracy comparison (V2 baseline vs Augmented) ---")
print(f"{'':>20}  {'V2 baseline':>16}  {'+DXY/TNX':>16}")
print("-" * 58)
for offset in [1, 2, 3]:
    r1 = eval_col(df_sig, f't+{offset}')
    r2 = eval_col(df_aug, f't+{offset}')
    print(f"  {f't+{offset}':>15}  {r1.get('accuracy',0)*100:>7.1f}% ({r1.get('total',0):>3})  {r2.get('accuracy',0)*100:>7.1f}% ({r2.get('total',0):>3})")

# DXY/TNX conflict filter count
conflict_count = len(df_aug[df_aug['tier'].isin(['conflict_risk_on', 'conflict_risk_off'])])
print(f"\nDXY/TNX conflict filters applied: {conflict_count}")

# ════════════════════════════════════════════
# PART B: Real skew validation
# ════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART B: Real Skew Validation (premium_log.db)")
print("=" * 70)

DB_PATH = os.path.join(SRC_DIR, '..', 'premium_log.db')
if not os.path.exists(DB_PATH):
    # Also try SRC_DIR
    DB_PATH = os.path.join(SRC_DIR, 'premium_log.db')

if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)

    # For each unique timestamp, find ATM put/call mid price at nearest expiry
    # Compute real_skew = (put_mid - call_mid) / (put_mid + call_mid)
    # Compare with proxy_skew = price / ema20 - 1

    timestamps = conn.execute(
        "SELECT DISTINCT ts, xsp_price FROM premium_log WHERE role='option' ORDER BY ts"
    ).fetchall()

    skew_records = []
    for ts, xsp_price in timestamps:
        # Get all option symbols at this timestamp
        rows_db = conn.execute(
            "SELECT expiry, opt_type, strike, mid FROM premium_log WHERE ts=? AND role='option'",
            (ts,)
        ).fetchall()

        # Group by expiry, find nearest to xsp_price
        expiries = set(r[0] for r in rows_db)
        best_expiry = None
        best_pair = None

        for expiry in expiries:
            # Get puts and calls at this expiry
            expiry_opts = [r for r in rows_db if r[0] == expiry]
            if len(expiry_opts) < 5:
                continue

            # Group by strike
            strikes = set(r[2] for r in expiry_opts)
            for strike in strikes:
                strike_opts = [r for r in expiry_opts if r[2] == strike]
                put = next((r[3] for r in strike_opts if r[1] == 'P'), None)
                call = next((r[3] for r in strike_opts if r[1] == 'C'), None)
                if put and call:
                    best_pair = (strike, expiry, put, call)
                    break
            if best_pair:
                break

        if best_pair:
            strike, expiry, put_mid, call_mid = best_pair
            total = put_mid + call_mid
            real_skew = (put_mid - call_mid) / total if total > 0 else 0
            skew_records.append({
                'ts': ts,
                'date': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M'),
                'xsp_price': xsp_price,
                'strike': strike,
                'expiry': expiry,
                'put_mid': put_mid,
                'call_mid': call_mid,
                'real_skew': real_skew,
            })

    df_skew = pd.DataFrame(skew_records)
    print(f"Real skew records: {len(df_skew)}")

    if len(df_skew) > 0:
        print(f"Real skew range: {df_skew['real_skew'].min():.4f} to {df_skew['real_skew'].max():.4f}")
        print(f"Real skew mean: {df_skew['real_skew'].mean():.4f}, std: {df_skew['real_skew'].std():.4f}")

        # Compute Price/EMA20 proxy skew from premium_log data
        proxy_skews = []
        for _, rec in df_skew.iterrows():
            # Get the close from backtest data for that day
            dt = datetime.fromtimestamp(rec['ts'])
            date_key = dt.strftime('%Y-%m-%d')
            # Align: find matching day in main df
            match = df[df.index.strftime('%Y-%m-%d') == date_key]
            if not match.empty:
                ema20 = match['ema_20'].iloc[0]
                px = match['price'].iloc[0]
                proxy = (px / ema20 - 1) * 100 if ema20 > 0 else 0
                proxy_skews.append(proxy)
            else:
                proxy_skews.append(np.nan)

        df_skew['proxy_skew'] = proxy_skews
        df_skew_valid = df_skew.dropna(subset=['proxy_skew'])

        if len(df_skew_valid) > 1:
            corr = df_skew_valid['real_skew'].corr(df_skew_valid['proxy_skew'])
            print(f"\nCorrelation: real_skew vs proxy_skew: r = {corr:.4f}")

            # Test real_skew directional signal
            # Group by day (take first snapshot of each day)
            df_skew_valid['day'] = df_skew_valid['date'].str[:10]
            daily_skew = df_skew_valid.groupby('day').first().reset_index()

            # For each day, check t+1 direction
            predictions = []
            for _, rec in daily_skew.iterrows():
                dt = datetime.strptime(rec['day'], '%Y-%m-%d')
                lookahead = dt + timedelta(days=1)
                # Skip weekends
                for _ in range(3):
                    if lookahead.weekday() >= 5:
                        lookahead += timedelta(days=1)
                    else:
                        break
                date_key = lookahead.strftime('%Y-%m-%d')
                match = df[df.index.strftime('%Y-%m-%d') == date_key]
                if not match.empty:
                    next_price = match['price'].iloc[0]
                    price_today = rec['xsp_price']
                    actual_dir = 'CALL' if next_price > price_today else 'PUT'
                    skew_dir = 'CALL' if rec['real_skew'] < 0 else 'PUT'  # negative skew = calls expensive = bullish
                    proxy_dir = 'CALL' if rec['proxy_skew'] < 0 else 'PUT'
                    predictions.append({
                        'day': rec['day'],
                        'real_skew': rec['real_skew'],
                        'proxy_skew': rec['proxy_skew'],
                        'actual_dir': actual_dir,
                        'skew_dir': skew_dir,
                        'proxy_dir': proxy_dir,
                        'skew_correct': skew_dir == actual_dir,
                        'proxy_correct': proxy_dir == actual_dir,
                    })

            df_pred = pd.DataFrame(predictions)
            if len(df_pred) > 0:
                skew_acc = df_pred['skew_correct'].mean() * 100
                proxy_acc = df_pred['proxy_correct'].mean() * 100
                print(f"Real skew   t+1 direction accuracy: {skew_acc:.1f}% ({df_pred['skew_correct'].sum()}/{len(df_pred)})")
                print(f"Proxy skew  t+1 direction accuracy: {proxy_acc:.1f}% ({df_pred['proxy_correct'].sum()}/{len(df_pred)})")

                # Detail
                print("\nDaily predictions:")
                print(df_pred.to_string(index=False))

    conn.close()
else:
    print(f"premium_log.db not found at {DB_PATH}")
    print("Skipping Part B.")

# ────────────────────────────────────────────
# Charts
# ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phase 2+3: DXY/TNX + Real Skew Analysis', fontsize=14, fontweight='bold')

# Chart 1: DXY vs XSP
ax1 = axes[0, 0]
dxy_sub = df[['dxy', 'price']].dropna().copy()
if len(dxy_sub) > 200:
    # Normalize
    dxy_norm = (dxy_sub['dxy'] / dxy_sub['dxy'].iloc[0] - 1) * 100
    xsp_norm = (dxy_sub['price'] / dxy_sub['price'].iloc[0] - 1) * 100
    ax1.plot(dxy_sub.index, dxy_norm, label='DXY %', alpha=0.7, color='#e74c3c')
    ax1.plot(dxy_sub.index, xsp_norm, label='XSP %', alpha=0.7, color='#3498db')
    ax1.set_title('DXY vs XSP (normalized)')
    ax1.legend(fontsize=8)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Chart 2: TNX vs XSP
ax2 = axes[0, 1]
tnx_sub = df[['tnx', 'price']].dropna().copy()
if len(tnx_sub) > 200:
    tnx_norm = (tnx_sub['tnx'] / tnx_sub['tnx'].iloc[0] - 1) * 100
    xsp_norm2 = (tnx_sub['price'] / tnx_sub['price'].iloc[0] - 1) * 100
    ax2.plot(tnx_sub.index, tnx_norm, label='TNX %', alpha=0.7, color='#2ecc71')
    ax2.plot(tnx_sub.index, xsp_norm2, label='XSP %', alpha=0.7, color='#3498db')
    ax2.set_title('TNX vs XSP (normalized)')
    ax2.legend(fontsize=8)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

# Chart 3: Real skew vs proxy (if available)
ax3 = axes[1, 0]
if 'df_skew' in dir() and len(df_skew_valid) > 1:
    ax3.scatter(df_skew_valid['proxy_skew'], df_skew_valid['real_skew'],
                alpha=0.6, c='#9b59b6')
    z = np.polyfit(df_skew_valid['proxy_skew'].values, df_skew_valid['real_skew'].values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_skew_valid['proxy_skew'].min(), df_skew_valid['proxy_skew'].max(), 50)
    ax3.plot(x_line, p(x_line), 'r--', alpha=0.7,
             label=f'R² = {corr**2:.4f}' if 'corr' in dir() else '')
    ax3.set_xlabel('Proxy Skew (Price/EMA20-1) %')
    ax3.set_ylabel('Real Skew')
    ax3.set_title('Real Skew vs Proxy Skew')
    ax3.legend()

# Chart 4: Accuracy comparison
ax4 = axes[1, 1]
offsets = [1, 2, 3]
acc_v2 = [eval_col(df_sig, f't+{o}').get('accuracy', 0) * 100 for o in offsets]
acc_aug = [eval_col(df_aug, f't+{o}').get('accuracy', 0) * 100 for o in offsets]
x = np.arange(len(offsets)); w = 0.3
ax4.bar(x - w/2, acc_v2, w, label='V2 baseline', color='#3498db', alpha=0.7)
ax4.bar(x + w/2, acc_aug, w, label='+DXY/TNX', color='#e67e22', alpha=0.8)
ax4.set_xticks(x); ax4.set_xticklabels([f't+{o}' for o in offsets])
ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax4.set_ylabel('Accuracy (%)')
ax4.set_title('Accuracy: V2 baseline vs +DXY/TNX')
ax4.legend(fontsize=8)
for i in range(len(offsets)):
    ax4.text(i - w/2, acc_v2[i] + 1, f'{acc_v2[i]:.1f}%', ha='center', fontsize=7, color='#3498db')
    ax4.text(i + w/2, acc_aug[i] + 1, f'{acc_aug[i]:.1f}%', ha='center', fontsize=7, color='#e67e22')

plt.tight_layout(rect=[0, 0, 1, 0.95])
chart_path = os.path.join(OUT_DIR, 'backtest_phase23.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n📊 图表: {chart_path}")
print("✅ Phase 2+3 分析完成")
