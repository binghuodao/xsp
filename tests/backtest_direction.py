"""Backtest market report direction accuracy using historical ^XSP + ^VIX data."""
import os, json
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime, timedelta

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SRC_DIR

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 120)

# ────────────────────────────────────────────
# 1. Fetch data
# ────────────────────────────────────────────
print("⏳ Fetching ^XSP data (2 years)...")
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

# Use adjusted close if available
close_col = 'Adj Close' if 'Adj Close' in xsp.columns else 'Close'
price_series = xsp[close_col]
vix_close = vix['Close'] if 'Close' in vix.columns else vix['Adj Close']
skew_close = skew_data['Close'] if 'Close' in skew_data.columns else skew_data['Adj Close']

# ────────────────────────────────────────────
# 2. Compute indicators (identical to app.py)
# ────────────────────────────────────────────
df = pd.DataFrame(index=price_series.index)
df['price'] = price_series

# VIX rank & percentile (rolling 2-year lookback)
vix_aligned = vix_close.reindex(df.index, method='ffill')
vi = vix_aligned.values.astype(float)
n_vix = len(vi)
vix2y = 504
vix_rank_arr = np.full(n_vix, np.nan)
vix_pct_arr = np.full(n_vix, np.nan)
for i in range(n_vix):
    start = max(0, i - vix2y)
    window = vi[start:i+1]
    if len(window) >= 100:
        vix_rank_arr[i] = np.nanpercentile(window, 100 * (1 - vi[i] / np.max(window)))
        # Actually compute rank correctly
        lower = np.sum(window <= vi[i]) / len(window) * 100
        vix_rank_arr[i] = lower
        vix_pct_arr[i] = np.sum(window <= vi[i]) / len(window) * 100
    else:
        vix_rank_arr[i] = 50.0
        vix_pct_arr[i] = 50.0
df['vix'] = vi
df['vix_rank'] = vix_rank_arr
df['vix_percentile'] = vix_pct_arr

# SKEW index
skew_aligned = skew_close.reindex(df.index, method='ffill')
df['skew_index'] = skew_aligned.values.astype(float)

# EMA20
df['ema_20'] = df['price'].ewm(span=20, min_periods=20).mean()

# Bollinger Bands (20,2)
def bb(df, col='price', period=20, std=2):
    sma = df[col].rolling(period).mean()
    sd = df[col].rolling(period).std()
    bbl = sma - std * sd
    bbu = sma + std * sd
    return sma, bbl, bbu

sma, df['bbl'], df['bbu'] = bb(df)
df['bw'] = (df['bbu'] - df['bbl']).fillna(0)
df['bbw'] = (df['bw'] / df['price'] * 100).fillna(0)

# Dev: (price - sma) / sma * 100 (or vs ema20?)
# In app.py: just gets hs.get('dev', 0), but from where? 
# We use (price - sma) / sma * 100
df['dev'] = ((df['price'] - sma) / sma * 100).fillna(0)

# ATR14
def atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# Need high/low data - use Close if no OHLC available
xsp_h = xsp['High'] if 'High' in xsp.columns else price_series
xsp_l = xsp['Low'] if 'Low' in xsp.columns else price_series
xsp_c = price_series
df['atr_14'] = atr(xsp_h, xsp_l, xsp_c, 14)

# ADX (14-period, using Close approximation)
def adx_func(high, low, close, period=14):
    # Simplified ADX using close only
    up = close.diff()
    down = -up
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=close.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=close.index)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_tr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr_tr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr_tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

adx_series, plus_di_series, minus_di_series = adx_func(xsp_h, xsp_l, xsp_c, 14)
df['adx'] = adx_series
df['di_diff'] = (plus_di_series - minus_di_series) / 100

# Efficiency Ratio (10-period)
def efficiency_ratio(close, period=10):
    change = close.diff(period).abs()
    volatility = close.diff().abs().rolling(period).sum()
    return (change / volatility).fillna(0)

df['er'] = efficiency_ratio(price_series, 10)

# Volatility Ratio (10-period)
def vol_ratio(close, period=10):
    high = close.rolling(period).max()
    low = close.rolling(period).min()
    vr = ((close - low) / (high - low).replace(0, np.nan)).fillna(0.5)
    return vr.clip(0, 1) * 2  # Scale to ~0-2 range

# RSI(14)
def rsi_indicator(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['vr'] = vol_ratio(price_series, 10)

# RSI(14) & Price/EMA20
df['rsi_14'] = rsi_indicator(price_series, 14)
df['price_ema20_pct'] = (df['price'] / df['ema_20'] - 1) * 100

# Support/Resistance (Bollinger Bands)
df['support'] = df['bbl']
df['resistance'] = df['bbu']



# ────────────────────────────────────────────
# 3. Score function (identical to app.py)
# ────────────────────────────────────────────
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

# ────────────────────────────────────────────
# 4. Direction logic (identical to app.py)
# ────────────────────────────────────────────
def get_direction(row):
    price = row['price']
    bbu = row['bbu']
    bbl = row['bbl']
    bw = row['bw']
    di_diff = row['di_diff']
    atr14 = row['atr_14']
    score = row['score']
    is_trend = score >= 50

    if bbu == bbl or bw <= 0 or np.isnan(bw):
        return None, 'insufficient_data'

    dup = (bbu - price) / bw * 100
    dlow = (price - bbl) / bw * 100
    if atr14 and atr14 > 0 and not np.isnan(atr14):
        near_threshold = atr14 * 0.60
    else:
        near_threshold = bw * 0.10
    near_top = (bbu - price) < near_threshold
    near_bottom = (price - bbl) < near_threshold

    if near_top and score >= 50:
        return 'PUT', f'贴BB上轨({dup:.0f}%)'
    elif near_bottom and score >= 50:
        return 'CALL', f'贴BB下轨({dlow:.0f}%)'
    elif near_top:
        return None, 'BB中段'
    elif near_bottom:
        return None, 'BB中段'
    elif is_trend:
        if di_diff > 0:
            return 'CALL', f'DI+({di_diff:.2f})'
        elif di_diff < 0:
            return 'PUT', f'DI-({di_diff:.2f})'
        else:
            return None, 'trend_neutral'
    else:
        return None, 'BB中段'

# ────────────────────────────────────────────
# 4b. Direction v2 — fusion logic
# ────────────────────────────────────────────
def get_direction_v2(row):
    price = row['price']
    bbu = row['bbu']
    bbl = row['bbl']
    bw = row['bw']
    ema20 = row['ema_20']
    di_diff = row['di_diff']
    atr14 = row['atr_14']
    score = row['score']
    vix_pct = row.get('vix_percentile', 50)

    if bbu == bbl or bw <= 0 or np.isnan(bw):
        return None, 'insufficient_data', None

    dup = (bbu - price) / bw * 100
    dlow = (price - bbl) / bw * 100
    if atr14 and atr14 > 0 and not np.isnan(atr14):
        near_threshold = atr14 * 0.60
    else:
        near_threshold = bw * 0.10
    near_top = (bbu - price) < near_threshold
    near_bottom = (price - bbl) < near_threshold

    is_trend = score >= 50
    near_bb_overall = near_top or near_bottom

    # Level 1: 趋势 (原有的非近轨趋势)
    if not near_bb_overall and is_trend:
        if di_diff > 0:
            return 'CALL', f'DI+({di_diff:.2f})', 'trend'
        elif di_diff < 0:
            return 'PUT', f'DI-({di_diff:.2f})', 'trend'
        else:
            return None, 'trend_neutral', None

    # Level 2: 近轨 + VIX高 → 确认反转
    if near_top and score >= 50 and vix_pct > 75:
        return 'PUT', f'贴BB上+VIX({vix_pct:.0f}%)', 'nearbb_vix'
    if near_bottom and score >= 50 and vix_pct > 75:
        return 'CALL', f'贴BB下+VIX({vix_pct:.0f}%)', 'nearbb_vix'

    # Level 3: 矛盾过滤 — DI diff 与近轨方向相反 → 不开仓
    if near_top and di_diff > 0:
        return None, '矛盾:近上+DI+', 'filtered'
    if near_bottom and di_diff < 0:
        return None, '矛盾:近下+DI-', 'filtered'

    # Level 4: 近轨 (原有逻辑) — 非矛盾 + 非VIX极端
    if near_top and score >= 50:
        return 'PUT', f'贴BB上轨({dup:.0f}%)', 'nearbb'
    if near_bottom and score >= 50:
        return 'CALL', f'贴BB下轨({dlow:.0f}%)', 'nearbb'
    if near_top or near_bottom:
        return None, 'BB中段', None   # score < 60

    return None, 'BB中段', None

# ────────────────────────────────────────────
# 4c. SKEW filter wrappers
# ────────────────────────────────────────────
def get_direction_v2_skew(row):
    """V2 direction + SKEW filter:
       PUT + low SKEW (<140) → skip (no tail risk, don't short)
       CALL + high SKEW (>155) → skip (tail risk, don't go long)"""
    d, r, t = get_direction_v2(row)
    if d is None:
        return None, r, None
    skew = row.get('skew_index', 146)
    if d == 'PUT' and skew < 140:
        return None, f'skew_filtered_PUT({skew:.0f})', 'skew_filtered'
    if d == 'CALL' and skew > 155:
        return None, f'skew_filtered_CALL({skew:.0f})', 'skew_filtered'
    return d, r + f'(SKEW={skew:.0f})', t

# ────────────────────────────────────────────
# 5. Rolling backtest
# ────────────────────────────────────────────
print("⏳ Running backtest...")

rows = df.to_dict('records')
dates = df.index.tolist()
signals = []
last_valid = None

for i in range(len(rows)):
    row = rows[i].copy()
    row['date'] = dates[i]
    if i < 20:  # Need warm-up for EMA/BB
        continue
    if np.isnan(row.get('ema_20', np.nan)) or np.isnan(row.get('bbl', np.nan)):
        continue
    if row['price'] <= 0:
        continue

    row['score'] = compute_score(row)
    if row['score'] is None:
        continue

    # Old direction logic
    direction, reason = get_direction(row)
    row['direction'] = direction
    row['reason'] = reason

    # New direction logic (fusion)
    direction_v2, reason_v2, tier_v2 = get_direction_v2(row)
    row['direction_v2'] = direction_v2
    row['reason_v2'] = reason_v2
    row['tier_v2'] = tier_v2

    # V2 + SKEW filter
    direction_v2s, reason_v2s, tier_v2s = get_direction_v2_skew(row)
    row['direction_v2s'] = direction_v2s
    row['reason_v2s'] = reason_v2s
    row['tier_v2s'] = tier_v2s

    # Check future price changes
    future_prices = {}
    for offset in [1, 2, 3, 4, 5]:
        j = i + offset
        if j < len(rows):
            future_prices[f't+{offset}'] = rows[j]['price']
        else:
            future_prices[f't+{offset}'] = None

    row['price_t'] = row['price']
    for k, v in future_prices.items():
        row[k] = v
    signals.append(row.copy())

df_signals = pd.DataFrame(signals).set_index('date')

# ────────────────────────────────────────────
# 6. Accuracy evaluation
# ────────────────────────────────────────────
def eval_col(df_sig, offset_key, dir_col):
    valid = df_sig[df_sig[dir_col].notna()].copy()
    if valid.empty:
        return {}, 0, 0, 0, 0
    future_col = offset_key
    valid = valid[valid[future_col].notna()]
    if valid.empty:
        return {}, 0, 0, 0, 0
    actual_change = valid[future_col].values - valid['price_t'].values
    actual_dir = np.where(actual_change > 0, 'CALL', 'PUT')
    correct = valid[dir_col].values == actual_dir
    n_correct = int(correct.sum())
    n_total = len(correct)
    acc = n_correct / n_total if n_total > 0 else 0
    tp = int(((valid[dir_col].values == 'CALL') & (actual_dir == 'CALL')).sum())
    fp = int(((valid[dir_col].values == 'CALL') & (actual_dir == 'PUT')).sum())
    tn = int(((valid[dir_col].values == 'PUT') & (actual_dir == 'PUT')).sum())
    fn = int(((valid[dir_col].values == 'PUT') & (actual_dir == 'CALL')).sum())
    return {'accuracy': round(acc, 4), 'correct': n_correct, 'total': n_total,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}, n_correct, n_total, tp, fp, tn, fn

# ── Old logic evaluation ──
results_old = {}
for offset in [1, 2, 3, 4, 5]:
    key = f't+{offset}'
    r, *_ = eval_col(df_signals, key, 'direction')
    results_old[key] = r

has_signal_old = df_signals[df_signals['direction'].notna()].copy()
near_bb_old = has_signal_old[has_signal_old['reason'].str.contains('贴BB', na=False)]
trend_old = has_signal_old[~has_signal_old['reason'].str.contains('贴BB', na=False)]
for name, subset in [('near_bb', near_bb_old), ('trend', trend_old)]:
    if not subset.empty:
        ac = subset['t+3'].values - subset['price_t'].values
        ad = np.where(ac > 0, 'CALL', 'PUT')
        c = subset['direction'].values == ad
        results_old[f'{name}_t+3'] = {'accuracy': round(c.mean(), 4), 'correct': int(c.sum()), 'total': len(c)}
        results_old[f'{name}_score_t+3'] = subset['score'].mean()

n_signals_old = len(has_signal_old)
n_total_all = len(df_signals)
skip_reasons_old = df_signals[df_signals['direction'].isna()]['reason'].value_counts().to_dict()

# ── New logic (fusion v2) evaluation ──
results_new = {}
for offset in [1, 2, 3, 4, 5]:
    key = f't+{offset}'
    r, *_ = eval_col(df_signals, key, 'direction_v2')
    results_new[key] = r

has_signal_new = df_signals[df_signals['direction_v2'].notna()].copy()
n_signals_new = len(has_signal_new)
skip_reasons_new = df_signals[df_signals['direction_v2'].isna()]['reason_v2'].value_counts().to_dict()

# Per-tier breakdown (new logic, t+3)
tier_labels = [
    ('trend', '趋势(原有)'),
    ('nearbb_vix', '近轨+VIX确认'),
    ('nearbb', '近轨(原有)'),
]
for tier_key, _ in tier_labels:
    subset = has_signal_new[has_signal_new['tier_v2'] == tier_key]
    if not subset.empty:
        ac = subset['t+3'].values - subset['price_t'].values
        ad = np.where(ac > 0, 'CALL', 'PUT')
        c = subset['direction_v2'].values == ad
        results_new[f'{tier_key}_t+3'] = {'accuracy': round(c.mean(), 4), 'correct': int(c.sum()), 'total': len(c)}
        results_new[f'{tier_key}_score'] = subset['score'].mean()
    else:
        results_new[f'{tier_key}_t+3'] = {'accuracy': 0, 'correct': 0, 'total': 0}
        results_new[f'{tier_key}_score'] = 0

# Comparison categories (新 / 旧 同口径)
for name, tiers in [('near_bb', ['nearbb_vix', 'nearbb']), ('trend', ['trend'])]:
    subset = has_signal_new[has_signal_new['tier_v2'].isin(tiers)]
    if not subset.empty:
        ac = subset['t+3'].values - subset['price_t'].values
        ad = np.where(ac > 0, 'CALL', 'PUT')
        c = subset['direction_v2'].values == ad
        results_new[f'{name}_cmp_t+3'] = {'accuracy': round(c.mean(), 4), 'correct': int(c.sum()), 'total': len(c)}
    else:
        results_new[f'{name}_cmp_t+3'] = {'accuracy': 0, 'correct': 0, 'total': 0}

# Monthly breakdown (new logic)
df_monthly_new = has_signal_new.copy()
df_monthly_new['month'] = pd.DatetimeIndex(df_monthly_new.index).strftime('%Y-%m')
monthly_accs_new = {}
for month, grp in df_monthly_new.groupby('month'):
    for offset in [1, 2, 3]:
        key = f't+{offset}'
        grp2 = grp[grp[key].notna()]
        if not grp2.empty:
            ac = grp2[key].values - grp2['price_t'].values
            ad = np.where(ac > 0, 'CALL', 'PUT')
            c = grp2['direction_v2'].values == ad
            monthly_accs_new.setdefault(month, {})[key] = round(c.mean() * 100, 1)

# ── V2 + SKEW evaluation ──
results_skew = {}
for offset in [1, 2, 3, 4, 5]:
    key = f't+{offset}'
    r, *_ = eval_col(df_signals, key, 'direction_v2s')
    results_skew[key] = r

has_signal_skew = df_signals[df_signals['direction_v2s'].notna()].copy()
n_signals_skew = len(has_signal_skew)
skip_reasons_skew = df_signals[df_signals['direction_v2s'].isna()]['reason_v2s'].value_counts().to_dict()

skew_filtered_count = len(df_signals[df_signals['tier_v2s'] == 'skew_filtered'])
print(f"SKEW filters applied: {skew_filtered_count}")

for tier_key, _ in tier_labels:
    subset = has_signal_skew[has_signal_skew['tier_v2s'] == tier_key]
    if not subset.empty:
        ac = subset['t+3'].values - subset['price_t'].values
        ad = np.where(ac > 0, 'CALL', 'PUT')
        c = subset['direction_v2s'].values == ad
        results_skew[f'{tier_key}_t+3'] = {'accuracy': round(c.mean(), 4), 'correct': int(c.sum()), 'total': len(c)}
    else:
        results_skew[f'{tier_key}_t+3'] = {'accuracy': 0, 'correct': 0, 'total': 0}

# ────────────────────────────────────────────
# 7. Text report
# ────────────────────────────────────────────
lines = []
lines.append("=" * 95)
lines.append("XSP 方向准确率回测 — 旧逻辑 vs V2 vs V2+SKEW")
lines.append("=" * 95)
lines.append(f"数据范围: {df.index[0].date()} → {df.index[-1].date()}")
lines.append(f"总交易日: {len(df)}")
lines.append("")

# ── Overall comparison ──
lines.append(f"{'持有窗口':>15}  {'旧逻辑':>16}  {'V2':>16}  {'V2+SKEW':>16}")
lines.append("-" * 70)
for offset in [1, 2, 3, 4, 5]:
    key = f't+{offset}'
    ro = results_old.get(key, {})
    rn = results_new.get(key, {})
    rs = results_skew.get(key, {})
    lines.append(f"  {key:>12}  {ro.get('accuracy',0)*100:>6.1f}% ({ro.get('total',0):>3})  "
                 f"{rn.get('accuracy',0)*100:>6.1f}% ({rn.get('total',0):>3})  "
                 f"{rs.get('accuracy',0)*100:>6.1f}% ({rs.get('total',0):>3})")

lines.append("")
lines.append("--- 分场景对比 (t+3) ---")
for label, old_key, cmp_key in [
    ('近轨信号', 'near_bb_t+3', 'near_bb_cmp_t+3'),
    ('趋势信号', 'trend_t+3', 'trend_cmp_t+3'),
]:
    ro = results_old.get(old_key, {})
    rn = results_new.get(cmp_key, {})
    lines.append(f"  {label:>12}: 旧 {ro.get('accuracy',0)*100:>5.1f}% ({ro.get('total',0)})  →  "
                 f"V2 {rn.get('accuracy',0)*100:>5.1f}% ({rn.get('total',0)})")


lines.append("")
lines.append("--- 融合v2 分Tier准确率 (t+3) ---")
for tier_key, tier_label in tier_labels:
    r = results_new.get(f'{tier_key}_t+3', {})
    acc = r.get('accuracy', 0)
    c = r.get('correct', 0)
    t = r.get('total', 0)
    score = results_new.get(f'{tier_key}_score', 0)
    if t > 0:
        lines.append(f"  {tier_label:>16}: {acc*100:>5.1f}%  ({c}/{t})  Score {score:.0f}")
    else:
        lines.append(f"  {tier_label:>16}: —  (无信号)")

lines.append("")
lines.append("--- V2+SKEW 分Tier准确率 (t+3) ---")
for tier_key, tier_label in tier_labels:
    r = results_skew.get(f'{tier_key}_t+3', {})
    acc = r.get('accuracy', 0)
    c = r.get('correct', 0)
    t = r.get('total', 0)
    if t > 0:
        lines.append(f"  {tier_label:>16}: {acc*100:>5.1f}%  ({c}/{t})")
    else:
        lines.append(f"  {tier_label:>16}: —  (无信号)")
lines.append(f"  {'SKEW过滤(新增)':>16}: —  ({skew_filtered_count} 次过滤)")

lines.append("")
lines.append(f"旧逻辑:  信号 {n_signals_old}/{n_total_all} ({n_signals_old/n_total_all*100:.1f}%)")
lines.append(f"V2:      信号 {n_signals_new}/{n_total_all} ({n_signals_new/n_total_all*100:.1f}%)")
lines.append(f"V2+SKEW:信号 {n_signals_skew}/{n_total_all} ({n_signals_skew/n_total_all*100:.1f}%)")

lines.append("")
lines.append("--- 旧逻辑跳过原因 ---")
for reason, cnt in sorted(skip_reasons_old.items(), key=lambda x: -x[1]):
    lines.append(f"  {reason}: {cnt} ({cnt/n_total_all*100:.1f}%)")

lines.append("")
lines.append("--- V2跳过原因 ---")
for reason, cnt in sorted(skip_reasons_new.items(), key=lambda x: -x[1]):
    lines.append(f"  {reason}: {cnt} ({cnt/n_total_all*100:.1f}%)")

lines.append("")
lines.append("--- V2+SKEW跳过原因 ---")
for reason, cnt in sorted(skip_reasons_skew.items(), key=lambda x: -x[1]):
    lines.append(f"  {reason}: {cnt} ({cnt/n_total_all*100:.1f}%)")

report = "\n".join(lines)
print(report)

with open(os.path.join(OUT_DIR, 'backtest_results.txt'), 'w') as f:
    f.write(report)

# ────────────────────────────────────────────
# 8. Charts
# ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('XSP Direction Accuracy — Old vs V2 vs V2+SKEW', fontsize=16, fontweight='bold')

# Chart 1: Accuracy comparison
ax1 = axes[0, 0]
offsets = [1, 2, 3, 4, 5]
old_accs = [results_old[f't+{o}'].get('accuracy', 0) * 100 for o in offsets]
v2_accs = [results_new[f't+{o}'].get('accuracy', 0) * 100 for o in offsets]
skew_accs = [results_skew[f't+{o}'].get('accuracy', 0) * 100 for o in offsets]
x = np.arange(len(offsets))
w = 0.25
bars1 = ax1.bar(x - w, old_accs, w, label='Old', color='steelblue', alpha=0.5)
bars2 = ax1.bar(x, v2_accs, w, label='V2', color='#2ecc71', alpha=0.7)
bars3 = ax1.bar(x + w, skew_accs, w, label='V2+SKEW', color='#e74c3c', alpha=0.8)
for bar, val in zip(bars1, old_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=7, color='steelblue')
for bar, val in zip(bars2, v2_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=7, color='#2ecc71')
for bar, val in zip(bars3, skew_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=7, color='#e74c3c')
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (random)')
ax1.set_xticks(x)
ax1.set_xticklabels([f't+{o}' for o in offsets])
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Direction Accuracy vs Hold Days')
ax1.set_ylim(0, max(max(old_accs), max(v2_accs), max(skew_accs)) + 10)
ax1.legend(fontsize=8)

# Chart 2: Monthly accuracy heatmap (new logic)
ax2 = axes[0, 1]
monthly_months_new = sorted(monthly_accs_new.keys())
months_show = monthly_months_new[-24:]
if len(months_show) > 0:
    t1_vals = [monthly_accs_new[m].get('t+1', 0) for m in months_show]
    t2_vals = [monthly_accs_new[m].get('t+2', 0) for m in months_show]
    t3_vals = [monthly_accs_new[m].get('t+3', 0) for m in months_show]
    heat_data = np.array([t1_vals, t2_vals, t3_vals])
    im = ax2.imshow(heat_data, aspect='auto', cmap='RdYlGn', vmin=20, vmax=80)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['t+1', 't+2', 't+3'])
    ax2.set_xticks(range(len(months_show)))
    ax2.set_xticklabels(months_show, rotation=45, ha='right', fontsize=8)
    ax2.set_title('Monthly Accuracy (Fusion v2)')
    fig.colorbar(im, ax=ax2, shrink=0.8)

# Chart 3: Signal type pie (new)
ax3 = axes[1, 0]
tier_cnt = has_signal_new['tier_v2'].value_counts()
pie_labels = []
pie_values = []
pie_colors = []
tier_colors = {'trend': '#3498db', 'nearbb_vix': '#9b59b6', 'nearbb': '#2ecc71'}
for tier_key, _ in tier_labels:
    cnt = int(tier_cnt.get(tier_key, 0))
    if cnt > 0:
        short_label = {'trend': '趋势', 'nearbb_vix': '近轨+VIX', 'nearbb': '近轨'}[tier_key]
        pie_labels.append(f'{short_label}\n({cnt})')
        pie_values.append(cnt)
        pie_colors.append(tier_colors.get(tier_key, '#95a5a6'))
skip_cnt = n_total_all - n_signals_new
if skip_cnt > 0:
    pie_labels.append(f'Skipped\n({skip_cnt})')
    pie_values.append(skip_cnt)
    pie_colors.append('#95a5a6')
if pie_values:
    ax3.pie(pie_values, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
ax3.set_title('Signal Distribution (Fusion v2)')

# Chart 4: Cumulative return (new logic)
ax4 = axes[1, 1]
has_signal_new_sorted = has_signal_new.sort_index()
if len(has_signal_new_sorted) > 0:
    for offset, label, color in [(1, 't+1', '#3498db'), (3, 't+3', '#2ecc71')]:
        col = f't+{offset}'
        sub = has_signal_new_sorted[has_signal_new_sorted[col].notna()].copy()
        if len(sub) > 0:
            ret = (sub[col].values - sub['price_t'].values) / sub['price_t'].values
            direction_val = np.where(sub['direction_v2'].values == 'CALL', 1, -1)
            daily_pnl = direction_val * ret * 10000
            cum = np.cumsum(daily_pnl)
            ax4.plot(cum, label=f'v2 {label}', color=color, alpha=0.8)
    # Overlay old t+3 for reference
    has_signal_old_sorted = has_signal_old.sort_index()
    if len(has_signal_old_sorted) > 0:
        sub = has_signal_old_sorted[has_signal_old_sorted['t+3'].notna()].copy()
        if len(sub) > 0:
            ret = (sub['t+3'].values - sub['price_t'].values) / sub['price_t'].values
            direction_val = np.where(sub['direction'].values == 'CALL', 1, -1)
            daily_pnl = direction_val * ret * 10000
            cum = np.cumsum(daily_pnl)
            ax4.plot(cum, label='old t+3', color='gray', alpha=0.4, linestyle='--')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Trades')
    ax4.set_ylabel('Cumulative PnL ($)')
    ax4.set_title('Simulated Cumulative Return (1 contract/trade)')
    ax4.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
chart_path = os.path.join(OUT_DIR, 'backtest_charts.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n📊 图表已保存: {chart_path}")
print(f"📄 报告已保存: {os.path.join(OUT_DIR, 'backtest_results.txt')}")
print("✅ 回测完成")
