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

# Use adjusted close if available
close_col = 'Adj Close' if 'Adj Close' in xsp.columns else 'Close'
price_series = xsp[close_col]
vix_close = vix['Close'] if 'Close' in vix.columns else vix['Adj Close']

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

df['vr'] = vol_ratio(price_series, 10)

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
     'bbw': [45, 30, 18, 10, 0], 'vr': [2.0, 1.3, 0.8, 0.5, 0]}

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
    is_trend = score >= 65

    if bbu == bbl or bw <= 0 or np.isnan(bw):
        return None, 'insufficient_data'

    dup = (bbu - price) / bw * 100
    dlow = (price - bbl) / bw * 100
    if atr14 and atr14 > 0 and not np.isnan(atr14):
        near_threshold = atr14 * 0.50
    else:
        near_threshold = bw * 0.10
    near_top = (bbu - price) < near_threshold
    near_bottom = (price - bbl) < near_threshold

    if near_top and score >= 60:
        return 'PUT', f'贴BB上轨({dup:.0f}%)'
    elif near_bottom and score >= 60:
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

    direction, reason = get_direction(row)
    if direction is None:
        # Still record for skip stats
        row['direction'] = None
        row['reason'] = reason
        row['score'] = row.get('score', 0)
        signals.append(row.copy())
        continue

    # Check future price changes
    future_prices = {}
    for offset in [1, 2, 3, 4, 5]:
        j = i + offset
        if j < len(rows):
            future_prices[f't+{offset}'] = rows[j]['price']
        else:
            future_prices[f't+{offset}'] = None

    row['direction'] = direction
    row['reason'] = reason
    row['price_t'] = row['price']
    for k, v in future_prices.items():
        row[k] = v
    signals.append(row.copy())

df_signals = pd.DataFrame(signals).set_index('date')

# ────────────────────────────────────────────
# 6. Accuracy evaluation
# ────────────────────────────────────────────
def eval_window(df_sig, offset_key):
    """Evaluate direction accuracy for a given t+N window."""
    valid = df_sig[df_sig['direction'].notna()].copy()
    if valid.empty:
        return {}, 0, 0, 0, 0
    future_col = offset_key
    valid = valid[valid[future_col].notna()]
    if valid.empty:
        return {}, 0, 0, 0, 0
    actual_change = valid[future_col].values - valid['price_t'].values
    actual_dir = np.where(actual_change > 0, 'CALL', 'PUT')
    correct = valid['direction'].values == actual_dir
    n_correct = int(correct.sum())
    n_total = len(correct)
    acc = n_correct / n_total if n_total > 0 else 0
    tp = int(((valid['direction'].values == 'CALL') & (actual_dir == 'CALL')).sum())
    fp = int(((valid['direction'].values == 'CALL') & (actual_dir == 'PUT')).sum())
    tn = int(((valid['direction'].values == 'PUT') & (actual_dir == 'PUT')).sum())
    fn = int(((valid['direction'].values == 'PUT') & (actual_dir == 'CALL')).sum())
    return {'accuracy': round(acc, 4), 'correct': n_correct, 'total': n_total,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}, n_correct, n_total, tp, fp, tn, fn

results = {}
for offset in [1, 2, 3, 4, 5]:
    key = f't+{offset}'
    r, *_ = eval_window(df_signals, key)
    results[key] = r

# Per-scenario breakdown (t+3)
has_signal = df_signals[df_signals['direction'].notna()].copy()
near_bb = has_signal[has_signal['reason'].str.contains('贴BB', na=False)]
trend = has_signal[~has_signal['reason'].str.contains('贴BB', na=False)]
for name, subset in [('near_bb', near_bb), ('trend', trend)]:
    if not subset.empty:
        actual_change = subset['t+3'].values - subset['price_t'].values
        actual_dir = np.where(actual_change > 0, 'CALL', 'PUT')
        correct = subset['direction'].values == actual_dir
        acc = correct.mean()
        n_correct = int(correct.sum())
        n_total = len(correct)
        results[f'{name}_t+3'] = {'accuracy': round(acc, 4), 'correct': n_correct, 'total': n_total}
        results[f'{name}_score_t+3'] = subset['score'].mean()

# Monthly breakdown
df_monthly = has_signal.copy()
df_monthly['month'] = pd.DatetimeIndex(df_monthly.index).strftime('%Y-%m')
monthly_accs = {}
for month, grp in df_monthly.groupby('month'):
    for offset in [1, 2, 3]:
        key = f't+{offset}'
        grp2 = grp[grp[key].notna()]
        if not grp2.empty:
            actual_change = grp2[key].values - grp2['price_t'].values
            actual_dir = np.where(actual_change > 0, 'CALL', 'PUT')
            correct = grp2['direction'].values == actual_dir
            monthly_accs.setdefault(month, {})[key] = round(correct.mean() * 100, 1)

# Skip stats
n_signals = len(has_signal)
n_total = len(df_signals)
n_skipped = n_total - n_signals
skip_reasons = df_signals[df_signals['direction'].isna()]['reason'].value_counts().to_dict()

# ────────────────────────────────────────────
# 7. Text report
# ────────────────────────────────────────────
lines = []
lines.append("=" * 60)
lines.append("XSP 市场早报方向准确率回测报告")
lines.append("=" * 60)
lines.append(f"数据范围: {df.index[0].date()} → {df.index[-1].date()}")
lines.append(f"总交易日: {len(df)}")
lines.append(f"信号天数: {n_signals} ({n_skipped} 天跳过)")
lines.append("")

lines.append("--- 跳过原因分布 ---")
for reason, cnt in sorted(skip_reasons.items(), key=lambda x: -x[1]):
    lines.append(f"  {reason}: {cnt} ({cnt/n_total*100:.1f}%)")

lines.append("")
lines.append(f"{'持有窗口':>10}  {'准确率':>8}  {'正确/总':>14}")
lines.append("-" * 40)
for offset in [1, 2, 3, 4, 5]:
    key = f't+{offset}'
    r = results.get(key, {})
    acc = r.get('accuracy', 0)
    c = r.get('correct', 0)
    t = r.get('total', 0)
    lines.append(f"{key:>10}  {acc*100:>7.1f}%  {c:>4}/{t:<4}")
    if offset in (1, 2, 3):
        cm = f"      TP={r.get('tp',0)} FP={r.get('fp',0)}  TN={r.get('tn',0)} FN={r.get('fn',0)}"
        lines.append(f"{'':>10}  {cm}")

lines.append("")
lines.append("--- 分场景 (t+3) ---")
for name, label in [('near_bb', '近轨信号'), ('trend', '趋势信号')]:
    r = results.get(f'{name}_t+3', {})
    acc = r.get('accuracy', 0)
    c = r.get('correct', 0)
    t = r.get('total', 0)
    score = results.get(f'{name}_score_t+3', 0)
    lines.append(f"  {label:>12}: {acc*100:>6.1f}%  ({c}/{t})  平均Score {score:.0f}")

lines.append("")
lines.append("--- 月度准确率 ---")
monthly_months = sorted(monthly_accs.keys())
lines.append(f"  {'月份':>8}  {'t+1':>7}  {'t+2':>7}  {'t+3':>7}  {'信号数':>7}")
lines.append("  " + "-" * 42)
for month in monthly_months:
    m = monthly_accs[month]
    cnt = len(has_signal[df_monthly['month'] == month])
    lines.append(f"  {month:>8}  {m.get('t+1',0):>6.1f}%  {m.get('t+2',0):>6.1f}%  {m.get('t+3',0):>6.1f}%  {cnt:>6}")

report = "\n".join(lines)
print(report)

with open(os.path.join(OUT_DIR, 'backtest_results.txt'), 'w') as f:
    f.write(report)

# ────────────────────────────────────────────
# 8. Charts
# ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('XSP Direction Accuracy Backtest', fontsize=16, fontweight='bold')

# Chart 1: Accuracy vs Hold Days
ax1 = axes[0, 0]
offsets = [1, 2, 3, 4, 5]
accs_plot = [results[f't+{o}'].get('accuracy', 0) * 100 for o in offsets]
bars = ax1.bar([f't+{o}' for o in offsets], accs_plot, color='steelblue', width=0.6)
for bar, val in zip(bars, accs_plot):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (random)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Direction Accuracy vs Hold Days')
ax1.set_ylim(0, max(accs_plot) + 8)
ax1.legend()

# Chart 2: Monthly accuracy heatmap
ax2 = axes[0, 1]
months_show = monthly_months[-24:]
if len(months_show) > 0:
    t1_vals = [monthly_accs[m].get('t+1', 0) for m in months_show]
    t2_vals = [monthly_accs[m].get('t+2', 0) for m in months_show]
    t3_vals = [monthly_accs[m].get('t+3', 0) for m in months_show]
    heat_data = np.array([t1_vals, t2_vals, t3_vals])
    im = ax2.imshow(heat_data, aspect='auto', cmap='RdYlGn', vmin=20, vmax=80)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['t+1', 't+2', 't+3'])
    ax2.set_xticks(range(len(months_show)))
    ax2.set_xticklabels(months_show, rotation=45, ha='right', fontsize=8)
    ax2.set_title('Monthly Accuracy Heatmap')
    fig.colorbar(im, ax=ax2, shrink=0.8)

# Chart 3: Signal type pie
ax3 = axes[1, 0]
near_bb_cnt = len(near_bb)
trend_cnt = len(trend)
skip_cnt = max(0, n_total - n_signals)
pie_labels = []
pie_values = []
pie_colors = []
if near_bb_cnt > 0:
    pie_labels.append(f'Near-Band\n({near_bb_cnt})')
    pie_values.append(near_bb_cnt)
    pie_colors.append('#2ecc71')
if trend_cnt > 0:
    pie_labels.append(f'Trend\n({trend_cnt})')
    pie_values.append(trend_cnt)
    pie_colors.append('#3498db')
if skip_cnt > 0:
    pie_labels.append(f'Skipped\n({skip_cnt})')
    pie_values.append(skip_cnt)
    pie_colors.append('#95a5a6')
ax3.pie(pie_values, labels=pie_labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
ax3.set_title('Signal Type Distribution')

# Chart 4: Cumulative return (simulated)
ax4 = axes[1, 1]
has_signal_sorted = has_signal.sort_index()
if len(has_signal_sorted) > 0:
    for offset, label, color in [(1, 't+1', '#3498db'), (2, 't+2', '#e67e22'), (3, 't+3', '#2ecc71')]:
        col = f't+{offset}'
        sub = has_signal_sorted[has_signal_sorted[col].notna()].copy()
        if len(sub) > 0:
            ret = (sub[col].values - sub['price_t'].values) / sub['price_t'].values
            direction_val = np.where(sub['direction'].values == 'CALL', 1, -1)
            daily_pnl = direction_val * ret * 10000  # 1 contract * return
            cum = np.cumsum(daily_pnl)
            ax4.plot(cum, label=label, color=color, alpha=0.8)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Trades')
    ax4.set_ylabel('Cumulative PnL ($)')
    ax4.set_title('Simulated Cumulative Return (1 contract/trade)')
    ax4.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
chart_path = os.path.join(OUT_DIR, 'backtest_charts.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n📊 图表已保存: {chart_path}")
print(f"📄 报告已保存: {os.path.join(OUT_DIR, 'backtest_results.txt')}")
print("✅ 回测完成")
