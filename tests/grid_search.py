"""Grid search: find optimal parameters for XSP direction signals."""
import os, sys, json, time, itertools
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = SRC_DIR

# ═══════════════════════════════════════════════
# 1. Fetch data (once)
# ═══════════════════════════════════════════════
print("⏳ Fetching data...")
xsp = yf.download("^XSP", period="3y", interval="1d", auto_adjust=False)
xsp.columns = [c[0] for c in xsp.columns]
xsp.index = pd.to_datetime(xsp.index)

vix = yf.download("^VIX", period="3y", interval="1d", auto_adjust=False)
vix.columns = [c[0] for c in vix.columns]
vix.index = pd.to_datetime(vix.index)

skew_data = yf.download("^SKEW", period="3y", interval="1d", auto_adjust=False)
skew_data.columns = [c[0] for c in skew_data.columns]
skew_data.index = pd.to_datetime(skew_data.index)

close_col = 'Adj Close' if 'Adj Close' in xsp.columns else 'Close'
price_series = xsp[close_col]
vix_close = vix['Close'] if 'Close' in vix.columns else vix['Adj Close']
skew_close = skew_data['Close'] if 'Close' in skew_data.columns else skew_data['Adj Close']

# ═══════════════════════════════════════════════
# 2. Compute indicators (once)
# ═══════════════════════════════════════════════
print("⏳ Computing indicators...")
df = pd.DataFrame(index=price_series.index)
df['price'] = price_series

# VIX rank & percentile
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
        lower = np.sum(window <= vi[i]) / len(window) * 100
        vix_rank_arr[i] = lower
        vix_pct_arr[i] = lower
    else:
        vix_rank_arr[i] = 50.0
        vix_pct_arr[i] = 50.0
df['vix'] = vi
df['vix_rank'] = vix_rank_arr
df['vix_percentile'] = vix_pct_arr

# SKEW
skew_aligned = skew_close.reindex(df.index, method='ffill')
df['skew_index'] = skew_aligned.values.astype(float)

# EMA20
df['ema_20'] = df['price'].ewm(span=20, min_periods=20).mean()

# Bollinger Bands
sma = df['price'].rolling(20).mean()
sd = df['price'].rolling(20).std()
df['bbl'] = sma - 2 * sd
df['bbu'] = sma + 2 * sd
df['bw'] = (df['bbu'] - df['bbl']).fillna(0)
df['bbw'] = (df['bw'] / df['price'] * 100).fillna(0)

# Dev
df['dev'] = ((df['price'] - sma) / sma * 100).fillna(0)

# ATR14
xsp_h = xsp['High'] if 'High' in xsp.columns else price_series
xsp_l = xsp['Low'] if 'Low' in xsp.columns else price_series
xsp_c = price_series
prev_close = xsp_c.shift(1)
tr = pd.concat([
    xsp_h - xsp_l,
    (xsp_h - prev_close).abs(),
    (xsp_l - prev_close).abs()
], axis=1).max(axis=1)
df['atr_14'] = tr.rolling(14).mean()

# ADX
up = xsp_c.diff()
down = -up
plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=xsp_c.index)
minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=xsp_c.index)
atr_tr = tr.rolling(14).mean()
plus_di = 100 * plus_dm.rolling(14).mean() / atr_tr
minus_di = 100 * minus_dm.rolling(14).mean() / atr_tr
dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
df['adx'] = dx.rolling(14).mean()
df['di_diff'] = (plus_di - minus_di) / 100

# ER
def er_func(close, period=10):
    change = close.diff(period).abs()
    vol = close.diff().abs().rolling(period).sum()
    return (change / vol).fillna(0)
df['er'] = er_func(price_series, 10)

# VR
def vr_func(close, period=10):
    h = close.rolling(period).max()
    l_ = close.rolling(period).min()
    vr = ((close - l_) / (h - l_).replace(0, np.nan)).fillna(0.5)
    return vr.clip(0, 1) * 2
df['vr'] = vr_func(price_series, 10)

# Support/Resistance
df['support'] = df['bbl']
df['resistance'] = df['bbu']

# ── Scoring function ──
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

# ═══════════════════════════════════════════════
# 3. Pre-compute scores for all rows
# ═══════════════════════════════════════════════
print("⏳ Computing scores...")
rows = df.to_dict('records')
dates = df.index.tolist()
for i in range(len(rows)):
    rows[i]['score'] = compute_score(rows[i])

# Pre-compute future returns
max_hold = 5
for n in range(1, max_hold + 1):
    df[f'ret_{n}'] = df['price'].shift(-n) / df['price'] - 1

# ═══════════════════════════════════════════════
# 4. Evaluate function
# ═══════════════════════════════════════════════
def evaluate(score_trend_th, score_nearbb_th, atr_mult, vix_pct_thresh, use_conflict):
    """Run direction logic with given params, return metrics dict."""
    results = []
    for i in range(len(rows)):
        if i < 20:
            continue
        row = rows[i]
        price = row['price']
        bbu = row['bbu']
        bbl = row['bbl']
        bw = row['bw']
        ema20 = row['ema_20']
        di_diff = row['di_diff']
        atr14 = row['atr_14']
        score = row['score']
        vix_pct = row.get('vix_percentile', 50)
        skew = row.get('skew_index', 146)

        if bbu == bbl or bw <= 0 or np.isnan(bw):
            continue
        if np.isnan(ema20) or np.isnan(bbl) or price <= 0:
            continue
        if np.isnan(score):
            continue

        # Near-BB check
        if atr14 and atr14 > 0 and not np.isnan(atr14):
            near_threshold = atr14 * atr_mult
        else:
            near_threshold = bw * 0.10
        near_top = (bbu - price) < near_threshold
        near_bottom = (price - bbl) < near_threshold
        near_bb = near_top or near_bottom

        is_trend = score >= score_trend_th

        # Level 1: Trend
        if not near_bb and is_trend:
            if di_diff > 0:
                direction = 'CALL'
                reason = 'trend'
            elif di_diff < 0:
                direction = 'PUT'
                reason = 'trend'
            else:
                continue
        # Level 2: Near-BB + VIX high
        elif near_top and score >= score_nearbb_th and vix_pct > vix_pct_thresh:
            direction = 'PUT'
            reason = 'nearbb_vix'
        elif near_bottom and score >= score_nearbb_th and vix_pct > vix_pct_thresh:
            direction = 'CALL'
            reason = 'nearbb_vix'
        # Level 3: Conflict filter
        elif use_conflict and near_top and di_diff > 0:
            continue
        elif use_conflict and near_bottom and di_diff < 0:
            continue
        # Level 4: Near-BB
        elif near_top and score >= score_nearbb_th:
            direction = 'PUT'
            reason = 'nearbb'
        elif near_bottom and score >= score_nearbb_th:
            direction = 'CALL'
            reason = 'nearbb'
        else:
            continue

        # SKEW filter
        if direction == 'PUT' and skew < 140:
            continue
        if direction == 'CALL' and skew > 155:
            continue

        results.append((i, direction, reason))

    # Evaluate accuracy at t+1..t+5
    n_days = len(rows)
    holdings = list(range(1, 6))
    acc = {}
    by_reason = {}

    for idx, direction, reason in results:
        for n in holdings:
            if idx + n < n_days:
                ret = df['ret_' + str(n)].iloc[idx]
                signed = ret if direction == 'CALL' else -ret
                correct = signed > 0
                acc.setdefault(n, {'correct': 0, 'total': 0, 'sum_win': 0.0, 'sum_loss': 0.0, 'win_count': 0, 'loss_count': 0})
                acc[n]['total'] += 1
                if correct:
                    acc[n]['correct'] += 1
                    acc[n]['sum_win'] += signed
                    acc[n]['win_count'] += 1
                else:
                    acc[n]['sum_loss'] += -signed
                    acc[n]['loss_count'] += 1

        by_reason.setdefault(reason, {'correct': 0, 'total': 0, 'n': 0, 'sum_win': 0.0, 'sum_loss': 0.0})
        by_reason[reason]['total'] += 1
        by_reason[reason]['n'] += 1
        # t+3 signed return
        if idx + 3 < n_days:
            ret3 = df['ret_3'].iloc[idx]
            signed3 = ret3 if direction == 'CALL' else -ret3
            if signed3 > 0:
                by_reason[reason]['correct'] += 1
                by_reason[reason]['sum_win'] += signed3
            else:
                by_reason[reason]['sum_loss'] += -signed3

    def _stats(a):
        t = a['total']
        if t == 0:
            return {'accuracy': 0, 'avg_ret': 0, 'profit_factor': 0}
        avg_ret = (a['sum_win'] - a['sum_loss']) / t * 100
        pf = a['sum_win'] / a['sum_loss'] if a['sum_loss'] > 0 else float('inf')
        return {
            'accuracy': round(a['correct'] / t * 100, 1),
            'avg_ret': round(avg_ret, 2),
            'profit_factor': round(pf, 2) if pf != float('inf') else 999,
        }

    return {
        'num_signals': len(results),
        'accuracy': {n: _stats(acc[n]) for n in holdings},
        'by_reason': {k: {
            'accuracy': round(v['correct'] / v['total'] * 100, 1) if v['total'] > 0 else 0,
            'count': v['n'],
            'avg_ret': round((v['sum_win'] - v['sum_loss']) / v['total'] * 100, 2) if v['total'] > 0 else 0,
            'profit_factor': round(v['sum_win'] / v['sum_loss'], 2) if v['sum_loss'] > 0 else 999,
        } for k, v in by_reason.items()}
    }

# ═══════════════════════════════════════════════
# 5. Grid search
# ═══════════════════════════════════════════════
print("⏳ Running grid search...")
score_trend_values = [55, 60, 65, 70]
score_nearbb_values = [50, 55, 60, 65]
atr_mult_values = [0.3, 0.4, 0.5, 0.6, 0.7]
vix_pct_values = [50, 65, 75, 85]
conflict_values = [True, False]

results = []
total = len(score_trend_values) * len(score_nearbb_values) * len(atr_mult_values) * len(vix_pct_values) * len(conflict_values)
start = time.time()

for st, sn, am, vp, cf in itertools.product(
    score_trend_values, score_nearbb_values, atr_mult_values, vix_pct_values, conflict_values
):
    r = evaluate(st, sn, am, vp, cf)
    results.append({
        'trend_th': st,
        'nearbb_th': sn,
        'atr_mult': am,
        'vix_pct': vp,
        'conflict': cf,
        'signals': r['num_signals'],
        't3': r['accuracy'].get(3, {}).get('accuracy', 0),
        't5': r['accuracy'].get(5, {}).get('accuracy', 0),
        'avg_ret_3': r['accuracy'].get(3, {}).get('avg_ret', 0),
        'pf_3': r['accuracy'].get(3, {}).get('profit_factor', 0),
        'by_reason': r['by_reason'],
    })

elapsed = time.time() - start
print(f"Done: {total} combos in {elapsed:.1f}s ({elapsed/total:.2f}s per combo)")

# ═══════════════════════════════════════════════
# 6. Report
# ═══════════════════════════════════════════════
dfr = pd.DataFrame(results)

# Filter: at least 30 signals
dfr = dfr[dfr['signals'] >= 30].copy()

# Sort by t+3 accuracy
top = dfr.sort_values('t3', ascending=False).head(30)

print("\n═══ Top 30 by t+3 accuracy (min 30 signals) ═══")
print(f"{'#':>3} {'trend':>5} {'nearbb':>6} {'atr':>4} {'vix':>4} {'cf':>4} {'sig':>4} {'t+3%':>5} {'t+5%':>5} {'avgR%':>6} {'PF':>5} {'tr%':>5} {'trN':>5} {'nbb%':>5}")
print("-" * 80)
for idx, row in top.iterrows():
    br = row.get('by_reason', {})
    trend_pct = br.get('trend', {}).get('accuracy', 0)
    trend_n = br.get('trend', {}).get('count', 0)
    nbb_pct = br.get('nearbb_vix', {}).get('accuracy', 0)
    nbb_n = br.get('nearbb_vix', {}).get('count', 0)
    print(f"{idx:>3} {row['trend_th']:>5} {row['nearbb_th']:>6} {row['atr_mult']:>4.1f} {row['vix_pct']:>4} {str(row['conflict'])[0]:>4} {row['signals']:>4} {row['t3']:>5.1f} {row['t5']:>5.1f} {row['avg_ret_3']:>6.2f} {row['pf_3']:>5.1f} {trend_pct:>4.0f}% {trend_n:>4} {nbb_pct:>4.0f}%")

# Also sort by t+5
top5 = dfr.sort_values('t5', ascending=False).head(20)
print("\n═══ Top 20 by t+5 accuracy ═══")
print(f"{'#':>3} {'trend':>5} {'nearbb':>6} {'atr':>4} {'vix':>4} {'cf':>4} {'sig':>4} {'t+3%':>5} {'t+5%':>5} {'avgR%':>6} {'PF':>5}")
print("-" * 65)
for idx, row in top5.iterrows():
    print(f"{idx:>3} {row['trend_th']:>5} {row['nearbb_th']:>6} {row['atr_mult']:>4.1f} {row['vix_pct']:>4} {str(row['conflict'])[0]:>4} {row['signals']:>4} {row['t3']:>5.1f} {row['t5']:>5.1f} {row['avg_ret_3']:>6.2f} {row['pf_3']:>5.1f}")

# Save full results
out = []
for _, row in dfr.iterrows():
    out.append({k: v for k, v in row.items() if k != 'by_reason'})
    out[-1]['by_reason'] = {k: {'accuracy': v['accuracy'], 'count': v['count'], 'avg_ret': v['avg_ret'], 'profit_factor': v['profit_factor']} for k, v in row['by_reason'].items()}

with open(os.path.join(OUT_DIR, 'grid_search_results.json'), 'w') as f:
    json.dump(out, f, indent=2)

print(f"\n📄 Full results saved: {os.path.join(OUT_DIR, 'grid_search_results.json')}")
print("✅ Grid search complete")
