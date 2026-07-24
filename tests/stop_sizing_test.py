"""
Backtest: stop-loss by signal type + smooth position sizing.
Read-only analysis script.
"""
import numpy as np, pandas as pd, yfinance as yf
import warnings; warnings.filterwarnings('ignore')
import pandas_ta as ta

print("Fetching data...")
xsp = yf.download('^XSP', period='3y', interval='1d', auto_adjust=False)
xsp.columns = [c[0] for c in xsp.columns]
xsp.index = pd.to_datetime(xsp.index)
price = xsp['Adj Close'] if 'Adj Close' in xsp.columns else xsp['Close']
high = xsp['High'] if 'High' in xsp.columns else price
low = xsp['Low'] if 'Low' in xsp.columns else price

vix = yf.download('^VIX', period='3y', interval='1d', auto_adjust=False)
vix.columns = [c[0] for c in vix.columns]
vix.index = pd.to_datetime(vix.index)
vix_close = vix['Close'] if 'Close' in vix.columns else vix['Adj Close']

skew_data = yf.download('^SKEW', period='3y', interval='1d', auto_adjust=False)
skew_data.columns = [c[0] for c in skew_data.columns]
skew_data.index = pd.to_datetime(skew_data.index)
skew_close = skew_data['Close'] if 'Close' in skew_data.columns else skew_data['Adj Close']

print("Computing indicators...")
df = pd.DataFrame(index=price.index)
df['price'] = price; df['high'] = high; df['low'] = low
df['ema_20'] = price.ewm(span=20, min_periods=20).mean()
sma = price.rolling(20).mean(); sd = price.rolling(20).std()
df['bbu'] = sma + 2*sd; df['bbl'] = sma - 2*sd
df['bw'] = (df['bbu'] - df['bbl']).fillna(0)
df['bbw'] = (df['bw'] / price * 100).fillna(0)
df['dev'] = ((price - sma) / sma * 100).fillna(0)
prev = price.shift(1)
tr = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
df['atr_14'] = tr.rolling(14).mean()
up = price.diff()
plus_dm = pd.Series(np.where((up > -up) & (up > 0), up, 0), index=price.index)
minus_dm = pd.Series(np.where((-up > up) & (-up > 0), -up, 0), index=price.index)
atr14_s = tr.rolling(14).mean()
df['di_diff'] = (100 * plus_dm.rolling(14).mean() / atr14_s - 100 * minus_dm.rolling(14).mean() / atr14_s) / 100
df['er'] = (price.diff(10).abs() / price.diff().abs().rolling(10).sum()).fillna(0)
vr_hi = price.rolling(10).max(); vr_lo = price.rolling(10).min()
df['vr'] = ((price - vr_lo) / (vr_hi - vr_lo).replace(0, np.nan)).fillna(0.5).clip(0, 1) * 2
adx_df = ta.adx(high, low, price, length=14)
df['adx'] = adx_df['ADX_14'].values

vix_aligned = vix_close.reindex(df.index, method='ffill')
vi = vix_aligned.values.astype(float)
vpa = np.full(len(vi), np.nan)
for i in range(len(vi)):
    start = max(0, i - 504)
    w = vi[start:i+1]
    vpa[i] = np.sum(w <= vi[i]) / len(w) * 100 if len(w) >= 100 else 50
df['vix_percentile'] = vpa
skew_aligned = skew_close.reindex(df.index, method='ffill')
df['skew_index'] = skew_aligned.values.astype(float)

# Score functions
W = {'adx': 0.3, 'er': 0.2, 'bbw': 0.15, 'dev': 0.15, 'vr': 0.1}
T = {'adx': [30, 25, 20, 15, 0], 'er': [.7, .55, .35, .2, 0],
     'bbw': [45, 30, 18, 10, 0], 'dev': [3.0, 1.5, 0.8, 0.3, 0],
     'vr': [2.0, 1.3, .8, .5, 0]}

def score_ts(v, th):
    for t, s in zip(th, [100, 75, 50, 25, 0]):
        if v >= t: return s
    return 0

def compute_score(row):
    t = 5
    for k, w in W.items():
        v = row.get(k)
        if v is None or np.isnan(v): t += 5; continue
        t += score_ts(abs(v) if k == 'dev' else v, T[k]) * w
    return round(t)

def get_signal(row):
    p = row['price']; u = row['bbu']; l = row['bbl']; bw = row['bw']
    dd = row['di_diff']; at = row['atr_14']; sc = row['score']; vp = row['vix_percentile']
    if u == l or bw <= 0 or np.isnan(bw): return None, None, None
    near_th = at * 0.60 if (at and at > 0 and not np.isnan(at)) else bw * 0.1
    nt = (u - p) < near_th; nb = (p - l) < near_th;     is_t = sc >= 50
    if not nt and not nb and is_t:
        return ('CALL', 'trend') if dd > 0 else ('PUT', 'trend') if dd < 0 else (None, None)
    if nt and sc >= 50 and vp > 75: return 'PUT', 'nearbb_vix'
    if nb and sc >= 50 and vp > 75: return 'CALL', 'nearbb_vix'
    if nt and dd > 0: return None, None
    if nb and dd < 0: return None, None
    if nt and sc >= 50: return 'PUT', 'nearbb'
    if nb and sc >= 50: return 'CALL', 'nearbb'
    return None, None

def get_tier(score, di_diff, skew):
    if abs(di_diff) > 0 and score >= 72 and ((skew < 145 if di_diff > 0 else skew > 145)):
        return 'strong'
    elif score >= 65: return 'normal'
    else: return 'weak'

def simulate_trade(entry_p, direction, i, rows, stop_pct, t_n=3):
    hit = False; exit_p = None
    stop_frac = stop_pct / 100.0
    for off in range(1, t_n + 1):
        j = i + off
        if j >= len(rows): break
        r = rows[j]
        if direction == 'CALL':
            adverse = (entry_p - r['low']) / entry_p
        else:
            adverse = (r['high'] - entry_p) / entry_p
        if adverse >= stop_frac:
            hit = True
            exit_p = entry_p * (1 - stop_frac) if direction == 'CALL' else entry_p * (1 + stop_frac)
            break
    if not hit:
        j3 = i + t_n
        if j3 < len(rows):
            exit_p = rows[j3]['price']
    if exit_p is None: return None
    if direction == 'CALL': pnl = (exit_p - entry_p) / entry_p
    else: pnl = (entry_p - exit_p) / entry_p
    return {'pnl_pct': pnl * 100, 'hit_stop': hit}

# Generate all signals
print("Generating signals...")
rows = df.to_dict('records')
signals = []
for i in range(20, len(rows) - 5):
    r = rows[i].copy()
    if r['price'] <= 0 or np.isnan(r.get('ema_20', np.nan)): continue
    r['score'] = compute_score(r)
    if r['score'] is None: continue
    dir_s, reason = get_signal(r)
    if dir_s is None: continue
    tier = get_tier(r['score'], r['di_diff'], r['skew_index'])
    signals.append({'i': i, 'dir': dir_s, 'reason': reason, 'tier': tier,
                    'score': r['score'], 'price': r['price']})
print(f"  Total signals: {len(signals)}")

# BACKTEST #1: Stop by signal type
print()
print("=" * 80)
print("BACKTEST #1: 止损按信号类型")
print("=" * 80)

scenarios = [
    ("当前(统一1.5%)", {'trend': 1.5, 'nearbb': 1.5, 'nearbb_vix': 1.5}),
    ("分类(trend=2.0%, nearbb_vix=1.0%)", {'trend': 2.0, 'nearbb': 2.0, 'nearbb_vix': 1.0}),
    ("更宽(trend=3.0%, nearbb_vix=1.0%)", {'trend': 3.0, 'nearbb': 3.0, 'nearbb_vix': 1.0}),
]

for label, stop_map in scenarios:
    results = []
    for s in signals:
        sp = stop_map.get(s['reason'], 1.5)
        trade = simulate_trade(s['price'], s['dir'], s['i'], rows, sp, 3)
        if trade:
            trade.update(s)
            results.append(trade)
    n = len(results)
    w = sum(1 for t in results if t['pnl_pct'] > 0)
    hits = sum(1 for t in results if t['hit_stop'])
    ar = np.mean([t['pnl_pct'] for t in results]) if results else 0
    aw = np.mean([t['pnl_pct'] for t in results if t['pnl_pct'] > 0]) if w else 0
    al = np.mean([t['pnl_pct'] for t in results if t['pnl_pct'] <= 0]) if (n - w) else 0
    pf = abs(aw / al) if al != 0 else 0
    print()
    print(f"  {label}: n={n}  win={w/n*100:.1f}%  hits={hits/n*100:.1f}%  avg_ret={ar:+.2f}%  PF={pf:.2f}")
    for rea in ['trend', 'nearbb_vix', 'nearbb']:
        sub = [t for t in results if t['reason'] == rea]
        if not sub: continue
        sw = sum(1 for t in sub if t['pnl_pct'] > 0)
        sn = len(sub)
        sh = sum(1 for t in sub if t['hit_stop'])
        sa = np.mean([t['pnl_pct'] for t in sub])
        print(f"    [{rea}] {sn}  win={sw/sn*100:.1f}%  hits={sh/sn*100:.1f}%  avg={sa:+.2f}%")

    # ETF $ estimate
    size_cur = {'weak': 2000, 'normal': 4000, 'strong': 4000}
    total_dollar = 0
    for t in results:
        amt = size_cur.get(t['tier'], 2000)
        total_dollar += amt * 3 * t['pnl_pct'] / 100
    print(f"    ETF 3年总收益: ${total_dollar:,.0f}  年化: ${total_dollar/3:,.0f}")

# BACKTEST #2: Smooth position sizing
print()
print("=" * 80)
print("BACKTEST #2: 平滑仓位")
print("=" * 80)

results = []
for s in signals:
    trade = simulate_trade(s['price'], s['dir'], s['i'], rows, 1.5, 3)
    if trade:
        trade.update(s)
        results.append(trade)

size_cur = {'weak': 2000, 'normal': 4000, 'strong': 4000}

def size_smooth(sc):
    if sc <= 59: return 1000
    elif sc <= 69: return 2000
    elif sc <= 74: return 3000
    else: return 4000

total_cur = sum(size_cur[t['tier']] * 3 * t['pnl_pct'] / 100 for t in results)
total_smo = sum(size_smooth(t['score']) * 3 * t['pnl_pct'] / 100 for t in results)

total_inv_cur = sum(size_cur[t['tier']] for t in results)
total_inv_smo = sum(size_smooth(t['score']) for t in results)

print(f"  当前三级: 总投资=${total_inv_cur:,.0f}  3年收益=${total_cur:,.0f}  年化=${total_cur/3:,.0f}")
print(f"  平滑分级: 总投资=${total_inv_smo:,.0f}  3年收益=${total_smo:,.0f}  年化=${total_smo/3:,.0f}")
print(f"  差异: 收益${total_smo - total_cur:+,.0f}  ({(total_smo/total_cur - 1)*100:+.1f}%)")

print()
print("  Score分布:")
for lo, hi in [(55, 59), (60, 64), (65, 69), (70, 74), (75, 100)]:
    sub = [t for t in results if lo <= t['score'] <= hi]
    if not sub:
        print(f"    {lo}-{hi}:  0 sig")
        continue
    rets = [t['pnl_pct'] for t in sub]
    cur_amt = size_cur.get(sub[0]['tier'], 2000)
    smo_amt = size_smooth(sub[0]['score'])
    cur_ret = sum(cur_amt * 3 * r / 100 for r in rets)
    smo_ret = sum(smo_amt * 3 * r / 100 for r in rets)
    print(f"    {lo}-{hi}:  {len(sub):>3} sig  cur=${cur_amt:>4} smo=${smo_amt:>4}  "
          f"avg_ret={np.mean(rets):+.2f}%  win={sum(1 for r in rets if r > 0)/len(rets)*100:.0f}%  "
          f"cur_ret=${cur_ret:+,.0f}  smo_ret=${smo_ret:+,.0f}")

print()
print("Done.")
