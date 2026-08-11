"""Macro regime study over the crash-layer ledger.

For each CRASH trade in trades_7y.csv, align entry-day macro indicators and
profile whether any slow-variable regime (rates / dollar / risk-appetite)
separates stop-loss trades from winners.

Pure-data study. Does NOT touch app.py or the harness. V9 baseline untouched.
"""
import os, argparse, csv
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SRC_DIR, 'sim_reports_full')
TRADES = os.path.join(OUT_DIR, 'trades_7y.csv')

MACRO = {
    'TNX': '^TNX',          # 10Y Treasury yield
    'FVX': '^FVX',          # 5Y
    'IRX': '^IRX',          # 13W T-bill
    'DXY': 'DX-Y.NYB',      # Dollar index
    'SPX': '^SPX',          # S&P 500
    'GLD': 'GLD',           # Gold
    'HYG': 'HYG',           # High yield corporate
}

ap = argparse.ArgumentParser()
ap.add_argument('--no-net', action='store_true', help='use cached CSV only, no downloads')
ap.add_argument('--period', default='7y')
args = ap.parse_args()

import yfinance as yf

def _cols(df):
    if df is None or df.empty:
        return pd.DataFrame()
    return df[['Open', 'Close']]

def _load(tk, period):
    csvp = os.path.join(OUT_DIR, f'_macro_{tk}_{period}.csv')
    if args.no_net and os.path.exists(csvp):
        df = pd.read_csv(csvp, index_col=0, parse_dates=True)
    else:
        df = _cols(yf.Ticker(MACRO[tk]).history(period=period, interval='1d', auto_adjust=False)).sort_index()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_csv(csvp)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()

macro = {}
for tk in MACRO:
    macro[tk] = _load(tk, args.period)
    if getattr(macro[tk].index, 'tz', None) is not None:
        macro[tk].index = macro[tk].index.tz_localize(None)
    print(f"  {tk:6} {MACRO[tk]:14} {len(macro[tk]):>4} rows {macro[tk].index[0].date()}→{macro[tk].index[-1].date()}")

trades = [r for r in csv.DictReader(open(TRADES)) if r['kind'] == 'CRASH']
trades.sort(key=lambda r: r['open'])
print(f"CRASH trades: {len(trades)}")

# ── feature engineering at entry date ────────────────────────────────
def _series(tk):
    return macro[tk]['Close']

def entry_row(t, tk, col='Close'):
    s = _series(tk)
    m = s[s.index <= pd.Timestamp(t).tz_localize(None) if not hasattr(macro[tk].index, 'tz') else s.index <= t]
    return float(m.iloc[-1]) if len(m) else None

rows = []
for r in trades:
    t0 = pd.Timestamp(r['open']).tz_localize(None)
    d = {'n': r['n'], 'open': r['open'], 'pnl': float(r['total_pnl']),
         'opt': float(r['opt_pnl']), 'etf': float(r['etf_pnl']),
         'res': r['result'], 'is_stop': r['result'].startswith('止损')}

    def _at(s):
        s = s.tz_localize(None) if getattr(s.index, 'tz', None) is not None else s
        m = s[s.index <= t0]
        return m

    tnx = _at(_series('TNX')); fvx = _at(_series('FVX')); irx = _at(_series('IRX'))
    dxy = _at(_series('DXY')); spx = _at(_series('SPX')); gld = _at(_series('GLD')); hyg = _at(_series('HYG'))
    m10 = tnx; m5 = fvx; m13 = irx
    md = dxy; ms = spx
    if len(m10) and len(m13):
        d['y10'] = float(m10.iloc[-1]); d['y13'] = float(m13.iloc[-1])
        d['curve'] = d['y10'] - d['y13']                       # 10Y-13W slope (2s10s-ish proxy)
        if len(m10) >= 20: d['y10_20d'] = float(m10.iloc[-1]) - float(m10.iloc[-20])
    if len(md):
        d['dxy'] = float(md.iloc[-1])
        if len(md) >= 20: d['dxy_20d'] = (float(md.iloc[-1]) / float(md.iloc[-20]) - 1) * 100
        if len(md) >= 60: d['dxy_60d'] = (float(md.iloc[-1]) / float(md.iloc[-60]) - 1) * 100
    if len(ms):
        d['spx'] = float(ms.iloc[-1])
        sma200 = ms.rolling(200).mean()
        s200 = sma200[sma200.index <= t0]
        if len(s200): d['spx_200d'] = (float(ms.iloc[-1]) / float(s200.iloc[-1]) - 1) * 100
    gh = hyg[hyg.index <= t0]; gg = gld[gld.index <= t0]
    if len(gh) and len(gg): d['hyg_gld'] = float(gh.iloc[-1]) / float(gg.iloc[-1])
    rows.append(d)

df = pd.DataFrame(rows)
out = os.path.join(OUT_DIR, 'macro_regime_trades.csv')
df.to_csv(out, index=False)
print(f"wrote {out}  ({len(df)} rows)")

# ── profile: stop-rate / avg PnL per regime bucket ───────────────────
def prof(df, key, lo, hi, lab):
    g = df[df[key].notna() & (df[key] >= lo) & (df[key] < hi)]
    if len(g) == 0:
        return
    print(f"  {lab:<28} n={len(g):>3} stop {g['is_stop'].mean()*100:>4.0f}% | avg {g['pnl'].mean():>+7.0f} | sum {g['pnl'].sum():>+8.0f}")

def buckets(key, edges, name):
    print(f"\n-- {name} ({key}) --")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        lab = f"[{lo}, {hi})" if i < len(edges) - 2 else f"≥{lo}"
        prof(df, key, lo, hi, lab)
    # overall
    g = df[df[key].notna()]
    print(f"  全体                              n={len(g):>3} stop {g['is_stop'].mean()*100:>4.0f}% | avg {g['pnl'].mean():>+7.0f} | sum {g['pnl'].sum():>+8.0f}")

buckets('curve', [-np.inf, -0.5, 0, 0.5, np.inf], '利率曲线 10Y-13W(%)')
buckets('y10', [-np.inf, 2, 3, 4, 5, np.inf], '10Y 绝对水平(%)')
buckets('y10_20d', [-np.inf, -0.2, 0, 0.2, 0.4, np.inf], '10Y 20d变化(pp)')
buckets('dxy_20d', [-np.inf, -1, 0, 1, 2, np.inf], 'DXY 20d变化(%)')
buckets('dxy_60d', [-np.inf, -3, 0, 3, 6, np.inf], 'DXY 60d变化(%)')
buckets('spx_200d', [-np.inf, -20, -10, 0, 10, np.inf], 'SPX 距200d均(%)')
buckets('hyg_gld', [-np.inf, 0.14, 0.16, 0.18, np.inf], 'HYG/GLD 比')

# ── rank each macro feature by stop-rate separation ──────────────────
print("\n\n===== 特征区分度排名 (高开率 vs stop率 由数据主导) =====")
features = ['curve', 'y10', 'y10_20d', 'dxy_20d', 'dxy_60d', 'spx_200d', 'hyg_gld']
for key in features:
    g = df[df[key].notna()]
    if len(g) < 30:
        print(f"  {key:<10} n={len(g)} 样本不足")
        continue
    med = g[key].median()
    hi = g[g[key] > med]; lo = g[g[key] <= med]
    sep = abs(hi['is_stop'].mean() - lo['is_stop'].mean()) * 100
    print(f"  {key:<10} n={len(g):>3} 上组stop {hi['is_stop'].mean()*100:>4.0f}% PnL {hi['pnl'].sum():>+8.0f} | 下组stop {lo['is_stop'].mean()*100:>4.0f}% PnL {lo['pnl'].sum():>+8.0f} | 分离 {sep:.0f}pp")
