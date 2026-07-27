"""Swing trade scan: BB-position + DI cross + BB-edge entries, short holds."""
import sys, os, collections
import numpy as np, pandas as pd, yfinance as yf

OUT = os.path.dirname(os.path.abspath(__file__))

def swing_bt(period='10y',
             # Entry signal config
             signal_a=False, signal_b=False, signal_c=False,
             a_bbp=0.60, a_mom=0.003,
             # Exit params
             hold_days=3, trail_pct=0.015, stop_pct=0.03,
             # Call-only baseline
             call_only=False,
             # Leverage
             lev=3):
    np.random.seed(0)
    xsp = yf.download('^XSP', period=period, interval='1d', progress=False)
    if isinstance(xsp.columns, pd.MultiIndex): xsp = xsp.droplevel('Ticker', axis=1)
    vix = yf.download('^VIX', period=period, interval='1d', progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix = vix.droplevel('Ticker', axis=1)
    vc = vix['Close'].reindex(xsp.index).ffill()
    xc = xsp['Close']; xh = xsp['High']; xl = xsp['Low']
    df = pd.DataFrame(index=xsp.index)
    df['price'] = xc; df['high'] = xh; df['low'] = xl

    # BB
    s20 = xc.rolling(20).mean(); bs = xc.rolling(20).std()
    df['bbu'] = s20 + 2*bs; df['bbl'] = s20 - 2*bs; df['bbm'] = s20
    df['bbp'] = (xc - df['bbl']) / (df['bbu'] - df['bbl'])

    # ATR
    tr = pd.concat([xh-xl, (xh-xc.shift(1)).abs(), (xl-xc.shift(1)).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()

    # ADX / DI
    up = xc.diff(); dn = -up
    pdm = pd.Series(np.where((up>dn)&(up>0), up, 0), index=xc.index)
    mdm = pd.Series(np.where((dn>up)&(dn>0), dn, 0), index=xc.index)
    a14 = tr.rolling(14).mean()
    pdi = 100*pdm.rolling(14).mean()/a14; mdi = 100*mdm.rolling(14).mean()/a14
    df['di_diff'] = (pdi-mdi)/100; df['di_diff_prev'] = df['di_diff'].shift(1).fillna(0)

    # RSI
    dlt = xc.diff(); gn = dlt.clip(lower=0); ls = (-dlt).clip(lower=0)
    ag = gn.rolling(14).mean(); al = ls.rolling(14).mean()
    df['rsi_14'] = (100-100/(1+ag/al.replace(0, np.nan))).fillna(50)

    # VIX
    df['vix'] = vc.values; df['vix_percentile'] = vc.rank(pct=True)*100

    # Short-term chg
    df['chg1'] = xc.pct_change()

    # Near-BB
    df['nt'] = df['price'] >= df['bbu'] - df['atr_14']*0.60
    df['nb'] = df['price'] <= df['bbl'] + df['atr_14']*0.60

    df = df.dropna().copy()

    def gd(r):
        bbp = r['bbp']; dd = r['di_diff']; ddp = r['di_diff_prev']
        rsi = r['rsi_14']; c1 = r['chg1']

        # Signal A: bbp + momentum
        if signal_a:
            if bbp > a_bbp and c1 < -a_mom:
                return 'SHORT', 'A_bbp_short'
            if bbp < (1-a_bbp) and c1 > a_mom:
                return 'LONG', 'A_bbp_long'

        # Signal B: DI cross
        if signal_b:
            if dd < 0 and ddp >= 0:
                return 'SHORT', 'B_di_cross'
            if dd > 0 and ddp <= 0:
                return 'LONG', 'B_di_cross'

        # Signal C: BB edge + RSI extreme
        if signal_c:
            if r['nt'] and rsi > 65:
                return 'SHORT', 'C_near_top'
            if r['nb'] and rsi < 35:
                return 'LONG', 'C_near_bot'

        return None, ''

    trades = []; pos = None
    for i in range(len(df)):
        row = df.iloc[i]; d, rr = gd(row); dt = df.index[i]

        # Check opposite signal → force exit
        if pos is not None and d is not None and d != pos['dir']:
            # Exit at close, enter new direction same day
            cs = float(row['price'])
            dr = cs / pos['pc'] - 1
            pos['ev'] *= (1 + lev*dr) if pos['dir'] == 'LONG' else (1 - lev*dr)
            pnl = pos['ev'] - 5000
            trades.append({'entry_date': pos['ed'], 'exit_date': dt,
                           'dir': pos['dir'], 'entry_price': pos['ep'],
                           'exit_price': round(cs, 2), 'pnl': round(pnl, 2),
                           'exit_type': 'opposite_signal', 'entry_reason': pos.get('reason','')})
            pos = None

        # Exit existing position
        if pos is not None and i > pos['ei']:
            ex = False; xp = None; xt = ''; cs = float(row['price'])
            dr = cs / pos['pc'] - 1
            pos['ev'] *= (1 + lev*dr) if pos['dir'] == 'LONG' else (1 - lev*dr)
            pos['pc'] = cs
            if pos['dir'] == 'LONG':
                pos['pp'] = max(pos['pp'], cs)
                if cs <= pos['pp'] * (1 - trail_pct): xp=cs; ex=True; xt='trail'
            else:
                pos['pp'] = min(pos['pp'], cs)
                if cs >= pos['pp'] * (1 + trail_pct): xp=cs; ex=True; xt='trail'
            # Fixed stop
            if not ex:
                lw = float(row['low']); hh = float(row['high'])
                if pos['dir'] == 'LONG' and lw <= pos['ep']*(1-stop_pct):
                    xp=pos['ep']*(1-stop_pct); ex=True; xt='fixed_stop'
                if pos['dir'] == 'SHORT' and hh >= pos['ep']*(1+stop_pct):
                    xp=pos['ep']*(1+stop_pct); ex=True; xt='fixed_stop'
            # Hold days
            if not ex and i - pos['ei'] >= hold_days:
                xp=cs; ex=True; xt='t+30'
            if ex:
                trades.append({'entry_date': pos['ed'], 'exit_date': dt,
                               'dir': pos['dir'], 'entry_price': pos['ep'],
                               'exit_price': round(xp, 2), 'pnl': round(pos['ev']-5000, 2),
                               'exit_type': xt, 'entry_reason': pos.get('reason','')})
                pos = None

        # Entry new position
        if pos is None and d is not None:
            pos = {'dir': d, 'ep': float(row['price']), 'ed': dt, 'ei': i,
                   'ev': 5000, 'pk': 5000, 'pc': float(row['price']),
                   'pp': float(row['price']), 'reason': rr}

    return trades


def summ(trades):
    n = len(trades)
    if n == 0: return {'n':0,'wr':0,'pnl':0,'longs':0,'shorts':0,'long_pnl':0,'short_pnl':0}
    w = sum(1 for t in trades if t['pnl'] > 0)
    pnl = sum(t['pnl'] for t in trades)
    longs = sum(1 for t in trades if t['dir'] == 'LONG')
    shorts = sum(1 for t in trades if t['dir'] == 'SHORT')
    lp = sum(t['pnl'] for t in trades if t['dir'] == 'LONG')
    sp = sum(t['pnl'] for t in trades if t['dir'] == 'SHORT')
    reas = collections.Counter(t['entry_reason'] for t in trades)
    return {'n':n,'wr':round(w/n*100,1),'pnl':pnl,
            'longs':longs,'shorts':shorts,'long_pnl':lp,'short_pnl':sp,
            'reasons':', '.join(f'{k}:{v}' for k,v in sorted(reas.items()))}


if __name__ == '__main__':
    devnull = open('/dev/null', 'w')
    old_out, old_err = sys.stdout, sys.stderr

    PERIODS = ['3y','5y','10y']
    ALL_RES = {}

    # Baseline (use put_strategy_search for T+30 CALL-only reference)
    print('=== BASELINE (T+30 CALL-only reference) ===')
    sys.path.insert(0, OUT)
    from put_strategy_search import run_bt as base_bt
    for p in PERIODS:
        sys.stdout, sys.stderr = devnull, devnull
        tr = base_bt(period=p, call_only=True, call_hold=30, call_trail=0.035)
        sys.stdout, sys.stderr = old_out, old_err
        s = summ(tr)
        print(f'  {p}: {s["n"]}tr WR{s["wr"]}% PnL${s["pnl"]:.0f}')
        ALL_RES[f'baseline|{p}'] = s

    # ─── SIGNAL A: bbp + momentum ───
    print('\n=== SIGNAL A: bbp + momentum ===')
    for bbp_th in [0.55, 0.60, 0.65, 0.70]:
        for mom in [0.002, 0.003, 0.005]:
            for hd in [2, 3, 5]:
                for tr in [0.01, 0.015]:
                    for sp in [0.02, 0.03]:
                        sys.stdout, sys.stderr = devnull, devnull
                        trs = swing_bt(period='10y', signal_a=True,
                                       a_bbp=bbp_th, a_mom=mom,
                                       hold_days=hd, trail_pct=tr, stop_pct=sp)
                        sys.stdout, sys.stderr = old_out, old_err
                        s = summ(trs)
                        label = f'A_bbp{bbp_th:.2f}_mom{mom:.3f}_h{hd}_tr{tr:.2f}_st{sp:.2f}'
                        ALL_RES[label] = s
                        print(f'  {label}: {s["n"]}tr s{s["shorts"]}/l{s["longs"]} WR{s["wr"]}% PnL${s["pnl"]:.0f}')

    # ─── SIGNAL B: DI cross ───
    print('\n=== SIGNAL B: DI cross ===')
    for hd in [2, 3, 5]:
        for tr in [0.01, 0.015]:
            for sp in [0.02, 0.03]:
                sys.stdout, sys.stderr = devnull, devnull
                trs = swing_bt(period='10y', signal_b=True,
                               hold_days=hd, trail_pct=tr, stop_pct=sp)
                sys.stdout, sys.stderr = old_out, old_err
                s = summ(trs)
                label = f'B_di_h{hd}_tr{tr:.2f}_st{sp:.2f}'
                ALL_RES[label] = s
                print(f'  {label}: {s["n"]}tr s{s["shorts"]}/l{s["longs"]} WR{s["wr"]}% PnL${s["pnl"]:.0f}')

    # ─── SIGNAL C: BB edge + RSI extreme ───
    print('\n=== SIGNAL C: BB edge + RSI extreme ===')
    for hd in [2, 3, 5]:
        for tr in [0.01, 0.015]:
            for sp in [0.02, 0.03]:
                sys.stdout, sys.stderr = devnull, devnull
                trs = swing_bt(period='10y', signal_c=True,
                               hold_days=hd, trail_pct=tr, stop_pct=sp)
                sys.stdout, sys.stderr = old_out, old_err
                s = summ(trs)
                label = f'C_edge_h{hd}_tr{tr:.2f}_st{sp:.2f}'
                ALL_RES[label] = s
                print(f'  {label}: {s["n"]}tr s{s["shorts"]}/l{s["longs"]} WR{s["wr"]}% PnL${s["pnl"]:.0f}')

    # ─── COMBINED: best A + best B + best C ───
    print('\n=== COMBINED: A+B+C ===')
    # Find best configs per signal type by 10y PnL
    best_a = max([(k,v['pnl'],v) for k,v in ALL_RES.items() if k.startswith('A_')], key=lambda x: x[1])
    best_b = max([(k,v['pnl'],v) for k,v in ALL_RES.items() if k.startswith('B_')], key=lambda x: x[1])
    best_c = max([(k,v['pnl'],v) for k,v in ALL_RES.items() if k.startswith('C_')], key=lambda x: x[1])
    print(f'  Best A: {best_a[0]} PnL${best_a[1]:.0f}')
    print(f'  Best B: {best_b[0]} PnL${best_b[1]:.0f}')
    print(f'  Best C: {best_c[0]} PnL${best_c[1]:.0f}')

    # Run each pair and all three together
    for sigs, slabel in [(['signal_a','signal_b'], 'A+B'),
                         (['signal_a','signal_c'], 'A+C'),
                         (['signal_b','signal_c'], 'B+C'),
                         (['signal_a','signal_b','signal_c'], 'A+B+C')]:
        kw = {s: True for s in sigs}
        for hd in [2, 3, 5]:
            for tr in [0.01, 0.015]:
                for sp in [0.02, 0.03]:
                    for p in PERIODS:
                        sys.stdout, sys.stderr = devnull, devnull
                        trs = swing_bt(period=p, hold_days=hd, trail_pct=tr, stop_pct=sp, **kw)
                        sys.stdout, sys.stderr = old_out, old_err
                        s = summ(trs)
                        label = f'{slabel}_h{hd}_tr{tr:.2f}_st{sp:.2f}|{p}'
                        ALL_RES[label] = s

    # ─── PRINT ALL 10y RESULTS ───
    print('\n========== ALL 10y RESULTS ==========')
    base_pnl = ALL_RES.get('baseline|10y', {}).get('pnl', 0)
    print(f'  Baseline (T+30 CALL): ${base_pnl:.0f}')
    rows = [(k,s) for k,s in ALL_RES.items() if '10y' in k]
    rows.sort(key=lambda x: -x[1]['pnl'])
    seen = set()
    for k, s in rows:
        if k in seen: continue
        if k.startswith('baseline'): continue
        seen.add(k)
        delta = s['pnl'] - base_pnl
        print(f'  {k:55s} PnL${s["pnl"]:+.0f} ({delta:+.0f}) '
              f'{s["n"]}tr s{s["shorts"]}/l{s["longs"]} WR{s["wr"]}%')

    # ─── PRINT BEST CONFIG PER-TRADE DETAIL ───
    print('\n========== PER-TRADE: BEST COMBOS ==========')
    # Hardcode best configs from results
    best_configs = [
        ('B+C_h3_tr0.01_st0.03', {'signal_b':True,'signal_c':True,'hold_days':3,'trail_pct':0.01,'stop_pct':0.03,'period':'10y'}),
        ('B+C_h5_tr0.01_st0.03', {'signal_b':True,'signal_c':True,'hold_days':5,'trail_pct':0.01,'stop_pct':0.03,'period':'10y'}),
        ('B+C_h3_tr0.01_st0.02', {'signal_b':True,'signal_c':True,'hold_days':3,'trail_pct':0.01,'stop_pct':0.02,'period':'10y'}),
    ]
    for label, kw in best_configs:
        sys.stdout, sys.stderr = devnull, devnull
        trs = swing_bt(period=kw['period'], signal_b=kw['signal_b'], signal_c=kw['signal_c'],
                       hold_days=kw['hold_days'], trail_pct=kw['trail_pct'], stop_pct=kw['stop_pct'])
        sys.stdout, sys.stderr = old_out, old_err
        s = summ(trs)
        print(f'\n--- {label}: {len(trs)}tr PnL${s["pnl"]:.0f} (L{s["longs"]}+S{s["shorts"]}) WR{s["wr"]}% ---')
        # Group by year
        by_year = collections.defaultdict(list)
        for t in trs:
            ed = t['entry_date']
            if hasattr(ed, 'strftime'): yr = ed.year
            else: yr = '?'
            by_year[yr].append(t)
        for yr in sorted(by_year.keys()):
            yt = by_year[yr]
            ypnl = sum(t['pnl'] for t in yt)
            yw = sum(1 for t in yt if t['pnl']>0)
            puts = sum(1 for t in yt if t['dir']=='SHORT')
            longs = sum(1 for t in yt if t['dir']=='LONG')
            print(f'  {yr}: {len(yt)}tr (SHORT{puts}+LONG{longs}) WR{yw/len(yt)*100:.0f}% PnL${ypnl:+.0f}')
            # Show a few notable trades
            for t in yt[:3]:
                ed = t['entry_date'].strftime('%Y-%m-%d') if hasattr(t['entry_date'],'strftime') else str(t['entry_date'])
                print(f'    {ed} {t["dir"]:5s} ep={t["entry_price"]:.1f} pnl=${t["pnl"]:+.0f} '
                      f'reason={t["entry_reason"]} exit={t["exit_type"]}')

    devnull.close()
