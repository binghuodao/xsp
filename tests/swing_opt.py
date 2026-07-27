"""Swing trade optimization: Stage 1 (ADX filter) + Stage 2 (confirm) + Stage 3 (VIX filter)."""
import sys, os, collections
import numpy as np, pandas as pd, yfinance as yf

def swing_bt(period='10y',
             signal_b=True, signal_c=True,
             adx_filter=0,     # 0=off, 20/25/30
             confirm='none',   # 'none','dual','double_day','price'
             vix_filter=0,     # 0=off, 20/25
             hold_days=3, trail_pct=0.01, stop_pct=0.03,
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

    s20 = xc.rolling(20).mean(); bs = xc.rolling(20).std()
    df['bbu'] = s20+2*bs; df['bbl'] = s20-2*bs
    df['bbp'] = (xc - df['bbl']) / (df['bbu'] - df['bbl'])
    tr = pd.concat([xh-xl,(xh-xc.shift(1)).abs(),(xl-xc.shift(1)).abs()],axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    up = xc.diff(); dn = -up
    pdm = pd.Series(np.where((up>dn)&(up>0), up, 0), index=xc.index)
    mdm = pd.Series(np.where((dn>up)&(dn>0), dn, 0), index=xc.index)
    a14 = tr.rolling(14).mean()
    pdi = 100*pdm.rolling(14).mean()/a14; mdi = 100*mdm.rolling(14).mean()/a14
    df['di_diff'] = (pdi-mdi)/100; df['di_diff_prev'] = df['di_diff'].shift(1).fillna(0)
    df['adx'] = 100*abs(pdi-mdi)/(pdi+mdi).rolling(14).mean()
    # Fix ADX compute
    dx = 100*abs(pdi-mdi)/(pdi+mdi)
    df['adx'] = dx.rolling(14).mean()
    dlt = xc.diff(); gn = dlt.clip(lower=0); ls = (-dlt).clip(lower=0)
    ag = gn.rolling(14).mean(); al = ls.rolling(14).mean()
    df['rsi_14'] = (100-100/(1+ag/al.replace(0,np.nan))).fillna(50)
    df['vix'] = vc.values
    df['nt'] = df['price'] >= df['bbu'] - df['atr_14']*0.60
    df['nb'] = df['price'] <= df['bbl'] + df['atr_14']*0.60
    df['chg1'] = xc.pct_change()
    df['close_gt_open'] = xc > xsp['Open']
    df['day_dir'] = np.sign(df['chg1'])
    # Previous day signal for double-day confirm
    df['prev_signal_short'] = 0; df['prev_signal_long'] = 0
    df = df.dropna().copy()

    # Track confirmation state
    prev_sig_short = 0; prev_sig_long = 0

    def gd(r):
        nonlocal prev_sig_short, prev_sig_long
        dd = r['di_diff']; ddp = r['di_diff_prev']
        rsi = r['rsi_14']; adx = r['adx']; vi = r['vix']
        bbp = r['bbp']; c1 = r['chg1']
        nt_r = r['nt']; nb_r = r['nb']
        close_gt_open = r['close_gt_open']

        # Raw signals
        sig_short = False; sig_long = False
        raw_short_reason = ''; raw_long_reason = ''

        # Signal B: DI cross
        if signal_b:
            if dd < 0 and ddp >= 0:
                sig_short = True; raw_short_reason = 'B_di'
            if dd > 0 and ddp <= 0:
                sig_long = True; raw_long_reason = 'B_di'

        # Signal C: BB edge + RSI
        if signal_c:
            if nt_r and rsi > 65:
                sig_short = True; raw_short_reason = 'C_edge'
            if nb_r and rsi < 35:
                sig_long = True; raw_long_reason = 'C_edge'

        # Stage 1: ADX filter - block counter-trend signals
        if adx_filter > 0 and adx > adx_filter:
            if dd > 0:  # strong uptrend → only long
                sig_short = False
            if dd < 0:  # strong downtrend → only short
                sig_long = False

        # Stage 3: VIX filter
        if vix_filter > 0:
            if vi > vix_filter:  # panic → only short
                sig_long = False
            if vi < (vix_filter - 10 if vix_filter == 25 else 15):  # complacent → only long
                # VIX < 15 or < 5 depending
                low_vix = 5 if vix_filter == 25 else 10
                if vi < low_vix:
                    sig_short = False

        # Stage 2: confirmation
        confirmed_short = sig_short; confirmed_long = sig_long

        if confirm == 'dual':
            # B and C must agree
            has_b_short = dd < 0 and ddp >= 0
            has_c_short = nt_r and rsi > 65
            has_b_long = dd > 0 and ddp <= 0
            has_c_long = nb_r and rsi < 35
            confirmed_short = (has_b_short and has_c_short)
            confirmed_long = (has_b_long and has_c_long)
            if confirmed_short: raw_short_reason = 'B+C_dual'
            if confirmed_long: raw_long_reason = 'B+C_dual'

        elif confirm == 'double_day':
            # Need same signal 2 days in a row
            confirmed_short = sig_short and prev_sig_short
            confirmed_long = sig_long and prev_sig_long
            if confirmed_short: raw_short_reason = raw_short_reason + '_2d'
            if confirmed_long: raw_long_reason = raw_long_reason + '_2d'

        elif confirm == 'price':
            # Price must confirm direction
            if sig_short:
                confirmed_short = not close_gt_open  # close < open (down day)
            if sig_long:
                confirmed_long = close_gt_open  # close > open (up day)

        # Update prev signals
        prev_sig_short = 1 if sig_short else 0
        prev_sig_long = 1 if sig_long else 0

        if confirmed_short:
            return 'SHORT', raw_short_reason
        if confirmed_long:
            return 'LONG', raw_long_reason
        return None, ''

    trades = []; pos = None
    for i in range(len(df)):
        row = df.iloc[i]; d, rr = gd(row); dt = df.index[i]

        # Opposite signal → exit
        if pos is not None and d is not None and d != pos['dir']:
            cs = float(row['price'])
            dr = cs / pos['pc'] - 1
            pos['ev'] *= (1 + lev*dr) if pos['dir'] == 'LONG' else (1 - lev*dr)
            trades.append({'entry_date': pos['ed'], 'exit_date': dt,
                           'dir': pos['dir'], 'entry_price': pos['ep'],
                           'exit_price': round(cs, 2), 'pnl': round(pos['ev']-5000, 2),
                           'exit_type': 'opposite_signal', 'entry_reason': pos.get('reason','')})
            pos = None

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
            if not ex:
                lw = float(row['low']); hh = float(row['high'])
                if pos['dir'] == 'LONG' and lw <= pos['ep']*(1-stop_pct):
                    xp=pos['ep']*(1-stop_pct); ex=True; xt='fixed_stop'
                if pos['dir'] == 'SHORT' and hh >= pos['ep']*(1+stop_pct):
                    xp=pos['ep']*(1+stop_pct); ex=True; xt='fixed_stop'
            if not ex and i - pos['ei'] >= hold_days:
                xp=cs; ex=True; xt='t+30'
            if ex:
                trades.append({'entry_date': pos['ed'], 'exit_date': dt,
                               'dir': pos['dir'], 'entry_price': pos['ep'],
                               'exit_price': round(xp, 2), 'pnl': round(pos['ev']-5000, 2),
                               'exit_type': xt, 'entry_reason': pos.get('reason','')})
                pos = None

        if pos is None and d is not None:
            pos = {'dir': d, 'ep': float(row['price']), 'ed': dt, 'ei': i,
                   'ev': 5000, 'pk': 5000, 'pc': float(row['price']),
                   'pp': float(row['price']), 'reason': rr}
            prev_sig_short = 0; prev_sig_long = 0

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
    return {'n':n,'wr':round(w/n*100,1),'pnl':pnl,
            'longs':longs,'shorts':shorts,'long_pnl':lp,'short_pnl':sp}

if __name__ == '__main__':
    devnull = open('/dev/null', 'w')
    old_out, old_err = sys.stdout, sys.stderr
    PERIODS = ['3y','5y','10y']
    ALL = {}

    # Baseline (B+C best from previous run)
    print('=== BASELINE: B+C h3 tr0.01 st0.03 ===')
    for p in PERIODS:
        sys.stdout, sys.stderr = devnull, devnull
        tr = swing_bt(period=p, signal_b=True, signal_c=True, hold_days=3, trail_pct=0.01, stop_pct=0.03)
        sys.stdout, sys.stderr = old_out, old_err
        s = summ(tr)
        ALL[f'BASELINE|{p}'] = s
        print(f'  {p}: {s["n"]}tr s{s["shorts"]}/l{s["longs"]} WR{s["wr"]}% PnL${s["pnl"]:.0f}')

    # STAGE 1: ADX filter
    print('\n=== STAGE 1: ADX filter ===')
    for af in [20, 25, 30]:
        for hd in [3, 5]:
            for tr in [0.01, 0.02]:
                for sp in [0.03]:
                    for p in PERIODS:
                        sys.stdout, sys.stderr = devnull, devnull
                        trs = swing_bt(period=p, adx_filter=af, hold_days=hd, trail_pct=tr, stop_pct=sp)
                        sys.stdout, sys.stderr = old_out, old_err
                        s = summ(trs)
                        key = f'S1_adx{af}_h{hd}_tr{tr:.2f}_st{sp:.2f}|{p}'
                        ALL[key] = s
                        if p == '10y':
                            print(f'  adx{af}_h{hd}_tr{tr:.2f}: {s["n"]}tr s{s["shorts"]}/l{s["longs"]} WR{s["wr"]}% PnL${s["pnl"]:.0f}')

    # STAGE 2: Confirmation modes
    print('\n=== STAGE 2: Confirmation ===')
    base_exit = [(3, 0.01), (3, 0.02), (5, 0.01)]
    for cm in ['dual', 'double_day', 'price']:
        for hd, tr in base_exit:
            for sp in [0.03]:
                for p in PERIODS:
                    sys.stdout, sys.stderr = devnull, devnull
                    trs = swing_bt(period=p, confirm=cm, hold_days=hd, trail_pct=tr, stop_pct=sp)
                    sys.stdout, sys.stderr = old_out, old_err
                    s = summ(trs)
                    key = f'S2_{cm}_h{hd}_tr{tr:.2f}_st{sp:.2f}|{p}'
                    ALL[key] = s
                    if p == '10y':
                        print(f'  {cm}_h{hd}_tr{tr:.2f}: {s["n"]}tr s{s["shorts"]}/l{s["longs"]} WR{s["wr"]}% PnL${s["pnl"]:.0f}')

    # STAGE 3: VIX filter
    print('\n=== STAGE 3: VIX filter ===')
    for vf in [20, 25]:
        for hd, tr in base_exit:
            for sp in [0.03]:
                for p in PERIODS:
                    sys.stdout, sys.stderr = devnull, devnull
                    trs = swing_bt(period=p, vix_filter=vf, hold_days=hd, trail_pct=tr, stop_pct=sp)
                    sys.stdout, sys.stderr = old_out, old_err
                    s = summ(trs)
                    key = f'S3_vix{vf}_h{hd}_tr{tr:.2f}_st{sp:.2f}|{p}'
                    ALL[key] = s
                    if p == '10y':
                        print(f'  vix{vf}_h{hd}_tr{tr:.2f}: {s["n"]}tr s{s["shorts"]}/l{s["longs"]} WR{s["wr"]}% PnL${s["pnl"]:.0f}')

    # COMBINED: best of each stage together
    print('\n=== COMBINED: Stage1+2+3 ===')
    # Pick the best from each stage and combine
    s1_best = max([(k, v) for k, v in ALL.items() if k.startswith('S1') and '10y' in k], key=lambda x: x[1]['pnl'])
    s2_best = max([(k, v) for k, v in ALL.items() if k.startswith('S2') and '10y' in k], key=lambda x: x[1]['pnl'])
    s3_best = max([(k, v) for k, v in ALL.items() if k.startswith('S3') and '10y' in k], key=lambda x: x[1]['pnl'])
    print(f'  Best S1: {s1_best[0]} PnL${s1_best[1]["pnl"]:.0f}')
    print(f'  Best S2: {s2_best[0]} PnL${s2_best[1]["pnl"]:.0f}')
    print(f'  Best S3: {s3_best[0]} PnL${s3_best[1]["pnl"]:.0f}')

    # Parse best configs
    def parse_key(k):
        parts = k.split('|')[0].split('_')
        kw = {'signal_b':True, 'signal_c':True, 'hold_days':3, 'trail_pct':0.01, 'stop_pct':0.03}
        for p in parts:
            if p.startswith('adx'): kw['adx_filter'] = int(p[3:])
            if p.startswith('dual'): kw['confirm'] = 'dual'
            if p.startswith('double_day'): kw['confirm'] = 'double_day'
            if p.startswith('price'): kw['confirm'] = 'price'
            if p.startswith('vix'): kw['vix_filter'] = int(p[3:])
            if p.startswith('h'): kw['hold_days'] = int(p[1:])
            if p.startswith('tr'): kw['trail_pct'] = float(p[2:])
            if p.startswith('st'): kw['stop_pct'] = float(p[2:])
        return kw

    # Try combining best S1 with best S2, best S1+S2, etc.
    combos_to_test = [
        ('S1+S2', {**parse_key(s1_best[0]), **{k:v for k,v in parse_key(s2_best[0]).items() if k in ('confirm',)}}),
        ('S1+S3', {**parse_key(s1_best[0]), **{k:v for k,v in parse_key(s3_best[0]).items() if k in ('vix_filter',)}}),
        ('S2+S3', {**parse_key(s2_best[0]), **{k:v for k,v in parse_key(s3_best[0]).items() if k in ('vix_filter',)}}),
        ('S1+S2+S3', {**parse_key(s1_best[0]), **{k:v for k,v in parse_key(s2_best[0]).items() if k in ('confirm',)},
                       **{k:v for k,v in parse_key(s3_best[0]).items() if k in ('vix_filter',)}}),
    ]
    for label, kw in combos_to_test:
        for p in PERIODS:
            sys.stdout, sys.stderr = devnull, devnull
            trs = swing_bt(period=p, **kw)
            sys.stdout, sys.stderr = old_out, old_err
            s = summ(trs)
            key = f'{label}|{p}'
            ALL[key] = s
            print(f'  {label}_{p}: {s["n"]}tr s{s["shorts"]}/l{s["longs"]} WR{s["wr"]}% PnL${s["pnl"]:.0f}')

    # FINAL RANKING (10y only)
    print('\n========== FINAL 10y RANKING ==========')
    base_pnl = ALL.get('BASELINE|10y', {}).get('pnl', 0)
    print(f'  Baseline (B+C h3 tr0.01): ${base_pnl:.0f}')
    rows = [(k, s) for k, s in ALL.items() if '10y' in k]
    rows.sort(key=lambda x: -x[1]['pnl'])
    seen = set()
    for k, s in rows:
        if k in seen: continue
        seen.add(k)
        delta = s['pnl'] - base_pnl
        print(f'  {k:55s} PnL${s["pnl"]:+.0f} ({delta:+.0f}) '
              f'{s["n"]}tr s{s["shorts"]}/l{s["longs"]} WR{s["wr"]}%')

    # PER-TRADE: Best overall config
    best_key = max([(k, v) for k, v in ALL.items() if '10y' in k and not k.startswith('BASE')],
                   key=lambda x: x[1]['pnl'])[0]
    print(f'\n========== PER-TRADE: {best_key} ==========')
    kw = parse_key(best_key)
    sys.stdout, sys.stderr = devnull, devnull
    trs = swing_bt(period='10y', **kw)
    sys.stdout, sys.stderr = old_out, old_err
    s = summ(trs)
    print(f'  Total: {len(trs)}tr PnL${s["pnl"]:.0f} WR{s["wr"]}% L{s["longs"]}+S{s["shorts"]}')

    # By year
    by_year = collections.defaultdict(list)
    for t in trs:
        yr = t['entry_date'].year if hasattr(t['entry_date'], 'year') else 0
        by_year[yr].append(t)
    for yr in sorted(by_year.keys()):
        yt = by_year[yr]; ypnl = sum(t['pnl'] for t in yt)
        yw = sum(1 for t in yt if t['pnl']>0)/len(yt)*100
        shorts = sum(1 for t in yt if t['dir']=='SHORT')
        longs = sum(1 for t in yt if t['dir']=='LONG')
        print(f'  {yr}: {len(yt)}tr S{shorts}+L{longs} WR{yw:.0f}% PnL${ypnl:+.0f}')
        for t in yt[:5]:
            ed = t['entry_date'].strftime('%m/%d') if hasattr(t['entry_date'],'strftime') else str(t['entry_date'])
            print(f'    {ed} {t["dir"]:5s} ep={t["entry_price"]:.0f} pnl=${t["pnl"]:+.0f} '
                  f'reason={t["entry_reason"]} exit={t["exit_type"]}')

    # Specific: Jul 2026
    print('\n--- Jul 2026 detail ---')
    jul = [t for t in trs if hasattr(t['entry_date'],'month') and t['entry_date'].month==7 and t['entry_date'].year==2026]
    for t in jul:
        ed = t['entry_date'].strftime('%m/%d')
        print(f'  {ed} {t["dir"]:5s} ep={t["entry_price"]:.0f} pnl=${t["pnl"]:+.0f} '
              f'reason={t["entry_reason"]} exit={t["exit_type"]}')

    devnull.close()
