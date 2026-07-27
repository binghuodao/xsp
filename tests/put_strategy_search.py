"""PUT strategy search: DI crossover + near-top fade + combined."""
import os, sys, collections, json
import numpy as np, pandas as pd, yfinance as yf

SRC = os.path.dirname(os.path.abspath(__file__))

def run_bt(period='10y',
           # CALL params (fixed baseline)
           call_trail=0.035, call_hold=30, call_lev=3, call_stop=0.05, call_nb_stop=0.01,
           # PUT entry method: 'di_cross', 'near_top', 'both'
           put_method='none',
           # PUT entry thresholds
           put_score=30, put_vix=15, put_hold=14, put_trail=0.10, put_stop=0.10, put_lev=1.5,
           # Method B: near-top fade
           top_rsi=65,
           # Force no PUT (for baseline)
           call_only=False):
    if call_only: put_method = 'none'
    np.random.seed(0)
    xsp = yf.download('^XSP', period=period, interval='1d', progress=False)
    if isinstance(xsp.columns, pd.MultiIndex): xsp = xsp.droplevel('Ticker', axis=1)
    vix = yf.download('^VIX', period=period, interval='1d', progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix = vix.droplevel('Ticker', axis=1)
    vc = vix['Close'].reindex(xsp.index).ffill()
    df = pd.DataFrame(index=xsp.index)
    xc = xsp['Close']; xh = xsp['High']; xl = xsp['Low']
    df['price'] = xc; df['high'] = xh; df['low'] = xl
    s20 = xc.rolling(20).mean(); bs = xc.rolling(20).std()
    df['bbu'] = s20 + 2*bs; df['bbl'] = s20 - 2*bs; df['sma50'] = xc.rolling(50).mean()
    tr = pd.concat([xh-xl, (xh-xc.shift(1)).abs(), (xl-xc.shift(1)).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    up = xc.diff(); dn = -up
    pdm = pd.Series(np.where((up>dn)&(up>0), up, 0), index=xc.index)
    mdm = pd.Series(np.where((dn>up)&(dn>0), dn, 0), index=xc.index)
    a14 = tr.rolling(14).mean()
    pdi = 100*pdm.rolling(14).mean()/a14; mdi = 100*mdm.rolling(14).mean()/a14
    dx = 100*abs(pdi-mdi)/(pdi+mdi)
    df['adx'] = dx.rolling(14).mean()
    df['di_diff'] = (pdi-mdi)/100; df['di_diff_prev'] = df['di_diff'].shift(1).fillna(0)
    chg = xc.diff(10).abs(); vol = xc.diff().abs().rolling(10).sum()
    df['er'] = (chg/vol).fillna(0)
    h10 = xc.rolling(10).max(); l10 = xc.rolling(10).min()
    vr = ((xc-l10)/(h10-l10).replace(0, np.nan)).fillna(0.5); df['vr'] = vr.clip(0,1)*2
    df['vix_percentile'] = vc.rank(pct=True)*100; df['vix'] = vc.values
    dlt = xc.diff(); gn = dlt.clip(lower=0); ls = (-dlt).clip(lower=0)
    ag = gn.rolling(14).mean(); al = ls.rolling(14).mean()
    df['rsi_14'] = (100-100/(1+ag/al.replace(0, np.nan))).fillna(50)

    def cs_call(r):
        s = 0; s += 20 if r['adx']>=25 else 10 if r['adx']>=20 else 0
        s += 20 if r['er']>=0.60 else 10 if r['er']>=0.45 else 0
        s += 20 if r['vr']>=1.2 else 10 if r['vr']>=1.0 else 0
        s += 20 if r['rsi_14']>=60 else 10 if r['rsi_14']>=55 else 0
        return round(min(s,100)*0.4+(r['adx']/60 if r['adx']>0 else 0)*100*0.3+(s/100)*30)
    def cs_put(r):
        s = 0; s += 20 if r['adx']>=25 else 10 if r['adx']>=20 else 0
        ea = abs(r['er']); s += 20 if ea>=0.60 else 10 if ea>=0.45 else 0
        v2 = 2-r['vr']; s += 20 if v2>=1.2 else 10 if v2>=1.0 else 0
        r2 = 100-r['rsi_14']; s += 20 if r2>=60 else 10 if r2>=55 else 0
        return round(min(s,100)*0.4+(r['adx']/60 if r['adx']>0 else 0)*100*0.3+(s/100)*30)
    df['score_call'] = df.apply(cs_call, axis=1); df['score_put'] = df.apply(cs_put, axis=1)
    df = df.dropna().copy()

    def gd(r):
        nt = r['price'] >= r['bbu']-r['atr_14']*0.60 if pd.notna(r['bbu']) else False
        nb = r['price'] <= r['bbl']+r['atr_14']*0.60 if pd.notna(r['bbl']) else False
        nbo = nt or nb; sc_c = r['score_call']; sc_p = r['score_put']
        dd = r['di_diff']; vp = r['vix_percentile']

        # L1 mid-BB zone (not near any BB edge)
        if not nbo:
            # CALL: di2days + SMA50 + score
            if dd > 0 and r['price'] > r['sma50'] and sc_c >= 50:
                if r['di_diff_prev'] <= 0: return None, 'L1_trend'
                return 'CALL', 'L1_trend'
            # PUT: DI cross (first day DI turns negative)
            if put_method in ('di_cross','both') and dd < 0 and sc_p >= put_score:
                if r['di_diff_prev'] >= 0 and r['vix'] > put_vix and r['price'] < r['sma50']:
                    return 'PUT', 'L1_di_cross'
            return None, 'L1_BB_mid'

        # L2: near-BB + high VIX
        if nt and sc_c >= 50 and vp > 75:
            if put_method in ('near_top','both') and sc_p >= put_score and r['rsi_14'] > top_rsi and r['vix'] < 20:
                return 'PUT', 'L2_nearbb_vix_put'
            return None, 'L2_nearbb_vix'
        if nb and sc_c >= 50 and vp > 75: return 'CALL', 'L2_nearbb_vix'

        # L3: near-top + dd>0 (conflict: price at top, still trending up)
        if nt and dd > 0:
            if put_method in ('near_top','both') and sc_p >= put_score and r['rsi_14'] > top_rsi and r['vix'] < 20:
                return 'PUT', 'L3_conflict_put'
            return None, 'L3_conflict'

        # L4: near-BB (single band), no other condition
        if nt and sc_c >= 50:
            if put_method in ('near_top','both') and sc_p >= put_score and r['rsi_14'] > top_rsi and r['vix'] < 20:
                return 'PUT', 'L4_nearbb_put'
            return None, 'L4_nearbb'
        if nb and sc_c >= 35 and (r['adx']<25 or r['rsi_14']<35): return 'CALL', 'L4_nearbb'

        if nt or nb: return None, 'L0_bbedge'
        return None, 'L0_mid'

    trades = []; pos = None
    for i in range(len(df)):
        row = df.iloc[i]; d, rr = gd(row); dt = df.index[i]
        is_nb = rr.startswith('L2') or rr.startswith('L4')
        # Exit
        if pos is not None and i > pos['ei']:
            ex = False; xp = None; xt = ''; cs = float(row['price'])
            dr = cs / pos['pc'] - 1
            lev = call_lev if pos['dir'] == 'CALL' else put_lev
            pos['ev'] *= (1 + lev*dr) if pos['dir'] == 'CALL' else (1 - lev*dr)
            pos['pc'] = cs
            if pos['dir'] == 'CALL':
                pos['pp'] = max(pos['pp'], cs)
                if cs <= pos['pp'] * (1 - call_trail): xp=cs; ex=True; xt='trail'
            else:
                pos['pp'] = min(pos['pp'], cs)
                if cs >= pos['pp'] * (1 + put_trail): xp=cs; ex=True; xt='trail'
            if not ex:
                lw = float(row['low']); hh = float(row['high'])
                if pos['dir'] == 'CALL':
                    sp = call_nb_stop if pos['nb'] else call_stop
                    if lw <= pos['ep']*(1-sp): xp=pos['ep']*(1-sp); ex=True; xt='fixed_stop'
                else:
                    sp = call_nb_stop if pos['nb'] else put_stop
                    if hh >= pos['ep']*(1+sp): xp=pos['ep']*(1+sp); ex=True; xt='fixed_stop'
            hold = call_hold if pos['dir'] == 'CALL' else put_hold
            if not ex and i - pos['ei'] >= hold: xp=cs; ex=True; xt='t+30'
            if ex:
                trades.append({
                    'entry_date': pos['ed'], 'exit_date': dt,
                    'dir': pos['dir'], 'entry_price': pos['ep'], 'exit_price': round(xp, 2),
                    'pnl': round(pos['ev'] - 5000, 2), 'exit_type': xt,
                    'entry_reason': pos.get('reason', ''),
                })
                pos = None
        # Entry
        if pos is None and d is not None:
            pos = {'dir': d, 'ep': float(row['price']), 'ed': dt, 'ei': i,
                   'ev': 5000, 'pk': 5000, 'pc': float(row['price']), 'pp': float(row['price']),
                   'nb': is_nb, 'reason': rr}
    return trades


def summary(trades):
    n = len(trades)
    if n == 0: return {'n':0,'wr':0,'pnl':0,'avg':0,'maxdd':0,'puts':0,'calls':0,'reasons':''}
    w = sum(1 for t in trades if t['pnl'] > 0)
    pnl = sum(t['pnl'] for t in trades)
    puts = sum(1 for t in trades if t['dir'] == 'PUT')
    calls = sum(1 for t in trades if t['dir'] == 'CALL')
    put_pnl = sum(t['pnl'] for t in trades if t['dir'] == 'PUT')
    call_pnl = sum(t['pnl'] for t in trades if t['dir'] == 'CALL')
    md = min(t['pnl'] for t in trades)
    reas = collections.Counter(t['entry_reason'] for t in trades)
    reas_s = ', '.join(f'{k}:{v}' for k,v in sorted(reas.items()))
    return {'n':n,'wr':round(w/n*100,1),'pnl':pnl,'avg':round(pnl/n,2),
            'maxdd':md,'puts':puts,'calls':calls,'put_pnl':put_pnl,
            'call_pnl':call_pnl,'reasons':reas_s}


def print_trades(trades, label=''):
    print(f'\n--- {label} ---')
    if not trades: print('  (none)')
    for t in trades:
        ed = t['entry_date']
        if hasattr(ed, 'strftime'): ed = ed.strftime('%Y-%m-%d')
        print(f"  {ed} {t['dir']:5s} ep={t['entry_price']:.1f} "
              f"pnl=${t['pnl']:+.0f} exit={t['exit_type']:12s} reason={t['entry_reason']}")


# ================================================================
# TEST RUNNER
# ================================================================
RESULTS = {}

def test(label, **kw):
    for p in ['3y','5y','10y']:
        try:
            trades = run_bt(period=p, **kw)
            s = summary(trades)
            key = f'{label}|{p}'
            RESULTS[key] = {'trades':trades, 'summary':s, 'params':kw}
        except Exception as e:
            print(f'  ERROR {label} {p}: {e}')

if __name__ == '__main__':
    devnull = open('/dev/null', 'w')
    old_out, old_err = sys.stdout, sys.stderr

    # ─── BASELINE (pure CALL) ───
    print('\n========== BASELINE: pure CALL ==========')
    for p in ['3y','5y','10y']:
        sys.stdout, sys.stderr = devnull, devnull
        tr = run_bt(period=p, call_only=True)
        sys.stdout, sys.stderr = old_out, old_err
        s = summary(tr)
        print(f'  {p}: {s["n"]}tr WR{s["wr"]}% PnL${s["pnl"]:.0f}')
        RESULTS[f'baseline|{p}'] = {'trades':tr, 'summary':s}

    # ─── METHOD A: DI cross ───
    # First pass: broad scan over key params
    print('\n========== METHOD A: DI crossover (first pass) ==========')
    a_combos = [
        (30, 15, 1.0, 0.10, 14), (35, 15, 1.0, 0.10, 14),
        (30, 15, 1.5, 0.10, 14), (30, 15, 1.0, 0.07, 14),
        (30, 15, 1.0, 0.10, 21), (30, 18, 1.0, 0.10, 14),
        (30, 15, 1.0, 0.15, 14), (30, 15, 1.0, 0.10, 30),
    ]
    for ps, vx, pl, pt, ph in a_combos:
        sys.stdout, sys.stderr = devnull, devnull
        tr = run_bt(period='10y', put_method='di_cross', put_score=ps, put_vix=vx,
                    put_lev=pl, put_trail=pt, put_hold=ph, put_stop=0.10)
        sys.stdout, sys.stderr = old_out, old_err
        s = summary(tr)
        label = f'A_sc{ps}_vx{vx}_lev{pl}_tr{pt}_h{ph}'
        print(f'  {label}: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
              f'WR{s["wr"]}% PnL${s["pnl"]:.0f} (PUT${s["put_pnl"]:.0f}+CALL${s["call_pnl"]:.0f})')
        RESULTS[label] = {'trades':tr, 'summary':s}

    # ─── METHOD B: near-top fade ───
    print('\n========== METHOD B: near-top fade (first pass) ==========')
    b_combos = [
        (30, 65, 0.5, 0.05, 10), (35, 65, 0.5, 0.05, 10),
        (30, 60, 0.5, 0.05, 10), (30, 70, 0.5, 0.05, 10),
        (30, 65, 1.0, 0.05, 10), (30, 65, 0.5, 0.07, 10),
        (30, 65, 0.5, 0.05, 14),
    ]
    for ps, rs, pl, pt, ph in b_combos:
        sys.stdout, sys.stderr = devnull, devnull
        tr = run_bt(period='10y', put_method='near_top', put_score=ps, top_rsi=rs,
                    put_lev=pl, put_trail=pt, put_hold=ph, put_stop=0.05)
        sys.stdout, sys.stderr = old_out, old_err
        s = summary(tr)
        label = f'B_sc{ps}_rs{rs}_lev{pl}_tr{pt}_h{ph}'
        print(f'  {label}: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
              f'WR{s["wr"]}% PnL${s["pnl"]:.0f} (PUT${s["put_pnl"]:.0f}+CALL${s["call_pnl"]:.0f})')
        RESULTS[label] = {'trades':tr, 'summary':s}

    # ─── METHOD C: both ───
    print('\n========== METHOD C: both ==========')
    c_combos = [
        (30, 15, 1.0, 0.10, 14, 65),
        (30, 15, 0.5, 0.10, 14, 65),
        (35, 15, 1.0, 0.10, 14, 65),
        (30, 18, 1.0, 0.10, 14, 65),
    ]
    for ps, vx, pl, pt, ph, rs in c_combos:
        sys.stdout, sys.stderr = devnull, devnull
        tr = run_bt(period='10y', put_method='both',
                    put_score=ps, put_vix=vx, put_lev=pl,
                    put_trail=pt, put_hold=ph, put_stop=0.10,
                    top_rsi=rs)
        sys.stdout, sys.stderr = old_out, old_err
        s = summary(tr)
        label = f'C_sc{ps}_vx{vx}_lev{pl}_tr{pt}_h{ph}_rs{rs}'
        print(f'  {label}: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
              f'WR{s["wr"]}% PnL${s["pnl"]:.0f} (PUT${s["put_pnl"]:.0f}+CALL${s["call_pnl"]:.0f})')
        RESULTS[label] = {'trades':tr, 'summary':s}

    # ─── ALL configs sorted ───
    print('\n========== ALL 10y results sorted by PnL ==========')
    baseline = RESULTS.get('baseline|10y', {}).get('summary', {}).get('pnl', 0)
    rows = [(k,v['summary']) for k,v in RESULTS.items()]
    rows.sort(key=lambda x: -x[1]['pnl'])
    print(f'  Baseline (pure CALL): PnL ${baseline:.0f}')
    for label, s in rows:
        delta = s['pnl'] - baseline
        print(f'  {label:45s} PnL${s["pnl"]:+.0f} ({delta:+.0f}) '
              f'{s["n"]}tr {s["puts"]}PUT+{s["calls"]}CALL WR{s["wr"]}%')

    # ─── SECOND PASS: expand promising directions ───
    print('\n========== SECOND PASS: expand best params ==========')
    a_best = {'put_score':35, 'put_vix':15, 'put_lev':1.0, 'put_trail':0.10, 'put_hold':14}
    # DI cross: lower VIX threshold to get more PUTs; vary hold/trail
    for vx in [12, 15]:
        for ph in [10, 14, 21]:
            for pt in [0.07, 0.10]:
                sys.stdout, sys.stderr = devnull, devnull
                tr = run_bt(period='10y', put_method='di_cross',
                            put_score=35, put_vix=vx, put_lev=1.0,
                            put_trail=pt, put_hold=ph, put_stop=0.10)
                sys.stdout, sys.stderr = old_out, old_err
                s = summary(tr)
                label = f'A2_sc35_vx{vx}_lev1.0_tr{pt}_h{ph}'
                print(f'  {label}: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
                      f'WR{s["wr"]}% PnL${s["pnl"]:.0f} (PUT${s["put_pnl"]:.0f}+CALL${s["call_pnl"]:.0f})')
                RESULTS[label] = {'trades':tr, 'summary':s}

    # Near-top: try score=25 for more PUTs with RS>70
    for ps in [25, 30]:
        for pl in [0.5, 1.0]:
            sys.stdout, sys.stderr = devnull, devnull
            tr = run_bt(period='10y', put_method='near_top',
                        put_score=ps, top_rsi=70, put_lev=pl,
                        put_trail=0.05, put_hold=10, put_stop=0.05)
            sys.stdout, sys.stderr = old_out, old_err
            s = summary(tr)
            label = f'B2_sc{ps}_rs70_lev{pl}_tr0.05_h10'
            print(f'  {label}: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
                  f'WR{s["wr"]}% PnL${s["pnl"]:.0f} (PUT${s["put_pnl"]:.0f}+CALL${s["call_pnl"]:.0f})')
            RESULTS[label] = {'trades':tr, 'summary':s}

    # Both: score=35, try VIX=12 for DI cross
    for vx in [12, 15]:
        for pl in [0.5, 1.0]:
            sys.stdout, sys.stderr = devnull, devnull
            tr = run_bt(period='10y', put_method='both',
                        put_score=35, put_vix=vx, put_lev=pl,
                        put_trail=0.10, put_hold=14, put_stop=0.10, top_rsi=70)
            sys.stdout, sys.stderr = old_out, old_err
            s = summary(tr)
            label = f'C2_sc35_vx{vx}_lev{pl}_tr0.1_h14_rs70'
            print(f'  {label}: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
                  f'WR{s["wr"]}% PnL${s["pnl"]:.0f} (PUT${s["put_pnl"]:.0f}+CALL${s["call_pnl"]:.0f})')
            RESULTS[label] = {'trades':tr, 'summary':s}

    # ─── THIRD PASS: test best on 3y and 5y ───
    print('\n========== THIRD PASS: best configs on 3y/5y/10y ==========')
    best_configs = [
        ('A_best', 'di_cross', {'put_score':35, 'put_vix':15, 'put_lev':1.0, 'put_trail':0.10, 'put_hold':14}),
        ('B_best', 'near_top', {'put_score':30, 'top_rsi':70, 'put_lev':0.5, 'put_trail':0.05, 'put_hold':10}),
        ('C_best', 'both',     {'put_score':35, 'put_vix':12, 'put_lev':1.0, 'put_trail':0.10, 'put_hold':14, 'top_rsi':70}),
    ]
    for cl, pm, pkw in best_configs:
        for p in ['3y','5y','10y']:
            sys.stdout, sys.stderr = devnull, devnull
            tr = run_bt(period=p, put_method=pm, **pkw)
            sys.stdout, sys.stderr = old_out, old_err
            s = summary(tr)
            print(f'  {cl}_{p}: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
                  f'WR{s["wr"]}% PnL${s["pnl"]:.0f} (PUT${s["put_pnl"]:.0f}+CALL${s["call_pnl"]:.0f})')
            RESULTS[f'{cl}|{p}'] = {'trades':tr, 'summary':s}

    # ─── FINAL RANKING ───
    print('\n========== FINAL RANKING (10y) ==========')
    baseline = RESULTS.get('baseline|10y', {}).get('summary', {}).get('pnl', 0)
    rows = [(k,v['summary']) for k,v in RESULTS.items()]
    rows.sort(key=lambda x: -x[1]['pnl'])
    print(f'  Baseline: ${baseline:.0f}')
    for label, s in rows[:20]:
        delta = s['pnl'] - baseline
        print(f'  {label:50s} PnL${s["pnl"]:+.0f} ({delta:+.0f}) '
              f'{s["n"]}tr {s["puts"]}PUT+{s["calls"]}CALL WR{s["wr"]}%')

    # ─── PER-TRADE: best configs ───
    print('\n========== PER-TRADE DETAIL ==========')
    for bl in ['A_best|10y', 'B_best|10y', 'C_best|10y']:
        if bl in RESULTS:
            tr = RESULTS[bl]['trades']
            puts = [t for t in tr if t['dir'] == 'PUT']
            calls = [t for t in tr if t['dir'] == 'CALL']
            print(f'\n--- {bl} ({len(tr)} tr: {len(puts)}PUT + {len(calls)}CALL) ---')
            for t in puts:
                ed = t['entry_date']
                if hasattr(ed, 'strftime'): ed = ed.strftime('%Y-%m-%d')
                print(f'  PUT {ed} ep={t["entry_price"]:.1f} pnl=${t["pnl"]:+.0f} '
                      f'exit={t["exit_type"]:10s} reason={t["entry_reason"]}')
            # Print PUT-adjacent context: what CALL trades were replaced by PUT
            base_trades = RESULTS.get('baseline|10y', {}).get('trades', [])
            base_dates = {t['entry_date'] for t in base_trades}
            new_dates = {t['entry_date'] for t in tr}
            lost_calls = base_dates - new_dates
            if lost_calls:
                print(f'  CALL trades blocked by PUTs ({len(lost_calls)}):')
                for t in base_trades:
                    if t['entry_date'] in lost_calls:
                        ed = t['entry_date']
                        if hasattr(ed, 'strftime'): ed = ed.strftime('%Y-%m-%d')
                        print(f'    {ed} ep={t["entry_price"]:.1f} pnl=${t["pnl"]:+.0f} reason={t["entry_reason"]}')

    devnull.close()
