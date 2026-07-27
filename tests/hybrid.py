"""Hybrid backtest: trend (T+30 CALL) takes priority, swing (B+C) fills gaps."""
import sys, os, collections
import numpy as np, pandas as pd, yfinance as yf

OUT = os.path.dirname(os.path.abspath(__file__))

def hybrid_bt(period='10y',
              # Trend params
              trend_hold=30, trend_trail=0.035,
              # Swing params
              swing_hold=5, swing_trail=0.01, swing_stop=0.03,
               swing_confirm='none',
               swing_adx_filter=0,
               swing_direction_filter=False,
               lev=3):
    np.random.seed(0)
    xsp = yf.download('^XSP', period=period, interval='1d', progress=False)
    if isinstance(xsp.columns, pd.MultiIndex): xsp = xsp.droplevel('Ticker', axis=1)
    vix = yf.download('^VIX', period=period, interval='1d', progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix = vix.droplevel('Ticker', axis=1)
    vc = vix['Close'].reindex(xsp.index).ffill()
    xc = xsp['Close']; xh = xsp['High']; xl = xsp['Low']; xo = xsp['Open']
    df = pd.DataFrame(index=xsp.index)
    df['price'] = xc; df['high'] = xh; df['low'] = xl

    s20 = xc.rolling(20).mean(); bs = xc.rolling(20).std()
    df['bbu'] = s20+2*bs; df['bbl'] = s20-2*bs; df['sma50'] = xc.rolling(50).mean()
    df['bbp'] = (xc - df['bbl']) / (df['bbu'] - df['bbl'])
    tr = pd.concat([xh-xl, (xh-xc.shift(1)).abs(), (xl-xc.shift(1)).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    up = xc.diff(); dn = -up
    pdm = pd.Series(np.where((up>dn)&(up>0), up, 0), index=xc.index)
    mdm = pd.Series(np.where((dn>up)&(dn>0), dn, 0), index=xc.index)
    a14 = tr.rolling(14).mean()
    pdi = 100*pdm.rolling(14).mean()/a14; mdi = 100*mdm.rolling(14).mean()/a14
    df['di_diff'] = (pdi-mdi)/100; df['di_diff_prev'] = df['di_diff'].shift(1).fillna(0)
    dx = 100*abs(pdi-mdi)/(pdi+mdi)
    df['adx'] = dx.rolling(14).mean()
    dlt = xc.diff(); gn = dlt.clip(lower=0); ls = (-dlt).clip(lower=0)
    ag = gn.rolling(14).mean(); al = ls.rolling(14).mean()
    df['rsi_14'] = (100-100/(1+ag/al.replace(0, np.nan))).fillna(50)
    df['vix'] = vc.values; df['vix_percentile'] = vc.rank(pct=True)*100
    df['nt'] = df['price'] >= df['bbu'] - df['atr_14']*0.60
    df['nb'] = df['price'] <= df['bbl'] + df['atr_14']*0.60
    df['chg1'] = xc.pct_change()

    # Trend scoring (from put_strategy_search)
    chg = xc.diff(10).abs(); vol = xc.diff().abs().rolling(10).sum()
    df['er'] = (chg/vol).fillna(0)
    h10 = xc.rolling(10).max(); l10 = xc.rolling(10).min()
    vr = ((xc-l10)/(h10-l10).replace(0, np.nan)).fillna(0.5); df['vr'] = vr.clip(0,1)*2

    def cs_call(r):
        s = 0; s += 20 if r['adx']>=25 else 10 if r['adx']>=20 else 0
        s += 20 if r['er']>=0.60 else 10 if r['er']>=0.45 else 0
        s += 20 if r['vr']>=1.2 else 10 if r['vr']>=1.0 else 0
        s += 20 if r['rsi_14']>=60 else 10 if r['rsi_14']>=55 else 0
        return round(min(s,100)*0.4+(r['adx']/60 if r['adx']>0 else 0)*100*0.3+(s/100)*30)

    df['score_call'] = df.apply(cs_call, axis=1)
    df = df.dropna().copy()

    # Confirmation state tracker
    prev_sig_short = 0; prev_sig_long = 0

    def gd(r):
        nonlocal prev_sig_short, prev_sig_long
        dd = r['di_diff']; ddp = r['di_diff_prev']
        sc_c = r['score_call']; rsi = r['rsi_14']
        adx = r['adx']; vi = r['vix']; vp = r['vix_percentile']
        nt_r = r['nt']; nb_r = r['nb']

        # ─── Trend entry signal ───
        trend_signal = None
        nbo = nt_r or nb_r
        if not nbo:
            if dd > 0 and r['price'] > r['sma50'] and sc_c >= 50:
                if ddp <= 0:
                    pass  # first DI>0 day → don't enter yet
                else:
                    trend_signal = ('CALL', 'L1_trend')
        if nb_r and sc_c >= 50 and vp > 75:
            trend_signal = ('CALL', 'L2_nearbb_vix')
        if nb_r and sc_c >= 35 and (adx < 25 or rsi < 35):
            trend_signal = ('CALL', 'L4_nearbb')

        # ─── Swing entry signals (B+C) ───
        sig_short = False; sig_long = False
        raw_short_reason = ''; raw_long_reason = ''

        # Signal B: DI cross
        if dd < 0 and ddp >= 0:
            sig_short = True; raw_short_reason = 'B_di'
        if dd > 0 and ddp <= 0:
            sig_long = True; raw_long_reason = 'B_di'

        # Signal C: BB edge + RSI
        if nt_r and rsi > 65:
            sig_short = True; raw_short_reason = 'C_edge'
        if nb_r and rsi < 35:
            sig_long = True; raw_long_reason = 'C_edge'

        # ADX filter for swing
        if swing_adx_filter > 0 and adx > swing_adx_filter:
            if dd > 0: sig_short = False
            if dd < 0: sig_long = False

        # Confirmation
        confirmed_short = sig_short; confirmed_long = sig_long
        if swing_confirm == 'double_day':
            confirmed_short = sig_short and prev_sig_short
            confirmed_long = sig_long and prev_sig_long
        elif swing_confirm == 'price':
            if sig_short and r['chg1'] < 0: confirmed_short = True
            else: confirmed_short = False
            if sig_long and r['chg1'] > 0: confirmed_long = True
            else: confirmed_long = False

        prev_sig_short = 1 if sig_short else 0
        prev_sig_long = 1 if sig_long else 0

        # Directional filter: only swing WITH the larger trend
        if swing_direction_filter:
            above_sma = r['price'] > r['sma50']
            if confirmed_long and not above_sma:
                confirmed_long = False
            if confirmed_short and above_sma:
                confirmed_short = False

        swing_signal = None
        if confirmed_short: swing_signal = ('SHORT', raw_short_reason)
        if confirmed_long: swing_signal = ('LONG', raw_long_reason)

        return trend_signal, swing_signal

    trades = []
    trend_pos = None  # {'dir':'CALL',...}
    swing_pos = None  # {'dir':'LONG'/'SHORT',...}

    for i in range(len(df)):
        row = df.iloc[i]; trend_sig, swing_sig = gd(row); dt = df.index[i]

        # ─── Trend position exit ───
        if trend_pos is not None and i > trend_pos['ei']:
            ex = False; xp = None; xt = ''; cs = float(row['price'])
            dr = cs / trend_pos['pc'] - 1
            trend_pos['ev'] *= 1 + lev*dr
            trend_pos['pc'] = cs
            trend_pos['pp'] = max(trend_pos['pp'], cs)
            if cs <= trend_pos['pp'] * (1 - trend_trail):
                xp = cs; ex = True; xt = 'trail'
            if not ex:
                lw = float(row['low'])
                if lw <= trend_pos['ep'] * (1 - 0.05):
                    xp = trend_pos['ep']*(1-0.05); ex = True; xt = 'fixed_stop'
            if not ex and i - trend_pos['ei'] >= trend_hold:
                xp = cs; ex = True; xt = 't+30'

            if ex:
                trades.append({
                    'entry_date': trend_pos['ed'], 'exit_date': dt,
                    'dir': trend_pos['dir'], 'type': 'trend',
                    'entry_price': trend_pos['ep'], 'exit_price': round(xp, 2),
                    'pnl': round(trend_pos['ev'] - 5000, 2),
                    'exit_type': xt, 'entry_reason': trend_pos.get('reason', ''),
                })
                trend_pos = None

        # ─── Swing position exit ───
        if swing_pos is not None and i > swing_pos['ei']:
            ex = False; xp = None; xt = ''; cs = float(row['price'])
            dr = cs / swing_pos['pc'] - 1
            swing_pos['ev'] *= (1 + lev*dr) if swing_pos['dir'] == 'LONG' else (1 - lev*dr)
            swing_pos['pc'] = cs

            # Opposite signal → force exit (before other exit checks)
            if swing_sig is not None and swing_sig[0] != swing_pos['dir']:
                xp = cs; ex = True; xt = 'opposite_signal'

            if not ex:
                if swing_pos['dir'] == 'LONG':
                    swing_pos['pp'] = max(swing_pos['pp'], cs)
                    if cs <= swing_pos['pp'] * (1 - swing_trail):
                        xp = cs; ex = True; xt = 'trail'
                else:
                    swing_pos['pp'] = min(swing_pos['pp'], cs)
                    if cs >= swing_pos['pp'] * (1 + swing_trail):
                        xp = cs; ex = True; xt = 'trail'

            if not ex:
                lw = float(row['low']); hh = float(row['high'])
                if swing_pos['dir'] == 'LONG' and lw <= swing_pos['ep']*(1-swing_stop):
                    xp = swing_pos['ep']*(1-swing_stop); ex = True; xt = 'fixed_stop'
                if swing_pos['dir'] == 'SHORT' and hh >= swing_pos['ep']*(1+swing_stop):
                    xp = swing_pos['ep']*(1+swing_stop); ex = True; xt = 'fixed_stop'

            if not ex and i - swing_pos['ei'] >= swing_hold:
                xp = cs; ex = True; xt = 't+30'

            if ex:
                trades.append({
                    'entry_date': swing_pos['ed'], 'exit_date': dt,
                    'dir': swing_pos['dir'], 'type': 'swing',
                    'entry_price': swing_pos['ep'], 'exit_price': round(xp, 2),
                    'pnl': round(swing_pos['ev'] - 5000, 2),
                    'exit_type': xt, 'entry_reason': swing_pos.get('reason', ''),
                })
                swing_pos = None

        # ─── Trend entry (priority) ───
        if trend_pos is None and trend_sig is not None and trend_sig[0] is not None:
            # If swing is active, close it first
            if swing_pos is not None:
                cs = float(row['price'])
                dr = cs / swing_pos['pc'] - 1
                swing_pos['ev'] *= (1 + lev*dr) if swing_pos['dir'] == 'LONG' else (1 - lev*dr)
                trades.append({
                    'entry_date': swing_pos['ed'], 'exit_date': dt,
                    'dir': swing_pos['dir'], 'type': 'swing',
                    'entry_price': swing_pos['ep'], 'exit_price': round(cs, 2),
                    'pnl': round(swing_pos['ev'] - 5000, 2),
                    'exit_type': 'preempted_by_trend', 'entry_reason': swing_pos.get('reason', ''),
                })
                swing_pos = None
            trend_pos = {'dir': trend_sig[0], 'ep': float(row['price']), 'ed': dt, 'ei': i,
                         'ev': 5000, 'pk': 5000, 'pc': float(row['price']),
                         'pp': float(row['price']), 'reason': trend_sig[1]}

        # ─── Swing entry (only if no trend position) ───
        if trend_pos is None and swing_pos is None and swing_sig is not None and swing_sig[0] is not None:
            swing_pos = {'dir': swing_sig[0], 'ep': float(row['price']), 'ed': dt, 'ei': i,
                         'ev': 5000, 'pk': 5000, 'pc': float(row['price']),
                         'pp': float(row['price']), 'reason': swing_sig[1]}

    return trades


def summ(trades):
    n = len(trades)
    if n == 0: return {'n':0,'wr':0,'pnl':0}
    w = sum(1 for t in trades if t['pnl'] > 0)
    pnl = sum(t['pnl'] for t in trades)
    trend = [t for t in trades if t.get('type') == 'trend']
    swing = [t for t in trades if t.get('type') == 'swing']
    return {'n':n,'wr':round(w/n*100,1),'pnl':pnl,
            'trend':len(trend),'trend_pnl':sum(t['pnl'] for t in trend),
            'swing':len(swing),'swing_pnl':sum(t['pnl'] for t in swing),
            'shorts':sum(1 for t in swing if t['dir']=='SHORT'),
            'longs':sum(1 for t in swing if t['dir']=='LONG')}


if __name__ == '__main__':
    devnull = open('/dev/null', 'w')
    old_out, old_err = sys.stdout, sys.stderr
    PERIODS = ['3y','5y','10y']

    BASE_KW = {}  # default hybrid_bt kwargs

    # ─── Step 1: Compare trend_hold=30 vs 14 ───
    print('=== STEP 1: trend_hold=30 vs 14 ===')
    for label, kw in [
        ('trend30_swingS1_adx25_h5', {'swing_adx_filter':25, 'swing_hold':5}),
        ('trend14_swingS1_adx25_h5', {'trend_hold':14, 'swing_adx_filter':25, 'swing_hold':5}),
        ('trend30_swingB+C_baseline_h3', {}),
        ('trend14_swingB+C_baseline_h3', {'trend_hold':14}),
    ]:
        for p in PERIODS:
            sys.stdout, sys.stderr = devnull, devnull
            tr = hybrid_bt(period=p, **kw)
            sys.stdout, sys.stderr = old_out, old_err
            s = summ(tr)
            print(f'  {label}|{p:3s}: {s["n"]}tr PnL${s["pnl"]:.0f} '
                  f'(trend{s["trend"]}tr${s["trend_pnl"]:.0f}+swing{s["swing"]}tr${s["swing_pnl"]:.0f})')

    # ─── Step 2: Add directional filter ───
    print('\n=== STEP 2: directional filter (SMA50) ===')
    for label, kw in [
        ('trend30_nofilter_B+C_h3', {}),
        ('trend30_direction_B+C_h3', {'swing_direction_filter':True}),
        ('trend30_nofilter_S1_adx25_h5', {'swing_adx_filter':25, 'swing_hold':5}),
        ('trend30_direction_S1_adx25_h5', {'swing_adx_filter':25, 'swing_hold':5, 'swing_direction_filter':True}),
    ]:
        for p in PERIODS:
            sys.stdout, sys.stderr = devnull, devnull
            tr = hybrid_bt(period=p, **kw)
            sys.stdout, sys.stderr = old_out, old_err
            s = summ(tr)
            print(f'  {label}|{p:3s}: {s["n"]}tr PnL${s["pnl"]:.0f} '
                  f'(trend{s["trend"]}tr${s["trend_pnl"]:.0f}+swing{s["swing"]}tr${s["swing_pnl"]:.0f}) '
                  f'swing_short{s["shorts"]}+long{s["longs"]} WR{s["wr"]}%')

    # ─── Step 4: More swing configs in hybrid mode ───
    print('\n=== STEP 4: more swing configs in hybrid (trend30, no direciton) ===')
    for label, kw in [
        ('S1_adx25_h5',           {'swing_adx_filter':25, 'swing_hold':5}),
        ('S1_adx20_h5',           {'swing_adx_filter':20, 'swing_hold':5}),
        ('S1_adx25_h3',           {'swing_adx_filter':25, 'swing_hold':3}),
        ('S2_double_day_h5',      {'swing_confirm':'double_day', 'swing_hold':5}),
        ('S2_double_day_h3',      {'swing_confirm':'double_day', 'swing_hold':3}),
        ('S1+S2_adx25_dd_h5',     {'swing_adx_filter':25, 'swing_confirm':'double_day', 'swing_hold':5}),
        ('B+C_baseline_h3',       {}),
        ('B+C_baseline_h5',       {'swing_hold':5}),
    ]:
        for p in PERIODS:
            sys.stdout, sys.stderr = devnull, devnull
            tr = hybrid_bt(period=p, **kw)
            sys.stdout, sys.stderr = old_out, old_err
            s = summ(tr)
            print(f'  {label}|{p:3s}: {s["n"]}tr PnL${s["pnl"]:.0f} '
                  f'(trend{s["trend"]}tr${s["trend_pnl"]:.0f}+swing{s["swing"]}tr${s["swing_pnl"]:.0f})')

    print('\n✅ Tests complete')
    devnull.close()
