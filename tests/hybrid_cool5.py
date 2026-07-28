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
                swing_direction_filter=True,
               short_etf='SPXU'):  # 'SPXU'(3x bear), 'SDS'(2x bear), 'SH'(1x bear), 'none'(skip shorts)
    np.random.seed(0)
    xsp = yf.download('^XSP', period=period, interval='1d', progress=False)
    if isinstance(xsp.columns, pd.MultiIndex): xsp = xsp.droplevel('Ticker', axis=1)
    vix = yf.download('^VIX', period=period, interval='1d', progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix = vix.droplevel('Ticker', axis=1)
    vc = vix['Close'].reindex(xsp.index).ffill()
    spxl = yf.download('SPXL', period=period, interval='1d', progress=False)
    if isinstance(spxl.columns, pd.MultiIndex): spxl = spxl.droplevel('Ticker', axis=1)
    spxl_c = spxl['Close'].reindex(xsp.index).ffill()

    # Download bear ETF (for swing shorts). Use SPXU if 'none' just to avoid error — unused.
    bear_ticker = short_etf if short_etf != 'none' else 'SPXU'
    bear = yf.download(bear_ticker, period=period, interval='1d', progress=False)
    if isinstance(bear.columns, pd.MultiIndex): bear = bear.droplevel('Ticker', axis=1)
    bear_c = bear['Close'].reindex(xsp.index).ffill()
    skew = yf.download('^SKEW', period=period, interval='1d', progress=False)
    if isinstance(skew.columns, pd.MultiIndex): skew = skew.droplevel('Ticker', axis=1)
    skew_c = skew['Close'].reindex(xsp.index).ffill().fillna(146)
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
    df['skew'] = skew_c.values
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
    sma200_full = xc.rolling(200).mean()
    df['sma200'] = sma200_full.reindex(df.index)

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

        # Directional filter: no swing LONG when both short and long term are bearish
        if swing_direction_filter:
            if confirmed_long and r['price'] < r['sma50'] and r['price'] < r['sma200']:
                confirmed_long = False

        swing_signal = None
        if confirmed_short: swing_signal = ('SHORT', raw_short_reason)
        if confirmed_long: swing_signal = ('LONG', raw_long_reason)

        return trend_signal, swing_signal

    trades = []; last_di_exit_i = None
    trend_pos = None  # {'dir':'CALL',...}
    swing_pos = None  # {'dir':'LONG'/'SHORT',...}

    for i in range(len(df)):
        row = df.iloc[i]; trend_sig, swing_sig = gd(row); dt = df.index[i]
        dd = row['di_diff']; ddp = row['di_diff_prev']
        vi = float(row['vix']); vp = float(row['vix_percentile'])

        # ─── Trend position exit ───
        if trend_pos is not None and i > trend_pos['ei']:
            ex = False; xp = None; xt = ''; cs = float(row['price'])
            is_call = trend_pos['dir'] == 'CALL'

            # DI reversal exit, only during elevated VIX (bear market panic)
            if (vp > 80 or vi > 28):
                if (is_call and dd < 0 and ddp < 0) or (not is_call and dd > 0 and ddp > 0):
                    xp = cs; ex = True; xt = 'di_reversal'; last_di_exit_i = i

            if not ex and is_call:
                trend_pos['pp'] = max(trend_pos['pp'], cs)
                if cs <= trend_pos['pp'] * (1 - trend_trail):
                    xp = cs; ex = True; xt = 'trail'
            if not ex and not is_call:
                trend_pos['pp'] = min(trend_pos['pp'], cs)
                if cs >= trend_pos['pp'] * (1 + trend_trail):
                    xp = cs; ex = True; xt = 'trail'
            if not ex:
                lw = float(row['low']); hh = float(row['high'])
                if is_call and lw <= trend_pos['ep'] * (1 - 0.05):
                    xp = trend_pos['ep']*(1-0.05); ex = True; xt = 'fixed_stop'
                if not is_call and hh >= trend_pos['ep'] * (1 + 0.05):
                    xp = trend_pos['ep']*(1+0.05); ex = True; xt = 'fixed_stop'
            if not ex and i - trend_pos['ei'] >= trend_hold:
                xp = cs; ex = True; xt = 't+30'

            if ex:
                etf_exit = float(spxl_c.loc[df.index[i]]) if is_call else float(bear_c.loc[df.index[i]])
                pnl = round(5000 * (etf_exit / trend_pos['etf_entry'] - 1), 2)
                trades.append({
                    'entry_date': trend_pos['ed'], 'exit_date': dt,
                    'dir': trend_pos['dir'], 'type': 'trend',
                    'entry_price': trend_pos['ep'], 'exit_price': round(xp, 2),
                    'pnl': pnl,
                    'exit_type': xt, 'entry_reason': trend_pos.get('reason', ''),
                })
                trend_pos = None

        # ─── Swing position exit ───
        if swing_pos is not None and i > swing_pos['ei']:
            ex = False; xp = None; xt = ''; cs = float(row['price'])

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
                etf_exit = float(spxl_c.loc[df.index[i]]) if swing_pos['dir'] == 'LONG' else float(bear_c.loc[df.index[i]])
                pnl = round(5000 * (etf_exit / swing_pos['etf_entry'] - 1), 2)
                trades.append({
                    'entry_date': swing_pos['ed'], 'exit_date': dt,
                    'dir': swing_pos['dir'], 'type': 'swing',
                    'entry_price': swing_pos['ep'], 'exit_price': round(xp, 2),
                    'pnl': pnl,
                    'exit_type': xt, 'entry_reason': swing_pos.get('reason', ''),
                })
                swing_pos = None

        # ─── Trend entry (priority) ───
        if trend_pos is None and trend_sig is not None and trend_sig[0] is not None:
            # Cooling period: no re-enter for 5 days after DI reversal exit
            if last_di_exit_i is not None and i - last_di_exit_i <= 5:
                trend_sig = None
            # If swing is active, close it first
            if swing_pos is not None:
                etf_exit = float(spxl_c.loc[df.index[i]]) if swing_pos['dir'] == 'LONG' else float(bear_c.loc[df.index[i]])
                pnl = round(5000 * (etf_exit / swing_pos['etf_entry'] - 1), 2)
                trades.append({
                    'entry_date': swing_pos['ed'], 'exit_date': dt,
                    'dir': swing_pos['dir'], 'type': 'swing',
                    'entry_price': swing_pos['ep'], 'exit_price': round(float(row['price']), 2),
                    'pnl': pnl,
                    'exit_type': 'preempted_by_trend', 'entry_reason': swing_pos.get('reason', ''),
                })
                swing_pos = None
            trend_pos = {'dir': trend_sig[0], 'ep': float(row['price']), 'ed': dt, 'ei': i,
                         'pp': float(row['price']), 'reason': trend_sig[1],
                         'etf_entry': float(spxl_c.loc[df.index[i]]) if trend_sig[0] == 'CALL' else float(bear_c.loc[df.index[i]])}

        # ─── Swing entry (only if no trend position) ───
        if trend_pos is None and swing_pos is None and swing_sig is not None and swing_sig[0] is not None:
            if short_etf == 'none' and swing_sig[0] == 'SHORT':
                pass  # skip shorts when short_etf='none'
            else:
                swing_pos = {'dir': swing_sig[0], 'ep': float(row['price']), 'ed': dt, 'ei': i,
                             'pp': float(row['price']), 'reason': swing_sig[1],
                             'etf_entry': float(spxl_c.loc[df.index[i]]) if swing_sig[0] == 'LONG' else float(bear_c.loc[df.index[i]])}

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

    swing_base = {'swing_confirm': 'double_day', 'swing_hold': 5}

    def by_year(trades):
        grp = collections.defaultdict(list)
        for t in trades:
            yr = t['entry_date'].year if hasattr(t['entry_date'], 'strftime') else '?'
            grp[yr].append(t)
        return grp

    def yr_line(yt, label=''):
        n = len(yt)
        pnl = sum(t['pnl'] for t in yt)
        w = sum(1 for t in yt if t['pnl'] > 0)
        trend = sum(1 for t in yt if t.get('type') == 'trend')
        swing = sum(1 for t in yt if t.get('type') == 'swing')
        s = sum(1 for t in yt if t.get('type') == 'swing' and t['dir'] == 'SHORT')
        l = sum(1 for t in yt if t.get('type') == 'swing' and t['dir'] == 'LONG')
        label_part = f'{label}: ' if label else ''
        return f'  {label_part}{n:2d}tr PnL${pnl:>+6.0f} WR{w/n*100:.0f}% (趋势{trend}+摆动{swing} S{s}+L{l})'

    def print_yr_table(trades, header_prefix=''):
        grp = by_year(trades)
        for yr in sorted(grp.keys()):
            print(yr_line(grp[yr], str(yr)))
        total_n = len(trades)
        if total_n:
            total_pnl = sum(t['pnl'] for t in trades)
            total_w = sum(1 for t in trades if t['pnl'] > 0)
            total_trend = sum(1 for t in trades if t.get('type') == 'trend')
            total_swing = sum(1 for t in trades if t.get('type') == 'swing')
            total_s = sum(1 for t in trades if t.get('type') == 'swing' and t['dir'] == 'SHORT')
            total_l = sum(1 for t in trades if t.get('type') == 'swing' and t['dir'] == 'LONG')
            print(f'  合计: {total_n:2d}tr PnL${total_pnl:>+6.0f} WR{total_w/total_n*100:.0f}% (趋势{total_trend}+摆动{total_swing} S{total_s}+L{total_l})')

    # ─── 1. 纯趋势（无摆动）───
    print('=== 1. 纯趋势（无摆动）===')
    sys.stdout, sys.stderr = devnull, devnull
    tr_trend_raw = hybrid_bt(period='10y', **swing_base, short_etf='none')
    sys.stdout, sys.stderr = old_out, old_err
    print_yr_table([t for t in tr_trend_raw if t.get('type') == 'trend'])

    # ─── 2-5. 混合 + 不同做空工具 ───
    print('\n=== 2-5. 混合 S2_double_day_h5 + 不同做空工具 ===')
    scenarios = [
        ('SPXU(3x)', {}),
        ('SDS(2x)',  {'short_etf': 'SDS'}),
        ('SH(1x)',   {'short_etf': 'SH'}),
        ('做多only',  {'short_etf': 'none'}),
    ]
    for label, kw in scenarios:
        print(f'\n--- {label} ---')
        full_kw = {**swing_base, **kw}
        sys.stdout, sys.stderr = devnull, devnull
        tr = hybrid_bt(period='10y', **full_kw)
        sys.stdout, sys.stderr = old_out, old_err
        print_yr_table(tr)

    # ─── Directional filter 对比 ───
    print('\n========== Directional Filter 对比 (S2_double_day_h5 + SPXU) ==========')
    for df_lbl, df_kw in [('旧 (dir_filter=False)', {'swing_direction_filter': False}),
                           ('新 (dir_filter=True)',  {})]:
        print(f'\n--- {df_lbl} ---')
        full_kw = {**swing_base, **df_kw}
        sys.stdout, sys.stderr = devnull, devnull
        tr = hybrid_bt(period='10y', **full_kw)
        sys.stdout, sys.stderr = old_out, old_err
        print_yr_table(tr)

    print('\n✅ 完成')
    devnull.close()
