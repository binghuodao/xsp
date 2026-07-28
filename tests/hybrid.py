"""Pure trend CALL strategy. Swing removed 2026-07-28:
   6y backtest: 33 swing trades net −$266. Signal B (DI cross) never
   fired with double_day. Signal C (BB edge+RSI) net −$266 total.
   Swing added complexity with zero net benefit vs pure trend."""
import sys, os, collections
import numpy as np, pandas as pd, yfinance as yf

OUT = os.path.dirname(os.path.abspath(__file__))

def hybrid_bt(period='10y', trend_hold=30, trend_trail=0.035):
    np.random.seed(0)
    xsp = yf.download('^XSP', period=period, interval='1d', progress=False)
    if isinstance(xsp.columns, pd.MultiIndex): xsp = xsp.droplevel('Ticker', axis=1)
    vix = yf.download('^VIX', period=period, interval='1d', progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix = vix.droplevel('Ticker', axis=1)
    vc = vix['Close'].reindex(xsp.index).ffill()
    spxl = yf.download('SPXL', period=period, interval='1d', progress=False)
    if isinstance(spxl.columns, pd.MultiIndex): spxl = spxl.droplevel('Ticker', axis=1)
    spxl_c = spxl['Close'].reindex(xsp.index).ffill()
    xc = xsp['Close']; xh = xsp['High']; xl = xsp['Low']; xo = xsp['Open']
    df = pd.DataFrame(index=xsp.index)
    df['price'] = xc; df['high'] = xh; df['low'] = xl

    s20 = xc.rolling(20).mean(); bs = xc.rolling(20).std()
    df['bbu'] = s20+2*bs; df['bbl'] = s20-2*bs; df['sma50'] = xc.rolling(50).mean()
    df['sma50_slope'] = df['sma50'].diff(5)
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

    # Trend scoring
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

    def gd(r):
        dd = r['di_diff']; ddp = r['di_diff_prev']
        sc_c = r['score_call']; rsi = r['rsi_14']
        adx = r['adx']; vp = r['vix_percentile']
        nt_r = r['nt']; nb_r = r['nb']

        if not (nt_r or nb_r):
            if dd > 0 and r['price'] > r['sma50'] and sc_c >= 50:
                if r['sma50'] < r['sma200'] and r['price'] > r['sma200'] and r['sma50_slope'] < 2:
                    pass
                elif ddp <= 0:
                    pass
                else:
                    return ('CALL', 'L1_trend')
        if nb_r and sc_c >= 50 and vp > 75:
            return ('CALL', 'L2_nearbb_vix')
        if nb_r and sc_c >= 35 and (adx < 25 or rsi < 35) and not (adx >= 25 and r['price'] < r['sma50']):
            return ('CALL', 'L4_nearbb')
        return (None, None)

    trades = []
    trend_pos = None

    for i in range(len(df)):
        row = df.iloc[i]; trend_sig = gd(row); dt = df.index[i]
        dd = row['di_diff']; ddp = row['di_diff_prev']
        vi = float(row['vix']); vp = float(row['vix_percentile'])

        # ─── Trend exit ───
        if trend_pos is not None and i > trend_pos['ei']:
            ex = False; xp = None; xt = ''; cs = float(row['price'])

            if (vp > 80 or vi > 28) and dd < 0 and ddp < 0:
                xp = cs; ex = True; xt = 'di_reversal'

            if not ex:
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
                etf_exit = float(spxl_c.loc[df.index[i]])
                pnl = round(5000 * (etf_exit / trend_pos['etf_entry'] - 1), 2)
                trades.append({
                    'entry_date': trend_pos['ed'], 'exit_date': dt,
                    'dir': 'CALL', 'type': 'trend',
                    'entry_price': trend_pos['ep'], 'exit_price': round(xp, 2),
                    'pnl': pnl,
                    'exit_type': xt, 'entry_reason': trend_pos.get('reason', ''),
                })
                trend_pos = None

        # ─── Trend entry ───
        if trend_pos is None and trend_sig is not None and trend_sig[0] is not None:
            trend_pos = {'dir': 'CALL', 'ep': float(row['price']), 'ed': dt, 'ei': i,
                         'pp': float(row['price']), 'reason': trend_sig[1],
                         'etf_entry': float(spxl_c.loc[df.index[i]])}

    return trades


def summ(trades):
    n = len(trades)
    if n == 0: return {'n':0,'wr':0,'pnl':0}
    w = sum(1 for t in trades if t['pnl'] > 0)
    return {'n':n,'wr':round(w/n*100,1),'pnl':sum(t['pnl'] for t in trades)}


if __name__ == '__main__':
    devnull = open('/dev/null', 'w')
    old_out, old_err = sys.stdout, sys.stderr

    def by_year(trades):
        grp = collections.defaultdict(list)
        for t in trades:
            yr = t['entry_date'].year if hasattr(t['entry_date'], 'strftime') else '?'
            grp[yr].append(t)
        return grp

    def print_yr_table(trades):
        grp = by_year(trades)
        for yr in sorted(grp.keys()):
            yt = grp[yr]
            n = len(yt)
            pnl = sum(t['pnl'] for t in yt)
            w = sum(1 for t in yt if t['pnl'] > 0)
            print(f'  {yr}  {n:2d}tr  PnL${pnl:>+7.0f}  WR{w/n*100:.0f}%')
        total_n = len(trades)
        if total_n:
            total_pnl = sum(t['pnl'] for t in trades)
            total_w = sum(1 for t in trades if t['pnl'] > 0)
            print(f'  合计  {total_n:2d}tr  PnL${total_pnl:>+7.0f}  WR{total_w/total_n*100:.0f}%')

    sys.stdout, sys.stderr = devnull, devnull
    tr = hybrid_bt(period='10y')
    sys.stdout, sys.stderr = old_out, old_err
    print('=== 趋势CALL（纯策略）===')
    print_yr_table(tr)
    print('\n✅ 完成')
    devnull.close()
