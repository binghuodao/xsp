"""Pure trend CALL strategy. Swing removed 2026-07-28:
   6y backtest: 33 swing trades net −$266. Signal B (DI cross) never
   fired with double_day. Signal C (BB edge+RSI) net −$266 total.
   Swing added complexity with zero net benefit vs pure trend."""
import sys, os, collections, math
import numpy as np, pandas as pd, yfinance as yf

OUT = os.path.dirname(os.path.abspath(__file__))

def _s5(v):
    return round(v / 5) * 5

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_call_price(S, K, T, sigma, r=0.05):
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)

def bs_call_delta(S, K, T, sigma, r=0.05):
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

def find_strike_for_delta(S, target_delta, T, sigma, r=0.05):
    lo, hi = S * 0.6, S * 1.1
    for _ in range(30):
        mid = (lo + hi) / 2
        d = bs_call_delta(S, mid, T, sigma, r)
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return _s5((lo + hi) / 2)

def hybrid_bt(period='10y', trend_hold=30, trend_trail=0.03, naked_size=0, naked_delta=None,
               opt_dte=14, roll_dte=2, opt_spread=1, spread_width=15, skip_bbp=0.80):
    """Trend: $5k SPXL 3x + 1 rolling CALL position.
    opt_spread=0 → naked CALL (opt_dte DTE).
    opt_spread=1 → CALL debit spread (opt_dte DTE, long ATM K1, short K1+spread_width).
    Roll when remaining DTE ≤ roll_dte. Option shares ETF exits (no own stop).
    skip_bbp>0 → skip NEW entry when BB%>skip_bbp (high-band filter); ≤0 disables."""
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
            if not ex:
                if trend_pos['pp'] < trend_pos['ep'] * 1.005:
                    if cs <= trend_pos['ep'] * (1 - 0.02):
                        xp = cs; ex = True; xt = 'entry_trail'
            if not ex and i - trend_pos['ei'] >= trend_hold:
                xp = cs; ex = True; xt = 't+30'

            if ex:
                etf_exit = float(spxl_c.loc[df.index[i]])
                etf_pnl = round(5000 * (etf_exit / trend_pos['etf_entry'] - 1), 2)
                opt_pnl = trend_pos.get('opt_pnl', 0)
                if naked_size > 0 and 'opt_K' in trend_pos:
                    opt_days = i - trend_pos['opt_entry_idx']
                    T_rem = max(opt_dte / 365 - opt_days / 365, 1 / 365)
                    if opt_spread:
                        exit_opt = (bs_call_price(cs, trend_pos['opt_K'], T_rem, trend_pos['opt_sigma'])
                                    - bs_call_price(cs, trend_pos['opt_K2'], T_rem, trend_pos['opt_sigma']))
                    else:
                        exit_opt = bs_call_price(cs, trend_pos['opt_K'], T_rem, trend_pos['opt_sigma'])
                    opt_pnl += round(max((exit_opt - trend_pos['opt_entry']) * 100 * naked_size,
                                         -trend_pos['opt_entry'] * 100 * naked_size), 2)
                trades.append({
                    'entry_date': trend_pos['ed'], 'exit_date': dt,
                    'dir': 'CALL_spread' if opt_spread else 'CALL', 'type': 'trend',
                    'entry_price': trend_pos['ep'], 'exit_price': round(xp, 2),
                    'pnl': etf_pnl + opt_pnl,
                    'etf_pnl': etf_pnl, 'opt_pnl': opt_pnl,
                    'entry_opt_cost': trend_pos.get('opt_entry', 0),
                    'num_rolls': trend_pos.get('num_rolls', 0),
                    'exit_type': xt, 'entry_reason': trend_pos.get('reason', ''),
                })
                trend_pos = None

        # ─── Option rolling: close when DTE≤roll_dte, open new opt_dte spread ───
        if trend_pos is not None and naked_size > 0 and 'opt_K' in trend_pos:
            opt_days = i - trend_pos['opt_entry_idx']
            if opt_days >= opt_dte - roll_dte:
                T_rem = max(opt_dte / 365 - opt_days / 365, 1 / 365)
                if opt_spread:
                    roll_close = (bs_call_price(cs, trend_pos['opt_K'], T_rem, trend_pos['opt_sigma'])
                                  - bs_call_price(cs, trend_pos['opt_K2'], T_rem, trend_pos['opt_sigma']))
                else:
                    roll_close = bs_call_price(cs, trend_pos['opt_K'], T_rem, trend_pos['opt_sigma'])
                roll_pnl = round(max((roll_close - trend_pos['opt_entry']) * 100 * naked_size,
                                     -trend_pos['opt_entry'] * 100 * naked_size), 2)
                trend_pos['opt_pnl'] += roll_pnl
                sigma = max(float(row['vix']), 10) / 100
                T = opt_dte / 365
                if naked_delta is not None:
                    strike = find_strike_for_delta(cs, naked_delta, T, sigma)
                else:
                    strike = _s5(cs)
                trend_pos['opt_K'] = strike
                if opt_spread:
                    trend_pos['opt_K2'] = strike + spread_width
                    trend_pos['opt_entry'] = (bs_call_price(cs, strike, T, sigma)
                                              - bs_call_price(cs, strike + spread_width, T, sigma))
                else:
                    trend_pos['opt_entry'] = bs_call_price(cs, strike, T, sigma)
                trend_pos['opt_sigma'] = sigma
                trend_pos['opt_entry_idx'] = i
                trend_pos['num_rolls'] += 1

        # ─── Trend entry ───
        if trend_pos is None and trend_sig is not None and trend_sig[0] is not None \
                and (skip_bbp <= 0 or float(row['bbp']) <= skip_bbp):
            trend_pos = {'dir': 'CALL', 'ep': float(row['price']), 'ed': dt, 'ei': i,
                         'pp': float(row['price']), 'reason': trend_sig[1],
                         'etf_entry': float(spxl_c.loc[df.index[i]])}
            if naked_size > 0:
                sigma = max(float(row['vix']), 10) / 100
                T = opt_dte / 365
                if naked_delta is not None:
                    strike = find_strike_for_delta(float(row['price']), naked_delta, T, sigma)
                else:
                    strike = _s5(float(row['price']))
                trend_pos['opt_K'] = strike
                if opt_spread:
                    trend_pos['opt_K2'] = strike + spread_width
                    trend_pos['opt_entry'] = (bs_call_price(float(row['price']), strike, T, sigma)
                                              - bs_call_price(float(row['price']), strike + spread_width, T, sigma))
                else:
                    trend_pos['opt_entry'] = bs_call_price(float(row['price']), strike, T, sigma)
                trend_pos['opt_sigma'] = sigma
                trend_pos['opt_entry_idx'] = i
                trend_pos['opt_pnl'] = 0
                trend_pos['num_rolls'] = 0

    return trades


def mr_bt(period='10y', pricing='gamma_theta', entry_cost=3.50, green_buffer=0.003, stop_pct=0.02, etf_size=0):
    """Mean reversion CALL: nb + RSI<30 + VIX>20, hold max 3d.
    green_buffer=0.003: exit when XSP > entry×1.003 (skip barely-green losses)
    stop_pct=0.02: exit when XSP ≤ entry×0.98 (cap single loss)
    etf_size > 0: also buy $etf_size SPXL 3x ETF, track combined PnL
    
    pricing:
      'gamma_theta' — gamma-theta P&L model, entry=$entry_cost
      'bs_atm'      — Black-Scholes pricing, ATM strike (round to 5)
      'bs_otm'      — Black-Scholes, delta ~0.35 strike
    Exit: first day XSP closes above entry, force exit day 3."""
    xsp = yf.download('^XSP', period=period, interval='1d', progress=False)
    if isinstance(xsp.columns, pd.MultiIndex): xsp = xsp.droplevel('Ticker', axis=1)
    vix = yf.download('^VIX', period=period, interval='1d', progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix = vix.droplevel('Ticker', axis=1)
    xc = xsp['Close']; xh = xsp['High']; xl = xsp['Low']
    vc = vix['Close'].reindex(xc.index).ffill()
    spxl = yf.download('SPXL', period=period, interval='1d', progress=False)
    if isinstance(spxl.columns, pd.MultiIndex): spxl = spxl.droplevel('Ticker', axis=1)
    spxl_c = spxl['Close'].reindex(xc.index).ffill()

    df = pd.DataFrame(index=xc.index)
    df['price'] = xc
    s20 = xc.rolling(20).mean(); bs = xc.rolling(20).std()
    df['bbu'] = s20+2*bs; df['bbl'] = s20-2*bs
    tr = pd.concat([xh-xl, (xh-xc.shift(1)).abs(), (xl-xc.shift(1)).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['nb'] = df['price'] <= df['bbl'] + df['atr_14']*0.60
    dlt = xc.diff(); gn = dlt.clip(lower=0); ls = (-dlt).clip(lower=0)
    ag = gn.rolling(14).mean(); al = ls.rolling(14).mean()
    df['rsi_14'] = (100-100/(1+ag/al.replace(0, np.nan))).fillna(50)
    df['vix'] = vc.values
    df = df.dropna().copy(); df = df[df.index >= pd.Timestamp('2021-03-01')]

    is_bs = pricing.startswith('bs_')
    mc = entry_cost * 100 if not is_bs else 0
    T = 7 / 365
    trades = []; pos = None; max_days = 3

    for i in range(len(df)):
        row = df.iloc[i]; dt = df.index[i]; cs = float(row['price'])

        # MR exit
        if pos is not None:
            la = i - pos['ei']
            if la > 0:
                ex = False; xp = None; xt = ''
                if stop_pct and cs <= pos['ep'] * (1 - stop_pct):
                    xp = cs; ex = True; xt = 'mr_stop'
                elif green_buffer and cs > pos['ep'] * (1 + green_buffer) and la <= max_days:
                    xp = cs; ex = True; xt = 'mr_green'
                elif la >= max_days:
                    xp = cs; ex = True; xt = 'mr_time'

                if ex:
                    if is_bs:
                        cal_days = (dt - pos['ed']).days
                        T_rem = max(T - cal_days / 365, 1 / 365)
                        exit_opt = bs_call_price(xp, pos['K'], T_rem, pos['sigma'])
                        pnl = round(max((exit_opt - pos['entry_opt']) * 100, -pos['entry_opt'] * 100), 2)
                    else:
                        pt = xp - pos['ep']
                        ed = min(0.5 + 0.08 * abs(pt), 1.0)
                        ad = (0.5 + ed) / 2
                        dp = ad * pt * 100
                        gp = 0
                        if pt > 0:
                            sp = (1.0 - 0.5) / 0.08
                            ep_g = min(abs(pt), sp)
                            gp = 0.5 * 0.08 * (ep_g ** 2) * 100
                        tp = 0.35 * la * 100
                        pnl = round(max(dp + gp - tp, -mc), 2)

                    if not xt:
                        xt = 'mr_green' if cs > pos['ep'] else 'mr_time'
                    etf_pnl = 0
                    if etf_size > 0 and 'etf_entry' in pos:
                        spxl_exit = float(spxl_c.loc[dt]) if dt in spxl_c.index else float(spxl_c.iloc[-1])
                        etf_pnl = round(etf_size * (spxl_exit / pos['etf_entry'] - 1), 2)
                    trades.append({
                        'entry_date': pos['ed'], 'exit_date': dt,
                        'dir': 'CALL', 'type': 'mr',
                        'entry_price': pos['ep'], 'exit_price': round(xp, 2),
                        'pnl': pnl, 'exit_type': xt,
                        'entry_reason': 'MR_nb_rsi_vix',
                        'entry_opt_cost': pos.get('entry_opt', entry_cost) * (1 if is_bs else 1),
                        'etf_pnl': etf_pnl,
                    })
                    pos = None

        # MR entry (only when no position, signal day, not consecutive)
        if pos is None and row['rsi_14'] < 30 and row['vix'] > 20:
            if i == 0 or not (df.iloc[i-1]['rsi_14'] < 30 and df.iloc[i-1]['vix'] > 20):
                if is_bs:
                    sigma = float(row['vix']) / 100
                    if pricing == 'bs_otm':
                        strike = find_strike_for_delta(cs, 0.35, T, sigma)
                    else:
                        strike = _s5(cs)
                    opt_price = bs_call_price(cs, strike, T, sigma)
                    pos = {'ep': cs, 'ed': dt, 'ei': i,
                           'K': strike, 'entry_opt': opt_price, 'sigma': sigma}
                else:
                    pos = {'ep': cs, 'ed': dt, 'ei': i}
                if etf_size > 0:
                    pos['etf_entry'] = float(spxl_c.loc[dt]) if dt in spxl_c.index else float(spxl_c.iloc[-1])

    return trades


def crash_bt(period='10y', vix_req=False, drop_thresh=0.005, etf_size=2000, spread_width=15,
             stop_pct=0.025, trail_pct=0, green_buffer=0.0, max_hold=4, T_dte=21):
    """Crash bounce: XSP drop>drop_thresh → 1 CALL spread (ATM + ATM+spread_width, T_dte DTE) + $2k SPXL.
    Stop exits: fixed stop (stop_pct) and/or trailing stop (trail_pct from peak).
    Green exit: entry × (1+green_buffer). Time exit: T+max_hold.
    Max loss = net debit paid. Max gain = (spread_width×100) - debit."""
    xsp = yf.download('^XSP', period=period, interval='1d', progress=False)
    if isinstance(xsp.columns, pd.MultiIndex): xsp = xsp.droplevel('Ticker', axis=1)
    vix = yf.download('^VIX', period=period, interval='1d', progress=False)
    if isinstance(vix.columns, pd.MultiIndex): vix = vix.droplevel('Ticker', axis=1)
    xc = xsp['Close']; xh = xsp['High']; xl = xsp['Low']
    vc = vix['Close'].reindex(xc.index).ffill()
    spxl = yf.download('SPXL', period=period, interval='1d', progress=False)
    if isinstance(spxl.columns, pd.MultiIndex): spxl = spxl.droplevel('Ticker', axis=1)
    spxl_c = spxl['Close'].reindex(xc.index).ffill()

    df = pd.DataFrame(index=xc.index)
    df['price'] = xc
    tr = pd.concat([xh-xl, (xh-xc.shift(1)).abs(), (xl-xc.shift(1)).abs()], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    dlt = xc.diff(); gn = dlt.clip(lower=0); ls = (-dlt).clip(lower=0)
    ag = gn.rolling(14).mean(); al = ls.rolling(14).mean()
    df['rsi_14'] = (100-100/(1+ag/al.replace(0, np.nan))).fillna(50)
    df['vix'] = vc.values; df['chg1'] = xc.pct_change()
    df = df.dropna().copy(); df = df[df.index >= pd.Timestamp('2021-03-01')]

    T = T_dte / 365
    trades = []; pos = None

    for i in range(len(df)):
        row = df.iloc[i]; dt = df.index[i]; cs = float(row['price'])

        # ── Crash bounce exit ──
        if pos is not None:
            la = i - pos['ei']
            if la > 0:
                ex = False; xp = None; xt = ''
                pos['pp'] = max(pos.get('pp', cs), cs)
                trail_active = trail_pct > 0 and pos['pp'] > pos['ep']
                trail_stop = pos['pp'] * (1 - trail_pct) if trail_active else None
                fixed_stop = pos['ep'] * (1 - stop_pct)
                effective_stop = max(fixed_stop, trail_stop) if trail_stop else fixed_stop
                if cs <= effective_stop:
                    xp = cs; ex = True; xt = 'trail' if (trail_stop and cs <= trail_stop and trail_active and cs > fixed_stop) else 'stop'
                elif cs > pos['ep'] * (1 + green_buffer) and la <= max_hold:
                    xp = cs; ex = True; xt = 'green'
                elif la >= max_hold:
                    xp = cs; ex = True; xt = 'time'

                if ex:
                    cal_days = (dt - pos['ed']).days
                    T_rem = max(T - cal_days / 365, 1 / 365)
                    e1 = bs_call_price(xp, pos['K1'], T_rem, pos['sigma'])
                    e2 = bs_call_price(xp, pos['K2'], T_rem, pos['sigma'])
                    opt_pnl = round(max((e1 - e2 - pos['debit']) * 100, -pos['debit'] * 100), 2)
                    etf_pnl = 0
                    if etf_size > 0 and 'etf_entry' in pos:
                        spxl_exit = float(spxl_c.loc[dt]) if dt in spxl_c.index else float(spxl_c.iloc[-1])
                        etf_pnl = round(etf_size * (spxl_exit / pos['etf_entry'] - 1), 2)
                    trades.append({
                        'entry_date': pos['ed'], 'exit_date': dt,
                        'dir': 'CALL_spread', 'type': 'crash',
                        'entry_price': pos['ep'], 'exit_price': round(xp, 2),
                        'pnl': opt_pnl, 'exit_type': xt,
                        'entry_reason': 'crash_vix_drop',
                        'entry_opt_cost': pos['debit'],
                        'etf_pnl': etf_pnl,
                    })
                    pos = None

        # ── Crash bounce entry ──
        entry_ok = (i > 0 and row['chg1'] < -drop_thresh and (not vix_req or row['vix'] > 20))
        if pos is None and entry_ok:
            if i == 0 or not (df.iloc[i-1]['chg1'] < -drop_thresh):
                sigma = float(row['vix']) / 100 if row['vix'] > 5 else 0.20
                if sigma > 0.01:
                    K1 = _s5(cs)
                    K2 = K1 + spread_width
                    e1 = bs_call_price(cs, K1, T, sigma)
                    e2 = bs_call_price(cs, K2, T, sigma)
                    debit = e1 - e2
                    if debit > 0.01:
                        pos = {'ep': cs, 'ed': dt, 'ei': i,
                               'K1': K1, 'K2': K2, 'debit': debit, 'sigma': sigma}
                        if etf_size > 0:
                            pos['etf_entry'] = float(spxl_c.loc[dt]) if dt in spxl_c.index else float(spxl_c.iloc[-1])

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
    tr_atm = hybrid_bt(period='10y', naked_size=1)
    tr_otm = hybrid_bt(period='10y', naked_size=1, naked_delta=0.35)
    mr_gt = mr_bt(period='10y', pricing='gamma_theta')
    mr_atm = mr_bt(period='10y', pricing='bs_atm')
    mr_otm = mr_bt(period='10y', pricing='bs_otm')
    mr_combo = mr_bt(period='10y', pricing='bs_atm', etf_size=2000)
    crash = crash_bt(period='10y', etf_size=2000)
    sys.stdout, sys.stderr = old_out, old_err

    print('=== 趋势CALL（纯策略）===')
    print_yr_table(tr)
    print()

    print('=== 趋势CALL: 纯ETF vs ETF+滚动价差ATM vs ETF+滚动价差OTM ===')
    for label, td in [('纯ETF', tr), ('+价差ATM', tr_atm), ('+价差OTM', tr_otm)]:
        n = len(td)
        if n == 0:
            print(f'  {label:<12}  0tr')
            continue
        etf_pnl = sum(t.get('etf_pnl', t['pnl']) for t in td)
        opt_pnl = sum(t.get('opt_pnl', 0) for t in td)
        tot_pnl = etf_pnl + opt_pnl
        w = sum(1 for t in td if (t.get('etf_pnl', t['pnl']) + t.get('opt_pnl', 0)) > 0)
        rolls = sum(t.get('num_rolls', 0) for t in td)
        avg_etf = etf_pnl / n
        avg_opt = opt_pnl / n
        avg_tot = avg_etf + avg_opt
        avg_cost = sum(t.get('entry_opt_cost', 0) for t in td) / n
        max_loss = min(t.get('etf_pnl', t['pnl']) + t.get('opt_pnl', 0) for t in td)
        extra = f'  均成本${avg_cost*100:.0f}/口  最大亏${max_loss:+.0f}' if td[0].get('opt_pnl', 0) is not None and any('opt_pnl' in t for t in td) else ''
        print(f'  {label:<12}  {n:2d}tr  ETF${etf_pnl:>+8,.0f}  OPT${opt_pnl:>+8,.0f}  合计${tot_pnl:>+8,.0f}  WR{w/n*100:.0f}%  |  滚{rolls}次  均单ETF${avg_etf:+.0f}  OPT${avg_opt:+.0f}  ${avg_tot:+.0f}{extra}')
    print()

    for label, mr in [('原始gamma-theta ($3.50)', mr_gt),
                      ('BS ATM (VIX入场价)', mr_atm),
                      ('BS OTM Δ0.35 (VIX入场价)', mr_otm)]:
        print(f'=== 均值回归CALL — {label} ===')
        print_yr_table(mr)
        avg_pnl = sum(t['pnl'] for t in mr) / len(mr) if mr else 0
        avg_cost = sum(t.get('entry_opt_cost', 0) for t in mr) / len(mr) if mr else 0
        print(f'  均单PnL ${avg_pnl:+.0f}  均成本 ${avg_cost*100:.0f}/口')
        print()

    print('=== 均值回归CALL + $2k SPXL（BS ATM）===')
    # Show combined PnL (CALL + ETF) per year
    combo_by_yr = by_year(mr_combo)
    for yr in sorted(combo_by_yr.keys()):
        yt = combo_by_yr[yr]
        n = len(yt)
        pnl = sum(t['pnl'] + t.get('etf_pnl', 0) for t in yt)
        w = sum(1 for t in yt if t['pnl'] + t.get('etf_pnl', 0) > 0)
        print(f'  {yr}  {n:2d}tr  PnL${pnl:>+7.0f}  WR{w/n*100:.0f}%')
    total_n = len(mr_combo)
    total_pnl = sum(t['pnl'] + t.get('etf_pnl', 0) for t in mr_combo)
    total_w = sum(1 for t in mr_combo if t['pnl'] + t.get('etf_pnl', 0) > 0)
    print(f'  合计  {total_n:2d}tr  PnL${total_pnl:>+7.0f}  WR{total_w/total_n*100:.0f}%')
    avg_opt = sum(t['pnl'] for t in mr_combo) / len(mr_combo) if mr_combo else 0
    avg_etf = sum(t.get('etf_pnl', 0) for t in mr_combo) / len(mr_combo) if mr_combo else 0
    avg_tot = avg_opt + avg_etf
    print(f'  均单: CALL${avg_opt:+.0f} + ETF${avg_etf:+.0f} = ${avg_tot:+.0f}')
    print()

    tpnl = sum(t['pnl'] for t in tr_atm)
    mc_opt = sum(t['pnl'] for t in mr_combo)
    mc_etf = sum(t.get('etf_pnl', 0) for t in mr_combo)
    mc_tot = mc_opt + mc_etf
    print(f'\n✅ 完成  趋势(ETF+价差)${tpnl:+>7.0f}  '
          f'MR_GT${sum(t["pnl"] for t in mr_gt):+>7.0f}  '
          f'MR_ATM${sum(t["pnl"] for t in mr_atm):+>7.0f}  '
          f'MR_OTM${sum(t["pnl"] for t in mr_otm):+>7.0f}')
    print(f'  MR组合(CALL+$2k ETF)${mc_tot:+>7.0f}')

    print()
    print('=== 崩盘反弹CALL价差15点 21DTE + $2k SPXL (0.5%drop, 无VIX, 2.5%stop, 首阳即出, T+4) ===')
    crash_by_yr = by_year(crash)
    for yr in sorted(crash_by_yr.keys()):
        yt = crash_by_yr[yr]
        n = len(yt)
        pnl = sum(t['pnl'] + t.get('etf_pnl', 0) for t in yt)
        w = sum(1 for t in yt if t['pnl'] + t.get('etf_pnl', 0) > 0)
        print(f'  {yr}  {n:2d}tr  PnL${pnl:>+7.0f}  WR{w/n*100:.0f}%')
    cn = len(crash)
    cp = sum(t['pnl'] + t.get('etf_pnl', 0) for t in crash)
    cw = sum(1 for t in crash if t['pnl'] + t.get('etf_pnl', 0) > 0)
    print(f'  合计  {cn:2d}tr  PnL${cp:>+7.0f}  WR{cw/cn*100:.0f}%')
    cavg_opt = sum(t['pnl'] for t in crash) / cn if crash else 0
    cavg_etf = sum(t.get('etf_pnl', 0) for t in crash) / cn if crash else 0
    max_loss = min(t['pnl'] + t.get('etf_pnl', 0) for t in crash)
    max_opt_loss = min(t['pnl'] for t in crash)
    avg_cost = sum(t.get('entry_opt_cost', 0) for t in crash) / cn if crash else 0
    print(f'  均单: CALL价差${cavg_opt:+.0f} + ETF${cavg_etf:+.0f} = ${cavg_opt+cavg_etf:+.0f}')
    print(f'  最大亏 ${max_loss:+.0f} (OPT${max_opt_loss:+.0f})  均成本 ${avg_cost*100:.0f}/口')
    print()

    # Combined 3-strategy comparison (each row = strategy total incl. ETF)
    tpnl = sum(t['pnl'] for t in tr_atm)
    mc_opt = sum(t['pnl'] for t in mr_combo)
    mc_etf = sum(t.get('etf_pnl', 0) for t in mr_combo)
    mc_tot = mc_opt + mc_etf
    total_pnl = tpnl + mc_tot + cp
    tn = len(tr_atm); mn = len(mr_combo); ccn = cn
    tw = sum(1 for t in tr_atm if t['pnl'] > 0)
    mw = sum(1 for t in mr_combo if t['pnl'] + t.get('etf_pnl', 0) > 0)
    print('=== 三策略对比 ===')
    print(f'  {"策略":<22} {"笔数":>4} {"总PnL":>10} {"WR":>5}')
    print(f'  趋势($5k SPXL+价差)                {tn:4d} ${tpnl:>+8,.0f} {tw/tn*100:>4.0f}%')
    print(f'  MR(CALL+$2k ETF)             {mn:4d} ${mc_tot:>+8,.0f} {mw/mn*100:>4.0f}%')
    print(f'  崩盘(CALL价差15点+$2k) {ccn:4d} ${cp:>+8,.0f} {cw/ccn*100:>4.0f}%')
    print(f'  {"合计":<22} {tn+mn+ccn:4d} ${total_pnl:>+8,.0f}')

    print(f'\n✅ 完成  趋势${tpnl:+>7.0f}  '
          f'MR_GT${sum(t["pnl"] for t in mr_gt):+>7.0f}  '
          f'MR_ATM${sum(t["pnl"] for t in mr_atm):+>7.0f}  '
          f'MR_OTM${sum(t["pnl"] for t in mr_otm):+>7.0f}')
    print(f'  MR组合${mc_tot:+>7.0f}  崩盘${cp:+>7.0f}  总策略${total_pnl:+>7.0f}')
    devnull.close()
