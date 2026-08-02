"""Real-data simulation of the three strategies' daily reports.

Each strategy runs on its OWN real historical signal day:
  - Trend  : 2026-06-03  (real CALL trend day)
  - MR     : 2026-03-27  (real RSI<30 + VIX>20 day)
  - Crash  : auto-scanned most recent XSP single-day drop > 0.5%

All inputs are real from yfinance; the option chain is BS-synthesized with the
real VIX of that day as sigma (yfinance has no historical option chains).

Usage:  python3 tests/sim_reports.py > tests/sim_reports_results.txt
"""
import sys, os, tempfile, datetime
from datetime import date, timedelta
from unittest.mock import MagicMock

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _SCRIPT_DIR)

# fake moomoo before importing app
class _FakeMoomoo:
    class OpenQuoteContext:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a, **kw): pass
    class SubType:
        ORDER_BOOK = 1
    RET_OK = 0
sys.modules['moomoo'] = _FakeMoomoo()

import pandas as pd
import pandas_ta as ta
import yfinance as yf
import app
import pricing

TREND_DATE = date(2026, 6, 3)
MR_DATE = date(2026, 3, 27)

# ── 1. download real history once ──
def _cols(df):
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_localize(None)
    return df

print("Downloading real history...")
xsp = _cols(yf.Ticker('^XSP').history(period='8mo')).sort_index()
vix = _cols(yf.Ticker('^VIX').history(period='1y')).sort_index()
spy = _cols(yf.Ticker('SPY').history(period='8mo')).sort_index()
spxl = _cols(yf.Ticker('SPXL').history(period='8mo')).sort_index()
skew = _cols(yf.Ticker('^SKEW').history(period='1mo')).sort_index()
for df in (xsp, vix, spy, spxl, skew):
    df.drop_duplicates(inplace=True)

# full-period indicator series
adx_df = ta.adx(spy['High'], spy['Low'], spy['Close'], length=14)
bb_df = ta.bbands(spy['Close'], length=20, std=2)
rsi_s = ta.rsi(spy['Close'], length=14)
sma20_s = spy['Close'].rolling(20).mean()
avg_vol_s = spy['Volume'].rolling(20).mean()
ema20p_s = spy['Close'].ewm(span=20, adjust=False).mean()

W = {'adx': .3, 'er': .2, 'bbw': .15, 'dev': .15, 'vr': .1}
T = {'adx': [30, 25, 20, 15, 0], 'er': [.7, .55, .35, .2, 0],
     'bbw': [45, 30, 18, 10, 0], 'dev': [3.0, 1.5, 0.8, 0.3, 0],
     'vr': [2.0, 1.3, .8, .5, 0]}

def _st(v, th):
    for t, s in zip(th, [100, 75, 50, 25, 0]):
        if v >= t:
            return s
    return 0

def build_snapshot(asof):
    """Rebuild app.historical_stats / price / spxl / prev_close as of `asof` (real data)."""
    x_i = xsp[xsp.index <= pd.Timestamp(asof)]
    v_i = vix[vix.index <= pd.Timestamp(asof)]
    s_i = spy[spy.index <= pd.Timestamp(asof)]
    sk_i = skew[skew.index <= pd.Timestamp(asof)]
    hs = dict(app.historical_stats)
    cv = float(v_i['Close'].iloc[-1])
    hs['vix'] = cv
    hs['vix_rank'] = float((cv - v_i['Close'].min()) / (v_i['Close'].max() - v_i['Close'].min())) * 100
    hs['vix_percentile'] = float((v_i['Close'] < cv).mean()) * 100
    if len(sk_i):
        hs['skew_index'] = float(sk_i['Close'].iloc[-1])
    closes = x_i['Close']; highs = x_i['High']; lows = x_i['Low']
    tr = pd.concat([highs - lows, (highs - closes.shift(1)).abs(), (lows - closes.shift(1)).abs()], axis=1).max(axis=1)
    hs['atr_14'] = float(tr.iloc[-14:].mean())
    hs['ema_20'] = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    if len(closes) >= 50:
        hs['sma50'] = float(closes.rolling(50).mean().iloc[-1])
        s50 = closes.rolling(50).mean()
        hs['sma50_slope'] = float(s50.iloc[-1] - s50.iloc[-6]) if len(s50.dropna()) >= 6 else 0
    hs['sma200'] = float(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else 0.0
    a_i = adx_df[adx_df.index <= pd.Timestamp(asof)]
    b_i = bb_df[bb_df.index <= pd.Timestamp(asof)]
    hs['adx'] = float(a_i['ADX_14'].iloc[-1])
    dmp = float(a_i['DMP_14'].iloc[-1]) if 'DMP_14' in a_i.columns else 0.0
    dmn = float(a_i['DMN_14'].iloc[-1]) if 'DMN_14' in a_i.columns else 0.0
    hs['di_diff'] = round((dmp - dmn) / 100, 3)
    prev_a = adx_df[adx_df.index < pd.Timestamp(asof)]
    if len(prev_a):
        hs['di_diff_prev'] = round((float(prev_a['DMP_14'].iloc[-1]) - float(prev_a['DMN_14'].iloc[-1])) / 100, 3)
    scl = s_i['Close']
    hs['er'] = float(abs(scl.iloc[-1] - scl.iloc[-11]) / scl.diff().abs().tail(10).sum())
    up, mid, lo = b_i.iloc[:, 2].iloc[-1], b_i.iloc[:, 1].iloc[-1], b_i.iloc[:, 0].iloc[-1]
    hs['bbw'] = float((up - lo) / mid * 100)
    hs['support'] = float(lo)
    hs['resistance'] = float(up)
    sm_i = sma20_s[sma20_s.index <= pd.Timestamp(asof)]
    hs['dev'] = float((scl.iloc[-1] - sm_i.iloc[-1]) / sm_i.iloc[-1] * 100)
    av_i = avg_vol_s[avg_vol_s.index <= pd.Timestamp(asof)]
    hs['vr'] = float(s_i['Volume'].iloc[-1] / av_i.iloc[-1])
    r_i = rsi_s[rsi_s.index <= pd.Timestamp(asof)]
    hs['rsi_14'] = float(r_i.iloc[-1])
    ep_i = ema20p_s[ema20p_s.index <= pd.Timestamp(asof)]
    hs['price_ema20_pct'] = float((scl.iloc[-1] / ep_i.iloc[-1] - 1) * 100)
    hs['last_updated'] = 0
    app.historical_stats = hs
    app.latest_data['index']['price'] = float(x_i['Close'].iloc[-1])
    p_prev = xsp[xsp.index < pd.Timestamp(asof)]
    app._get_xsp_prev_close = lambda: float(p_prev['Close'].iloc[-1])
    sp = spxl[spxl.index <= pd.Timestamp(asof)]
    spxl_price = float(sp['Close'].iloc[-1])
    return float(x_i['Close'].iloc[-1]), spxl_price, float(cv)

def make_chain(price, sigma, asof):
    """BS-synthesized option chain: expiries +7/+14/+21 days, strikes 700-810."""
    opts = {}
    for n in (7, 14, 21):
        exp = asof + timedelta(days=n)
        exp_str = exp.strftime('%Y-%m-%d')
        ds = exp.strftime('%y%m%d')
        T = n / 365.0
        for strike in range(700, 815, 5):
            for ot in ('C', 'P'):
                bs = pricing.black_scholes(price, strike, T, 0.05, sigma, ot)
                eps = 0.5
                d = (pricing.black_scholes(price + eps, strike, T, 0.05, sigma, ot)
                     - pricing.black_scholes(price - eps, strike, T, 0.05, sigma, ot)) / (2 * eps)
                sym = f"US.XSP{ds}{ot}{int(strike * 1000)}"
                opts[sym] = {'symbol': sym, 'strike': strike, 'expiry': exp_str,
                             'opt_type': ot, 'bid': round(bs * 0.98, 2), 'ask': round(bs * 1.02, 2),
                             'mid': round(bs, 2), 'delta': round(d, 4), 'gamma': 0.01,
                             'theta': -0.02, 'vega': 0.05, 'iv': sigma, 'is_watched': False,
                             'open_interest': 1000}
    app.latest_data['options'] = opts

# ── 2. auto-pick most recent crash signal day (XSP single-day drop > 0.5%) ──
def find_crash_day():
    closes = xsp['Close']
    for i in range(len(closes) - 1, 0, -1):
        chg = (closes.iloc[i] - closes.iloc[i - 1]) / closes.iloc[i - 1]
        if chg < -0.005:
            return xsp.index[i].date(), xsp.index[i - 1].date(), chg
    return None, None, None

CRASH_DATE, CRASH_PREV_DATE, CRASH_CHG = find_crash_day()

# ── 3. mock clock ──
def _clock_for(asof):
    mn = type('MN', (), {})()
    mn.syd_dt = datetime.datetime.combine(asof, datetime.time(21, 30, 0))
    mn.et_dt = datetime.datetime.combine(asof, datetime.time(7, 30, 0))
    real_dt = app.datetime
    class _FakeDatetime:
        def __getattr__(self, name):
            if name == 'now':
                def _now(tz=None):
                    if tz is app.ET_TZ:
                        return mn.et_dt.replace(tzinfo=app.ET_TZ)
                    if tz is app.S_TZ:
                        return mn.syd_dt.replace(tzinfo=app.S_TZ)
                    return datetime.datetime.now(tz)
                return _now
            return getattr(real_dt, name)
    return _FakeDatetime()

app.socketio = MagicMock()
app.send_telegram = lambda m: print(m)

def reset_state():
    for a in ('_active_position_date', '_entry_price', '_peak_price', '_etf_entry_price', '_etf_peak_price',
              '_prev_report_direction', '_mr_entry_date', '_mr_entry_price', '_mr_etf_entry_price',
              '_crash_entry_date', '_crash_entry_price', '_crash_k1', '_crash_k2', '_crash_debit',
              '_crash_sigma', '_crash_etf_entry', '_trend_opt_expiry', '_trend_opt_strike',
              '_trend_opt_strike2', '_trend_opt_entry', '_trend_opt_entry_date', '_trend_opt_sigma'):
        setattr(app, a, None)
    app._crash_etf_scaled = False
    app._trend_opt_pnl = 0.0
    app._latest_report = {}
    app._morning_report_date = ''
    app._evening_report_date = ''
    app._prev_report_score = 0
    app.user_watchlist = []
    app.POSITION_FILE = tempfile.mktemp(suffix='.json')
    app.WATCHLIST_FILE = tempfile.mktemp(suffix='.json')

def scenario(date, title, setup):
    reset_state()
    app.datetime = _clock_for(date)
    price, spxl_p, vix_p = build_snapshot(date)
    make_chain(price, vix_p / 100.0, date)
    app.latest_data['index']['price'] = price
    spxl = setup(price, spxl_p)
    app._etf_price_cache['SPXL'] = spxl
    print(); print("=" * 78); print("SCENARIO:", title); print("=" * 78)
    app.send_market_report('morning', force=False)
    print()
    print("[direction]", app._latest_report.get('direction'), "| reason:", app._latest_report.get('reason'))

# ── 4. scenarios ──
# 1 baseline (trend day, no position)
scenario(TREND_DATE, "1. 基线（无持仓，真实 06-03 信号）", lambda p, s: s)

# 2 trend open
def _trend_open(p, s):
    app._entry_price = p; app._peak_price = p
    app._etf_entry_price = s; app._etf_peak_price = s
    app._active_position_date = TREND_DATE
    app._prev_report_direction = 'CALL'; app._prev_report_score = 62
    return s
scenario(TREND_DATE, "2. 趋势·开仓（entry=peak=XSP 755.37, ETF=SPXL）", _trend_open)

# 3 trend higher peak
def _trend_peak(p, s):
    app._entry_price = p; app._peak_price = p + 10
    app._etf_entry_price = s; app._etf_peak_price = s + 3.5
    app._active_position_date = TREND_DATE - timedelta(days=2)
    app._prev_report_direction = 'CALL'; app._prev_report_score = 62
    app.latest_data['index']['price'] = p - 3
    return s + 1.5
scenario(TREND_DATE, "3. 趋势·新高跟踪（peak XSP +10 / SPXL +3.5, 现价 -3）", _trend_peak)

# 4 trend trail trigger
def _trend_trig(p, s):
    app._entry_price = p; app._peak_price = p + 45
    app._etf_entry_price = s; app._etf_peak_price = s + 18
    app._active_position_date = TREND_DATE - timedelta(days=5)
    app._prev_report_direction = 'CALL'; app._prev_report_score = 62
    app.latest_data['index']['price'] = p - 5
    return s - 12
scenario(TREND_DATE, "4. 趋势·触发平仓（peak +45, 现价 -5 < 跟踪 -3%）", _trend_trig)

# 5 MR holding (signal 03-27, holding report 03-30)
def _mr(p, s):
    x_i = xsp[xsp.index <= pd.Timestamp(MR_DATE)]
    sp_i = spxl[spxl.index <= pd.Timestamp(MR_DATE)]
    app._mr_entry_date = MR_DATE
    app._mr_entry_price = float(x_i['Close'].iloc[-1])
    app._mr_etf_entry_price = float(sp_i['Close'].iloc[-1])
    return s
scenario(date(2026, 3, 30), "5. MR持仓（03-27 入场, 03-30 早报）", _mr)

# 6 crash not scaled
def _crash(p, s):
    app._crash_entry_date = CRASH_DATE
    app._crash_entry_price = p
    k1 = app._s5(p); k2 = k1 + 15
    app._crash_k1 = k1; app._crash_k2 = k2
    app._crash_sigma = app.historical_stats.get('vix', 20) / 100.0
    T21 = 21 / 365.0
    r = 0.05
    e1 = pricing.black_scholes(p, k1, T21, r, app._crash_sigma, 'C')
    e2 = pricing.black_scholes(p, k2, T21, r, app._crash_sigma, 'C')
    app._crash_debit = e1 - e2
    app._crash_etf_entry = s
    app._crash_etf_scaled = False
    return s
scenario(CRASH_DATE, "6. 崩盘·未缩放（信号日入场）", _crash)

# 7 crash scaled
def _crash_scaled(p, s):
    app._crash_entry_date = CRASH_DATE
    app._crash_entry_price = p
    app._crash_k1 = None; app._crash_k2 = None; app._crash_debit = None; app._crash_sigma = None
    app._crash_etf_entry = s
    app._crash_etf_scaled = True
    return s
scenario(CRASH_DATE, "7. 崩盘·已缩放（期权已平, 剩$1k SPXL续持）", _crash_scaled)

print()
print("CRASH day auto-picked:", CRASH_DATE, "chg", f"{CRASH_CHG*100:.2f}%")
