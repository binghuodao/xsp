"""Full-history replay of the app's own signal engine -> real daily reports.

For EVERY trading day we rebuild app state (indicators +
BS option chain + mock evening clock) and call app.send_market_report('evening'),
so the app's position state machine runs continuously exactly like production.

Canonical backtest: python3 tests/sim_reports_full.py --no-net --period 7y
  = full data window (2021-06-01 -> today, ~5.4y; ^XSP data caps at 2021-03-01).
  Default --period 3y keeps the daily-review window used for day-to-day checks.

Output — batched into tests/sim_reports_full/ for easy lookup:
  index_{period}.txt    master index: per-strategy trade table (open/close/result/file:line)
  backtest_stats_{period}.txt  per-year PnL/WR table (BS repricing, r=5%)
  sim_rpt_YYYY.txt     full "state changed" reports per year (split to H1/H2 if large)
  Compact hold lines:  unchanged hold days are 1 line (date · D+n · #trade · peak · price · stop)

"State unchanged" = position fingerprint identical to yesterday AND no significant
close/trigger event. Those days emit only the compact line (or nothing if no position).

Usage:  python3 tests/sim_reports_full.py        (writes tests/sim_reports_full/*)
        python3 tests/sim_reports_full.py --no-net   (skip downloads; reuse cached csv)
"""
import sys, os, tempfile, datetime, json, argparse
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

OUT_DIR = os.path.join(_SCRIPT_DIR, 'sim_reports_full')
os.makedirs(OUT_DIR, exist_ok=True)
SPLIT_LINES = 3000          # year file bigger than this -> split H1/H2
WARMUP_DAYS = 60            # skip indicator warmup days
PERIOD = '3y'

# ═══════════════════════════════ 1. data ═══════════════════════════════
def _cols(df):
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_localize(None)
    return df

def _load(cache, ticker, period):
    csv = os.path.join(OUT_DIR, f'_{ticker}_{period}.csv')
    if cache and os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
    else:
        df = _cols(yf.Ticker(ticker).history(period=period)).sort_index()
        df.to_csv(csv)
    df = _cols(df).sort_index()
    df.drop_duplicates(inplace=True)
    return df

ap = argparse.ArgumentParser()
ap.add_argument('--no-net', action='store_true', help='use cached CSV only, no downloads')
ap.add_argument('--period', default='3y', help='yfinance download period (3y default; use 7y for the 6y backtest window)')
ap.add_argument('--crash-mode', default='V9', help='crash exit variant: V9 止损日期权续持 (default, production: stop closes ETF only, option rides to 收复入场价 or T+21 expiry兜底, multiple residuals can coexist) | V5 首阴+盈利保护 (prior default) | V4 首阴 (optimal at $2k) | V6 首阴+3天限 | V7 首阴+连阳2 | V8 首阴/二次首阳混合 | V0 baseline | V1 strict T+4 | V2 half-reset | V3 full-close')
ap.add_argument('--crash-half', type=float, default=0.125, help='crash ETF fraction sold at 首阳 (default 0.125 = V8d sell $625 keep $4375); 0.25 sell $1.25k keep $3.75k, 0.5 V4 legacy sell $2.5k keep $2.5k')
ap.add_argument('--stop-pct', type=float, default=0.025, help='crash XSP stop line = entry*(1-pct) (default 0.025 = -2.5%%)')
ap.add_argument('--drop-thresh', type=float, default=0.005, help='crash signal XSP daily-drop threshold (default 0.005 = 0.5%%)')
ap.add_argument('--stop-cooldown', type=int, default=0, help='days to block new crash entries after a crash stop-loss (default 0 = off)')
ap.add_argument('--reentry-pct', type=float, default=1.0, help='V4 re-entry trigger: price <= entry*this (default 1.0 = retrace to entry)')
ap.add_argument('--dte', type=int, default=21, help='crash CALL spread days-to-expiry (default 21)')
ap.add_argument('--spread-w', type=int, default=15, help='crash CALL spread width k2-k1 (default 15)')
ap.add_argument('--etf-stop', type=float, default=0.0, help='crash SPXL separate stop pct: exit ETF leg when SPXL <= entry*(1-pct), option rides (default 0 = off)')
ap.add_argument('--layer-priority', default='crash_mr_trend', help='delayed-open priority: crash_mr_trend (default, 崩盘优先) | mr_crash_trend (MR 优先承接恐慌日)')
ap.add_argument('--risk-gate', default='none', help='bear-regime gate (research): none | b200 (close<SMA200) | b200slope (close<SMA200 & SMA200 falling) | vix80 (VIX 252d pct>80) | macd (XSP MACD death cross) | engulf (bearish engulfing) | s3red (3 consecutive red days)')
ap.add_argument('--risk-mult', type=float, default=1.0, help='crash size multiplier when gate ON: 0 = skip entry, 0<mult<1 = scale PnL (default 1.0 = gate inert)')
ap.add_argument('--opt-mult', type=float, default=1.0, help='crash option-leg PnL multiplier (ETF leg unchanged): 1.0 baseline, 0.5 half-size option, 0 = ETF-only')
ap.add_argument('--outdir', default=OUT_DIR, help='output dir for index/stats/report files (default: tests/sim_reports_full)')
ap.add_argument('--stats-only', action='store_true', help='skip per-year sim_rpt batch files; write only index + backtest_stats')
ap.add_argument('--warmup', type=int, default=60, help='indicator warmup days to skip (default 60; 0 = replay from data start 2021-03-01)')
args = ap.parse_args()

PERIOD = args.period
CRASH_MODE = args.crash_mode
CRASH_HALF = args.crash_half
STOP_PCT = args.stop_pct
REENTRY_PCT = args.reentry_pct
DROP_THRESH = args.drop_thresh
STOP_COOLDOWN = args.stop_cooldown
DTE = args.dte
SPREAD_W = args.spread_w
ETF_STOP = args.etf_stop
LAYER_PRIORITY = args.layer_priority
RISK_GATE = args.risk_gate
RISK_MULT = args.risk_mult
OPT_MULT = args.opt_mult
RESULT_DIR = args.outdir
STATS_ONLY = args.stats_only
os.makedirs(RESULT_DIR, exist_ok=True)
WARMUP_DAYS = args.warmup
REPLAY_START = pd.Timestamp('2020-08-01')   # canonical backtest floor; ^XSP data caps at 2021-03-01, so window = data-available

print(f"Loading real history ({PERIOD})...")
xsp = _load(args.no_net, '^XSP', PERIOD)
vix = _load(args.no_net, '^VIX', PERIOD)
spy = _load(args.no_net, 'SPY', PERIOD)
spxl = _load(args.no_net, 'SPXL', PERIOD)
skew = _load(args.no_net, '^SKEW', PERIOD)

adx_df = ta.adx(spy['High'], spy['Low'], spy['Close'], length=14)
bb_df = ta.bbands(spy['Close'], length=20, std=2)
rsi_s = ta.rsi(spy['Close'], length=14)
sma20_s = spy['Close'].rolling(20).mean()
avg_vol_s = spy['Volume'].rolling(20).mean()
ema20p_s = spy['Close'].ewm(span=20, adjust=False).mean()

# ── risk-gate point-in-time features (XSP index regime; slice by index<=asof in build_snapshot) ──
_xc = xsp['Close']
sma200_s = _xc.rolling(200).mean()
macd_s = _xc.ewm(span=12, adjust=False).mean() - _xc.ewm(span=26, adjust=False).mean()
macdsig_s = macd_s.ewm(span=9, adjust=False).mean()
_ox, _cx = xsp['Open'], xsp['Close']
engulf_s = (_cx < _ox) & (_cx.shift(1) > _ox.shift(1)) & (_ox >= _cx.shift(1)) & (_cx <= _ox.shift(1))
s3red_s = (_cx < _cx.shift(1)) & (_cx.shift(1) < _cx.shift(2)) & (_cx.shift(2) < _cx.shift(3))
vix_pct_252_s = vix['Close'].rolling(252).rank(pct=True) * 100

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
    app._crash_size_mult = RISK_MULT
    if RISK_GATE == 'none':
        flag = False
    else:
        s200_i = sma200_s[sma200_s.index <= pd.Timestamp(asof)]
        s200 = float(s200_i.iloc[-1]) if len(s200_i) else None
        s200_prev = float(s200_i.iloc[-21]) if len(s200_i) >= 21 else None
        cur = float(x_i['Close'].iloc[-1])
        if RISK_GATE == 'b200':
            flag = s200 is not None and cur < s200
        elif RISK_GATE == 'b200slope':
            flag = s200 is not None and s200_prev is not None and cur < s200 and (s200 - s200_prev) < 0
        elif RISK_GATE == 'vix80':
            v252 = vix_pct_252_s[vix_pct_252_s.index <= pd.Timestamp(asof)]
            flag = bool(v252.iloc[-1] > 80) if len(v252) else False
        elif RISK_GATE == 'macd':
            m_i = macd_s[macd_s.index <= pd.Timestamp(asof)]
            s_i2 = macdsig_s[macdsig_s.index <= pd.Timestamp(asof)]
            flag = bool(m_i.iloc[-1] < s_i2.iloc[-1]) if len(m_i) and len(s_i2) else False
        elif RISK_GATE == 'engulf':
            e_i = engulf_s[engulf_s.index <= pd.Timestamp(asof)]
            flag = bool(e_i.iloc[-1]) if len(e_i) else False
        elif RISK_GATE == 's3red':
            r3 = s3red_s[s3red_s.index <= pd.Timestamp(asof)]
            flag = bool(r3.iloc[-1]) if len(r3) else False
        else:
            raise ValueError(f'unknown --risk-gate {RISK_GATE}')
    app._risk_off_active = (lambda f=bool(flag): f)
    app._crash_etf_size = int(ETF_SIZE['CRASH'] * (RISK_MULT if flag else 1.0))
    sp = spxl[spxl.index <= pd.Timestamp(asof)]
    spxl_price = float(sp['Close'].iloc[-1])
    return float(x_i['Close'].iloc[-1]), spxl_price, float(cv)

def make_chain(price, sigma, asof):
    """BS-synthesized option chain: expiries +7/+14/+21 days, dynamic strike band."""
    opts = {}
    lo = int(price * 0.92 // 5) * 5 - 5
    hi = int(price * 1.10 // 5) * 5 + 10
    for n in (7, 14, 21):
        exp = asof + timedelta(days=n)
        exp_str = exp.strftime('%Y-%m-%d')
        ds = exp.strftime('%y%m%d')
        T = n / 365.0
        for strike in range(lo, hi + 1, 5):
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

# ═══════════════════════════════ 2. harness ═══════════════════════════════
def _clock_for(asof):
    mn = type('MN', (), {})()
    mn.syd_dt = datetime.datetime.combine(asof + timedelta(days=1), datetime.time(6, 30, 0))
    mn.et_dt = datetime.datetime.combine(asof, datetime.time(16, 30, 0))
    real_dt = datetime.datetime
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
_captured = {}
def _capture(m):
    _captured['msg'] = m
app.send_telegram = _capture

def init_state():
    for a in ('_active_position_date', '_entry_price', '_peak_price', '_etf_entry_price', '_etf_peak_price',
              '_prev_report_direction', '_mr_entry_date', '_mr_entry_price', '_mr_etf_entry_price',
              '_crash_entry_date', '_crash_entry_price', '_crash_k1', '_crash_k2', '_crash_debit',
              '_crash_sigma', '_crash_etf_entry', '_crash_half_date', '_crash_reentry_date',
              '_trend_opt_expiry', '_trend_opt_strike',
              '_trend_opt_strike2', '_trend_opt_entry', '_trend_opt_entry_date', '_trend_opt_sigma'):
        setattr(app, a, None)
    app._crash_resids = []
    app._crash_etf_scaled = False
    app._crash_reentry = False
    app._crash_exit_mode = CRASH_MODE
    app._crash_half_pct = CRASH_HALF
    app._crash_stop_pct = STOP_PCT
    app._crash_reentry_pct = REENTRY_PCT
    app._crash_dte = DTE
    app._crash_spread_w = SPREAD_W
    app._crash_drop_thresh = DROP_THRESH
    app._crash_etf_stop_pct = ETF_STOP
    app._crash_etf_out = False
    app._layer_priority = LAYER_PRIORITY
    app._crash_stop_cooldown = STOP_COOLDOWN
    app._crash_stop_date = None
    app._crash_size_mult = RISK_MULT
    app._crash_etf_size = int(ETF_SIZE['CRASH'] * RISK_MULT)
    app._risk_off_active = lambda: False
    app._trend_opt_pnl = 0.0
    app._latest_report = {}
    app._morning_report_date = ''
    app._evening_report_date = ''
    app._prev_report_score = 0
    app.user_watchlist = []
    app.POSITION_FILE = tempfile.mktemp(suffix='.json')
    app.WATCHLIST_FILE = tempfile.mktemp(suffix='.json')
    app._etf_price_cache = {}
    app.historical_stats = {}

def state_fp():
    return (app._prev_report_direction,
            app._active_position_date,
            app._trend_opt_expiry, app._trend_opt_strike, app._trend_opt_strike2,
            app._mr_entry_date,
            app._crash_entry_date,
            app._crash_etf_scaled,
            app._crash_reentry,
            app._crash_etf_out)

def infer_blocked(msg, r):
    return (r.get('direction') == 'CALL' and '暂缓' in msg and '★ 做多 ETF' not in msg)

# ═══════════════════════════════ 3. trade ledger ═══════════════════════════════
ledger = {'TREND': [], 'MR': [], 'CRASH': []}
_open_ref = {'TREND': None, 'MR': None, 'CRASH': None}  # n of currently-open trade
ETF_SIZE = {'TREND': 5000, 'MR': 2000, 'CRASH': 5000}

def open_trade(kind, asof):
    n = len(ledger[kind]) + 1
    ledger[kind].append({'n': n, 'kind': kind, 'open': asof, 'open_p': None,
                         'close': None, 'close_p': None, 'result': None,
                         'file': None, 'line': None, 'rolls': 0,
                         'etf_entry': None, 'etf_pnl': 0.0, 'opt_pnl': 0.0, 'segs': [],
                         'mr_K': None, 'mr_sigma': None, 'mr_expiry': None, 'mr_opt_entry': None,
                         'k1': None, 'k2': None, 'debit': None, 'sigma': None,
                         'half_etf': 0.0, 'half_date': None, 'etf_out': False, 'size_mult': 1.0,
                         'resid_entry': None, 'resid_expiry': None, 'resid': None})
    _open_ref[kind] = n
    return n

def _bs_spread(price, k1, k2, T, sigma):
    if T <= 0: T = 1 / 365.0
    if not sigma or sigma <= 0: sigma = 0.20
    return pricing.black_scholes(price, k1, T, 0.05, sigma, 'C') - pricing.black_scholes(price, k2, T, 0.05, sigma, 'C')

def _bs_call(price, K, T, sigma):
    if T <= 0: T = 1 / 365.0
    if not sigma or sigma <= 0: sigma = 0.20
    return pricing.black_scholes(price, K, T, 0.05, sigma, 'C')

def _seg_from_app():
    if not app._trend_opt_expiry:
        return None
    return {'expiry': app._trend_opt_expiry,
            'k1': app._trend_opt_strike, 'k2': app._trend_opt_strike2,
            'debit': app._trend_opt_entry, 'sigma': app._trend_opt_sigma,
            'open': app._trend_opt_entry_date}

def _seg_close_pnl(seg, asof, price):
    """Realized PnL of one spread segment closed at `asof` close."""
    if not seg or not seg.get('k1') or not seg.get('k2') or seg.get('debit') is None:
        return 0.0
    exp = datetime.datetime.strptime(seg['expiry'], '%Y-%m-%d').date()
    T_rem = max((exp - asof).days / 365.0, 1 / 365.0)
    close_d = _bs_spread(price, seg['k1'], seg['k2'], T_rem, seg.get('sigma'))
    return max((close_d - seg['debit']), -seg['debit']) * 100

def book_spread(asof, price, prev_seg, now_seg):
    """Accrue trend spread segment PnL into the currently-open TREND trade.
    Roll/close when the segment changed (expiry or strikes differ); persist otherwise."""
    t = cur_trade('TREND')
    if not t:
        return
    same = (prev_seg is not None and now_seg is not None
            and prev_seg.get('expiry') == now_seg.get('expiry')
            and prev_seg.get('k1') == now_seg.get('k1'))
    if prev_seg and not same:
        t['opt_pnl'] = (t.get('opt_pnl') or 0) + _seg_close_pnl(prev_seg, asof, price)
    if now_seg and not same:
        t['segs'] = list(t.get('segs') or []) + [now_seg]

def cur_trade(kind):
    if _open_ref[kind] is None:
        return None
    for t in ledger[kind]:
        if t['n'] == _open_ref[kind]:
            return t
    return None

def close_trade(kind, asof, price, result):
    t = cur_trade(kind)
    if t:
        t['close'] = asof
        t['close_p'] = price
        t['result'] = result
        t['hold_days'] = max(len(pd.bdate_range(t['open'], asof)) - 1, 0)
    _open_ref[kind] = None

def settle_residuals(asof, price):
    """V9: settle any crash options riding past their XSP stop-loss (close on 收复入场价/首阳 or T+21 expiry)."""
    done = []
    for t in ledger['CRASH']:
        r = t.get('resid')
        if not r:
            continue
        days_left = (r['expiry'] - asof).days
        if price <= r['entry'] and days_left > 0:
            continue
        T_rem = max(days_left / 365.0, 1 / 365.0)
        close_d = _bs_spread(price, r['k1'], r['k2'], T_rem, r['sigma'])
        t['opt_pnl'] = max((close_d - r['debit']), -r['debit']) * 100 * t.get('size_mult', 1.0) * OPT_MULT
        reason = '残期收复' if price > r['entry'] else '到期兜底'
        t['resid'] = None
        done.append(f'CRASH#{t["n"]} 残期{reason}')
    return done

def trend_close_result(alerts):
    s = ' | '.join(alerts)
    if 't+30' in s: return 't+30强制平仓'
    if '滚动CALL价差已平仓' in s: return '趋势结束平价差'
    if '跟踪' in s: return '跟踪-3%触发'
    if '入场硬止损' in s: return '入场硬止损-2%'
    if '方向已由' in s: return '方向转变'
    if 'BB中段' in s: return 'BB中段/综合分不足'
    return '方向转空'

# ═══════════════════════════════ 4. assertions ═══════════════════════════════
def check_day(asof, msg, r, price, blocked, failures):
    f = []
    d = r.get('direction')
    # direction line consistency
    if d == 'CALL' and '→ 方向: CALL' not in msg:
        f.append('方向行缺失(CALL)')
    if d is None and 'BB中段' not in msg:
        f.append('方向行缺失(None)')
    # zero tree rows (tree removed)
    if '树' in msg:
        f.append('报告含树行')
    # blocked behavior
    if blocked:
        if '★ 做多 ETF' in msg: f.append('blocked日现ETF线')
        if '🛑' in msg: f.append('blocked日现止损线')
        if '暂缓' not in msg: f.append('blocked日缺暂缓提示')
        if d != 'CALL': f.append('blocked日方向异常')
        if app._active_position_date is not None: f.append('blocked日误开仓')
    # trend stop formula
    if r.get('stop_loss') and app._entry_price:
        e = app._entry_price
        is_nearbb = bool(r.get('reason')) and '贴BB' in r.get('reason')
        pct = 0.01 if is_nearbb else 0.05
        exp_fixed = e * (1 - pct)
        if f"固定 ${exp_fixed:.2f} (-{pct*100:.0f}%" not in msg:
            f.append(f'固定止损不符(期望{e*(1-pct):.2f})')
    # MR formula
    if r.get('mr_entry_price'):
        exp_stop = r['mr_entry_price'] * 0.98
        exp_green = r['mr_entry_price'] * 1.003
        if f"止损 ${exp_stop:.2f} (-2%)" not in msg: f.append('MR止损值不符')
        if f"首阳 ${exp_green:.2f} (+0.3%)" not in msg: f.append('MR首阳值不符')
        if r.get('mr_days') is not None and r['mr_days'] >= 3:
            if not any(x in msg for x in ('MR已持3天', 'MR首阳', 'MR跌穿')):
                f.append('MR到期未平仓')
    # crash formula
    if r.get('crash_entry_price'):
        exp_stop = r['crash_entry_price'] * (1 - STOP_PCT)
        exp_green = r['crash_entry_price'] * 1.0
        if f"止损 ${exp_stop:.2f} (-{STOP_PCT:.1%})" not in msg: f.append('崩盘止损值不符')
        if f"首阳 ${exp_green:.2f}" not in msg: f.append('崩盘首阳值不符')
        cf = r.get('crash_force_days')
        if cf is None:
            cf = r.get('crash_days')
        if cf is not None and cf >= 4:
            if not any(x in msg for x in ('已持4天', '跌穿止损', '首阳', '首阴', '二次首阳')):
                f.append('崩盘到期未平仓')
    for x in f:
        failures.append(f"[{asof}] {x}")


# ═══════════════════════════════ 5. replay ═══════════════════════════════
def main():
    init_state()
    trading_days = list(xsp.index)
    if REPLAY_START is not None:
        trading_days = [d for d in trading_days if d >= REPLAY_START]
    if len(trading_days) <= WARMUP_DAYS:
        sys.exit('not enough history')
    trading_days = trading_days[WARMUP_DAYS:]
    print(f"Replaying {len(trading_days)} trading days  {trading_days[0].date()} → {trading_days[-1].date()}")

    records = []          # one entry per emitted day
    stats = {'full': 0, 'compact': 0, 'skipped': 0, 'opens': 0, 'closes': 0}
    failures = []

    prev_fp = None
    prev_dir = None
    prev_blocked = None
    prev_spread_seg = None

    for i, day in enumerate(trading_days):
        asof = day.date()
        price, spxl_p, vix_p = build_snapshot(asof)
        make_chain(price, vix_p / 100.0, asof)
        app.latest_data['index']['price'] = price
        app._etf_price_cache['SPXL'] = spxl_p
        app.datetime = _clock_for(asof)
        _captured.clear()
        app.send_market_report('evening', force=False)
        msg = _captured.get('msg', '')
        if not msg:
            stats['skipped'] += 1
            continue
        r = app._latest_report
        now_fp = state_fp()
        d = r.get('direction')
        blocked = infer_blocked(msg, r)
        alerts = r.get('close_alerts', []) or []
        sig_alerts = [l for l in alerts if 'BB中段' not in l]   # ignore daily background hint
        now_spread_seg = _seg_from_app()

        ev = []
        # trend open/close (supports same-day close+reopen: prev_fp[1] != now_fp[1])
        if prev_fp is not None:
            tr_changed = prev_fp[1] != now_fp[1]
            if tr_changed and prev_fp[1] is not None:
                n = cur_trade('TREND')['n'] if cur_trade('TREND') else None
                t = cur_trade('TREND')
                if t and t.get('etf_entry'):
                    t['etf_pnl'] = ETF_SIZE['TREND'] * (spxl_p / t['etf_entry'] - 1)
                book_spread(asof, price, prev_spread_seg, now_spread_seg)
                close_trade('TREND', asof, price, trend_close_result(alerts))
                ev.append(f'TREND#{n} 平仓')
            if tr_changed and now_fp[1] is not None:
                n = open_trade('TREND', asof)
                t = cur_trade('TREND')
                if t: t['etf_entry'] = spxl_p
                ev.append(f'TREND#{n} 开仓')
            if prev_fp[2] is None and now_fp[2] is not None:
                ev.append('价差开')
            elif prev_fp[2] is not None and now_fp[2] is None:
                ev.append('价差平')
            if '滚仓' in msg:
                t = cur_trade('TREND')
                if t: t['rolls'] += 1
                ev.append('价差滚仓')
            mr_changed = prev_fp[5] != now_fp[5]
            if mr_changed and prev_fp[5] is not None:
                n = cur_trade('MR')['n'] if cur_trade('MR') else None
                t = cur_trade('MR')
                if t:
                    if t.get('mr_K') is not None and t.get('mr_opt_entry') is not None:
                        T_rem = max((t['mr_expiry'] - asof).days / 365.0, 1 / 365.0)
                        exit_opt = _bs_call(price, t['mr_K'], T_rem, t.get('mr_sigma'))
                        t['opt_pnl'] = max((exit_opt - t['mr_opt_entry']), -t['mr_opt_entry']) * 100
                    if t.get('etf_entry'):
                        t['etf_pnl'] = ETF_SIZE['MR'] * (spxl_p / t['etf_entry'] - 1)
                res = ('首阳+0.3%' if 'MR首阳' in msg else '止损-2%' if 'MR跌穿' in msg else '3天强制平')
                close_trade('MR', asof, price, res)
                ev.append(f'MR#{n} 平仓')
            if mr_changed and now_fp[5] is not None:
                n = open_trade('MR', asof)
                t = cur_trade('MR')
                if t:
                    t['etf_entry'] = spxl_p
                    t['mr_K'] = app._s5(app._mr_entry_price or price)
                    t['mr_sigma'] = vix_p / 100.0
                    t['mr_expiry'] = asof + timedelta(days=7)
                    t['mr_opt_entry'] = _bs_call(price, t['mr_K'], 7 / 365.0, t['mr_sigma'])
                ev.append(f'MR#{n} 开仓')
            cr_changed = prev_fp[6] != now_fp[6]
            if cr_changed and prev_fp[6] is not None:
                n = cur_trade('CRASH')['n'] if cur_trade('CRASH') else None
                t = cur_trade('CRASH')
                if t:
                    sm = t.get('size_mult', 1.0)
                    osm = sm * OPT_MULT
                    cal = (asof - t['open']).days
                    T_rem = max(DTE / 365.0 - cal / 365.0, 1 / 365.0)
                    if t.get('etf_out'):
                        if t.get('half_date') is None and t.get('k1') and t.get('k2') and t.get('debit') is not None:
                            close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                            t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
                    elif t.get('half_date'):
                        base = t.get('re_entry_spxl') or t.get('etf_entry')
                        rem = (ETF_SIZE['CRASH'] * (1 - CRASH_HALF)) * (spxl_p / base - 1) if base else 0
                        t['etf_pnl'] = ((t.get('half_etf') or 0) + rem) * sm
                    else:
                        if t.get('k1') and t.get('k2') and t.get('debit') is not None:
                            if CRASH_MODE == 'V9' and '崩盘跌穿' in msg:
                                t['resid'] = {'k1': t['k1'], 'k2': t['k2'], 'debit': t['debit'],
                                              'sigma': t.get('sigma') or 0.20,
                                              'entry': t['resid_entry'], 'expiry': t['resid_expiry']}
                            else:
                                close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                                t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
                        if t.get('etf_entry'):
                            t['etf_pnl'] = ETF_SIZE['CRASH'] * (spxl_p / t['etf_entry'] - 1) * sm
                res = ('首阴清仓' if '崩盘首阴' in msg else '二次首阳清仓' if '崩盘二次首阳' in msg else '首阳退半' if '崩盘首阳' in msg
                       else '止损-2.5%' if '崩盘跌穿' in msg else '4天强制平')
                close_trade('CRASH', asof, price, res)
                ev.append(f'CRASH#{n} 平仓')
            if cr_changed and now_fp[6] is not None:
                n = open_trade('CRASH', asof)
                t = cur_trade('CRASH')
                if t:
                    t['etf_entry'] = spxl_p
                    t['k1'] = app._crash_k1; t['k2'] = app._crash_k2
                    t['debit'] = app._crash_debit; t['sigma'] = app._crash_sigma
                    t['size_mult'] = RISK_MULT if app._risk_off_active() else 1.0
                    t['resid_entry'] = price
                    t['resid_expiry'] = asof + timedelta(days=DTE)
                ev.append(f'CRASH#{n} 开仓')
            if not prev_fp[7] and now_fp[7]:
                t = cur_trade('CRASH')
                if t:
                    sm = t.get('size_mult', 1.0)
                    osm = sm * OPT_MULT
                    cal = (asof - t['open']).days
                    T_rem = max(DTE / 365.0 - cal / 365.0, 1 / 365.0)
                    if t.get('k1') and t.get('k2') and t.get('debit') is not None:
                        close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                        t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
                    if not t.get('etf_out') and t.get('etf_entry'):
                        t['half_etf'] = (ETF_SIZE['CRASH'] * CRASH_HALF) * (spxl_p / t['etf_entry'] - 1) * sm
                    t['half_date'] = asof
                ev.append('崩盘退半')
            if not prev_fp[8] and now_fp[8]:
                t = cur_trade('CRASH')
                if t:
                    t['re_entry_spxl'] = spxl_p
                ev.append('崩盘再进')
            if not prev_fp[9] and now_fp[9]:
                t = cur_trade('CRASH')
                if t and not t.get('etf_out'):
                    sm = t.get('size_mult', 1.0)
                    if t.get('half_date'):
                        base = t.get('re_entry_spxl') or t.get('etf_entry')
                        rem = (ETF_SIZE['CRASH'] * (1 - CRASH_HALF)) * (spxl_p / base - 1) if base else 0
                        t['etf_pnl'] = ((t.get('half_etf') or 0) + rem) * sm
                    elif t.get('etf_entry'):
                        t['etf_pnl'] = ETF_SIZE['CRASH'] * (spxl_p / t['etf_entry'] - 1) * sm
                    t['etf_out'] = True
                ev.append('崩盘ETF止损')
            if prev_blocked is not None and prev_blocked != blocked:
                ev.append('高位拦截' if blocked else '拦截解除')
            if prev_dir is not None and d != prev_dir:
                ev.append(f'方向 {prev_dir}→{d}')
        if sig_alerts:
            ev.append('平仓提示')
        book_spread(asof, price, prev_spread_seg, now_spread_seg)
        ev.extend(settle_residuals(asof, price))
        fp_changed = (prev_fp is not None and now_fp != prev_fp)

        is_first = (i == 0)
        if ev or fp_changed or is_first:
            check_day(asof, msg, r, price, blocked, failures)
            records.append({'asof': asof, 'full': True,
                            'tag': ' | '.join(ev) if ev else ('SEED' if is_first else '状态变化'),
                            'msg': msg, 'dir': d, 'score': r.get('score'), 'blocked': blocked})
            stats['full'] += 1
        else:
            holds = [(k, cur_trade(k)) for k in ('TREND', 'MR', 'CRASH') if cur_trade(k)]
            if holds:
                line_parts = []
                for k, t in holds:
                    if k == 'TREND' and r.get('stop_loss'):
                        sl = r['stop_loss'][0].split('|')[0].strip().replace('止损(基准) ', '')
                        dd = max(len(pd.bdate_range(t['open'], asof)) - 1, 0)
                        line_parts.append(f"TREND#{t['n']} D+{dd} peak {app._peak_price:.2f} 现价 {price:.2f} 止损 {sl}")
                    elif k == 'MR' and r.get('mr_entry_price'):
                        dd = r.get('mr_days', 0)
                        line_parts.append(f"MR#{t['n']} D+{dd} 现价 {price:.2f} 止损 {r['mr_stop']:.2f} 首阳 {r['mr_green']:.2f}")
                    elif k == 'CRASH' and r.get('crash_entry_price'):
                        dd = r.get('crash_days', 0)
                        line_parts.append(f"CRASH#{t['n']} D+{dd} 现价 {price:.2f} 止损 {r['crash_stop']:.2f} 首阳 {r['crash_green']:.2f}")
                if line_parts:
                    records.append({'asof': asof, 'full': False, 'tag': '', 'msg': ' | '.join(line_parts),
                                    'dir': d, 'score': r.get('score'), 'blocked': blocked})
                    stats['compact'] += 1
            else:
                stats['skipped'] += 1

        prev_fp = now_fp
        prev_dir = d
        prev_blocked = blocked
        prev_spread_seg = now_spread_seg

    stats['opens'] = sum(len(ledger[k]) for k in ledger)
    stats['closes'] = sum(1 for k in ledger for t in ledger[k] if t.get('close'))

    # ═══════════════════════════════ 6. render batch files ═══════════════════════════════
    def _rec_lines(rec):
        if rec['full']:
            d = rec['dir']; sc = rec.get('score')
            tag = rec['tag']
            head = f"[{rec['asof']} {rec['asof'].strftime('%a')}]  {tag}  方向{str(d)} 综合{sc}"
            sep = '─' * 70
            return [head, sep] + rec['msg'].split('\n')
        return [f"    {rec['asof']}  {rec['msg']}"]

    # group records by year
    by_year = {}
    for rec in records:
        by_year.setdefault(rec['asof'].year, []).append(rec)

    file_specs = []   # (file_path, [lines], [ (asof, rec_index_global) ])
    rec_file = {}     # id(rec) -> (filename, start_line)
    for yr, rcs in sorted(by_year.items()):
        lines_all = []
        for rc in rcs:
            lines_all.extend(_rec_lines(rc))
        split = len(lines_all) > SPLIT_LINES
        if split:
            h1 = [rc for rc in rcs if rc['asof'].month <= 6]
            h2 = [rc for rc in rcs if rc['asof'].month > 6]
            halves = [(f'sim_rpt_{yr}_H1.txt', h1), (f'sim_rpt_{yr}_H2.txt', h2)]
        else:
            halves = [(f'sim_rpt_{yr}.txt', rcs)]
        for fn, sub in halves:
            body = []
            body.append('═' * 70)
            body.append(f'XSP 盘后晚报 — 全历史重放（{trading_days[0].date()} → {trading_days[-1].date()}）  文件={fn}')
            body.append(f'完整晚报 = 状态变化日(开仓/触发/平仓/拦截/方向变) | 缩进行 = 持有无变化')
            body.append(f'生成: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}  | 数据: yfinance {PERIOD} | 全部盘后 16:30 ET')
            body.append('═' * 70)
            body.append('')
            start_line = len(body) + 1
            for rc in sub:
                if not STATS_ONLY:
                    rec_file[id(rc)] = (fn, start_line)
                ls = _rec_lines(rc)
                body.extend(ls)
                start_line += len(ls)
            if not STATS_ONLY:
                with open(os.path.join(RESULT_DIR, fn), 'w') as f:
                    f.write('\n'.join(body) + '\n')
                print(f"  wrote {fn}: {len(sub)} entries, {len(body)} lines")

    # fill trade file/line refs + open prices from the actual report
    def _price_from_rec(rec, asof):
        # open price = the report's 现价 line (2nd block line) fallback: parse msg header
        for line in rec['msg'].split('\n'):
            if '现价 $' in line:
                try:
                    return float(line.split('现价 $')[1].split()[0])
                except Exception:
                    pass
        return None

    for t in ledger['TREND'] + ledger['MR'] + ledger['CRASH']:
        for rc in records:
            if rc['asof'] == t['open']:
                t['open_p'] = _price_from_rec(rc, t['open']) or t['open_p']
            if rc['asof'] == t['close']:
                t['close_p'] = _price_from_rec(rc, t['close']) or t['close_p']
        for rc in records:
            if rc['asof'] == t['open'] and id(rc) in rec_file:
                t['file'], t['line'] = rec_file[id(rc)]
            if rc['asof'] == t['close'] and id(rc) in rec_file:
                t['file_close'], t['line_close'] = rec_file[id(rc)]

    # ═══════════════════════════════ 7. master index ═══════════════════════════════
    idx = []
    idx.append('═' * 70)
    idx.append('XSP 三策略 全历史重放 — 交易总表')
    idx.append(f'重放区间: {trading_days[0].date()} → {trading_days[-1].date()}  ({len(trading_days)} 交易日)')
    idx.append(f'完整晚报 {stats["full"]} 日 | 紧凑持有行 {stats["compact"]} 日 | 跳过 {stats["skipped"]} 日')
    idx.append('查找: 定位列给出 文件:行号; 全部为盘后晚报(16:30 ET)')
    idx.append('═' * 70)
    idx.append('')
    idx.append(f'── TREND 趋势 ETF($5k SPXL)+14DTE CALL价差  共 {len(ledger["TREND"])} 笔 ──')
    idx.append('  #  开仓日        入场价    平仓日        出场价    天数  滚仓  结果            定位')
    for t in ledger['TREND']:
        fl = f"{t.get('file')}:{t.get('line')}" if t.get('file') else '-'
        cl = f"{t.get('file_close')}:{t.get('line_close')}" if t.get('file_close') else '-'
        idx.append(f"  {t['n']:<3}{t['open']}  {t['open_p'] or 0:8.2f}  {str(t['close']):<10}  {t['close_p'] or 0:8.2f}  {t.get('hold_days','-'):>4}  {t.get('rolls',0):>3}  {str(t.get('result','')):<14} {fl}")
        idx.append(f"       平仓定位: {cl}")
    idx.append('')
    idx.append(f'── MR 裸买CALL 7DTE (RSI<30+VIX>20)  共 {len(ledger["MR"])} 笔 ──')
    idx.append('  #  开仓日        入场价    平仓日        出场价    天数  结果            定位')
    for t in ledger['MR']:
        fl = f"{t.get('file')}:{t.get('line')}" if t.get('file') else '-'
        idx.append(f"  {t['n']:<3}{t['open']}  {t['open_p'] or 0:8.2f}  {str(t['close']):<10}  {t['close_p'] or 0:8.2f}  {t.get('hold_days','-'):>4}  {str(t.get('result','')):<14} {fl}")
    idx.append('')
    idx.append(f'── CRASH 崩盘 CALL价差15点21DTE+$5k SPXL  共 {len(ledger["CRASH"])} 笔 ──')
    idx.append('  #  开仓日        入场价    平仓日        出场价    天数  结果            定位')
    for t in ledger['CRASH']:
        fl = f"{t.get('file')}:{t.get('line')}" if t.get('file') else '-'
        idx.append(f"  {t['n']:<3}{t['open']}  {t['open_p'] or 0:8.2f}  {str(t['close']):<10}  {t['close_p'] or 0:8.2f}  {t.get('hold_days','-'):>4}  {str(t.get('result','')):<14} {fl}")
    idx.append('')
    idx.append('说明: 平仓结果=触发类型; 趋势平仓=方向转空/BB中段/跟踪/硬止损; MR 3天/崩盘4天=强制平')

    index_path = os.path.join(RESULT_DIR, f'index_{PERIOD}.txt')
    with open(index_path, 'w') as f:
        f.write('\n'.join(idx) + '\n')
    print(f"  wrote index_{PERIOD}.txt: {len(ledger['TREND'])} TREND, {len(ledger['MR'])} MR, {len(ledger['CRASH'])} CRASH trades")

    # ═══════════════════════════════ 7.6 per-trade PnL ledger ═══════════════════════════════
    trade_path = os.path.join(RESULT_DIR, f'trades_{PERIOD}.csv')
    with open(trade_path, 'w') as f:
        f.write('kind,n,open,open_p,close,close_p,hold_days,result,opt_pnl,etf_pnl,total_pnl\n')
        for t in ledger['TREND'] + ledger['MR'] + ledger['CRASH']:
            op = t.get('open_p') or 0
            cp = t.get('close_p') or 0
            hd = t.get('hold_days', '')
            opt = t.get('opt_pnl') or 0
            etf = t.get('etf_pnl') or 0
            f.write(f"{t['kind']},{t['n']},{t['open']},{op:.2f},{str(t.get('close'))},{cp:.2f},{hd},{t.get('result')},{opt:.2f},{etf:.2f},{opt + etf:.2f}\n")
    print(f"  wrote trades_{PERIOD}.csv: {len(ledger['TREND']) + len(ledger['MR']) + len(ledger['CRASH'])} trades")

    # ═══════════════════════════════ 7.5 backtest stats ═══════════════════════════════
    def _trade_pnl(t):
        return (t.get('etf_pnl') or 0) + (t.get('opt_pnl') or 0)

    def _trade_cost(t):
        if t['kind'] == 'TREND':
            segs = t.get('segs') or []
            if not segs:
                return 0.0
            return sum((seg.get('debit') or 0) for seg in segs) / len(segs) * 100
        if t['kind'] == 'MR':
            return (t.get('mr_opt_entry') or 0) * 100
        return (t.get('debit') or 0) * 100 * t.get('size_mult', 1.0)

    bt = []
    bt.append('═' * 70)
    bt.append(f'XSP 三策略 全历史重放 — 回测统计  ({trading_days[0].date()} → {trading_days[-1].date()}, {len(trading_days)} 交易日)')
    bt.append(f'数据: yfinance {PERIOD} | 口径: app 重放逐笔 PnL, 期权 BS 重定价 r=5%, 费用未计')
    bt.append(f'崩盘出场模式: {CRASH_MODE} (首阳退半 {CRASH_HALF:.0%} | 止损-{STOP_PCT:.1%} | 再进≤入场×{REENTRY_PCT:.2f} | 价差{SPREAD_W}点 {DTE}DTE)')
    if RISK_GATE != 'none':
        bt.append(f'熊市门: {RISK_GATE} × mult {RISK_MULT:g} (ON=门亮时崩盘仓缩至 {RISK_MULT:.0%})')
    bt.append('═' * 70)
    bt.append('')
    LAYER_NAME = {'TREND': '趋势 ETF($5k SPXL)+14DTE CALL价差',
                  'MR': 'MR 裸买CALL 7DTE ($2k SPXL)',
                  'CRASH': '崩盘 CALL价差15点21DTE+$5k SPXL'}
    for kind in ('TREND', 'MR', 'CRASH'):
        trs = ledger[kind]
        closed = [t for t in trs if t.get('close')]
        bt.append(f'── {LAYER_NAME[kind]}  共 {len(trs)} 笔 (已平 {len(closed)}) ──')
        bt.append('  年度    笔数    总PnL     胜率    均成本   最大亏    滚仓')
        byyr = {}
        for t in trs:
            byyr.setdefault(t['open'].year, []).append(t)
        for yr in sorted(byyr):
            ts = byyr[yr]
            pnl = sum(_trade_pnl(t) for t in ts)
            closes = [t for t in ts if t.get('close')]
            w = sum(1 for t in closes if _trade_pnl(t) > 0)
            wr = f"{w}/{len(closes)}" if closes else '-'
            cost = sum(_trade_cost(t) for t in ts) / len(ts) if ts else 0
            ml = min((_trade_pnl(t) for t in ts), default=0)
            rl = sum(t.get('rolls') or 0 for t in ts)
            opt_y = sum((t.get('opt_pnl') or 0) for t in ts)
            etf_y = sum((t.get('etf_pnl') or 0) for t in ts)
            bt.append(f'  {yr}    {len(ts):>3}   {pnl:>9.0f}   {wr:>5}   {cost:>7.1f}   {ml:>8.0f}   {rl:>4}   opt{opt_y:>+8.0f} etf{etf_y:>+8.0f}')
        pnl = sum(_trade_pnl(t) for t in trs)
        closes = [t for t in trs if t.get('close')]
        w = sum(1 for t in closes if _trade_pnl(t) > 0)
        wr = f"{w}/{len(closes)}" if closes else '-'
        cost = sum(_trade_cost(t) for t in trs) / len(trs) if trs else 0
        ml = min((_trade_pnl(t) for t in trs), default=0)
        rl = sum(t.get('rolls') or 0 for t in trs)
        hd = [t.get('hold_days', 0) for t in trs if t.get('close')]
        avh = f"{sum(hd) / len(hd):.1f}" if hd else '-'
        bt.append(f'  合计   {len(trs):>3}   {pnl:>9.0f}   {wr:>5}   {cost:>7.1f}   {ml:>8.0f}   {rl:>4}  均持{avh}d')
        opt_pnl = sum((t.get('opt_pnl') or 0) for t in trs)
        etf_pnl = sum((t.get('etf_pnl') or 0) for t in trs)
        bt.append(f'  腿拆分  期权 {opt_pnl:>+9.0f} | ETF {etf_pnl:>+9.0f} | 合计 {pnl:>+9.0f}')
        bt.append('')
    allt = ledger['TREND'] + ledger['MR'] + ledger['CRASH']
    tot_closed = [t for t in allt if t.get('close')]
    tot_pnl = sum(_trade_pnl(t) for t in allt)
    w = sum(1 for t in tot_closed if _trade_pnl(t) > 0)
    bt.append(f'── 三层合计  共 {len(allt)} 笔 (已平 {len(tot_closed)})  总PnL ${tot_pnl:,.0f}  胜率 {w}/{len(tot_closed)} ──')
    bt.append('')
    bt.append('说明:')
    bt.append('  · PnL = ETF仓位($5k/$2k/$5k SPXL) + 期权仓位(BS 重定价); 期权段滚动以滚仓日结算旧段')
    bt.append(f'  · 首阳退半: 崩盘期权当日全额结算, ETF 退 {CRASH_HALF:.0%}(${ETF_SIZE["CRASH"]*CRASH_HALF:.0f}); 首阴(V4)/二次首阳(V6-8)/止损/4天 再结剩余 ${ETF_SIZE["CRASH"]*(1-CRASH_HALF):.0f}')
    bt.append('  · MR 信号日=恐慌日(RSI<30+VIX>20), 与崩盘开仓日高度重叠; app 有 direction 闸+崩盘互斥')
    bt.append('    +2026-07-31 强制互斥, 故 MR 低频属结构性(panic 日优先被崩盘层承接), 非回测误差')
    bt_stats_path = os.path.join(RESULT_DIR, f'backtest_stats_{PERIOD}.txt')
    with open(bt_stats_path, 'w') as f:
        f.write('\n'.join(bt) + '\n')
    print(f"  wrote backtest_stats_{PERIOD}.txt")

    # ═══════════════════════════════ 8. report ═══════════════════════════════
    print()
    print('═' * 70)
    print(f"重放完成: 完整晚报 {stats['full']} | 紧凑 {stats['compact']} | 跳过 {stats['skipped']}")
    print(f"开仓 {stats['opens']} | 平仓 {stats['closes']}")
    if failures:
        print(f"❌ 断言失败 {len(failures)} 条:")
        for x in failures[:60]:
            print("   " + x)
        if len(failures) > 60:
            print(f"   ... 共 {len(failures)} 条")
        sys.exit(1)
    print("✅ 全部断言通过")


if __name__ == '__main__':
    main()
