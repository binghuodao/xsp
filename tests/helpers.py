"""Test helpers: shared utilities for golden tests and other test modules."""

import sys
import os
import tempfile
import datetime
from datetime import date, timedelta
from unittest.mock import MagicMock

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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _SCRIPT_DIR)


# ════════════════════════════════════════════════════════════
# Back-compat helpers used by existing tests (conftest.py, test_market_report.py)
# ════════════════════════════════════════════════════════════

def make_opt(sym, strike, expiry, opt_type, mid, delta, iv=0.15):
    return {
        'symbol': sym, 'strike': strike, 'expiry': expiry,
        'opt_type': opt_type, 'bid': round(mid * 0.9, 2),
        'ask': round(mid * 1.1, 2), 'mid': mid,
        'delta': delta, 'gamma': 0.01, 'theta': -0.02,
        'vega': 0.05, 'iv': iv, 'is_watched': False,
        'open_interest': 1000,
    }


def opt_sym(expiry_yyyymmdd, opt_type, strike):
    d = expiry_yyyymmdd.replace('-', '')[2:]
    return f"US.XSP{d}{opt_type}{int(strike * 1000)}"


def make_option_chain(ema20=750.0, expiry_7='2026-07-24', expiry_14='2026-07-31'):
    """Standard option chain for testing — two expiries, strikes ±20 around ema20."""
    opts = {}
    for expiry, width in [(expiry_7, 5.0), (expiry_14, 7.0)]:
        for strike in range(int(ema20) - 20, int(ema20) + 21, 5):
            for ot in ('P', 'C'):
                sym = opt_sym(expiry, ot, strike)
                moneyness = (strike - ema20) / ema20
                delta = -0.5 + moneyness * (2.0 if expiry == expiry_7 else 1.5) if ot == 'P' \
                        else 0.5 - moneyness * (2.0 if expiry == expiry_7 else 1.5)
                delta = max(min(delta, 0.99 if ot == 'C' else -0.01),
                            -0.99 if ot == 'P' else 0.01)
                mid = max(0.05, width - abs(strike - ema20) * (width / 25))
                opts[sym] = make_opt(sym, strike, expiry, ot, round(mid, 2), round(delta, 4))
    return opts


def std_hs(**overrides):
    """Standard historical_stats dict. Override any key via kwargs."""
    hs = {
        'vix': 14.0, 'vix_rank': 30.0, 'vix_percentile': 35.0,
        'atr_14': 8.0, 'ema_20': 750.0, 'skew': 0.0,
        'adx': 18.0, 'er': 0.30, 'bbw': 3.5, 'dev': 3.0, 'vr': 1.0,
        'support': 740.0, 'resistance': 760.0,
        'bw': 20.0, 'bbl': 740.0, 'bbu': 760.0,
        'di_diff': 0.0,
        'di_diff_prev': 0.1,
        'sma50': 700.0,
        'rsi_14': 50.0,
        'price_ema20_pct': 0.0,
        'skew_index': 146.0,
    }
    hs.update(overrides)
    return hs


def make_wl_entry(date, short, mid, long, opt_type, entry='', strategy='xmas'):
    """Christmas-tree watchlist entry."""
    return {
        'date': date, 'short': str(short), 'mid': str(mid),
        'long': str(long), 'opt_type': opt_type,
        'entry': str(entry) if entry else '', 'strategy': strategy,
    }

OUT_DIR = os.path.join(_SCRIPT_DIR, 'sim_reports_full')
os.makedirs(OUT_DIR, exist_ok=True)


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


def _make_snapshot_funcs(xsp, vix, spy, spxl, skew, adx_df, bb_df, rsi_s, sma20_s, avg_vol_s, ema20p_s, sma200_s, macd_s, macdsig_s, engulf_s, s3red_s, vix_pct_252_s):
    """Create build_snapshot and make_chain functions bound to loaded data."""

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
        closes = x_i['Close']
        highs = x_i['High']
        lows = x_i['Low']
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
        _prev_c = float(p_prev['Close'].iloc[-1]) if len(p_prev) else None
        _curr_c = float(x_i['Close'].iloc[-1]) if len(x_i) else None
        _curr_d = x_i.index[-1].date() if len(x_i) else None
        _prev_d = p_prev.index[-1].date() if len(p_prev) else None
        app._get_xsp_prev_close = lambda _pc=_prev_c: _pc
        app._get_xsp_closes = lambda _cc=_curr_c, _pc=_prev_c: (_cc, _pc)
        app._get_xsp_closes_with_dates = lambda _cc=_curr_c, _cd=_curr_d, _pc=_prev_c, _pd=_prev_d: (_cc, _cd, _pc, _pd)
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

    return build_snapshot, make_chain


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


def _reset_app_state():
    """Reset all app global state to fresh."""
    for a in ('_active_position_date', '_entry_price', '_peak_price', '_etf_entry_price', '_etf_peak_price',
              '_prev_report_direction', '_mr_entry_date', '_mr_entry_price', '_mr_etf_entry_price',
              '_crash_entry_date', '_crash_entry_price', '_crash_k1', '_crash_k2', '_crash_debit',
              '_crash_sigma', '_crash_etf_entry', '_crash_half_date', '_crash_reentry_date',
              '_crash_opt_reopen_date',
              '_trend_opt_expiry', '_trend_opt_strike',
              '_trend_opt_strike2', '_trend_opt_entry', '_trend_opt_entry_date', '_trend_opt_sigma'):
        setattr(app, a, None)
    app._crash_resids = []
    app._crash_etf_scaled = False
    app._crash_reentry = False
    app._crash_opt_reopened = False
    app._crash_exit_mode = app.CRASH_MODE if hasattr(app, 'CRASH_MODE') else 'V10'
    app._crash_half_pct = getattr(app, 'CRASH_HALF', 0.0625)
    app._crash_yin_pct = getattr(app, 'CRASH_YIN', 0.75)
    app._crash_yin_scaled = False
    app._crash_yin_date = None
    app._crash_stop_pct = getattr(app, 'STOP_PCT', 0.025)
    app._crash_reentry_pct = getattr(app, 'REENTRY_PCT', 1.0)
    app._crash_dte = getattr(app, 'DTE', 21)
    app._crash_spread_w = getattr(app, 'SPREAD_W', 15)
    app._crash_drop_thresh = getattr(app, 'DROP_THRESH', 0.005)
    app._crash_etf_stop_pct = getattr(app, 'ETF_STOP', 0.0)
    app._crash_etf_out = False
    app.CRASH_BOUNCE_THRESH = getattr(app, 'BOUNCE_THRESH', None)
    app._layer_priority = getattr(app, '_layer_priority', 'crash_mr_trend')
    app._crash_stop_cooldown = getattr(app, 'STOP_COOLDOWN', 0)
    app._crash_stop_date = None
    app._crash_size_mult = getattr(app, 'RISK_MULT', 1.0)
    app._crash_etf_size = int(5000 * getattr(app, 'RISK_MULT', 1.0))
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


def setup_production_environment(no_net=True, period='7y'):
    """Load all historical data and precompute indicators. Returns data bundle."""
    print(f"Loading real history ({period})...")
    xsp = _load(no_net, '^XSP', period)
    vix = _load(no_net, '^VIX', period)
    spy = _load(no_net, 'SPY', period)
    spxl = _load(no_net, 'SPXL', period)
    skew = _load(no_net, '^SKEW', period)

    # Precompute indicators
    adx_df = ta.adx(spy['High'], spy['Low'], spy['Close'], length=14)
    bb_df = ta.bbands(spy['Close'], length=20, std=2)
    rsi_s = ta.rsi(spy['Close'], length=14)
    sma20_s = spy['Close'].rolling(20).mean()
    avg_vol_s = spy['Volume'].rolling(20).mean()
    ema20p_s = spy['Close'].ewm(span=20, adjust=False).mean()

    # Risk-gate features (XSP index regime)
    _xc = xsp['Close']
    sma200_s = _xc.rolling(200).mean()
    macd_s = _xc.ewm(span=12, adjust=False).mean() - _xc.ewm(span=26, adjust=False).mean()
    macdsig_s = macd_s.ewm(span=9, adjust=False).mean()
    _ox, _cx = xsp['Open'], xsp['Close']
    engulf_s = (_cx < _ox) & (_cx.shift(1) > _ox.shift(1)) & (_ox >= _cx.shift(1)) & (_cx <= _ox.shift(1))
    s3red_s = (_cx < _cx.shift(1)) & (_cx.shift(1) < _cx.shift(2)) & (_cx.shift(2) < _cx.shift(3))
    vix_pct_252_s = vix['Close'].rolling(252).rank(pct=True) * 100

    # Create snapshot functions bound to this data
    build_snapshot, make_chain = _make_snapshot_funcs(
        xsp, vix, spy, spxl, skew,
        adx_df, bb_df, rsi_s, sma20_s, avg_vol_s, ema20p_s,
        sma200_s, macd_s, macdsig_s, engulf_s, s3red_s, vix_pct_252_s
    )

    return {
        'xsp': xsp, 'vix': vix, 'spy': spy, 'spxl': spxl, 'skew': skew,
        'build_snapshot': build_snapshot,
        'make_chain': make_chain,
    }


def run_production_to_date(target_date_str, env=None, config=None):
    """
    Run production app logic for a specific date.
    
    Args:
        target_date_str: Date string 'YYYY-MM-DD'
        env: Pre-loaded environment from setup_production_environment() (optional)
        config: Dict of config overrides for app globals (optional)
            e.g., {'CRASH_MODE': 'V10', 'BOUNCE_THRESH': 1.003, 'RISK_GATE': 'b200slope', ...}
    
    Returns:
        (report_dict, msg_str) - app._latest_report and captured telegram message
    """
    if env is None:
        env = setup_production_environment()
    
    target_date = pd.Timestamp(target_date_str).date()
    
    # Apply config overrides
    if config:
        for k, v in config.items():
            setattr(app, k, v)
    
    # Reset app state fresh
    _reset_app_state()
    
    # Setup harness
    app.socketio = MagicMock()
    _captured = {}
    def _capture(m):
        _captured['msg'] = m
    app.send_telegram = _capture
    
    # Build snapshot for target date
    price, spxl_price, vix_val = env['build_snapshot'](target_date)
    env['make_chain'](price, vix_val / 100.0, target_date)
    app.latest_data['index']['price'] = price
    app._etf_price_cache['SPXL'] = spxl_price
    app.datetime = _clock_for(target_date)
    
    # Run production evening report
    _captured.clear()
    app.send_market_report('evening', force=False)
    msg = _captured.get('msg', '')
    
    # Return a copy of the report (not the live reference)
    report = dict(app._latest_report)
    
    return report, msg


def run_production_sequence(dates, env=None, config=None):
    """
    Run production app logic for a sequence of dates (maintains state across days).
    
    Args:
        dates: List of date strings 'YYYY-MM-DD'
        env: Pre-loaded environment
        config: Config overrides
    
    Returns:
        List of (date, report_dict, msg_str)
    """
    if env is None:
        env = setup_production_environment()
    
    if config:
        for k, v in config.items():
            setattr(app, k, v)
    
    _reset_app_state()
    
    app.socketio = MagicMock()
    _captured = {}
    def _capture(m):
        _captured['msg'] = m
    app.send_telegram = _capture
    
    results = []
    for date_str in dates:
        target_date = pd.Timestamp(date_str).date()
        
        price, spxl_price, vix_val = env['build_snapshot'](target_date)
        env['make_chain'](price, vix_val / 100.0, target_date)
        app.latest_data['index']['price'] = price
        app._etf_price_cache['SPXL'] = spxl_price
        app.datetime = _clock_for(target_date)
        
        _captured.clear()
        app.send_market_report('evening', force=False)
        msg = _captured.get('msg', '')
        
        results.append((date_str, dict(app._latest_report), msg))
    
    return results