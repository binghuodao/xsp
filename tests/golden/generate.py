#!/usr/bin/env python3
"""Generate golden fixtures from current production behavior."""

import sys
import os
import json
import tempfile
import datetime
import argparse
from datetime import date, timedelta
from unittest.mock import MagicMock

# Setup path and fake moomoo before importing app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

# Output directory
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim_reports_full')
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


# Canonical days covering all production paths (15 days)
CANONICAL_DAYS = [
    "2021-06-16",  # Crash open (first trade)
    "2022-01-05",  # Crash open + carry setup
    "2022-01-11",  # 首阳退半 (SPXL half exit)
    "2022-01-13",  # 崩盘再进 (recovery reentry)
    "2022-01-18",  # 止损平仓 + 同价续持 (carry)
    "2022-01-21",  # Stop loss close
    "2024-01-02",  # 4天强制平 (early 2024)
    "2024-07-25",  # 首阴清仓 (full close)
    "2024-08-01",  # 止损 + 残期续持 (V9)
    "2024-08-14",  # 残期收复
    "2026-03-27",  # (synthetic) MR open - will be skipped if no signal
    "2026-04-28",  # Trend open (CALL)
    "2026-04-29",  # Trend close (趋势结束平价差)
    "2026-05-15",  # Recent crash open
    "2026-05-26",  # 4天强制平
]

# Production config (what runs live)
PROD_CONFIG = {
    'CRASH_MODE': 'V10',
    'CRASH_HALF': 0.0625,
    'CRASH_YIN': 0.75,
    'STOP_PCT': 0.025,
    'DROP_THRESH': 0.005,
    'REENTRY_PCT': 1.0,
    'DTE': 21,
    'SPREAD_W': 15,
    'ETF_STOP': 0.0,
    'LAYER_PRIORITY': 'crash_mr_trend',
    'RISK_GATE': 'b200slope',
    'RISK_MULT': 1.0,
    'Y10_GATE': 0.0,
    'OPT_MULT': 1.0,
    'OPT_STANDALONE': 'off',
    'OPT_TP': 1.70,
    'OPT_SL_RESID': 0.0,
    'OPT_SL_DECAY': 0.0,
    'OPT_SL_ADAPTIVE': False,
    'BOUNCE_THRESH': 1.003,
    'CRASH_BOUNCE_THRESH': 1.003,
    'WARMUP_DAYS': 60,
}


def extract_key_report_fields(report):
    """Extract the key fields we want to assert on."""
    key_fields = [
        'direction', 'score', 'reason', 'close_alerts',
        'crash_entry_date', 'crash_entry_price', 'crash_k1', 'crash_k2',
        'crash_debit', 'crash_sigma', 'crash_etf_entry', 'crash_etf_stop',
        'crash_green', 'crash_stop', 'crash_days', 'crash_force_days',
        'mr_entry_price', 'mr_stop', 'mr_green', 'mr_days',
        'trend_opt_expiry', 'trend_opt_strike', 'trend_opt_strike2',
        'trend_opt_entry', 'trend_opt_entry_date', 'trend_opt_sigma',
        'stop_loss', 'peak_price', 'entry_price', 'etf_entry_price',
        'etf_peak_price', 'active_position_date', 'prev_report_direction',
    ]
    return {k: report.get(k) for k in key_fields if k in report}


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
    app._crash_exit_mode = PROD_CONFIG['CRASH_MODE']
    app._crash_half_pct = PROD_CONFIG['CRASH_HALF']
    app._crash_yin_pct = PROD_CONFIG['CRASH_YIN']
    app._crash_yin_scaled = False
    app._crash_yin_date = None
    app._crash_stop_pct = PROD_CONFIG['STOP_PCT']
    app._crash_reentry_pct = PROD_CONFIG['REENTRY_PCT']
    app._crash_dte = PROD_CONFIG['DTE']
    app._crash_spread_w = PROD_CONFIG['SPREAD_W']
    app._crash_drop_thresh = PROD_CONFIG['DROP_THRESH']
    app._crash_etf_stop_pct = PROD_CONFIG['ETF_STOP']
    app._crash_etf_out = False
    app.CRASH_BOUNCE_THRESH = PROD_CONFIG['BOUNCE_THRESH']
    app._layer_priority = PROD_CONFIG['LAYER_PRIORITY']
    app._crash_stop_cooldown = PROD_CONFIG.get('STOP_COOLDOWN', 0)
    app._crash_stop_date = None
    app._crash_size_mult = PROD_CONFIG['RISK_MULT']
    app._crash_etf_size = int(5000 * PROD_CONFIG['RISK_MULT'])
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


def build_snapshot(asof, xsp, vix, spy, spxl, skew, adx_df, bb_df, rsi_s, sma20_s, avg_vol_s, ema20p_s,
                   sma200_s, macd_s, macdsig_s, engulf_s, s3red_s, vix_pct_252_s):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-net', action='store_true', help='use cached CSV only')
    parser.add_argument('--period', default='7y', help='yfinance period')
    parser.add_argument('--outdir', default='tests/golden', help='output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Apply production config to app globals
    for k, v in PROD_CONFIG.items():
        setattr(app, k, v)

    print("Loading data...")
    xsp = _load(args.no_net, '^XSP', args.period)
    vix = _load(args.no_net, '^VIX', args.period)
    spy = _load(args.no_net, 'SPY', args.period)
    spxl = _load(args.no_net, 'SPXL', args.period)
    skew = _load(args.no_net, '^SKEW', args.period)

    # Precompute indicators
    adx_df = ta.adx(spy['High'], spy['Low'], spy['Close'], length=14)
    bb_df = ta.bbands(spy['Close'], length=20, std=2)
    rsi_s = ta.rsi(spy['Close'], length=14)
    sma20_s = spy['Close'].rolling(20).mean()
    avg_vol_s = spy['Volume'].rolling(20).mean()
    ema20p_s = spy['Close'].ewm(span=20, adjust=False).mean()

    _xc = xsp['Close']
    sma200_s = _xc.rolling(200).mean()
    macd_s = _xc.ewm(span=12, adjust=False).mean() - _xc.ewm(span=26, adjust=False).mean()
    macdsig_s = macd_s.ewm(span=9, adjust=False).mean()
    _ox, _cx = xsp['Open'], xsp['Close']
    engulf_s = (_cx < _ox) & (_cx.shift(1) > _ox.shift(1)) & (_ox >= _cx.shift(1)) & (_cx <= _ox.shift(1))
    s3red_s = (_cx < _cx.shift(1)) & (_cx.shift(1) < _cx.shift(2)) & (_cx.shift(2) < _cx.shift(3))
    vix_pct_252_s = vix['Close'].rolling(252).rank(pct=True) * 100

    # Get trading days
    trading_days = list(xsp.index)
    REPLAY_START = pd.Timestamp('2020-08-01')
    trading_days = [d for d in trading_days if d >= REPLAY_START]
    WARMUP_DAYS = PROD_CONFIG['WARMUP_DAYS']
    trading_days = trading_days[WARMUP_DAYS:]

    # Convert canonical days to Timestamps for comparison
    canonical_set = {pd.Timestamp(d).date() for d in CANONICAL_DAYS}

    # Setup harness
    app.socketio = MagicMock()
    _captured = {}
    def _capture(m):
        _captured['msg'] = m
    app.send_telegram = _capture

    # Reset state
    _reset_app_state()

    print(f"Replaying {len(trading_days)} trading days...")
    fixtures_generated = 0

    for i, day in enumerate(trading_days):
        asof = day.date()
        
        if asof in canonical_set:
            print(f"  Capturing {asof}...")

        price, spxl_p, vix_p = build_snapshot(
            asof, xsp, vix, spy, spxl, skew,
            adx_df, bb_df, rsi_s, sma20_s, avg_vol_s, ema20p_s,
            sma200_s, macd_s, macdsig_s, engulf_s, s3red_s, vix_pct_252_s
        )
        make_chain(price, vix_p / 100.0, asof)
        app.latest_data['index']['price'] = price
        app._etf_price_cache['SPXL'] = spxl_p
        app.datetime = _clock_for(asof)

        _captured.clear()
        app.send_market_report('evening', force=False)
        msg = _captured.get('msg', '')

        if asof in canonical_set:
            report = dict(app._latest_report)
            fixture = {
                'date': str(asof),
                'msg': msg,
                'report': extract_key_report_fields(report),
            }
            filepath = os.path.join(args.outdir, f"{asof}.json")
            with open(filepath, 'w') as f:
                json.dump(fixture, f, indent=2, ensure_ascii=False)
            print(f"    -> saved {filepath}")
            fixtures_generated += 1

    print(f"\nGenerated {fixtures_generated} fixtures in {args.outdir}")


if __name__ == '__main__':
    main()