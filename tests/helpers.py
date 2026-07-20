"""Test helper functions — reusable across test modules."""


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
        'adx': 18.0, 'er': 0.30, 'bbw': 3.5, 'dev': 0.0, 'vr': 1.0,
        'support': 740.0, 'resistance': 760.0,
        'bw': 20.0, 'bbl': 740.0, 'bbu': 760.0,
        'di_diff': 0.0,
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
