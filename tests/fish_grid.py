"""Fish-order grid test: empty-window intraday dip-catcher.

Idea (user, 2026-08-15): during a true 3-layer empty window (no TREND/MR/CRASH
position riding into the day), place a limit buy for SPXL at yesterday's close
* (1 - trigger).  If today's SPXL Low touches the limit it fills at the limit
price with `fish_sh = max(round(size / limit), 1)` shares.

Close resolution (mirrors the crash-layer signal XSP close drop < -0.5%):
  * XSP close change < -0.5%  -> CONVERT: this is a crash-open day.  Fishing
    merges into the shared 5K crash position (crash still opens its entire
    round(5000/close) shares at close; incremental basis difference vs buying
    all 5K at close = fish_sh * (close - limit)).
  * otherwise                 -> SCALP: sell at today's close (same-day flat).

Per-fill incremental PnL is fish_sh * (SPXL_close - limit) in both buckets;
the bucket only decides realized-vs-merged semantics.

Empty-window definition: a day is busy when a position was carried into it,
i.e. exists a trade with open < day <= close (positions open/close at the
close print, so the open date itself is still fishable).
"""

import argparse
import pandas as pd

CSV = 'tests/sim_reports_full'
TRIGGERS = [0.025, 0.030, 0.035, 0.040, 0.045]


def load(period):
    x = pd.read_csv(f'{CSV}/_^XSP_{period}.csv', parse_dates=['Date']).sort_values('Date')
    s = pd.read_csv(f'{CSV}/_SPXL_{period}.csv', parse_dates=['Date']).sort_values('Date')
    t = pd.read_csv(f'{CSV}/trades_{period}.csv')
    d = pd.merge(x, s, on='Date', suffixes=('_x', '_s'))
    # busy if carried into the day: open < day <= close (any kind), or same-day
    # open only (position not yet held at open) is still fishable.
    busy = pd.Series(False, index=d.index)
    for _, r in t.iterrows():
        op = pd.Timestamp(r['open'])
        cl = pd.Timestamp(r['close']) if pd.notna(r['close']) else op
        m = (d['Date'] > op) & (d['Date'] <= cl)
        busy |= m
    d['busy'] = busy
    return d


def run(d, size, trigger):
    prev = d['Close_s'].shift(1)
    limit = prev * (1 - trigger)
    filled = d[(~d['busy']) & (d['Low_s'] <= limit)].copy()
    if filled.empty:
        return 0, 0, 0.0, 0, 0.0
    fish_sh = (size / limit.loc[filled.index]).round().clip(lower=1).astype(int)
    pnl = fish_sh * (filled['Close_s'] - limit.loc[filled.index])
    x_prev_all = d['Close_x'].shift(1)
    x_chg_v = (filled['Close_x'] / x_prev_all.loc[filled.index] - 1)
    is_conv = x_chg_v < -0.005
    scalp_pnl = pnl[~is_conv].sum()
    conv_pnl = pnl[is_conv].sum()
    n_scalp = int((~is_conv).sum())
    n_conv = int(is_conv.sum())
    return n_scalp, n_conv, float(scalp_pnl), int(filled.shape[0]), float(conv_pnl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=float, default=0, help='single size (0 = all in grid)')
    ap.add_argument('--period', nargs='+', default=['3y', '7y'])
    args = ap.parse_args()
    sizes = [1500, 2000] if args.size == 0 else [args.size]
    print(f'{"period":5s} {"size":>5s} {"trg%":>5s} {"fills":>5s} '
          f'{"scalp_n":>7s} {"scalp$":>8s} {"conv_n":>6s} {"conv$":>9s} {"net$":>9s}')
    for period in args.period:
        d = load(period)
        for size in sizes:
            for trg in TRIGGERS:
                n_scalp, n_conv, sp, n_fill, cp = run(d, size, trg)
                net = sp + cp
                print(f'{period:5s} {int(size):5d} {trg * 100:4.1f}% {n_fill:5d} '
                      f'{n_scalp:7d} {sp:8.0f} {n_conv:6d} {cp:9.0f} {net:9.0f}')


if __name__ == '__main__':
    main()