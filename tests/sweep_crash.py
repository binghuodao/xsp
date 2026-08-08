#!/usr/bin/env python3
"""Crash-layer parameter sweep driver.

Runs the canonical 7y replay (full 6y window) for each config and prints a
one-line summary.  Parameters are mapped onto sim_reports_full.py CLI args:

  --crash-half   -> half (ETF fraction sold at 首阳)
  --reentry-pct  -> reentry
  --stop-pct     -> stop_pct (XSP stop line = entry*(1-pct))
  --drop-thresh  -> drop_thresh (crash signal XSP daily-drop threshold)
  --stop-cooldown-> stop_cooldown (days blocked after a crash stop-loss)
  --crash-mode   -> mode (V4 default)

Usage:
  python3 tests/sweep_crash.py --cfg "stop_pct=0.03" \
    --cfg "stop_pct=0.04" --cfg "drop_thresh=0.01" ...

Judged on the full 6y sample (2021-06 -> today); 2022 listed as the bear-year
reference.  Everything except the three-layer row is the CRASH layer only.
"""
import subprocess, re, sys, os

SCRIPT = os.path.join(os.path.dirname(__file__), 'sim_reports_full.py')
STATS = os.path.join(os.path.dirname(__file__), 'sim_reports_full', 'backtest_stats_7y.txt')
ALIAS = {'half': '--crash-half', 'reentry': '--reentry-pct', 'stop_pct': '--stop-pct',
         'drop_thresh': '--drop-thresh', 'stop_cooldown': '--stop-cooldown', 'mode': '--crash-mode',
         'dte': '--dte', 'spread_w': '--spread-w', 'etf_stop': '--etf-stop', 'priority': '--layer-priority'}

def run_once(cfg):
    extra = []
    for k, v in cfg.items():
        extra += [ALIAS[k], str(v)]
    r = subprocess.run([sys.executable, SCRIPT, '--no-net', '--stats-only', '--period', '7y'] + extra,
                       capture_output=True, text=True)
    if r.returncode != 0:
        return dict(cfg=cfg, n='ERR', tot=None, y2022=None, tl=None, mx=None, wr='', leg='')
    txt = open(STATS).read()
    years = {}
    cs = txt.split('── 崩盘', 1)[1] if '── 崩盘' in txt else ''
    cs = cs.split('说明:', 1)[0]
    m = re.search(r'CALL价差15点21DTE\+\$5k SPXL\s+共 (\d+) 笔', cs)
    n = int(m.group(1)) if m else None
    m = re.search(r'\s+合计\s+(\d+)\s+(-?\d+)\s+(\d+/\d+)\s+([\d.]+)\s+(-?\d+)\s+(\d+)\s+均持([\d.]+)d', cs)
    tot = int(m.group(2)) if m else None
    wr = m.group(3) if m else ''
    mx = int(m.group(5)) if m else None
    for ym in re.finditer(r'\s+(\d{4})\s+(\d+)\s+(-?\d+)\s+(\d+/\d+)\s+([\d.]+)\s+(-?\d+)\s+(\d+)\s+opt\s+(-?\d+)\s+etf\s+(-?\d+)', cs):
        years[int(ym.group(1))] = (int(ym.group(2)), int(ym.group(3)))
    y2022 = years.get(2022)
    m = re.search(r'三层合计\s+共 (\d+) 笔 \(已平 \d+\)\s+总PnL \$(-?[\d,]+)', txt)
    tl = int(m.group(2).replace(',', '')) if m else None
    legm = re.search(r'腿拆分\s+期权\s+([+-]?\d+)\s+\|\s+ETF\s+([+-]?\d+)\s+\|\s+合计\s+([+-]?\d+)', cs)
    leg = f"opt{legm.group(1)}/etf{legm.group(2)}" if legm else ''
    return dict(cfg=cfg, n=n, tot=tot, y2022=y2022, tl=tl, mx=mx, wr=wr, leg=leg)

def fmt(c):
    cfg = ','.join(f'{k}={v}' for k, v in c['cfg'].items()) or 'BASE'
    y22 = f"{c['y2022'][1]}/{c['y2022'][0]}tr" if c['y2022'] else '?'
    return (f"{cfg:<38} n={c['n']:<4} crash6y={c['tot']:>+7}  2022={y22:>12} "
            f"maxLoss={c['mx']:>6} three=+{c['tl'] if c['tl'] else 0:<6} {c['wr']:<8} {c['leg']}")

if __name__ == '__main__':
    cfgs = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--cfg':
            cfg = {}
            for pair in sys.argv[i + 1].split(','):
                if not pair:
                    continue
                k, v = pair.split('=')
                try:
                    cfg[k] = float(v) if '.' in v or 'e' in v.lower() else int(v)
                except ValueError:
                    cfg[k] = v
            cfgs.append(cfg)
            i += 2
        else:
            sys.exit(f'unknown arg: {sys.argv[i]}')
    if not cfgs:
        cfgs = [{}]  # base
    print(f"{'config':<38} n      crash6y    2022            maxLoss three-layer WR      leg")
    for cfg in cfgs:
        print(fmt(run_once(cfg)))
