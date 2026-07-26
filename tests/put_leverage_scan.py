"""Leverage scan for best PUT configs."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.put_strategy_search import run_bt, summary, print_trades

devnull = open('/dev/null', 'w')
old_out, old_err = sys.stdout, sys.stderr

# Baseline first
for p in ['3y','5y','10y']:
    sys.stdout, sys.stderr = devnull, devnull
    tr = run_bt(period=p, call_only=True)
    sys.stdout, sys.stderr = old_out, old_err
    s = summary(tr)
    print(f'baseline {p}: {s["n"]}tr PnL${s["pnl"]:.0f}')

print('\n=== DI cross: leverage scan (sc35, vx15, tr0.10, h21) ===')
for pl in [1.0, 2.0, 3.0]:
    for p in ['3y','5y','10y']:
        sys.stdout, sys.stderr = devnull, devnull
        tr = run_bt(period=p, put_method='di_cross',
                    put_score=35, put_vix=15, put_lev=pl,
                    put_trail=0.10, put_hold=21, put_stop=0.10)
        sys.stdout, sys.stderr = old_out, old_err
        s = summary(tr)
        print(f'  di_cross lev{pl} {p}: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
              f'WR{s["wr"]}% PnL${s["pnl"]:.0f} (PUT${s["put_pnl"]:.0f}+CALL${s["call_pnl"]:.0f})')
        # Print per-trade
        puts = [t for t in tr if t['dir'] == 'PUT']
        for t in puts:
            ed = t['entry_date']
            if hasattr(ed, 'strftime'): ed = ed.strftime('%Y-%m-%d')
            print(f'    PUT {ed} ep={t["entry_price"]:.1f} pnl=${t["pnl"]:+.0f} '
                  f'exit={t["exit_type"]} reason={t["entry_reason"]}')

print('\n=== DI cross: score threshold scan (lev1.0, vx15, tr0.10, h21) ===')
for ps in [32, 35, 38, 40]:
    for p in ['10y']:
        sys.stdout, sys.stderr = devnull, devnull
        tr = run_bt(period=p, put_method='di_cross',
                    put_score=ps, put_vix=15, put_lev=1.0,
                    put_trail=0.10, put_hold=21, put_stop=0.10)
        sys.stdout, sys.stderr = old_out, old_err
        s = summary(tr)
        print(f'  di_cross sc{ps} 10y: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
              f'WR{s["wr"]}% PnL${s["pnl"]:.0f} (PUT${s["put_pnl"]:.0f}+CALL${s["call_pnl"]:.0f})')
        puts = [t for t in tr if t['dir'] == 'PUT']
        for t in puts:
            ed = t['entry_date']
            if hasattr(ed, 'strftime'): ed = ed.strftime('%Y-%m-%d')
            print(f'    PUT {ed} ep={t["entry_price"]:.1f} pnl=${t["pnl"]:+.0f} '
                  f'exit={t["exit_type"]} reason={t["entry_reason"]}')

print('\n=== DI cross: hold scan (sc35, vx15, lev1.0, tr0.10) ===')
for ph in [14, 21, 30]:
    for p in ['10y']:
        sys.stdout, sys.stderr = devnull, devnull
        tr = run_bt(period=p, put_method='di_cross',
                    put_score=35, put_vix=15, put_lev=1.0,
                    put_trail=0.10, put_hold=ph, put_stop=0.10)
        sys.stdout, sys.stderr = old_out, old_err
        s = summary(tr)
        print(f'  di_cross h{ph} 10y: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
              f'WR{s["wr"]}% PnL${s["pnl"]:.0f} (PUT${s["put_pnl"]:.0f}+CALL${s["call_pnl"]:.0f})')

print('\n=== Near-top fade: leverage scan (sc30, rs70, tr0.05, h10) ===')
for pl in [0.5, 1.0, 2.0]:
    for p in ['3y','5y','10y']:
        sys.stdout, sys.stderr = devnull, devnull
        tr = run_bt(period=p, put_method='near_top',
                    put_score=30, top_rsi=70, put_lev=pl,
                    put_trail=0.05, put_hold=10, put_stop=0.05)
        sys.stdout, sys.stderr = old_out, old_err
        s = summary(tr)
        print(f'  near_top lev{pl} {p}: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
              f'WR{s["wr"]}% PnL${s["pnl"]:.0f} (PUT${s["put_pnl"]:.0f}+CALL${s["call_pnl"]:.0f})')
        puts = [t for t in tr if t['dir'] == 'PUT']
        for t in puts:
            ed = t['entry_date']
            if hasattr(ed, 'strftime'): ed = ed.strftime('%Y-%m-%d')
            print(f'    PUT {ed} ep={t["entry_price"]:.1f} pnl=${t["pnl"]:+.0f} '
                  f'exit={t["exit_type"]} reason={t["entry_reason"]}')

print('\n=== Combined both: leverage scan (sc35, vx15, rs70, tr0.10, h14) ===')
for pl in [0.5, 1.0, 2.0]:
    for p in ['3y','5y','10y']:
        sys.stdout, sys.stderr = devnull, devnull
        tr = run_bt(period=p, put_method='both',
                    put_score=35, put_vix=15, top_rsi=70, put_lev=pl,
                    put_trail=0.10, put_hold=14, put_stop=0.10)
        sys.stdout, sys.stderr = old_out, old_err
        s = summary(tr)
        print(f'  both lev{pl} {p}: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
              f'WR{s["wr"]}% PnL${s["pnl"]:.0f} (PUT${s["put_pnl"]:.0f}+CALL${s["call_pnl"]:.0f})')
        puts = [t for t in tr if t['dir'] == 'PUT']
        for t in puts:
            ed = t['entry_date']
            if hasattr(ed, 'strftime'): ed = ed.strftime('%Y-%m-%d')
            print(f'    PUT {ed} ep={t["entry_price"]:.1f} pnl=${t["pnl"]:+.0f} '
                  f'exit={t["exit_type"]} reason={t["entry_reason"]}')

devnull.close()
