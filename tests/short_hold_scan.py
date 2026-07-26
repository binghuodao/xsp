"""Short-hold scan: T+3 to T+14 for CALL and PUT."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.put_strategy_search import run_bt, summary

devnull = open('/dev/null', 'w')
old_out, old_err = sys.stdout, sys.stderr

RESULTS = {}
HOLDS = [3, 5, 7, 10, 14]
CALL_TRAILS = [1.5, 2.0, 2.5, 3.0, 3.5]
PUT_TRAILS = [3, 5, 7, 10]
PUT_LEVS = [1.0, 2.0, 3.0]
PERIODS = ['3y','5y','10y']

# ─── 1. CALL-only baseline per hold/trail ───
print('=== 1. CALL scan (hold × trail) ===')
for h in HOLDS:
    for tr in CALL_TRAILS:
        for p in PERIODS:
            sys.stdout, sys.stderr = devnull, devnull
            trs = run_bt(period=p, call_only=True,
                         call_hold=h, call_trail=tr/100,
                         call_lev=3, call_stop=0.05, call_nb_stop=0.01)
            sys.stdout, sys.stderr = old_out, old_err
            s = summary(trs)
            key = f'C_h{h}_tr{tr}|{p}'
            RESULTS[key] = s
            if p == PERIODS[-1]:
                print(f'  C_h{h}_tr{tr}: {s["n"]}tr WR{s["wr"]}% PnL${s["pnl"]:.0f}')

# ─── 2. PUT-only (DI cross) scan ───
print('\n=== 2. PUT scan (DI cross, hold × trail × lev) ===')
for h in HOLDS:
    for tr in PUT_TRAILS:
        for lev in PUT_LEVS:
            for p in PERIODS:
                sys.stdout, sys.stderr = devnull, devnull
                trs = run_bt(period=p, put_method='di_cross',
                             put_score=35, put_vix=15,
                             put_lev=lev, put_trail=tr/100, put_hold=h, put_stop=0.10,
                             call_only=False)
                sys.stdout, sys.stderr = old_out, old_err
                s = summary(trs)
                key = f'P_h{h}_tr{tr}_lev{lev}|{p}'
                RESULTS[key] = s
                if p == PERIODS[-1]:
                    pp = s['put_pnl']
                    print(f'  P_h{h}_tr{tr}_lev{lev}: {s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
                          f'PnL${s["pnl"]:.0f} (PUT${pp:+.0f})')

# ─── 3. Best CALL configs per hold ───
print(f'\n=== 3. Best CALL per hold (10y PnL) ===')
baseline_30_35 = RESULTS.get('C_h30_tr3.5|10y', {}).get('pnl', 0)
if not baseline_30_35:
    # fallback: compute T+30 / 3.5% baseline
    sys.stdout, sys.stderr = devnull, devnull
    trs = run_bt(period='10y', call_only=True, call_hold=30, call_trail=0.035)
    sys.stdout, sys.stderr = old_out, old_err
    s = summary(trs)
    baseline_30_35 = s['pnl']
    print(f'  T+30/3.5%: PnL${s["pnl"]:.0f} {s["n"]}tr WR{s["wr"]}%')

best_calls = {}
for h in HOLDS:
    best_pnl = -999999
    best_tr = None
    best_s = None
    for tr in CALL_TRAILS:
        s = RESULTS.get(f'C_h{h}_tr{tr}|10y', {})
        pnl = s.get('pnl', -999999)
        if pnl > best_pnl:
            best_pnl = pnl
            best_tr = tr
            best_s = s
    best_calls[h] = (best_tr, best_s)
    delta = best_pnl - baseline_30_35
    print(f'  T+{h}/trail{best_tr}%: PnL${best_pnl:.0f} ({delta:+.0f}) '
          f'{best_s["n"]}tr WR{best_s["wr"]}%')

# ─── 4. Best PUT configs per hold ───
print(f'\n=== 4. Best PUT per hold (10y PnL delta vs baseline CALL) ===')
best_puts = {}
for h in HOLDS:
    best_delta = -999999
    best_tr = None
    best_lev = None
    best_s = None
    for tr in PUT_TRAILS:
        for lev in PUT_LEVS:
            s = RESULTS.get(f'P_h{h}_tr{tr}_lev{lev}|10y', {})
            if not s: continue
            pnl = s.get('pnl', -999999)
            delta = pnl - baseline_30_35
            if delta > best_delta:
                best_delta = delta
                best_tr = tr
                best_lev = lev
                best_s = s
    best_puts[h] = (best_tr, best_lev, best_s)
    pp = best_s.get('put_pnl', 0)
    print(f'  T+{h}/trail{best_tr}%/lev{best_lev}: PnL${best_s["pnl"]:.0f} '
          f'({best_delta:+.0f}) PUT${pp:+.0f}')

# ─── 5. Best combined configs (top 3 holds) ───
print(f'\n=== 5. Combined CALL+PUT: top 3 hold days ===')
# Best calls: find the hold with max CALL PnL
call_rank = sorted(best_calls.items(), key=lambda x: -x[1][1]['pnl'])
print(f'  Top CALL holds: {[(h,tr,s["pnl"]) for h,(tr,s) in call_rank[:4]]}')

# For each hold, combine best CALL trail with best PUT config
for h, (c_tr, c_s) in call_rank[:4]:
    p_tr, p_lev, p_s = best_puts.get(h, (None, None, None))
    if not p_s: continue
    
    # Run combined with both best CALL and best PUT for this hold
    for p in PERIODS:
        sys.stdout, sys.stderr = devnull, devnull
        trs = run_bt(period=p, put_method='di_cross',
                     put_score=35, put_vix=15,
                     put_lev=p_lev, put_trail=p_tr/100, put_hold=h, put_stop=0.10,
                     call_hold=h, call_trail=c_tr/100,
                     call_lev=3, call_stop=0.05, call_nb_stop=0.01)
        sys.stdout, sys.stderr = old_out, old_err
        s = summary(trs)
        key = f'COMBO_h{h}_cTr{c_tr}_pTr{p_tr}_pLev{p_lev}|{p}'
        RESULTS[key] = s
        # Print 10y only detail
        if p == '10y':
            pp = s['put_pnl']; cp = s['call_pnl']
            total = s['pnl']
            print(f'  T+{h}/cTr{c_tr}%/pTr{p_tr}%/pLev{p_lev} {p}: '
                  f'{s["n"]}tr ({s["puts"]}PUT+{s["calls"]}CALL) '
                  f'WR{s["wr"]}% PnL${total:.0f} (PUT${pp:+.0f}+CALL${cp:.0f})')
            # Per-trade detail for PUTs
            if 'trades' in locals() and trs:
                puts = [t for t in trs if t['dir'] == 'PUT']
                for t in puts:
                    ed = t['entry_date']
                    if hasattr(ed, 'strftime'): ed = ed.strftime('%Y-%m-%d')
                    print(f'    PUT {ed} ep={t["entry_price"]:.1f} pnl=${t["pnl"]:+.0f} '
                          f'reason={t["entry_reason"]} exit={t["exit_type"]}')

# ─── 6. Final ranking table ───
print(f'\n========== FINAL 10y RANKING ==========')
print(f'  Baseline T+30/3.5% (CALL only): ${baseline_30_35:.0f}')
rows = [(k, s) for k, s in RESULTS.items() if '10y' in k]
rows.sort(key=lambda x: -x[1]['pnl'])
for k, s in rows:
    delta = s['pnl'] - baseline_30_35
    puts = s.get('puts', 0)
    print(f'  {k:45s} PnL${s["pnl"]:+.0f} ({delta:+.0f}) '
          f'{s["n"]}tr {s["puts"]}PUT+{s["calls"]}CALL WR{s["wr"]}%')

# ─── 7. Per-trade detail for top 3 combos ───
print(f'\n========== TOP 3 COMBOS PER-TRADE ==========')
combos = [(k, s) for k, s in RESULTS.items() if k.startswith('COMBO') and '10y' in k]
combos.sort(key=lambda x: -x[1]['pnl'])
for k, s in combos[:3]:
    print(f'\n--- {k}: {s["n"]}tr PnL${s["pnl"]:.0f} ---')
    # We don't have trades stored in RESULTS (only summary), re-run
    # Parse params from key
    parts = k.split('|')[0].split('_')
    h = int(parts[1])
    c_tr = float(parts[2].replace('cTr','')) / 100
    p_tr = float(parts[3].replace('pTr','')) / 100
    p_lev = float(parts[4].replace('pLev',''))
    sys.stdout, sys.stderr = devnull, devnull
    trs = run_bt(period='10y', put_method='di_cross',
                 put_score=35, put_vix=15,
                 put_lev=p_lev, put_trail=p_tr, put_hold=h, put_stop=0.10,
                 call_hold=h, call_trail=c_tr,
                 call_lev=3, call_stop=0.05, call_nb_stop=0.01)
    sys.stdout, sys.stderr = old_out, old_err
    puts = [t for t in trs if t['dir'] == 'PUT']
    for t in puts:
        ed = t['entry_date']
        if hasattr(ed, 'strftime'): ed = ed.strftime('%Y-%m-%d')
        print(f'  PUT {ed} ep={t["entry_price"]:.1f} pnl=${t["pnl"]:+.0f} '
              f'reason={t["entry_reason"]} exit={t["exit_type"]}')

devnull.close()
