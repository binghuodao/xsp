# New main function with morning/evening/hold reports
def main():
    global _crash_etf_carry
    init_state()
    trading_days = list(xsp.index)
    if REPLAY_START is not None:
        trading_days = [d for d in trading_days if d >= REPLAY_START]
    if len(trading_days) <= WARMUP_DAYS:
        sys.exit('not enough history')
    trading_days = trading_days[WARMUP_DAYS:]
    print(f"Replaying {len(trading_days)} trading days  {trading_days[0].date()} → {trading_days[-1].date()}")

    # Separate record lists for morning, evening, and hold lines
    morning_records = []
    evening_records = []
    hold_records = []
    # Global records (combined) for backward compatibility with index/trade files
    records = []
    stats = {'full': 0, 'compact': 0, 'skipped': 0, 'opens': 0, 'closes': 0}
    failures = []
    trace_rows = []       # --trace-trend per-day diagnostics

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

        # ── MORNING REPORT ──
        _captured.clear()
        app.send_market_report('morning', force=False)
        morning_msg = _captured.get('msg', '')
        morning_r = app._latest_report.copy()
        morning_d = morning_r.get('direction')
        morning_blocked = infer_blocked(morning_msg, morning_r)

        # Capture morning-specific alerts (同价续持等)
        morning_specific_alerts = []
        if '同价续持' in morning_msg:
            if '免卖' in morning_msg:
                morning_specific_alerts.append('同价续持免卖')
            elif '加开' in morning_msg:
                morning_specific_alerts.append('同价续持加开')
            else:
                morning_specific_alerts.append('同价续持')

        morning_msg_clean = morning_msg
        morning_r_clean = morning_r.copy()

        # ── EVENING REPORT ──
        _captured.clear()
        app.send_market_report('evening', force=False)
        evening_msg = _captured.get('msg', '')
        if not evening_msg:
            stats['skipped'] += 1
            continue
        r = app._latest_report
        now_fp = state_fp()
        d = r.get('direction')
        blocked = infer_blocked(evening_msg, r)
        t_now = cur_trade('CRASH')
        if t_now and t_now.get('n') in _carry_display:
            cd = _carry_display[t_now['n']]
            if cd['fresh'] != cd['blend']:
                evening_msg = evening_msg.replace(f"入场${cd['fresh']:.2f}", f"入场${cd['blend']:.2f}")

        alerts = r.get('close_alerts', []) or []
        sig_alerts = [l for l in alerts if 'BB中段' not in l]
        now_spread_seg = _seg_from_app()

        # ── TREND 哑火诊断: 逐日分类 (--trace-trend) ──
        if TRACE_TREND:
            hs = app.historical_stats
            bbl = hs.get('support', 0); bbu = hs.get('resistance', 0)
            bw = bbu - bbl if (bbl and bbu and bbu > bbl) else 1
            dlow_v = (price - bbl) / bw * 100
            atr14 = hs.get('atr_14', 0)
            nt = atr14 * 0.60 if atr14 and atr14 > 0 else bw * 0.10
            near_top = (bbu - price) < nt
            near_bottom = (price - bbl) < nt
            if d == 'CALL':
                if dlow_v > 80:
                    tb = '高位暂缓'
                elif '崩盘/MR层占用' in msg:
                    tb = '三层互斥'
                elif now_fp[1] is not None:
                    tb = '已在TREND仓'
                else:
                    tb = '漏开'
            elif d is None:
                sc = r.get('score') or 0
                tb = '融合吃掉' if sc >= 50 else '非趋势'
            else:
                tb = f'方向{d}'
            trace_rows.append({'asof': asof, 'score': r.get('score'), 'direction': d,
                               'reason': r.get('reason'), 'dlow': dlow_v,
                               'near_top': near_top, 'near_bottom': near_bottom,
                               'blocked': blocked, 'bucket': tb})

        ev = []
        # trend open/close (supports same-day close+reopen: prev_fp[1] != now_fp[1])
        if prev_fp is not None:
            tr_changed = prev_fp[1] != now_fp[1]
            if tr_changed and prev_fp[1] is not None:
                n = cur_trade('TREND')['n'] if cur_trade('TREND') else None
                t = cur_trade('TREND')
                if t and t.get('etf_entry'):
                    t['etf_pnl'] = (t.get('etf_shares') or 0) * (spxl_p - t['etf_entry'])
                book_spread(asof, price, prev_spread_seg, now_spread_seg)
                close_trade('TREND', asof, price, trend_close_result(alerts), spxl_p)
                ev.append(f'TREND#{n} 平仓')
            if tr_changed and now_fp[1] is not None:
                n = open_trade('TREND', asof)
                t = cur_trade('TREND')
                if t:
                    t['etf_entry'] = spxl_p
                    t['etf_shares'] = max(round(ETF_SIZE['TREND'] / spxl_p), 1)
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
                        t['etf_pnl'] = (t.get('etf_shares') or 0) * (spxl_p - t['etf_entry'])
                res = ('首阳+0.3%' if 'MR首阳' in msg else '止损-2%' if 'MR跌穿' in msg else '3天强制平')
                close_trade('MR', asof, price, res, spxl_p)
                ev.append(f'MR#{n} 平仓')
            if mr_changed and now_fp[5] is not None:
                n = open_trade('MR', asof)
                t = cur_trade('MR')
                if t:
                    t['etf_entry'] = spxl_p
                    t['etf_shares'] = max(round(ETF_SIZE['MR'] / spxl_p), 1)
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
                    if OPT_STANDALONE == 'tp' and not t.get('opt_closed') and not t.get('resid') \
                            and t.get('k1') and t.get('k2') and t.get('debit') is not None:
                        t['resid'] = {'k1': t['k1'], 'k2': t['k2'], 'debit': t['debit'],
                                      'sigma': t.get('sigma') or 0.20,
                                      'entry': t['resid_entry'], 'expiry': t['resid_expiry'],
                                      'indep': True}
                        ev.append('期权转残值续持 (独立TP/SL/到期)')
                    if t.get('etf_out'):
                        if t.get('half_date') is None and t.get('k1') and t.get('k2') and t.get('debit') is not None and not _opt_done(t) and not t.get('resid'):
                            close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                            t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
                        elif t.get('half_date'):
                            base = t.get('re_entry_spxl') or t.get('etf_entry')
                            keep = t.get('keep_sh')
                            if keep is None:
                                sh = t.get('etf_shares') or 0
                                keep = sh - _leg_round(sh, CRASH_HALF)
                                t['keep_sh'] = keep
                            fill = _etf_stop_fill(base, asof) if ('崩盘跌穿' in msg and base) else spxl_p
                            fill = fill if fill is not None else spxl_p
                            rem = keep * (fill - base) if base else 0
                            t['etf_pnl'] = ((t.get('half_etf') or 0) + rem) * sm
                        elif t.get('yin_sh') is not None:
                            yin_pct = t.get('yin_pct') or CRASH_YIN
                            sh = t.get('etf_shares') or 0
                            yin_sh = t.get('yin_sh')
                            if yin_sh is None:
                                yin_sh = _leg_round(sh, CRASH_YIN)
                                t['yin_sh'] = yin_sh
                                t['yin_keep_sh'] = sh - yin_sh
                            keep = t.get('yin_keep_sh')
                            if keep is None:
                                keep = sh - yin_sh
                            if t.get('re_yin_px'):
                                rem = keep * (t['re_yin_px'] - t['etf_entry']) * sm
                                full_fill = _etf_stop_fill(t['re_yin_px'], asof) if '崩盘跌穿' in msg else spxl_p
                                full_fill = full_fill if full_fill is not None else spxl_p
                                full = sh * (full_fill - t['re_yin_px']) * sm
                                t['etf_pnl'] = t['yin_etf'] + rem + full
                            else:
                                final = _etf_stop_fill(t['etf_entry'], asof) if '崩盘跌穿' in msg else spxl_p
                                final = final if final is not None else spxl_p
                                t['etf_pnl'] = t['yin_etf'] + keep * (final - t['etf_entry']) * sm
                            if not t.get('re_yin_px') and t.get('k1') and t.get('k2') and t.get('debit') is not None and not _opt_done(t) and not t.get('resid'):
                                if CRASH_MODE in ('V9', 'V10', 'V11') and '崩盘跌穿' in msg:
                                    t['resid'] = {'k1': t['k1'], 'k2': t['k2'], 'debit': t['debit'],
                                                  'sigma': t.get('sigma') or 0.20,
                                                  'entry': t['resid_entry'], 'expiry': t['resid_expiry']}
                                else:
                                    close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                                    t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
                        else:
                            if t.get('k1') and t.get('k2') and t.get('debit') is not None and not _opt_done(t) and not t.get('resid'):
                                if CRASH_MODE in ('V9', 'V10', 'V11') and '崩盘跌穿' in msg:
                                    t['resid'] = {'k1': t['k1'], 'k2': t['k2'], 'debit': t['debit'],
                                                  'sigma': t.get('sigma') or 0.20,
                                                  'entry': t['resid_entry'], 'expiry': t['resid_expiry']}
                                else:
                                    close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                                    t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
                            if t.get('etf_entry'):
                                final = _etf_stop_fill(t['etf_entry'], asof) if '崩盘跌穿' in msg else spxl_p
                                if final is None:
                                    _crash_etf_carry = {'shares': t.get('etf_shares') or 0,
                                                        'entry': t['etf_entry'],
                                                        'fresh_entry': spxl_p}
                                    t['etf_pnl'] = 0.0
                                else:
                                    t['etf_pnl'] = (t.get('etf_shares') or 0) * (final - t['etf_entry']) * sm
                        if t.get('reopened'):
                            rcal = (asof - t['reopen_date']).days
                            rT_rem = max(DTE / 365.0 - rcal / 365.0, 1 / 365.0)
                            rclose = _bs_spread(price, t['reopen_k1'], t['reopen_k2'], rT_rem, t['reopen_sigma'])
                            ropnl = max((rclose - t['reopen_debit']), -t['reopen_debit']) * 100 * osm
                            t['opt_pnl'] = (t.get('opt_pnl') or 0) + ropnl
                        res = ('首阴清仓' if ('崩盘首阴' in msg and '崩盘首阴续持' not in msg) else '二次首阳清仓' if '崩盘二次首阳' in msg else '首阳退半' if '崩盘首阳' in msg
                               else '止损-2.5%' if '崩盘跌穿' in msg else '4天强制平')
                        close_trade('CRASH', asof, price, res, spxl_p)
                        ev.append(f'CRASH#{n} 平仓')
                    if cr_changed and now_fp[6] is not None:
                        n = open_trade('CRASH', asof)
                        t = cur_trade('CRASH')
                        if t:
                            if _crash_etf_carry:
                                carry = _crash_etf_carry
                                t['etf_shares'] = max(round(ETF_SIZE['CRASH'] / spxl_p), 1)
                                add = max(t['etf_shares'] - carry['shares'], 0)
                                t['etf_entry'] = (carry['entry'] * carry['shares'] + spxl_p * add) / t['etf_shares']
                                _carry_display[t['n']] = {'fresh': carry['fresh_entry'], 'blend': t['etf_entry']}
                                _crash_etf_carry = None
                            else:
                                t['etf_entry'] = spxl_p
                                t['etf_shares'] = max(round(ETF_SIZE['CRASH'] / spxl_p), 1)
                            t['k1'] = app._crash_k1; t['k2'] = app._crash_k2
                            t['debit'] = app._crash_debit; t['sigma'] = app._crash_sigma
                            t['size_mult'] = RISK_MULT if app._risk_off_active() else 1.0
                            t['resid_entry'] = price
                            t['resid_expiry'] = asof + timedelta(days=DTE)
                        ev.append(f'CRASH#{n} 开仓')
                    if not prev_fp[7] and now_fp[7]:
                        t = cur_trade('CRASH')
                        if t and prev_fp[10] is True:
                            sm = t.get('size_mult', 1.0)
                            osm = sm * OPT_MULT
                            cal = (asof - t['open']).days
                            rT_rem = max(DTE / 365.0 - cal / 365.0, 1 / 365.0)
                            if t.get('k1') and t.get('k2') and t.get('debit') is not None and OPT_STANDALONE != 'tp':
                                close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                                t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
                                t['opt_closed'] = True
                                t['k1'] = None; t['k2'] = None; t['debit'] = None
                            t['re_yin_px'] = spxl_p
                            ev.append('崩盘收复再进')
                        elif t:
                            sm = t.get('size_mult', 1.0)
                            osm = sm * OPT_MULT
                            cal = (asof - t['open']).days
                            T_rem = max(DTE / 365.0 - cal / 365.0, 1 / 365.0)
                            if t.get('k1') and t.get('k2') and t.get('debit') is not None and OPT_STANDALONE != 'tp':
                                close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                                t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
                                t['opt_closed'] = True
                                t['k1'] = None; t['k2'] = None; t['debit'] = None
                            if not t.get('etf_out') and t.get('etf_entry'):
                                sh = t.get('etf_shares') or 0
                                half_sh = _leg_round(sh, CRASH_HALF)
                                t['half_sh'] = half_sh
                                t['keep_sh'] = sh - half_sh
                                t['half_etf'] = half_sh * (spxl_p - t['etf_entry']) * sm
                            t['half_date'] = asof
                            ev.append('崩盘退半')
                        if not prev_fp[10] and now_fp[10]:
                            t = cur_trade('CRASH')
                            if t and t.get('etf_entry'):
                                sm = t.get('size_mult', 1.0)
                                t['yin_pct'] = CRASH_YIN
                                sh = t.get('etf_shares') or 0
                                yin_sh = _leg_round(sh, CRASH_YIN)
                                t['yin_sh'] = yin_sh
                                t['yin_keep_sh'] = sh - yin_sh
                                t['yin_etf'] = yin_sh * (spxl_p - t['etf_entry']) * sm
                                ev.append('崩盘首阴清')
                            if not prev_fp[8] and now_fp[8]:
                                t = cur_trade('CRASH')
                                if t:
                                    t['re_entry_spxl'] = spxl_p
                                    if CRASH_MODE == 'V11' and app._crash_opt_reopened:
                                        t['reopened'] = True
                                        t['reopen_k1'] = app._crash_k1; t['reopen_k2'] = app._crash_k2
                                        t['reopen_debit'] = app._crash_debit
                                        t['reopen_sigma'] = app._crash_sigma or 0.20
                                        t['reopen_date'] = asof
                                ev.append('崩盘再进')
                            if not prev_fp[9] and now_fp[9]:
                                t = cur_trade('CRASH')
                                if t and not t.get('etf_out'):
                                    sm = t.get('size_mult', 1.0)
                                    if t.get('half_date'):
                                        base = t.get('re_entry_spxl') or t.get('etf_entry')
                                        keep = t.get('keep_sh')
                                        if keep is None:
                                            keep = (t.get('etf_shares') or 0) - _leg_round(t.get('etf_shares') or 0, CRASH_HALF)
                                        fill = _etf_stop_fill(base, asof) if base else spxl_p
                                        fill = fill if fill is not None else spxl_p
                                        rem = keep * (fill - base) if base else 0
                                        t['etf_pnl'] = ((t.get('half_etf') or 0) + rem) * sm
                                    elif t.get('etf_entry'):
                                        _fill = _etf_stop_fill(t['etf_entry'], asof)
                                        _fill = _fill if _fill is not None else spxl_p
                                        t['etf_pnl'] = (t.get('etf_shares') or 0) * (_fill - t['etf_entry']) * sm
                                    t['etf_out'] = True
                                ev.append('崩盘ETF止损')
                            if prev_blocked is not None and prev_blocked != blocked:
                                ev.append('高位拦截' if blocked else '拦截解除')
                            if prev_dir is not None and d != prev_dir:
                                ev.append(f'方向 {prev_dir}→{d}')
                            if sig_alerts:
                                ev.append('平仓提示')
                            book_spread(asof, price, prev_spread_seg, now_spread_seg)
                            if OPT_STANDALONE == 'tp':
                                t = cur_trade('CRASH')
                                if t and not t.get('opt_closed') and not t.get('resid') and t.get('k1') and t.get('k2') and t.get('debit') is not None:
                                    sm = t.get('size_mult', 1.0)
                                    osm = sm * OPT_MULT
                                    cal = (asof - t['open']).days
                                    T_rem = max(DTE / 365.0 - cal / 365.0, 1 / 365.0)
                                    close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                                    tp_line = t['debit'] * OPT_TP
                                    sl_line = t['debit'] * OPT_SL_RESID if OPT_SL_RESID > 0 else 0
                                    if close_d >= tp_line:
                                        t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
                                        t['opt_closed'] = True
                                        ev.append(f'期权独立止盈 (价值${close_d:.2f}≥${tp_line:.2f})')
                                    elif sl_line > 0 and close_d <= sl_line and (not OPT_SL_ADAPTIVE or price <= t.get('resid_entry', price)):
                                        t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
                                        t['opt_closed'] = True
                                        ev.append(f'期权残值止损 (价值${close_d:.2f}≤${sl_line:.2f})')
                                elif OPT_STANDALONE == 'tp-v9':
                                    t = cur_trade('CRASH')
                                    scaled_away = t and (bool(t.get('half_date')) or t.get('yin_sh') is not None)
                                    if t and not t.get('opt_closed') and not scaled_away and t.get('k1') and t.get('k2') and t.get('debit') is not None:
                                        sm = t.get('size_mult', 1.0)
                                        osm = sm * OPT_MULT
                                        cal = (asof - t['open']).days
                                        T_rem = max(DTE / 365.0 - cal / 365.0, 1 / 365.0)
                                        close_d = _bs_spread(price, t['k1'], t['k2'], T_rem, t.get('sigma'))
                                        tp_line = t['debit'] * OPT_TP
                                        if close_d >= tp_line:
                                            t['opt_pnl'] = max((close_d - t['debit']), -t['debit']) * 100 * osm
                                            t['opt_closed'] = True
                                            ev.append(f'期权独立止盈 (价值${close_d:.2f}≥${tp_line:.2f})')
                                        ev.extend(settle_residuals(asof, price))
                                        fp_changed = (prev_fp is not None and now_fp != prev_fp)
                                        gate_blocked = '被利率闸门拦截' in msg

                                        is_first = (i == 0)
                                        if ev or fp_changed or is_first or gate_blocked:
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

                                            # === HOLD RECORDS FOR INTERMEDIATE POSITIONS ===
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
                            hold_records.append({'asof': asof, 'full': False, 'tag': '', 'msg': ' | '.join(line_parts),
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

            # ... rest of the function remains the same ...


# ═══════════════════════════════ 6. render batch files ═══════════════════════════════
# Modified to write separate files for morning, evening, and hold records
def _rec_lines(rec):
    if rec['full']:
        d = rec['dir']; sc = rec.get('score')
        tag = rec['tag']
        head = f"[{rec['asof']} {rec['asof'].strftime('%a')}]  {tag}  方向{str(d)} 综合{sc}"
        sep = '─' * 70
        return [head, sep] + rec['msg'].split('\n')
    return [f"    {rec['asof']}  {rec['msg']}"]

# Group records by year and type
morning_by_year = {}
evening_by_year = {}
hold_by_year = {}

for rec in morning_records:
    morning_by_year.setdefault(rec['asof'].year, []).append(rec)
for rec in evening_records:
    evening_by_year.setdefault(rec['asof'].year, []).append(rec)
for rec in hold_records:
    hold_by_year.setdefault(rec['asof'].year, []).append(rec)

# Write morning reports
for yr, rcs in sorted(morning_by_year.items()):
    for fn, sub in [(f'sim_rpt_{yr}_morning.txt', rcs)]:
        body = []
        body.append('═' * 70)
        body.append(f'XSP 盘前早报 — 全历史重放 文件={fn}')
        body.append(f'生成: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}  | 数据: yfinance {PERIOD} | 全部盘前 09:30 ET')
        body.append('═' * 70)
        body.append('')
        start_line = len(body) + 1
        for rc in rcs:
            if not STATS_ONLY:
                rec_file[id(rc)] = (fn, start_line)
            ls = _rec_lines(rc)
            body.extend(ls)
            start_line += len(ls)
        if not STATS_ONLY:
            with open(os.path.join(RESULT_DIR, fn), 'w') as f:
                f.write('\n'.join(body) + '\n')
            print(f"  wrote {fn}: {len(sub)} entries, {len(body)} lines")

# Write evening reports
for yr, rcs in sorted(evening_by_year.items()):
    for fn, sub in [(f'sim_rpt_{yr}_evening.txt', rcs)]:
        body = []
        body.append('═' * 70)
        body.append(f'XSP 盘后晚报 — 全历史重放 文件={fn}')
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

# Write hold records
for yr, rcs in sorted(hold_by_year.items()):
    for fn, sub in [(f'sim_rpt_{yr}_hold.txt', rcs)]:
        body = []
        body.append('═' * 70)
        body.append(f'XSP 持仓状态变化 — 全历史重放 文件={fn}')
        body.append(f'生成: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}  | 数据: yfinance {PERIOD}')
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