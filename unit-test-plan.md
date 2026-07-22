# XSP 早晚报单元测试计划

测试框架：pytest  
测试文件：`tests/test_market_report.py`  
目标函数：`app.send_market_report()`（纯逻辑部分，mock 掉 yfinance / moomoo / requests）

---

## 1. 测试数据准备

所有用例需要 mock 以下依赖：

| 依赖 | mock 方式 | 说明 |
|---|---|---|
| `requests.post` (Telegram) | `unittest.mock.patch` | 断言 URL + payload，不真实发送 |
| `yfinance` | `unittest.mock.patch` | `fast_info.last_price`, `history()` 返回预设 DataFrame |
| `moomoo OpenQuoteContext` | `unittest.mock.patch` | `get_option_chain` 返回预设期权链 |
| `datetime.now(ET_TZ)` | `freezegun` 或手动 mock | 控制当前时间，测试定时/去重/周末逻辑 |
| `datetime.now(S_TZ)` | 同上 | 控制悉尼时间，测试报告窗口 |
| `user_watchlist` | 直接赋值全局变量 | 测试平仓提示 |
| `latest_data["options"]` | 直接赋值 | 测试组合价计算 |
| `historical_stats` | 直接赋值 | 测试趋势指标计算 |

建议用 `@pytest.fixture` 统一初始化 mock 环境。

---

## 2. 开仓逻辑测试

### 2.1 趋势鉴别

| # | 输入条件 | 预期 |
|---|---|---|
| 1 | ADX≥25, VR≥1.2, 其他正常 | `is_trend = True`（综合分 ≥ 65） |
| 2 | ADX<20, VR<1.0, 其他正常 | `is_trend = False`（综合分 < 65） |
| 3 | ADX=40, VR=0.5（高趋势但低 VR） | 边界：综合分可能仍 ≥ 65 |
| 4 | VIX rank=0（极端低波） | 不影响趋势判断 |

### 2.2 方向判断

| # | 场景 | 输入 | 预期 `direction` | 预期 `reason` |
|---|---|---|---|---|
| 5 | 震荡 + 近 BB 上轨 | `BBU-price < ATR14×60%` | `PUT` | `贴BB上轨(95%)` |
| 6 | 震荡 + 近 BB 下轨 | `price-BBL < ATR14×60%` | `CALL` | `贴BB下轨(5%)` |
| 7 | 震荡 + BB 中段 | 不满足近轨条件，且非趋势 | `None` | `BB 中段` |
| 8 | 单边上升趋势 | `is_trend=True`, `skew=3.5` | `CALL` | `Skew 3.5` |
| 9 | 单边下降趋势 | `is_trend=True`, `skew=-2.1` | `PUT` | `Skew -2.1` |
| 10 | 近轨阈值：ATR14 可用 | ATR14=8.0, BW=40, 价差=2.0 | `PUT` (2.0 < 8.0×0.3=2.4) |
| 11 | 近轨阈值：ATR14 不可用 | ATR14=None, BW=40, BW×10%=4.0, 价差=3.0 | `PUT` (3.0 < 4.0) |
| 12 | 近轨阈值：ATR14 不可用且价差超 BW×10% | ATR14=None, 价差=5.0 > 4.0 | 不触发近轨条件 |

### 2.3 树行权价计算

| # | 方向 | `ema20` | 预期 `s` | 预期 `m` | 预期 `l` |
|---|---|---|---|---|---|
| 13 | PUT | 750.0 | 755 (m+10) | 745 (ema20-5) | 740 (m-5) |
| 14 | CALL | 750.0 | 745 (m-10) | 755 (ema20+5) | 760 (m+5) |
| 15 | PUT, is_trend=True | 750.0 | off=-5, m=745 | 755 (m+10) | 740 (m-5) |
| 16 | CALL, is_trend=True | 750.0 | off=+5, m=755 | 745 (m-10) | 760 (m+5) |

### 2.4 裸买单腿行权价

| # | 方向 | 预期 Δ | 符号规则 |
|---|---|---|---|
| 17 | CALL | `_find_delta_strike(7DTE, 0.35, 'C')` | Δ≈0.35 |
| 18 | PUT | `_find_delta_strike(7DTE, -0.35, 'P')` | Δ≈-0.35 |

### 2.5 ETF 参考

| # | `direction` | 预期 |
|---|---|---|
| 19 | PUT | `★ 做空 ETF: SH(1x) / SDS(2x) / SPXS(3x)` |
| 20 | CALL | `★ 做多 ETF: SPYM(1x) / SSO(2x) / SPXL(3x)` |
| 21 | None | 不出现 ETF 行 |

---

## 3. 平仓提示测试

### 3.1 watchlist 迭代

mock `user_watchlist` 包含以下条目：

| 用例 | date | short | mid | long | opt | entry | strategy |
|---|---|---|---|---|---|---|---|
| A | 260720 | 755 | 745 | 740 | P | 1.50 | xmas |
| B | 260721 | 760 | 750 | 745 | P | 0.80 | xmas |
| C | 260722 | 745 | 755 | 760 | C | "" | xmas |
| D | 260717 | 740 | — | — | P | 2.00 | naked |
| E | 260720 | 750 | — | — | P | 0.50 | naked |

mock `latest_data["options"]` 返回对应的组合价 / 中间价。

### 3.2 平仓条件

| # | 目标条目 | 模拟条件 | 预期 alert |
|---|---|---|---|
| 22 | A | `dte=3` | `仅剩3天到期` |
| 23 | B | `pnl_pct=60%` (当前组合价-入场>50%) | `盈利60%` |
| 24 | D | `pnl_pct=-55%` | `亏损55%` |
| 25 | A | `direction='CALL'`, `g_opt='P'` | `方向冲突` |
| 26 | D | `pnl=1.6`, `max_loss=2.0`, `cur_loss=0.4×100=40% < 80%` | 不触发（80% 阈值未到） |
| 27 | D | `pnl=1.8`, `max_loss=2.0`, `cur_loss=0.2×100=20%`, `cur_loss/max_loss=10%` | 不触发 |
| 28 | D | `pnl=浮动亏损达80%` | `浮亏达最大损失80%` |
| 29 | — | `now_et.weekday()==4` (周五) | `周末持仓风险` |
| 30 | — | `score<65`, `direction=None` | `价格在BB中段，综合分不足，建议减少仓位` |
| 31 | — | `_prev_report_score=72`, `score=58` | `趋势结束（72→58），建议平仓` |
| 32 | — | `_prev_report_direction='PUT'`, `direction='CALL'` | `方向已由PUT转为CALL，建议平仓` |
| 33 | — | 首次启动（`_prev_report_score=0`），`score=72` | 不触发（上期<65） |
| 34 | — | `score=80`, `direction=None` | 不触发 BB 中段减仓（score≥65） |

### 3.3 `direction=None` 时平仓仍运行

| # | 场景 | 预期 |
|---|---|---|
| 35 | `direction=None`，watchlist 有 A（DTE≤3），`score=60` | `close_lines` 包含 DTE 提示 + BB 中段减仓提示 |
| 36 | `direction=None`，watchlist 为空 | `close_lines` 为空，只发标题行 |

---

## 4. 组合价公式测试

| # | 类型 | opt_type | 输入 mid | 预期公式 | 结果 |
|---|---|---|---|---|---|
| 37 | xmas | PUT | S=2.0, M=1.0, L=0.5 | `S + 2L - 3M = 2.0 + 1.0 - 3.0 = 0.0` | 0.0 |
| 38 | xmas | CALL | S=2.0, M=1.0, L=0.5 | `S + 2L - 3M = 2.0 + 1.0 - 3.0 = 0.0` | 0.0 |
| 39 | xmas | PUT | S=3.0, M=1.0, L=0.5 | `3.0 + 1.0 - 3.0 = 1.0` | 1.0 (credit) |
| 40 | xmas | PUT | S=1.0, M=2.0, L=1.5 | `1.0 + 3.0 - 6.0 = -2.0` | -2.0 (debit) |

---

## 5. 定时与去重测试

| # | 场景 | 预期行为 |
|---|---|---|
| 41 | 悉尼时间 `21:30`，`_morning_report_date != today`，工作日 | 生成早报，发 Telegram，设置 dedup |
| 42 | 悉尼时间 `21:30`，`_morning_report_date == today` | 跳过（第一天已发） |
| 43 | 悉尼时间 `22:00` | 跳过（不在窗口 21:25-21:35） |
| 44 | 悉尼时间 `09:30` | 生成晚报 (`dte_adj=1`) |
| 45 | 悉尼时间周六 | 跳过（`weekday()>=5`） |
| 46 | 悉尼时间周日 | 跳过（`weekday()>=5`） |
| 47 | `force=True`, `_latest_report` 为空 | 生成报告，不发 Telegram，设 `_latest_report` |
| 48 | `force=True`, `_latest_report` 已存在 | 不覆盖？（注：当前代码不检查，重新生成） |

---

## 6. 边界与异常测试

| # | 场景 | 预期 |
|---|---|---|
| 49 | `historical_stats` 中 `ema_20` 为 0 或缺失 | 函数不崩溃，使用兜底值 |
| 50 | `hs.get('atr_14')` 为 0 或 None | 降级到 `BW×10%` |
| 51 | `latest_data["options"]` 为空 | close_lines 不崩溃，`for g in user_watchlist` 的 `o_s = None` 被 `if o_s and o_m and o_l` 跳过 |
| 52 | `_find_n_dte_expiry` 找不到合适的到期日（如非交易日） | 返回 `None`，树和裸买部分跳过 |
| 53 | `compute_combo_price` 中某个 leg mid 为 None | 不纳入计算，`cur_mid` 保持 None，跳过该条目 |
| 54 | watchlist 条目缺少 `date` 或 `short` 字段 | `continue` 跳过，不崩溃 |
| 55 | watchlist 条目 `entry` 字段无法转为 float | `try/except` 跳过 |
| 56 | 用户北京时区访问（非 ET/S_TZ） | 报告标题显示 ET 时间，定时基于 S_TZ |
| 57 | 跨日场景：ET 23:00（悉尼次日 13:00） | 晚报已过窗口，早报未到窗口，都不触发 |

---

## 7. 报告输出格式测试

| # | 场景 | 预期检验 |
|---|---|---|
| 58 | `direction='PUT'`, `is_trend=True` | 标题行含 icon+score，ETF 行，树行（7DTE），裸买行（7DTE, Δ≈-0.35），方向行 |
| 59 | `direction='CALL'`, `is_trend=True` | ETF 行，树行（7DTE），裸买行（7DTE, Δ≈0.35） |
| 60 | `direction='PUT'`, `is_trend=False` | 树行（off=0），无裸买行 |
| 61 | `direction=None` | 方向行 `BB中段，不开仓`，无 ETF/树/裸买 |
| 62 | `close_lines` 非空 | 追加 `━━━ 平仓提示 ━━━` 及各 alert |
| 63 | `close_lines` 为空 | 不追加平仓提示段 |
| 64 | 报告标题包含 ET 日期时间 | `Thu 2026-07-16 19:30 ET` 格式 |

---

## 8. 执行建议

1. 每个用例用 `@pytest.mark.parametrize` 保持可读性
2. 对 `send_market_report` 用 **白盒测试**（mock 内部数据）而非集成测试
3. 先实现 mock 固件（fixture），再逐个实现用例
4. 跑全部用例预期时间 < 30 秒
5. 建议并行实现顺序：
   - 2.1 ~ 2.5（开仓逻辑，共 21 个用例）
   - 5（定时去重，共 8 个用例）
   - 3（平仓提示，共 15 个用例）
   - 4（组合价公式，共 4 个用例）
   - 7（输出格式，共 7 个用例）
   - 6（边界异常，共 9 个用例）
