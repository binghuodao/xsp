"""Calibrate skew proxy (price/EMA20) against real option-chain skew from premium_log.db."""
import os, sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SRC_DIR, '..', 'premium_log.db')

print("=" * 60)
print("Skew 校准分析: proxy(price/EMA20) vs real(option mid)")
print("=" * 60)

# ── 1. Load premium_log data ──
conn = sqlite3.connect(DB_PATH)
df_raw = pd.read_sql_query(
    "SELECT ts, opt_symbol, strike, expiry, opt_type, mid, xsp_price, trade_date "
    "FROM premium_log WHERE role='option' AND mid > 0",
    conn
)
conn.close()
print(f"\n📦 premium_log 记录数: {len(df_raw)}")
print(f"   日期范围: {df_raw['trade_date'].min()} → {df_raw['trade_date'].max()}")

# ── 2. For each timestamp, compute real skew ──
# Groups by ts
snapshots = []
for ts, grp in df_raw.groupby('ts'):
    row = grp.iloc[0]
    xsp_price = row['xsp_price']
    if pd.isna(xsp_price) or xsp_price <= 0:
        continue

    # Find the expiry closest to 7 DTE from trade_date
    trade_date = row['trade_date']
    exps = grp['expiry'].unique()
    best_exp = None
    best_diff = 999
    for e in exps:
        try:
            d = (pd.to_datetime(e) - pd.to_datetime(trade_date)).days
            if 5 <= d <= 14 and abs(d - 7) < abs(best_diff):
                best_diff = d - 7
                best_exp = e
        except:
            continue
    if best_exp is None:
        continue

    # Filter to best expiry
    exp_grp = grp[grp['expiry'] == best_exp]

    # Find ATM strike
    puts = exp_grp[exp_grp['opt_type'] == 'P']
    calls = exp_grp[exp_grp['opt_type'] == 'C']
    if puts.empty or calls.empty:
        continue

    # ATM = strike closest to xsp_price
    all_strikes = sorted(exp_grp['strike'].unique())
    atm_strike = min(all_strikes, key=lambda s: abs(s - xsp_price))

    # Allow ±5 range for ATM
    if abs(atm_strike - xsp_price) > 5:
        continue

    atm_put = puts[puts['strike'] == atm_strike]
    atm_call = calls[calls['strike'] == atm_strike]
    if atm_put.empty or atm_call.empty:
        continue

    put_mid = atm_put['mid'].iloc[0]
    call_mid = atm_call['mid'].iloc[0]
    if put_mid <= 0 or call_mid <= 0:
        continue

    # Real skew: normalized difference
    total = put_mid + call_mid
    real_skew = (put_mid - call_mid) / total if total > 0 else 0

    snapshots.append({
        'ts': ts,
        'trade_date': trade_date,
        'xsp_price': xsp_price,
        'expiry': best_exp,
        'atm_strike': atm_strike,
        'put_mid': put_mid,
        'call_mid': call_mid,
        'real_skew': real_skew,
    })

df_skew = pd.DataFrame(snapshots)
print(f"\n📊 有效快照数: {len(df_skew)}  (含 ATM put/call mid)")
print(f"   Real skew 范围: {df_skew['real_skew'].min():.4f} ~ {df_skew['real_skew'].max():.4f}")
print(f"   Real skew 均值: {df_skew['real_skew'].mean():.4f}")

# ── 3. Compute proxy skew (price/EMA20) ──
print("\n⏳ 获取 ^XSP EMA20...")
xsp = yf.download("^XSP", period="1y", interval="1d", auto_adjust=False)
xsp.columns = [c[0] for c in xsp.columns]
close_col = 'Adj Close' if 'Adj Close' in xsp.columns else 'Close'
xsp['ema_20'] = xsp[close_col].ewm(span=20, min_periods=20).mean()

# Merge: for each snapshot date, find EMA20
df_skew['date'] = pd.to_datetime(df_skew['trade_date'])
xsp_ema = xsp[['ema_20']].copy()
xsp_ema.index = pd.to_datetime(xsp_ema.index)
df_skew = df_skew.merge(xsp_ema, left_on='date', right_index=True, how='left')
df_skew['proxy_skew'] = (df_skew['xsp_price'] / df_skew['ema_20'] - 1) * 100

valid = df_skew.dropna(subset=['real_skew', 'proxy_skew'])
print(f"\n   有效回归样本数: {len(valid)}")

# ── 4. Linear regression (numpy-based) ──
x_vals = valid['proxy_skew'].values
y_vals = valid['real_skew'].values

# OLS: y = a + b*x
A = np.vstack([x_vals, np.ones_like(x_vals)]).T
b, a = np.linalg.lstsq(A, y_vals, rcond=None)[0]
y_pred = a + b * x_vals

# R²
ss_res = np.sum((y_vals - y_pred) ** 2)
ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

# Pearson correlation
corr = np.corrcoef(x_vals, y_vals)[0, 1]
p_val = 0.0  # simplified

print(f"\n{'─' * 50}")
print(f"📈 回归结果")
print(f"{'─' * 50}")
print(f"   r² (决定系数):     {r2:.4f}")
print(f"   Pearson r:         {corr:.4f}")
print(f"   斜率 (b):          {b:.4f}")
print(f"   截距 (a):          {a:.4f}")
print(f"   样本量:            {len(valid)}")
print(f"{'─' * 50}")

if r2 >= 0.5:
    print("   ✅ 强相关 — proxy 可用，趋势方向结果可信")
elif r2 >= 0.2:
    print("   ⚠️  弱相关 — proxy 有参考价值，趋势方向结果需打折解读")
else:
    print("   ❌ 几乎无关 — proxy 无效，趋势方向准确率≈随机噪音")

# ── 5. Charts ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Skew Proxy Calibration: Price/EMA20 vs Real ATM Skew', fontsize=14, fontweight='bold')

# Scatter
ax1 = axes[0]
ax1.scatter(valid['proxy_skew'], valid['real_skew'], alpha=0.5, s=15)
x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
y_range = a + b * x_range
ax1.plot(x_range, y_range, 'r-', linewidth=2, label=f'OLS (r²={r2:.3f})')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
ax1.set_xlabel('Proxy Skew: (Price/EMA20 - 1) × 100 (%)')
ax1.set_ylabel('Real Skew: (Put Mid - Call Mid) / Total')
ax1.legend()
ax1.set_title(f'Regression: n={len(valid)}, slope={b:.4f}')

# Time series
ax2 = axes[1]
time_data = valid.sort_values('ts')
ax2.plot(range(len(time_data)), time_data['real_skew'].values, 'b-', alpha=0.7, label='Real Skew', linewidth=1)
ax2.plot(range(len(time_data)), time_data['proxy_skew'].values / 20, 'r-', alpha=0.7, label='Proxy/20 (scaled)', linewidth=1)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax2.set_xlabel('Snapshot (chronological)')
ax2.set_ylabel('Skew Value')
ax2.set_title('Real vs Proxy Skew Over Time')
ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.93])
chart_path = os.path.join(SRC_DIR, 'skew_calibration.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
print(f"\n📊 图表: {chart_path}")

# Text report
report_lines = [
    "=" * 60,
    "Skew 校准分析报告",
    "=" * 60,
    f"数据源: premium_log.db (role='option')",
    f"快照日期: {df_skew['trade_date'].min()} → {df_skew['trade_date'].max()}",
    f"有效快照数: {len(df_skew)}",
    f"回归样本数: {len(valid)}",
    "",
    f"r²:          {r2:.4f}",
    f"Pearson r:   {corr:.4f}",
    f"斜率:        {b:.4f}",
    f"截距:        {a:.4f}",
    f"Real Skew范围: {df_skew['real_skew'].min():.4f} ~ {df_skew['real_skew'].max():.4f}",
    f"Proxy Skew范围: {valid['proxy_skew'].min():.2f}% ~ {valid['proxy_skew'].max():.2f}%",
    "",
]
if r2 >= 0.5:
    report_lines.append("结论: 强相关 — proxy可用")
elif r2 >= 0.2:
    report_lines.append("结论: 弱相关 — proxy有参考价值")
else:
    report_lines.append("结论: 几乎无关 — proxy无效")

report_path = os.path.join(SRC_DIR, 'skew_calibration.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(report_lines) + '\n')
print(f"📄 报告: {report_path}")
print("✅ 校准完成")
