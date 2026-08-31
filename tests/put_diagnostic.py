#!/usr/bin/env python3
"""
Comprehensive PUT trading diagnostic for ^XSP.
Analyzes 10 years of data with indicators, scores, forward returns, and strategy simulation.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', '{:.2f}'.format)

# ============================================================
# 1. DOWNLOAD DATA
# ============================================================
print("=" * 100)
print("STEP 1: DOWNLOADING 10 YEARS OF ^XSP AND ^VIX DATA")
print("=" * 100)

end_date = datetime.now()
start_date = end_date - timedelta(days=365*11 + 200)

print(f"Fetching from {start_date.date()} to {end_date.date()}...")

xsp = yf.download('^XSP', start=start_date, end=end_date, auto_adjust=True)
vix = yf.download('^VIX', start=start_date, end=end_date, auto_adjust=True)

# Flatten MultiIndex columns
if isinstance(xsp.columns, pd.MultiIndex):
    xsp.columns = xsp.columns.get_level_values(0)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

# Align: keep only dates present in both
common_idx = xsp.index.intersection(vix.index)
xsp = xsp.loc[common_idx]
vix = vix.loc[common_idx]

print(f"  ^XSP rows: {len(xsp)}")
print(f"  ^VIX rows: {len(vix)}")

# ============================================================
# 2. BUILD DATAFRAME WITH ALL INDICATORS
# ============================================================
print("\n" + "=" * 100)
print("STEP 2: COMPUTING ALL INDICATORS")
print("=" * 100)

df = pd.DataFrame(index=xsp.index)
df['Open'] = xsp['Open']
df['High'] = xsp['High']
df['Low'] = xsp['Low']
df['Close'] = xsp['Close']
df['Volume'] = xsp['Volume']
df['VIX'] = vix['Close']

print(f"  Data range: {df.index.min().date()} to {df.index.max().date()}")
print(f"  Total trading days: {len(df)}")
print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
print(f"  VIX range: {df['VIX'].min():.2f} - {df['VIX'].max():.2f}")

# --- ADX, DI+/DI- (length=14) ---
adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
df['ADX'] = adx['ADX_14']
df['DI_plus'] = adx['DMP_14']
df['DI_minus'] = adx['DMN_14']
df['DI_diff'] = df['DI_plus'] - df['DI_minus']

# --- ATR (length=14) ---
df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

# --- SMA 50 ---
df['SMA50'] = ta.sma(df['Close'], length=50)

# --- Bollinger Bands (20,2) ---
bb = ta.bbands(df['Close'], length=20, std=2)
df['BB_upper'] = bb['BBU_20_2.0_2.0']
df['BB_middle'] = bb['BBM_20_2.0_2.0']
df['BB_lower'] = bb['BBL_20_2.0_2.0']
# BB%: where close sits within bands
denom = (df['BB_upper'] - df['BB_lower']).replace(0, np.nan)
df['BB_pct'] = ((df['Close'] - df['BB_lower']) / denom).clip(0, 1)

# nearBB: close within 2.5% of upper or lower band
df['nearBB'] = (
    ((df['Close'] >= df['BB_upper'] * 0.975) & (df['Close'] <= df['BB_upper'] * 1.025)) |
    ((df['Close'] >= df['BB_lower'] * 0.975) & (df['Close'] <= df['BB_lower'] * 1.025))
)

# --- ER (Efficiency Ratio, length=10) ---
df['ER'] = ta.er(df['Close'], length=10)

# --- VR (Volatility Ratio = TR / ATR) ---
df['TR'] = ta.true_range(df['High'], df['Low'], df['Close'])
df['VR'] = df['TR'] / df['ATR']
df['VR'] = df['VR'].clip(0, 5)

# --- RSI (length=14) ---
df['RSI'] = ta.rsi(df['Close'], length=14)

# --- VIX Percentile (252-day lookback) ---
def rolling_percentile(series, window=252):
    def _pct(x):
        if x.max() == x.min():
            return 50.0
        return (x[-1] - x.min()) / (x.max() - x.min()) * 100
    return series.rolling(window).apply(_pct, raw=True)

df['vix_pct'] = rolling_percentile(df['VIX'], window=252)

# ============================================================
# 3. COMPUTE CALL/PUT SCORES
# ============================================================
print("\n" + "=" * 100)
print("STEP 3: COMPUTING CALL AND PUT SCORES")
print("=" * 100)

# Each condition = 25 points. Score = sum(conditions met) * 25.
# CALL score: ER>=0.60, VR>=1.2, RSI>=60, ADX>=25
df['cs_call'] = (
    (df['ER'] >= 0.60).astype(int) +
    (df['VR'] >= 1.2).astype(int) +
    (df['RSI'] >= 60).astype(int) +
    (df['ADX'] >= 25).astype(int)
) * 25

# PUT score: abs(ER)>=0.60, 2-VR>=1.2 (VR<=0.8), 100-RSI>=60 (RSI<=40), ADX>=25
df['cs_put'] = (
    (df['ER'].abs() >= 0.60).astype(int) +
    ((2 - df['VR']) >= 1.2).astype(int) +
    ((100 - df['RSI']) >= 60).astype(int) +
    (df['ADX'] >= 25).astype(int)
) * 25

print(f"  CALL score stats: mean={df['cs_call'].mean():.1f}, min={df['cs_call'].min()}, max={df['cs_call'].max()}")
print(f"  PUT  score stats: mean={df['cs_put'].mean():.1f}, min={df['cs_put'].min()}, max={df['cs_put'].max()}")
print(f"  ER stats: mean={df['ER'].mean():.3f}, std={df['ER'].std():.3f}")
print(f"  VR stats: mean={df['VR'].mean():.3f}, std={df['VR'].std():.3f}")

# Condition hit rates
total_valid = df['ADX'].notna().sum()
print(f"\n  Condition hit rates (out of {total_valid} valid days):")
print(f"    ER>=0.60:       {(df['ER']>=0.60).sum()} ({(df['ER']>=0.60).mean()*100:.1f}%)")
print(f"    abs(ER)>=0.60:  {(df['ER'].abs()>=0.60).sum()} ({(df['ER'].abs()>=0.60).mean()*100:.1f}%)")
print(f"    VR>=1.2:        {(df['VR']>=1.2).sum()} ({(df['VR']>=1.2).mean()*100:.1f}%)")
print(f"    VR<=0.8:        {(df['VR']<=0.8).sum()} ({(df['VR']<=0.8).mean()*100:.1f}%)")
print(f"    RSI>=60:        {(df['RSI']>=60).sum()} ({(df['RSI']>=60).mean()*100:.1f}%)")
print(f"    RSI<=40:        {(df['RSI']<=40).sum()} ({(df['RSI']<=40).mean()*100:.1f}%)")
print(f"    ADX>=25:        {(df['ADX']>=25).sum()} ({(df['ADX']>=25).mean()*100:.1f}%)")

# ============================================================
# 4. CONFIGURATION FILTERS FOR PUT ENTRIES
# ============================================================
print("\n" + "=" * 100)
print("STEP 4: APPLYING PUT ENTRY CONFIGURATIONS")
print("=" * 100)

# Config A (strict)
config_a = (
    (~df['nearBB']) & (df['ADX'] >= 25) & (df['DI_diff'] < -0.02) &
    (df['VIX'] > 28) & (df['Close'] < df['SMA50']) & (df['vix_pct'] > 80)
)

# Config B (moderate)
config_b = (
    (~df['nearBB']) & (df['ADX'] >= 20) & (df['DI_diff'] < 0) &
    (df['VIX'] > 20) & (df['Close'] < df['SMA50'])
)

# Config C (loose)
config_c = (
    (~df['nearBB']) & (df['ADX'] >= 20) & (df['DI_diff'] < 0) &
    (df['VIX'] > 15) & (df['Close'] < df['SMA50'])
)

# Config D (putscore only)
config_d = (
    (~df['nearBB']) & (df['cs_put'] >= 35) & (df['DI_diff'] < 0) & (df['VIX'] > 20)
)

filters = {
    'Config A (strict)': config_a,
    'Config B (moderate)': config_b,
    'Config C (loose)': config_c,
    'Config D (putscore only)': config_d,
}

for name, cond in filters.items():
    count = cond.sum()
    pct = count / total_valid * 100 if total_valid > 0 else 0
    print(f"  {name}: {count} candidates ({pct:.2f}% of valid days)")

# ============================================================
# 5. FORWARD RETURNS
# ============================================================
print("\n" + "=" * 100)
print("STEP 5: COMPUTING FORWARD RETURNS")
print("=" * 100)

df['fwd_5d'] = df['Close'].shift(-5) / df['Close'] - 1
df['fwd_10d'] = df['Close'].shift(-10) / df['Close'] - 1
df['fwd_20d'] = df['Close'].shift(-20) / df['Close'] - 1

# ============================================================
# 6. DETAILED PUT CANDIDATE ANALYSIS (Config D)
# ============================================================
print("\n" + "=" * 100)
print("STEP 6: DETAILED PUT CANDIDATE ANALYSIS (Config D)")
print("=" * 100)

candidates = df[config_d].copy()
print(f"\n  Total PUT candidates (Config D): {len(candidates)}")

if len(candidates) == 0:
    print("  WARNING: No PUT candidates found. Exiting.")
    sys.exit(0)

# --- PnL Simulation ---
def simulate_put_trade(entry_idx, df, hold_days=30, trail_pct=0.10, stop_pct=0.10, leverage=3):
    entry_date = df.index[entry_idx]
    entry_price = df['Close'].iloc[entry_idx]
    
    max_idx = min(entry_idx + hold_days + 30, len(df) - 1)
    
    lowest_price = entry_price
    exit_idx = None
    exit_price = None
    exit_reason = 'hold'
    
    for i in range(entry_idx + 1, max_idx):
        current_price = df['Close'].iloc[i]
        
        if current_price < lowest_price:
            lowest_price = current_price
        
        # For PUT: trail stop activates when price rises trail_pct above lowest point
        if current_price >= lowest_price * (1 + trail_pct):
            exit_idx = i
            exit_price = current_price
            exit_reason = 'trail'
            break
        
        # Fixed stop: price rises stop_pct above entry (adverse move)
        if current_price >= entry_price * (1 + stop_pct):
            exit_idx = i
            exit_price = current_price
            exit_reason = 'stop'
            break
        
        # Time-based exit
        days_held = i - entry_idx
        if days_held >= hold_days:
            exit_idx = i
            exit_price = current_price
            exit_reason = 'time'
            break
    
    if exit_idx is None:
        exit_idx = min(entry_idx + hold_days, len(df) - 1)
        exit_price = df['Close'].iloc[exit_idx]
        exit_reason = 'time_max'
    
    # PnL for PUT: (entry - exit) / entry * leverage * 100
    price_change_pct = (entry_price - exit_price) / entry_price
    pnl_pct = price_change_pct * leverage * 100
    
    return {
        'entry_date': entry_date,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'pnl_pct': pnl_pct,
    }

# Simulate for all Config D candidates
sim_results = []
for idx in range(len(df)):
    if not config_d.iloc[idx]:
        continue
    sim = simulate_put_trade(idx, df, hold_days=30, trail_pct=0.10, stop_pct=0.10, leverage=3)
    sim_results.append(sim)

sim_df = pd.DataFrame(sim_results)
print(f"\n  PUT Trade Simulation (3x leverage, T+30 hold, 10% trail, 10% stop):")
print(f"  Trades: {len(sim_df)}")
print(f"  Avg PnL: {sim_df['pnl_pct'].mean():.2f}%")
print(f"  Median PnL: {sim_df['pnl_pct'].median():.2f}%")
print(f"  Winners: {(sim_df['pnl_pct']>0).sum()} ({(sim_df['pnl_pct']>0).mean()*100:.1f}%)")
print(f"  Losers:  {(sim_df['pnl_pct']<=0).sum()} ({(sim_df['pnl_pct']<=0).mean()*100:.1f}%)")
print(f"  Avg Win: {sim_df.loc[sim_df['pnl_pct']>0,'pnl_pct'].mean():.2f}%")
print(f"  Avg Loss: {sim_df.loc[sim_df['pnl_pct']<=0,'pnl_pct'].mean():.2f}%")
print(f"  Max Win: {sim_df['pnl_pct'].max():.2f}%")
print(f"  Max Loss: {sim_df['pnl_pct'].min():.2f}%")
print(f"  Exit reasons: {sim_df['exit_reason'].value_counts().to_dict()}")

# --- Forward Returns for PUT Candidates ---
candidate_fwd = df.loc[candidates.index, ['Close', 'VIX', 'ADX', 'DI_diff', 'RSI', 'cs_put', 'cs_call',
                                           'nearBB', 'SMA50', 'fwd_5d', 'fwd_10d', 'fwd_20d']].copy()

print(f"\n  Forward Returns Analysis for PUT Candidates:")
print(f"  Mean 5-day forward return:  {candidate_fwd['fwd_5d'].mean()*100:.2f}%")
print(f"  Mean 10-day forward return: {candidate_fwd['fwd_10d'].mean()*100:.2f}%")
print(f"  Mean 20-day forward return: {candidate_fwd['fwd_20d'].mean()*100:.2f}%")
print(f"  Median 5-day forward return:  {candidate_fwd['fwd_5d'].median()*100:.2f}%")
print(f"  Median 10-day forward return: {candidate_fwd['fwd_10d'].median()*100:.2f}%")
print(f"  Median 20-day forward return: {candidate_fwd['fwd_20d'].median()*100:.2f}%")

for period in ['fwd_5d', 'fwd_10d', 'fwd_20d']:
    neg = (candidate_fwd[period] < 0).sum()
    tot = candidate_fwd[period].notna().sum()
    print(f"  % {period} < 0 (price went down): {neg}/{tot} = {neg/tot*100:.1f}%" if tot > 0 else f"  % {period}: N/A")

# ============================================================
# 7. YEARLY SUMMARY TABLE
# ============================================================
print("\n" + "=" * 100)
print("STEP 7: YEARLY SUMMARY TABLE")
print("=" * 100)

years = candidates.index.year
summary_rows = []

for year in sorted(set(years)):
    mask = years == year
    yr_count = mask.sum()
    if yr_count == 0:
        continue
    
    yr_fwd = candidate_fwd.loc[mask]
    yr_sim = sim_df.iloc[np.where(mask)[0]] if len(sim_df) >= yr_count else pd.DataFrame()
    
    if len(yr_sim) > 0:
        avg_pnl = yr_sim['pnl_pct'].mean()
        winners = (yr_sim['pnl_pct'] > 0).sum()
        losers = (yr_sim['pnl_pct'] <= 0).sum()
    else:
        avg_pnl = float('nan')
        winners = 0
        losers = 0
    
    summary_rows.append({
        'Year': year,
        'PUT_Count': yr_count,
        'Avg_PnL%': f"{avg_pnl:.2f}" if not np.isnan(avg_pnl) else 'N/A',
        'Winners': winners,
        'Losers': losers,
        'WinRate%': f"{winners/(winners+losers)*100:.1f}" if (winners+losers) > 0 else 'N/A',
        'Avg5dFwd%': f"{yr_fwd['fwd_5d'].mean()*100:.2f}",
        'Avg10dFwd%': f"{yr_fwd['fwd_10d'].mean()*100:.2f}",
        'Avg20dFwd%': f"{yr_fwd['fwd_20d'].mean()*100:.2f}",
    })

summary_df = pd.DataFrame(summary_rows)
print(f"\n{summary_df.to_string(index=False)}")

# ============================================================
# 8. SPECIFIC TRADES IN 2020 AND 2022
# ============================================================
print("\n" + "=" * 100)
print("STEP 8: SPECIFIC PUT TRADES IN 2020 AND 2022 BEAR MARKETS")
print("=" * 100)

for bear_year in [2020, 2022]:
    bear_candidates = candidates[candidates.index.year == bear_year]
    if len(bear_candidates) == 0:
        print(f"\n  {bear_year}: No PUT candidates found.")
        continue
    
    print(f"\n  === {bear_year} - {len(bear_candidates)} PUT candidates ===")
    for date, row in bear_candidates.iterrows():
        sim_match = sim_df[sim_df['entry_date'] == date]
        pnl_str = f"{sim_match['pnl_pct'].values[0]:.2f}%" if len(sim_match) > 0 else "N/A"
        exit_reason = sim_match['exit_reason'].values[0] if len(sim_match) > 0 else "N/A"
        
        fwd5 = row.get('fwd_5d', np.nan)
        fwd10 = row.get('fwd_10d', np.nan)
        fwd20 = row.get('fwd_20d', np.nan)
        
        nb = "NB" if row['nearBB'] else "nt"
        
        print(f"  {date.date()} | ${row['Close']:.2f} | VIX={row['VIX']:.1f} | "
              f"ADX={row['ADX']:.1f} | DI={row['DI_diff']:.3f} | RSI={row['RSI']:.1f} | "
              f"cs_put={int(row['cs_put'])} | cs_call={int(row['cs_call'])} | "
              f"BB={nb} | PnL={pnl_str} | Exit={exit_reason} | "
              f"Fwd5d={fwd5*100:.2f}% | Fwd10d={fwd10*100:.2f}% | Fwd20d={fwd20*100:.2f}%")

# ============================================================
# 9. DIFFERENT HOLD PERIODS
# ============================================================
print("\n" + "=" * 100)
print("STEP 9: COMPARING HOLD PERIODS (T+30 vs T+60 vs T+90)")
print("=" * 100)

for hold in [30, 60, 90]:
    results = []
    for idx in range(len(df)):
        if not config_d.iloc[idx]:
            continue
        sim = simulate_put_trade(idx, df, hold_days=hold, trail_pct=0.10, stop_pct=0.10, leverage=3)
        results.append(sim)
    
    if len(results) == 0:
        print(f"\n  T+{hold}: No trades")
        continue
    
    rdf = pd.DataFrame(results)
    winners = (rdf['pnl_pct'] > 0).sum()
    losers = (rdf['pnl_pct'] <= 0).sum()
    total = len(rdf)
    print(f"\n  T+{hold} hold period ({total} trades):")
    print(f"    Avg PnL: {rdf['pnl_pct'].mean():.2f}%")
    print(f"    Median PnL: {rdf['pnl_pct'].median():.2f}%")
    print(f"    Win Rate: {winners/total*100:.1f}% ({winners}/{total})")
    print(f"    Avg Win: {rdf.loc[rdf['pnl_pct']>0,'pnl_pct'].mean():.2f}%")
    print(f"    Avg Loss: {rdf.loc[rdf['pnl_pct']<=0,'pnl_pct'].mean():.2f}%")
    print(f"    Best: {rdf['pnl_pct'].max():.2f}% | Worst: {rdf['pnl_pct'].min():.2f}%")

# ============================================================
# 10. COMPLETE LISTING - ALL PUT CANDIDATES
# ============================================================
print("\n" + "=" * 100)
print("STEP 10: COMPLETE LISTING - EVERY PUT CANDIDATE (Config D)")
print("=" * 100)

print(f"\n  Total PUT candidates: {len(candidates)}")
print(f"  {'─'*160}")
print(f"  Date       | Price   | VIX  | ADX  | DI      | RSI  | cs_put | cs_call | BB |  PnL%   | Fwd5d   | Fwd10d  | Fwd20d")
print(f"  {'─'*160}")

for date, row in candidates.iterrows():
    sim_match = sim_df[sim_df['entry_date'] == date]
    
    fwd5 = row.get('fwd_5d', np.nan)
    fwd10 = row.get('fwd_10d', np.nan)
    fwd20 = row.get('fwd_20d', np.nan)
    
    nb = "NB" if row['nearBB'] else "nt"
    pnl = f"{sim_match['pnl_pct'].values[0]:.2f}" if len(sim_match) > 0 else "N/A"
    
    print(f"  {date.date()} | ${row['Close']:<6.2f} | {row['VIX']:<4.1f} | "
          f"{row['ADX']:<4.1f} | {row['DI_diff']:<+.3f} | {row['RSI']:<4.1f} | "
          f"  {int(row['cs_put']):<3d}  |   {int(row['cs_call']):<3d}   | {nb} | "
          f"{pnl:>7s}% | {fwd5*100:>6.2f}% | {fwd10*100:>6.2f}% | {fwd20*100:>6.2f}%")

# ============================================================
# 11. FINAL COMPREHENSIVE SUMMARY
# ============================================================
print("\n" + "=" * 100)
print("FINAL COMPREHENSIVE SUMMARY")
print("=" * 100)

# Compute additional stats for the final summary
fwd5_neg = (candidate_fwd['fwd_5d'] < 0).sum()
fwd5_tot = candidate_fwd['fwd_5d'].notna().sum()
fwd10_neg = (candidate_fwd['fwd_10d'] < 0).sum()
fwd10_tot = candidate_fwd['fwd_10d'].notna().sum()
fwd20_neg = (candidate_fwd['fwd_20d'] < 0).sum()
fwd20_tot = candidate_fwd['fwd_20d'].notna().sum()

print(f"""
PUT DIAGNOSTIC SUMMARY for ^XSP
================================
Period: {df.index.min().date()} to {df.index.max().date()}
Total trading days: {len(df)}

CANDIDATE COUNTS:
  Config A (strict):        {config_a.sum()}
  Config B (moderate):      {config_b.sum()}
  Config C (loose):         {config_c.sum()}
  Config D (putscore only): {config_d.sum()}

PUT CANDIDATE FORWARD RETURNS (Config D):
  Avg 5-day forward return:  {candidate_fwd['fwd_5d'].mean()*100:.2f}%
  Avg 10-day forward return: {candidate_fwd['fwd_10d'].mean()*100:.2f}%
  Avg 20-day forward return: {candidate_fwd['fwd_20d'].mean()*100:.2f}%
  
  Negative 5d return:  {fwd5_neg}/{fwd5_tot} = {fwd5_neg/fwd5_tot*100:.1f}%
  Negative 10d return: {fwd10_neg}/{fwd10_tot} = {fwd10_neg/fwd10_tot*100:.1f}%
  Negative 20d return: {fwd20_neg}/{fwd20_tot} = {fwd20_neg/fwd20_tot*100:.1f}%

TRADE SIMULATION (3x lev, T+30, 10%% trail, 10%% stop):
  Total trades: {len(sim_df)}
  Average PnL: {sim_df['pnl_pct'].mean():.2f}%
  Win rate: {(sim_df['pnl_pct']>0).mean()*100:.1f}%
  Avg win: {sim_df.loc[sim_df['pnl_pct']>0,'pnl_pct'].mean():.2f}%
  Avg loss: {sim_df.loc[sim_df['pnl_pct']<=0,'pnl_pct'].mean():.2f}%
  Max win: {sim_df['pnl_pct'].max():.2f}%
  Max loss: {sim_df['pnl_pct'].min():.2f}%
  Exit reasons: {sim_df['exit_reason'].value_counts().to_dict()}

KEY FINDING - Structural Issue:
  If the PUT scoring system worked, PUT candidate days would be followed by
  predominantly NEGATIVE forward returns. The %Negative figures above tell us
  whether the system actually identifies downside moves.
  
  A PUT candidate count of {config_d.sum()} across {len(df)} days means these
  conditions fire {config_d.sum()/total_valid*100:.1f}% of the time. If the
  % of negative forward returns is near 50%%, the system has ZERO predictive
  power and is essentially random.
""")
