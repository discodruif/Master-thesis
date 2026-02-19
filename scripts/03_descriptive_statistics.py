#!/usr/bin/env python3
"""
03_descriptive_statistics.py
============================
Generate comprehensive descriptive statistics for the SPX options dataset.

Produces:
  1. Panel A: Full sample statistics (Table 1 in thesis)
  2. Panel B: Tradable universe statistics
  3. Panel C: By VIX regime (high/low)
  4. Panel D: By moneyness bucket
  5. Panel E: By maturity bucket
  6. Correlation matrix of features
  7. Time series of key variables

Output:
  data/descriptive_stats/table1_full_sample.csv
  data/descriptive_stats/table1_tradable.csv
  data/descriptive_stats/table_vix_regimes.csv
  data/descriptive_stats/table_moneyness.csv
  data/descriptive_stats/table_maturity.csv
  data/descriptive_stats/correlation_matrix.csv
  data/descriptive_stats/summary_report.txt
"""

import sys
from pathlib import Path

VENV_PATH = Path(__file__).parent.parent / "venv"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH / "lib" / "python3.12" / "site-packages"))

import numpy as np
import pandas as pd

PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
STATS_DIR = Path(__file__).parent.parent / "data" / "descriptive_stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)


def compute_descriptive_table(df, label="Full Sample"):
    """
    Compute descriptive statistics matching thesis proposal Table 4.4.
    Returns DataFrame with Mean, Std, P5, P25, P50, P75, P95, N.
    """
    variables = {
        'mid_price': 'Option mid price ($)',
        'impl_volatility': 'Implied volatility',
        'log_moneyness': 'Log-moneyness ln(K/F)',
        'days_to_expiry': 'Days to expiry',
        'delta': 'Delta',
        'abs_delta': '|Delta|',
        'gamma': 'Gamma',
        'vega': 'Vega',
        'theta': 'Theta',
        'bid_ask_spread': 'Bid-ask spread ($)',
        'rel_spread': 'Relative spread (%)',
        'volume': 'Volume (contracts)',
        'open_interest': 'Open interest',
        'log_volume': 'Log(1+volume)',
        'log_oi': 'Log(1+OI)',
        'iv_deviation': 'IV − RV(21d)',
        'vix_close': 'VIX level',
        'atm_iv': 'ATM IV',
        'iv_skew': 'IV skew (25Δ put − ATM)',
        'iv_term_slope': 'IV term slope',
        'ret_5d': 'SPX return (5d)',
        'ret_21d': 'SPX return (21d)',
        'rv_21d': 'Realized vol (21d)',
        'rv_63d': 'Realized vol (63d)',
    }
    
    # Add returns if available
    if 'opt_return' in df.columns:
        variables['opt_return'] = 'Option return (h-period)'
    if 'dh_return' in df.columns:
        variables['dh_return'] = 'Delta-hedged return (h-period)'
    
    rows = []
    for col, name in variables.items():
        if col not in df.columns:
            continue
        
        s = df[col].dropna()
        if len(s) == 0:
            continue
        
        # Scale percentages
        display_mult = 1
        if col in ['rel_spread']:
            display_mult = 100  # Show as percentage
        
        rows.append({
            'Variable': name,
            'Mean': s.mean() * display_mult,
            'Std': s.std() * display_mult,
            'P5': s.quantile(0.05) * display_mult,
            'P25': s.quantile(0.25) * display_mult,
            'Median': s.quantile(0.50) * display_mult,
            'P75': s.quantile(0.75) * display_mult,
            'P95': s.quantile(0.95) * display_mult,
            'N': int(len(s)),
        })
    
    stats_df = pd.DataFrame(rows)
    
    # Round for display
    for col in ['Mean', 'Std', 'P5', 'P25', 'Median', 'P75', 'P95']:
        stats_df[col] = stats_df[col].round(4)
    
    return stats_df


def compute_by_vix_regime(df):
    """Split by VIX regime and compute stats."""
    if 'vix_close' not in df.columns:
        return None
    
    median_vix = df['vix_close'].median()
    
    low_vix = df[df['vix_close'] <= median_vix]
    high_vix = df[df['vix_close'] > median_vix]
    
    stats_low = compute_descriptive_table(low_vix, "Low VIX")
    stats_low['Regime'] = f'Low VIX (≤ {median_vix:.1f})'
    
    stats_high = compute_descriptive_table(high_vix, "High VIX")
    stats_high['Regime'] = f'High VIX (> {median_vix:.1f})'
    
    return pd.concat([stats_low, stats_high], ignore_index=True)


def compute_by_moneyness(df):
    """Compute stats by moneyness bucket."""
    if 'log_moneyness' not in df.columns or 'cp_flag' not in df.columns:
        return None
    
    buckets = []
    
    # ATM: |m| < 0.03
    atm = df[df['log_moneyness'].abs() < 0.03]
    if len(atm) > 0:
        s = compute_descriptive_table(atm, "ATM")
        s['Bucket'] = 'ATM (|m| < 0.03)'
        s['N_obs'] = len(atm)
        buckets.append(s)
    
    # OTM Puts: m < -0.03 and CP=P
    otm_puts = df[(df['log_moneyness'] < -0.03) & (df['cp_flag'] == 'P')]
    if len(otm_puts) > 0:
        s = compute_descriptive_table(otm_puts, "OTM Puts")
        s['Bucket'] = 'OTM Puts (m < -0.03)'
        s['N_obs'] = len(otm_puts)
        buckets.append(s)
    
    # OTM Calls: m > 0.03 and CP=C
    otm_calls = df[(df['log_moneyness'] > 0.03) & (df['cp_flag'] == 'C')]
    if len(otm_calls) > 0:
        s = compute_descriptive_table(otm_calls, "OTM Calls")
        s['Bucket'] = 'OTM Calls (m > 0.03)'
        s['N_obs'] = len(otm_calls)
        buckets.append(s)
    
    # ITM Puts: m > 0.03 and CP=P
    itm_puts = df[(df['log_moneyness'] > 0.03) & (df['cp_flag'] == 'P')]
    if len(itm_puts) > 0:
        s = compute_descriptive_table(itm_puts, "ITM Puts")
        s['Bucket'] = 'ITM Puts'
        s['N_obs'] = len(itm_puts)
        buckets.append(s)
    
    # ITM Calls: m < -0.03 and CP=C
    itm_calls = df[(df['log_moneyness'] < -0.03) & (df['cp_flag'] == 'C')]
    if len(itm_calls) > 0:
        s = compute_descriptive_table(itm_calls, "ITM Calls")
        s['Bucket'] = 'ITM Calls'
        s['N_obs'] = len(itm_calls)
        buckets.append(s)
    
    if buckets:
        return pd.concat(buckets, ignore_index=True)
    return None


def compute_by_maturity(df):
    """Compute stats by maturity bucket."""
    if 'days_to_expiry' not in df.columns:
        return None
    
    buckets = []
    maturity_bins = [
        ('Near-term (7-30d)', 7, 30),
        ('Medium-term (31-90d)', 31, 90),
        ('Longer-term (91-365d)', 91, 365),
    ]
    
    for name, lo, hi in maturity_bins:
        subset = df[(df['days_to_expiry'] >= lo) & (df['days_to_expiry'] <= hi)]
        if len(subset) > 0:
            s = compute_descriptive_table(subset, name)
            s['Bucket'] = name
            s['N_obs'] = len(subset)
            buckets.append(s)
    
    if buckets:
        return pd.concat(buckets, ignore_index=True)
    return None


def compute_correlation_matrix(df, feature_cols):
    """Compute and save feature correlation matrix."""
    available = [c for c in feature_cols if c in df.columns]
    corr = df[available].corr()
    return corr


def generate_time_series_summary(df):
    """Generate monthly time-series summary of key variables."""
    df = df.copy()
    df['year_month'] = df['date'].dt.to_period('M')
    
    monthly = df.groupby('year_month').agg(
        n_observations=('mid_price', 'count'),
        n_unique_options=('optionid', 'nunique'),
        mean_mid_price=('mid_price', 'mean'),
        mean_iv=('impl_volatility', 'mean'),
        mean_spread_pct=('rel_spread', 'mean'),
        mean_volume=('volume', 'mean'),
        mean_oi=('open_interest', 'mean'),
        mean_vix=('vix_close', 'mean'),
    ).reset_index()
    
    monthly['year_month'] = monthly['year_month'].astype(str)
    
    return monthly


def generate_report(full_stats, tradable_stats, vix_stats, moneyness_stats, maturity_stats, df):
    """Generate a comprehensive text report."""
    lines = []
    lines.append("=" * 80)
    lines.append("  DESCRIPTIVE STATISTICS REPORT")
    lines.append("  SPX Options Dataset — Thesis: ML for Option Mispricing")
    lines.append("  Author: Maurits van Eck (ANR: 2062644)")
    lines.append("=" * 80)
    lines.append("")
    
    # Dataset overview
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"   Date range:         {df['date'].min().date()} to {df['date'].max().date()}")
    lines.append(f"   Trading days:       {df['date'].nunique():,}")
    lines.append(f"   Total observations: {len(df):,}")
    lines.append(f"   Unique options:     {df['optionid'].nunique():,}")
    lines.append(f"   Calls:              {(df['cp_flag']=='C').sum():,} ({100*(df['cp_flag']=='C').mean():.1f}%)")
    lines.append(f"   Puts:               {(df['cp_flag']=='P').sum():,} ({100*(df['cp_flag']=='P').mean():.1f}%)")
    lines.append(f"   Avg obs/day:        {len(df) / df['date'].nunique():,.0f}")
    lines.append("")
    
    # Key statistics
    lines.append("2. KEY VARIABLE STATISTICS (Full Sample)")
    lines.append("-" * 40)
    if full_stats is not None:
        for _, row in full_stats.iterrows():
            lines.append(f"   {row['Variable']:<35} Mean={row['Mean']:>10.4f}  Std={row['Std']:>10.4f}  "
                        f"Med={row['Median']:>10.4f}  N={row['N']:>10,}")
    lines.append("")
    
    # VIX regime comparison
    if vix_stats is not None:
        lines.append("3. VIX REGIME COMPARISON")
        lines.append("-" * 40)
        for regime in vix_stats['Regime'].unique():
            sub = vix_stats[vix_stats['Regime'] == regime]
            lines.append(f"\n   {regime}:")
            # Show key vars only
            for var in ['Implied volatility', 'Relative spread (%)', 'Volume (contracts)', 'IV − RV(21d)']:
                row = sub[sub['Variable'] == var]
                if len(row) > 0:
                    r = row.iloc[0]
                    lines.append(f"      {var:<35} Mean={r['Mean']:>10.4f}  Std={r['Std']:>10.4f}")
    lines.append("")
    
    # Moneyness buckets
    if moneyness_stats is not None:
        lines.append("4. MONEYNESS BUCKET COMPARISON")
        lines.append("-" * 40)
        for bucket in moneyness_stats['Bucket'].unique():
            sub = moneyness_stats[moneyness_stats['Bucket'] == bucket]
            n_obs = sub['N_obs'].iloc[0] if 'N_obs' in sub.columns else 'N/A'
            lines.append(f"\n   {bucket} (N={n_obs:,}):")
            for var in ['Implied volatility', 'Relative spread (%)', 'Volume (contracts)']:
                row = sub[sub['Variable'] == var]
                if len(row) > 0:
                    r = row.iloc[0]
                    lines.append(f"      {var:<35} Mean={r['Mean']:>10.4f}  Std={r['Std']:>10.4f}")
    lines.append("")
    
    # Maturity buckets
    if maturity_stats is not None:
        lines.append("5. MATURITY BUCKET COMPARISON")
        lines.append("-" * 40)
        for bucket in maturity_stats['Bucket'].unique():
            sub = maturity_stats[maturity_stats['Bucket'] == bucket]
            n_obs = sub['N_obs'].iloc[0] if 'N_obs' in sub.columns else 'N/A'
            lines.append(f"\n   {bucket} (N={n_obs:,}):")
            for var in ['Implied volatility', 'Relative spread (%)', 'Option mid price ($)']:
                row = sub[sub['Variable'] == var]
                if len(row) > 0:
                    r = row.iloc[0]
                    lines.append(f"      {var:<35} Mean={r['Mean']:>10.4f}  Std={r['Std']:>10.4f}")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("  Report generated by 03_descriptive_statistics.py")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  Descriptive Statistics Generator")
    print("=" * 70)
    
    # Load processed data
    print("\n📂 Loading processed data...")
    
    features_path = PROC_DIR / "spx_options_features.parquet"
    tradable_path = PROC_DIR / "spx_tradable_universe.parquet"
    
    if not features_path.exists():
        print(f"   ❌ {features_path} not found. Run 02_preprocess_data.py first.")
        sys.exit(1)
    
    df = pd.read_parquet(features_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"   Broad sample: {len(df):,} observations")
    
    tradable_df = None
    if tradable_path.exists():
        tradable_df = pd.read_parquet(tradable_path)
        tradable_df['date'] = pd.to_datetime(tradable_df['date'])
        print(f"   Tradable:     {len(tradable_df):,} observations")
    
    # === Panel A: Full sample ===
    print("\n📊 Panel A: Full sample statistics...")
    full_stats = compute_descriptive_table(df, "Full Sample")
    full_stats.to_csv(STATS_DIR / "table1_full_sample.csv", index=False)
    print(full_stats.to_string(index=False))
    
    # === Panel B: Tradable universe ===
    if tradable_df is not None:
        print("\n📊 Panel B: Tradable universe statistics...")
        tradable_stats = compute_descriptive_table(tradable_df, "Tradable Universe")
        tradable_stats.to_csv(STATS_DIR / "table1_tradable.csv", index=False)
    else:
        tradable_stats = None
    
    # === Panel C: VIX regimes ===
    print("\n📊 Panel C: VIX regime statistics...")
    vix_stats = compute_by_vix_regime(df)
    if vix_stats is not None:
        vix_stats.to_csv(STATS_DIR / "table_vix_regimes.csv", index=False)
    
    # === Panel D: Moneyness buckets ===
    print("\n📊 Panel D: Moneyness bucket statistics...")
    moneyness_stats = compute_by_moneyness(df)
    if moneyness_stats is not None:
        moneyness_stats.to_csv(STATS_DIR / "table_moneyness.csv", index=False)
    
    # === Panel E: Maturity buckets ===
    print("\n📊 Panel E: Maturity bucket statistics...")
    maturity_stats = compute_by_maturity(df)
    if maturity_stats is not None:
        maturity_stats.to_csv(STATS_DIR / "table_maturity.csv", index=False)
    
    # === Correlation matrix ===
    print("\n📊 Feature correlation matrix...")
    feature_cols = [
        'log_moneyness', 'log_ttm', 'impl_volatility', 'iv_deviation',
        'abs_delta', 'rel_spread', 'log_volume', 'log_oi',
        'vix_close', 'rv_21d', 'ret_5d', 'ret_21d',
        'atm_iv', 'iv_skew',
    ]
    corr = compute_correlation_matrix(df, feature_cols)
    corr.to_csv(STATS_DIR / "correlation_matrix.csv")
    print("   Top correlations (|r| > 0.5):")
    for i in range(len(corr)):
        for j in range(i+1, len(corr)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                print(f"      {corr.index[i]} × {corr.columns[j]}: {r:.3f}")
    
    # === Time series summary ===
    print("\n📊 Monthly time series summary...")
    ts_summary = generate_time_series_summary(df)
    ts_summary.to_csv(STATS_DIR / "monthly_time_series.csv", index=False)
    print(f"   → {len(ts_summary)} months")
    
    # === Generate report ===
    print("\n📝 Generating summary report...")
    report = generate_report(full_stats, tradable_stats, vix_stats, moneyness_stats, maturity_stats, df)
    report_path = STATS_DIR / "summary_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"   → Saved to {report_path}")
    print("\n" + report)
    
    print("\n" + "=" * 70)
    print("  ✅ Descriptive statistics complete!")
    print(f"  Files saved in: {STATS_DIR}")
    for f in sorted(STATS_DIR.iterdir()):
        print(f"    - {f.name}")
    print("=" * 70)


if __name__ == '__main__':
    main()
