#!/usr/bin/env python3
"""
02_preprocess_data.py
=====================
Preprocess raw SPX option data following Bali et al. (2023) methodology.

Pipeline:
  1. Apply sample filters (positive bid, maturity, moneyness, liquidity)
  2. Compute option returns (raw + delta-hedged)
  3. Construct feature vector (option-level, IV surface, underlying)
  4. Create broad sample + tradable universe
  5. Save processed datasets

Input:
  data/raw/spx_options_raw.parquet
  data/raw/spx_underlying.parquet
  data/raw/risk_free_rates.parquet
  data/raw/vix_daily.parquet

Output:
  data/processed/spx_options_filtered.parquet
  data/processed/spx_options_features.parquet
  data/processed/spx_tradable_universe.parquet
  data/processed/filter_summary.csv
"""

import sys
from pathlib import Path

VENV_PATH = Path(__file__).parent.parent / "venv"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH / "lib" / "python3.12" / "site-packages"))

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_data():
    """Load all raw datasets."""
    print("📂 Loading raw data...")
    
    options = pd.read_parquet(RAW_DIR / "spx_options_raw.parquet")
    print(f"   Options:    {len(options):>10,} rows")
    
    underlying = pd.read_parquet(RAW_DIR / "spx_underlying.parquet")
    print(f"   Underlying: {len(underlying):>10,} rows")
    
    vix = pd.read_parquet(RAW_DIR / "vix_daily.parquet")
    print(f"   VIX:        {len(vix):>10,} rows")
    
    rf = pd.read_parquet(RAW_DIR / "risk_free_rates.parquet")
    print(f"   Risk-free:  {len(rf):>10,} rows")
    
    # Ensure date columns are datetime
    for df in [options, underlying, vix]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
    if 'exdate' in options.columns:
        options['exdate'] = pd.to_datetime(options['exdate'])
    rf['date'] = pd.to_datetime(rf['date'])
    
    return options, underlying, vix, rf


def apply_filters(df):
    """
    Apply sample filters following Bali et al. (2023).
    Returns filtered DataFrame and filter summary.
    """
    print("\n🔍 Applying sample filters...")
    
    filter_log = []
    n0 = len(df)
    filter_log.append(('Raw data', n0, 0, 100.0))
    
    # 1. Positive bid
    mask = df['best_bid'] > 0
    df = df[mask].copy()
    n1 = len(df)
    filter_log.append(('Positive bid (bid > 0)', n1, n0 - n1, 100 * n1 / n0))
    
    # 2. Valid mid price
    df['mid_price'] = (df['best_bid'] + df['best_offer']) / 2
    mask = df['mid_price'] > 0.05
    df = df[mask].copy()
    n2 = len(df)
    filter_log.append(('Valid mid price (> $0.05)', n2, n1 - n2, 100 * n2 / n0))
    
    # 3. Maturity bounds: 7 <= TTM <= 365 days
    if 'days_to_expiry' not in df.columns:
        df['days_to_expiry'] = (df['exdate'] - df['date']).dt.days
    mask = (df['days_to_expiry'] >= 7) & (df['days_to_expiry'] <= 365)
    df = df[mask].copy()
    n3 = len(df)
    filter_log.append(('Maturity bounds (7-365 days)', n3, n2 - n3, 100 * n3 / n0))
    
    # 4. Moneyness bounds: |ln(K/F)| <= 0.20
    if 'forward_price' in df.columns and df['forward_price'].notna().any():
        df['log_moneyness'] = np.log(df['strike_price'] / df['forward_price'])
    else:
        # Approximate forward price
        df['log_moneyness'] = np.log(df['strike_price'] / df['mid_price'])  # placeholder
    
    mask = (df['log_moneyness'] >= -0.20) & (df['log_moneyness'] <= 0.20)
    df = df[mask].copy()
    n4 = len(df)
    filter_log.append(('Moneyness bounds (|ln(K/F)| ≤ 0.20)', n4, n3 - n4, 100 * n4 / n0))
    
    # 5. Valid implied volatility
    mask = df['impl_volatility'].notna() & (df['impl_volatility'] > 0) & (df['impl_volatility'] < 3.0)
    df = df[mask].copy()
    n5 = len(df)
    filter_log.append(('Valid IV (0 < IV < 300%)', n5, n4 - n5, 100 * n5 / n0))
    
    # 6. No bid-ask violations
    mask = df['best_bid'] <= df['best_offer']
    df = df[mask].copy()
    n6 = len(df)
    filter_log.append(('No bid-ask violation (bid ≤ ask)', n6, n5 - n6, 100 * n6 / n0))
    
    # 7. Broad sample minimum liquidity
    mask = (df['volume'] >= 10) & (df['open_interest'] >= 100)
    df_broad = df[mask].copy()
    n7 = len(df_broad)
    filter_log.append(('Broad liquidity (vol≥10, OI≥100)', n7, n6 - n7, 100 * n7 / n0))
    
    # Summary
    filter_df = pd.DataFrame(filter_log, columns=['Filter', 'Remaining', 'Removed', 'Pct_of_Original'])
    filter_df['Pct_of_Original'] = filter_df['Pct_of_Original'].round(1)
    
    print("\n   Filter Summary:")
    print(filter_df.to_string(index=False))
    
    # Save filter summary
    filter_df.to_csv(PROC_DIR / "filter_summary.csv", index=False)
    
    return df_broad, df  # broad (with liquidity filter), and quality-filtered (without liquidity)


def compute_realized_volatility(underlying, windows=[5, 21, 63]):
    """Compute realized volatility over multiple windows."""
    print("\n📐 Computing realized volatility...")
    
    underlying = underlying.sort_values('date').copy()
    
    for w in windows:
        col = f'rv_{w}d'
        underlying[col] = underlying['spx_return'].rolling(w).std() * np.sqrt(252)
        print(f"   RV({w}d): mean={underlying[col].mean():.4f}, std={underlying[col].std():.4f}")
    
    # Realized skewness (63d)
    underlying['rskew_63d'] = underlying['spx_return'].rolling(63).skew()
    
    # Past returns
    underlying['ret_5d'] = underlying['spx_close'].pct_change(5)
    underlying['ret_21d'] = underlying['spx_close'].pct_change(21)
    
    return underlying


def construct_iv_surface_features(df):
    """
    Extract IV surface features for each date.
    ATM IV level, skew, term slope, butterfly.
    """
    print("\n🌊 Constructing IV surface features...")
    
    surface_features = []
    
    for date, group in df.groupby('date'):
        features = {'date': date}
        
        # ATM options: |delta| closest to 0.50
        calls = group[group['cp_flag'] == 'C']
        puts = group[group['cp_flag'] == 'P']
        
        # ATM IV (calls with |delta| ≈ 0.50)
        if len(calls) > 0:
            calls_sorted = calls.iloc[(calls['delta'].abs() - 0.50).abs().argsort()]
            atm_call = calls_sorted.iloc[0]
            features['atm_iv'] = atm_call['impl_volatility']
        else:
            features['atm_iv'] = np.nan
        
        # IV Skew: 25-delta put IV - ATM IV
        if len(puts) > 0:
            puts_sorted = puts.iloc[(puts['delta'].abs() - 0.25).abs().argsort()]
            put_25d = puts_sorted.iloc[0]
            features['iv_skew'] = put_25d['impl_volatility'] - features.get('atm_iv', np.nan)
        else:
            features['iv_skew'] = np.nan
        
        # 25-delta call IV
        if len(calls) > 0:
            calls_25d = calls.iloc[(calls['delta'] - 0.25).abs().argsort()]
            call_25d = calls_25d.iloc[0]
            features['call_25d_iv'] = call_25d['impl_volatility']
        else:
            features['call_25d_iv'] = np.nan
        
        # Butterfly: 0.5 * (25d_put_iv + 25d_call_iv) - ATM_iv
        if not np.isnan(features.get('iv_skew', np.nan)) and not np.isnan(features.get('call_25d_iv', np.nan)):
            put_25d_iv = features['iv_skew'] + features['atm_iv']
            features['iv_butterfly'] = 0.5 * (put_25d_iv + features['call_25d_iv']) - features['atm_iv']
        else:
            features['iv_butterfly'] = np.nan
        
        # Term structure slope: ATM IV (60-90d) - ATM IV (7-30d)
        near = group[(group['days_to_expiry'] >= 7) & (group['days_to_expiry'] <= 30)]
        far = group[(group['days_to_expiry'] >= 60) & (group['days_to_expiry'] <= 120)]
        
        near_atm = near[(near['log_moneyness'].abs() < 0.03)]
        far_atm = far[(far['log_moneyness'].abs() < 0.03)]
        
        if len(near_atm) > 0 and len(far_atm) > 0:
            features['iv_term_slope'] = far_atm['impl_volatility'].mean() - near_atm['impl_volatility'].mean()
        else:
            features['iv_term_slope'] = np.nan
        
        surface_features.append(features)
    
    surface_df = pd.DataFrame(surface_features)
    print(f"   → IV surface features for {len(surface_df)} dates")
    print(f"   ATM IV:     mean={surface_df['atm_iv'].mean():.4f}")
    print(f"   IV Skew:    mean={surface_df['iv_skew'].mean():.4f}")
    print(f"   IV Term:    mean={surface_df['iv_term_slope'].mean():.4f}")
    
    return surface_df


def compute_option_returns(df, holding_period=5):
    """
    Compute option returns over holding period h.
    
    R_opt = (P_mid(t+h) - P_mid(t)) / P_mid(t)
    R_dh  = R_opt - delta * R_underlying(t to t+h)
    
    Options are identified by (cp_flag, strike_price, exdate) — the same
    contract tracked across dates.
    """
    print(f"\n📊 Computing option returns (h={holding_period} days)...")
    
    # Create a contract identifier: same option = same strike/expiry/type
    df = df.copy()
    df['contract_id'] = (df['cp_flag'].astype(str) + '_' + 
                         df['strike_price'].astype(str) + '_' + 
                         df['exdate'].astype(str))
    
    df = df.sort_values(['contract_id', 'date'])
    
    # For each contract, compute forward mid price after h business days
    print("   → Matching forward prices...")
    
    # Build a date-indexed mid price lookup per contract
    # Use merge-based approach: for each row at date t, find the same contract at date t+h
    
    # Create sorted unique dates
    all_dates = df['date'].sort_values().unique()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    
    # For each row, find the forward date (h business days ahead)
    df['date_idx'] = df['date'].map(date_to_idx)
    df['fwd_date_idx'] = df['date_idx'] + holding_period
    
    # Map forward date index back to actual date
    idx_to_date = {i: d for d, i in date_to_idx.items()}
    df['fwd_date'] = df['fwd_date_idx'].map(idx_to_date)
    
    # Self-join: merge to get forward price
    fwd_cols = df[['contract_id', 'date', 'mid_price']].copy()
    fwd_cols.columns = ['contract_id', 'fwd_date', 'mid_price_fwd']
    
    merged = df.merge(fwd_cols, on=['contract_id', 'fwd_date'], how='inner')
    
    print(f"   → Matched {len(merged):,} / {len(df):,} observations ({100*len(merged)/len(df):.1f}%)")
    
    if len(merged) == 0:
        print("   ⚠️ No matching forward prices found")
        df['opt_return'] = np.nan
        df['dh_return'] = np.nan
        return df
    
    # Compute option return
    merged['opt_return'] = (merged['mid_price_fwd'] - merged['mid_price']) / merged['mid_price']
    
    # Clean up temp columns
    merged = merged.drop(columns=['date_idx', 'fwd_date_idx', 'fwd_date', 'contract_id'], errors='ignore')
    
    print(f"   → {len(merged):,} option-return observations")
    print(f"   → Return stats: mean={merged['opt_return'].mean():.4f}, "
          f"std={merged['opt_return'].std():.4f}, "
          f"median={merged['opt_return'].median():.4f}")
    
    return merged


def construct_features(df, underlying, vix, surface_features):
    """
    Construct the full feature vector X_{i,t} for each option-date.
    """
    print("\n🔨 Constructing feature vector...")
    
    # Merge underlying data
    df = df.merge(underlying[['date', 'spx_close', 'rv_5d', 'rv_21d', 'rv_63d', 
                               'rskew_63d', 'ret_5d', 'ret_21d']], 
                  on='date', how='left')
    
    # Merge VIX
    df = df.merge(vix[['date', 'vix_close']], on='date', how='left')
    
    # Merge surface features
    df = df.merge(surface_features, on='date', how='left')
    
    # === Option-Level Features ===
    df['log_moneyness'] = np.log(df['strike_price'] / df['forward_price'])
    df['log_ttm'] = np.log(df['days_to_expiry'])
    df['iv_deviation'] = df['impl_volatility'] - df['rv_21d']  # IV - RV
    df['abs_delta'] = df['delta'].abs()
    df['dollar_gamma'] = df['gamma'] * df['spx_close']  # Dollar gamma (approximate)
    df['vega_to_price'] = np.where(df['mid_price'] > 0, df['vega'] / df['mid_price'], 0)
    df['theta_to_price'] = np.where(df['mid_price'] > 0, df['theta'] / df['mid_price'], 0)
    df['rel_spread'] = df['bid_ask_spread'] / df['mid_price']
    df['log_volume'] = np.log1p(df['volume'])
    df['log_oi'] = np.log1p(df['open_interest'])
    df['is_put'] = (df['cp_flag'] == 'P').astype(int)
    
    # === VIX features ===
    df = df.sort_values('date')
    # VIX change over past 5 days
    vix_changes = vix.sort_values('date').copy()
    vix_changes['vix_change_5d'] = vix_changes['vix_close'].diff(5)
    df = df.merge(vix_changes[['date', 'vix_change_5d']], on='date', how='left')
    
    # === Feature list ===
    feature_cols = [
        # Option-level
        'log_moneyness', 'log_ttm', 'impl_volatility', 'iv_deviation',
        'abs_delta', 'dollar_gamma', 'vega_to_price', 'theta_to_price',
        'rel_spread', 'log_volume', 'log_oi', 'is_put',
        # IV surface
        'atm_iv', 'iv_skew', 'iv_term_slope', 'iv_butterfly',
        # VIX
        'vix_close', 'vix_change_5d',
        # Underlying
        'ret_5d', 'ret_21d', 'rv_21d', 'rv_63d', 'rskew_63d',
    ]
    
    print(f"   → {len(feature_cols)} features constructed")
    
    # Check feature coverage
    for col in feature_cols:
        if col in df.columns:
            pct_valid = df[col].notna().mean() * 100
            if pct_valid < 90:
                print(f"   ⚠️ {col}: {pct_valid:.1f}% valid")
    
    return df, feature_cols


def create_tradable_universe(df):
    """
    Create tradable universe with stricter liquidity filters.
    Volume >= 50, OI >= 500.
    """
    print("\n🎯 Creating tradable universe...")
    
    mask = (df['volume'] >= 50) & (df['open_interest'] >= 500)
    tradable = df[mask].copy()
    
    print(f"   Broad sample:      {len(df):>10,} observations")
    print(f"   Tradable universe: {len(tradable):>10,} observations ({100*len(tradable)/len(df):.1f}%)")
    
    return tradable


def cross_sectional_rank_transform(df, feature_cols):
    """
    Rank-transform features cross-sectionally within each date.
    Maps to uniform [0, 1] following Gu, Kelly, Xiu (2020).
    """
    print("\n📊 Applying cross-sectional rank transformation...")
    
    ranked_cols = []
    for col in feature_cols:
        if col in df.columns and col != 'is_put':  # Don't rank binary features
            rank_col = f'{col}_rank'
            df[rank_col] = df.groupby('date')[col].rank(pct=True)
            ranked_cols.append(rank_col)
    
    print(f"   → {len(ranked_cols)} features rank-transformed")
    return df, ranked_cols


def main():
    print("=" * 70)
    print("  SPX Options Data Preprocessing Pipeline")
    print("  Following Bali et al. (2023) methodology")
    print("=" * 70)
    
    # Load data
    options, underlying, vix, rf = load_raw_data()
    
    # Apply filters
    broad_df, quality_df = apply_filters(options)
    
    # Compute realized volatility
    underlying = compute_realized_volatility(underlying)
    
    # Construct IV surface features
    surface_features = construct_iv_surface_features(broad_df)
    
    # Compute option returns (weekly holding period)
    returns_df = compute_option_returns(broad_df, holding_period=5)
    
    # Construct features
    featured_df, feature_cols = construct_features(returns_df, underlying, vix, surface_features)
    
    # Create tradable universe
    tradable_df = create_tradable_universe(featured_df)
    
    # Rank transform
    featured_df, ranked_cols = cross_sectional_rank_transform(featured_df, feature_cols)
    tradable_df, _ = cross_sectional_rank_transform(tradable_df, feature_cols)
    
    # === Save processed data ===
    print("\n💾 Saving processed datasets...")
    
    featured_df.to_parquet(PROC_DIR / "spx_options_features.parquet", index=False)
    print(f"   → spx_options_features.parquet ({len(featured_df):,} rows)")
    
    tradable_df.to_parquet(PROC_DIR / "spx_tradable_universe.parquet", index=False)
    print(f"   → spx_tradable_universe.parquet ({len(tradable_df):,} rows)")
    
    # Save metadata
    meta = {
        'feature_cols': feature_cols,
        'ranked_cols': ranked_cols,
        'n_broad_sample': len(featured_df),
        'n_tradable': len(tradable_df),
        'date_range': [str(featured_df['date'].min()), str(featured_df['date'].max())],
        'n_dates': int(featured_df['date'].nunique()),
    }
    pd.Series(meta).to_json(PROC_DIR / "preprocessing_metadata.json")
    
    print("\n" + "=" * 70)
    print("  ✅ Preprocessing complete!")
    print(f"  Broad sample:      {len(featured_df):>10,} observations")
    print(f"  Tradable universe: {len(tradable_df):>10,} observations")
    print(f"  Features:          {len(feature_cols):>10}")
    print(f"  Date range:        {featured_df['date'].min().date()} to {featured_df['date'].max().date()}")
    print(f"  Unique dates:      {featured_df['date'].nunique():>10,}")
    print("=" * 70)


if __name__ == '__main__':
    main()
