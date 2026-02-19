#!/usr/bin/env python3
"""
01b_generate_synthetic_data.py
==============================
Generate realistic synthetic SPX option data for pipeline development/testing.
Vectorized implementation for speed.

Usage:
  python3 01b_generate_synthetic_data.py [--n-dates 3750] [--seed 42]
"""

import argparse
import sys
import time
from pathlib import Path

VENV_PATH = Path(__file__).parent.parent / "venv"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH / "lib" / "python3.12" / "site-packages"))

import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_spx_path(n_dates, start_level=1100, annual_return=0.08, annual_vol=0.16, seed=42):
    """Generate realistic SPX price path with stochastic volatility."""
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    dates = pd.bdate_range('2010-01-04', periods=n_dates, freq='B')
    
    vol = np.zeros(n_dates)
    vol[0] = annual_vol
    prices = np.zeros(n_dates)
    prices[0] = start_level
    returns = np.zeros(n_dates)
    
    kappa, theta, xi, rho = 5.0, annual_vol, 0.3, -0.7
    
    for i in range(1, n_dates):
        dW1 = rng.normal(0, np.sqrt(dt))
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * rng.normal(0, np.sqrt(dt))
        vol[i] = max(0.05, vol[i-1] + kappa * (theta - vol[i-1]) * dt + xi * vol[i-1] * dW2)
        returns[i] = (annual_return - 0.5 * vol[i]**2) * dt + vol[i] * dW1
        prices[i] = prices[i-1] * np.exp(returns[i])
    
    return dates, prices, returns, vol


def vectorized_bs_price(S, K, T, r, sigma, cp_is_call):
    """Vectorized Black-Scholes pricing."""
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    price = np.where(cp_is_call, call_price, put_price)
    return np.maximum(price, 0.01)


def vectorized_bs_greeks(S, K, T, r, sigma, cp_is_call):
    """Vectorized Black-Scholes Greeks."""
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    call_delta = norm.cdf(d1)
    put_delta = call_delta - 1
    delta = np.where(cp_is_call, call_delta, put_delta)
    
    call_theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    put_theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    theta = np.where(cp_is_call, call_theta, put_theta)
    
    return delta, gamma, vega, theta


def generate_synthetic_options(n_dates=3750, seed=42):
    """Generate full synthetic SPX options dataset - vectorized."""
    rng = np.random.default_rng(seed)
    
    print("🔧 Generating synthetic SPX option data (vectorized)...")
    print(f"   Target: ~{n_dates} trading days (≈{n_dates/252:.0f} years)")
    t0 = time.time()
    
    dates, spx_prices, spx_returns, spot_vols = generate_spx_path(n_dates, seed=seed)
    print(f"   SPX range: {spx_prices.min():.0f} - {spx_prices.max():.0f}")
    
    vix = spot_vols * 100 * 1.15 + rng.normal(0, 1.5, n_dates)
    vix = np.clip(vix, 9, 85)
    
    # Risk-free rate path
    rf_rate = np.concatenate([
        np.linspace(0.003, 0.001, n_dates // 4),
        np.linspace(0.001, 0.025, n_dates // 4),
        np.linspace(0.025, 0.005, n_dates // 4),
        np.linspace(0.005, 0.045, n_dates - 3*(n_dates//4)),
    ]) + rng.normal(0, 0.001, n_dates)
    rf_rate = np.clip(rf_rate, 0.0001, 0.06)
    
    # Pre-generate all option rows using vectorized approach
    # For each date, generate ~200-400 option records (calls + puts across strikes/expiries)
    all_records = []
    option_id = 100000
    
    # Standard TTM grid
    ttm_grid = np.array([7, 14, 21, 30, 45, 60, 90, 120, 180, 270, 365])
    # Standard moneyness grid
    m_grid = np.linspace(-0.18, 0.18, 15)
    
    batch_size = 250  # Process in batches for progress reporting
    
    for batch_start in range(0, n_dates, batch_size):
        batch_end = min(batch_start + batch_size, n_dates)
        
        for d_idx in range(batch_start, batch_end):
            S = spx_prices[d_idx]
            r = rf_rate[d_idx]
            base_iv = spot_vols[d_idx] * 1.1
            
            # Select subset of TTMs (not all available every day)
            n_exp = min(len(ttm_grid), 6 + d_idx * 4 // n_dates)
            ttm_sel = rng.choice(ttm_grid, size=n_exp, replace=False)
            
            for ttm in ttm_sel:
                T = ttm / 365.0
                F = S * np.exp(r * T)
                
                # Generate strikes
                strikes = F * np.exp(m_grid)
                strikes = np.round(strikes / 5) * 5
                strikes = np.unique(strikes)
                
                n_strikes = len(strikes)
                if n_strikes == 0:
                    continue
                
                # Vectorize over strikes × 2 (call+put)
                n_opts = n_strikes * 2
                K_arr = np.repeat(strikes, 2)
                cp_arr = np.tile([True, False], n_strikes)  # True=Call, False=Put
                
                m_arr = np.log(K_arr / F)
                
                # IV with smile
                skew = 0.04 + 0.02 * (vix[d_idx] / 20 - 1)
                curv = 0.015 + 0.01 * (vix[d_idx] / 20 - 1)
                iv_arr = base_iv - skew * m_arr + curv * m_arr**2
                iv_arr += rng.normal(0, 0.003, n_opts)
                # Put skew enhancement
                put_otm = (~cp_arr) & (m_arr < 0)
                iv_arr[put_otm] *= 1.02
                iv_arr = np.clip(iv_arr, 0.05, 1.5)
                
                S_arr = np.full(n_opts, S)
                T_arr = np.full(n_opts, T)
                r_arr = np.full(n_opts, r)
                
                prices = vectorized_bs_price(S_arr, K_arr, T_arr, r_arr, iv_arr, cp_arr)
                delta, gamma, vega, theta = vectorized_bs_greeks(S_arr, K_arr, T_arr, r_arr, iv_arr, cp_arr)
                
                # Filter out near-zero prices
                valid = prices >= 0.05
                if not valid.any():
                    continue
                
                prices = prices[valid]
                K_arr = K_arr[valid]
                cp_arr = cp_arr[valid]
                m_arr = m_arr[valid]
                iv_arr = iv_arr[valid]
                delta = delta[valid]
                gamma = gamma[valid]
                vega = vega[valid]
                theta = theta[valid]
                n_valid = valid.sum()
                
                # Spreads
                atm_spread_pct = 0.02 + 0.01 * (vix[d_idx] / 20)
                otm_penalty = 0.05 * np.abs(m_arr) / 0.20
                spread_pct = np.clip(atm_spread_pct + otm_penalty, 0.005, 0.50)
                bid_ask = np.clip(prices * spread_pct, 0.05, prices * 0.40)
                best_bid = np.maximum(0.01, prices - bid_ask / 2)
                best_offer = prices + bid_ask / 2
                
                # Volume and OI
                base_vol = 200 * np.exp(-5 * m_arr**2) * (1 + 2 * d_idx / n_dates)
                volume = np.maximum(0, (base_vol * rng.lognormal(0, 1.5, n_valid)).astype(int))
                oi = np.maximum(0, (volume * rng.lognormal(2, 1, n_valid)).astype(int))
                
                option_ids = np.arange(option_id, option_id + n_valid)
                option_id += n_valid
                
                for i in range(n_valid):
                    all_records.append((
                        108105,                         # secid
                        dates[d_idx],                   # date
                        dates[d_idx] + pd.Timedelta(days=int(ttm)),  # exdate
                        'C' if cp_arr[i] else 'P',     # cp_flag
                        round(K_arr[i], 2),             # strike_price
                        round(best_bid[i], 2),          # best_bid
                        round(best_offer[i], 2),        # best_offer
                        int(volume[i]),                  # volume
                        int(oi[i]),                      # open_interest
                        round(iv_arr[i], 6),            # impl_volatility
                        round(delta[i], 6),             # delta
                        round(gamma[i], 8),             # gamma
                        round(vega[i], 4),              # vega
                        round(theta[i], 4),             # theta
                        100,                            # contract_size
                        round(F, 2),                    # forward_price
                        option_ids[i],                  # optionid
                        1,                              # index_flag
                        'E',                            # exercise_style
                        round(prices[i], 2),            # mid_price
                        round(best_offer[i] - best_bid[i], 2),  # bid_ask_spread
                        int(ttm),                       # days_to_expiry
                    ))
        
        elapsed = time.time() - t0
        pct = 100 * batch_end / n_dates
        print(f"   Progress: {batch_end}/{n_dates} ({pct:.0f}%) — {len(all_records):,} rows — {elapsed:.1f}s")
    
    columns = ['secid', 'date', 'exdate', 'cp_flag', 'strike_price', 'best_bid',
               'best_offer', 'volume', 'open_interest', 'impl_volatility', 'delta',
               'gamma', 'vega', 'theta', 'contract_size', 'forward_price', 'optionid',
               'index_flag', 'exercise_style', 'mid_price', 'bid_ask_spread', 'days_to_expiry']
    
    df = pd.DataFrame(all_records, columns=columns)
    elapsed = time.time() - t0
    print(f"\n   Total observations: {len(df):,} (generated in {elapsed:.1f}s)")
    
    # Save options data
    outpath = DATA_DIR / "spx_options_raw.parquet"
    df.to_parquet(outpath, index=False, engine='pyarrow')
    print(f"   → Saved to {outpath} ({outpath.stat().st_size / 1e6:.1f} MB)")
    
    # Save underlying
    und_df = pd.DataFrame({
        'date': dates,
        'spx_close': np.round(spx_prices, 2),
        'spx_return': np.round(spx_returns, 8),
    })
    und_path = DATA_DIR / "spx_underlying.parquet"
    und_df.to_parquet(und_path, index=False, engine='pyarrow')
    print(f"   → SPX underlying saved ({len(und_df)} rows)")
    
    # Save VIX
    vix_df = pd.DataFrame({
        'date': dates,
        'vix_close': np.round(vix, 2),
    })
    vix_path = DATA_DIR / "vix_daily.parquet"
    vix_df.to_parquet(vix_path, index=False, engine='pyarrow')
    print(f"   → VIX saved ({len(vix_df)} rows)")
    
    # Save risk-free rates (term structure)
    rf_rows = []
    tenors = [7, 14, 30, 60, 90, 180, 365]
    for d_idx in range(n_dates):
        for days in tenors:
            rf_rows.append({
                'date': dates[d_idx],
                'days': days,
                'rate': round(rf_rate[d_idx] + 0.001 * np.log(days/30), 6),
            })
    rf_df = pd.DataFrame(rf_rows)
    rf_path = DATA_DIR / "risk_free_rates.parquet"
    rf_df.to_parquet(rf_path, index=False, engine='pyarrow')
    print(f"   → Risk-free rates saved ({len(rf_df)} rows)")
    
    # Save Fama-French factors
    mktrf = spx_returns - rf_rate / 252
    ff_df = pd.DataFrame({
        'date': dates,
        'mktrf': np.round(mktrf, 6),
        'smb': np.round(rng.normal(0, 0.004, n_dates), 6),
        'hml': np.round(rng.normal(0, 0.004, n_dates), 6),
        'rf': np.round(rf_rate / 252, 6),
        'umd': np.round(rng.normal(0.0002, 0.005, n_dates), 6),
    })
    ff_path = DATA_DIR / "fama_french_factors.parquet"
    ff_df.to_parquet(ff_path, index=False, engine='pyarrow')
    print(f"   → Fama-French factors saved ({len(ff_df)} rows)")
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-dates', type=int, default=3750, help='Number of trading days')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 70)
    print("  Synthetic SPX Options Data Generator (Vectorized)")
    print("  For pipeline testing before WRDS access")
    print("=" * 70)
    
    df = generate_synthetic_options(n_dates=args.n_dates, seed=args.seed)
    
    print("\n" + "=" * 70)
    print("  ✅ Synthetic data generation complete!")
    print(f"  Total option-date observations: {len(df):,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Unique dates: {df['date'].nunique():,}")
    print("=" * 70)


if __name__ == '__main__':
    main()
