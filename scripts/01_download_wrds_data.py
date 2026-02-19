#!/usr/bin/env python3
"""
01_download_wrds_data.py
========================
Download SPX option data from OptionMetrics IvyDB via WRDS.

Prerequisites:
  1. WRDS account with OptionMetrics access (Tilburg University provides this)
  2. ~/.pgpass file with: wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD
  3. chmod 600 ~/.pgpass

Usage:
  python3 01_download_wrds_data.py [--start 2010-01-01] [--end 2024-12-31]

Output:
  data/raw/spx_options_raw.parquet       - Raw option quotes
  data/raw/spx_underlying.parquet        - SPX index levels
  data/raw/risk_free_rates.parquet       - Treasury rates
  data/raw/vix_daily.parquet             - VIX index
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure venv is active
VENV_PATH = Path(__file__).parent.parent / "venv"
if VENV_PATH.exists():
    sys.path.insert(0, str(VENV_PATH / "lib" / "python3.12" / "site-packages"))

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def connect_wrds():
    """Connect to WRDS PostgreSQL database."""
    import wrds
    try:
        db = wrds.Connection(wrds_username=os.environ.get('WRDS_USERNAME'))
        print(f"✅ Connected to WRDS as: {db.username}")
        return db
    except Exception as e:
        print(f"❌ WRDS connection failed: {e}")
        print("\nTo set up credentials:")
        print("  1. Create ~/.pgpass with: wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_USERNAME:YOUR_PASSWORD")
        print("  2. chmod 600 ~/.pgpass")
        print("  3. Or set WRDS_USERNAME env var and enter password when prompted")
        sys.exit(1)


def download_spx_options(db, start_date, end_date):
    """
    Download SPX option data from OptionMetrics IvyDB.
    
    Tables used:
      - optionm.opprcd: Option price data (daily)
      - optionm.secnmd: Security name/identifier mapping
    
    We use secid=108105 for SPX (S&P 500 Index).
    """
    print(f"\n📊 Downloading SPX options: {start_date} to {end_date}")
    
    # SPX secid in OptionMetrics
    # First verify the secid
    secid_query = """
    SELECT secid, effect_date, ticker, issuer
    FROM optionm.secnmd
    WHERE ticker = 'SPX'
    ORDER BY effect_date DESC
    LIMIT 5
    """
    print("  → Looking up SPX security ID...")
    secid_df = db.raw_sql(secid_query)
    print(f"  → Found SPX entries: {secid_df.to_string()}")
    
    spx_secid = secid_df['secid'].iloc[0]
    print(f"  → Using secid = {spx_secid}")
    
    # Download option price data in yearly chunks to manage memory
    all_chunks = []
    years = range(int(start_date[:4]), int(end_date[:4]) + 1)
    
    for year in years:
        yr_start = f"{year}-01-01"
        yr_end = f"{year}-12-31"
        
        # Clip to requested range
        if yr_start < start_date:
            yr_start = start_date
        if yr_end > end_date:
            yr_end = end_date
        
        query = f"""
        SELECT 
            o.secid,
            o.date,
            o.exdate,
            o.cp_flag,
            o.strike_price / 1000.0 AS strike_price,
            o.best_bid,
            o.best_offer,
            o.volume,
            o.open_interest,
            o.impl_volatility,
            o.delta,
            o.gamma,
            o.vega,
            o.theta,
            o.contract_size,
            o.forward_price,
            o.optionid,
            o.index_flag,
            o.exercise_style,
            (o.best_bid + o.best_offer) / 2.0 AS mid_price,
            (o.best_offer - o.best_bid) AS bid_ask_spread,
            o.exdate - o.date AS days_to_expiry
        FROM optionm.opprcd o
        WHERE o.secid = {spx_secid}
          AND o.date BETWEEN '{yr_start}' AND '{yr_end}'
          AND o.cp_flag IN ('C', 'P')
        ORDER BY o.date, o.exdate, o.strike_price, o.cp_flag
        """
        
        print(f"  → Downloading {year}...", end=" ", flush=True)
        t0 = time.time()
        chunk = db.raw_sql(query)
        elapsed = time.time() - t0
        print(f"  {len(chunk):,} rows ({elapsed:.1f}s)")
        all_chunks.append(chunk)
        
        # Rate limit - be nice to WRDS
        time.sleep(1)
    
    df = pd.concat(all_chunks, ignore_index=True)
    print(f"\n  → Total: {len(df):,} option-date observations")
    
    # Save
    outpath = DATA_DIR / "spx_options_raw.parquet"
    df.to_parquet(outpath, index=False, engine='pyarrow')
    print(f"  → Saved to {outpath} ({outpath.stat().st_size / 1e6:.1f} MB)")
    
    return df


def download_spx_underlying(db, start_date, end_date):
    """Download SPX index levels from OptionMetrics security price table."""
    print(f"\n📈 Downloading SPX underlying prices...")
    
    query = f"""
    SELECT date, close AS spx_close, return AS spx_return
    FROM optionm.secprd
    WHERE secid = (SELECT secid FROM optionm.secnmd WHERE ticker = 'SPX' LIMIT 1)
      AND date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
    """
    
    df = db.raw_sql(query)
    print(f"  → {len(df):,} daily observations")
    
    outpath = DATA_DIR / "spx_underlying.parquet"
    df.to_parquet(outpath, index=False, engine='pyarrow')
    print(f"  → Saved to {outpath}")
    
    return df


def download_risk_free_rates(db, start_date, end_date):
    """Download zero-coupon yield curve from OptionMetrics."""
    print(f"\n💰 Downloading risk-free rates...")
    
    query = f"""
    SELECT date, days, rate
    FROM optionm.zerocd
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date, days
    """
    
    df = db.raw_sql(query)
    print(f"  → {len(df):,} rate observations")
    
    outpath = DATA_DIR / "risk_free_rates.parquet"
    df.to_parquet(outpath, index=False, engine='pyarrow')
    print(f"  → Saved to {outpath}")
    
    return df


def download_vix(db, start_date, end_date):
    """Download VIX index data."""
    print(f"\n😨 Downloading VIX data...")
    
    # Try CBOE VIX from WRDS
    query = f"""
    SELECT date, close AS vix_close
    FROM cboe.cboe
    WHERE ticker = 'VIX'
      AND date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
    """
    
    try:
        df = db.raw_sql(query)
        print(f"  → {len(df):,} VIX observations from CBOE table")
    except Exception:
        # Fallback: compute from OptionMetrics 30-day IV
        print("  → CBOE table not available, trying alternative...")
        query = f"""
        SELECT caldt AS date, vix AS vix_close
        FROM cboe_exchange.cboe_vix
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY caldt
        """
        try:
            df = db.raw_sql(query)
            print(f"  → {len(df):,} VIX observations from cboe_exchange table")
        except Exception:
            print("  → VIX will be downloaded separately (Fred/CBOE)")
            df = pd.DataFrame(columns=['date', 'vix_close'])
    
    outpath = DATA_DIR / "vix_daily.parquet"
    df.to_parquet(outpath, index=False, engine='pyarrow')
    print(f"  → Saved to {outpath}")
    
    return df


def download_fama_french(db, start_date, end_date):
    """Download Fama-French factors from WRDS."""
    print(f"\n📉 Downloading Fama-French factors...")
    
    query = f"""
    SELECT date, mktrf, smb, hml, rf, umd
    FROM ff.fivefactors_daily
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date
    """
    
    try:
        df = db.raw_sql(query)
        print(f"  → {len(df):,} factor observations")
    except Exception:
        # Try alternative table
        query = f"""
        SELECT dateff AS date, mktrf, smb, hml, rf
        FROM ff.factors_daily
        WHERE dateff BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY dateff
        """
        try:
            df = db.raw_sql(query)
            # Try to get momentum separately
            mom_query = f"""
            SELECT dateff AS date, umd
            FROM ff.factors_daily
            WHERE dateff BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY dateff
            """
            try:
                mom_df = db.raw_sql(mom_query)
                df = df.merge(mom_df, on='date', how='left')
            except Exception:
                df['umd'] = np.nan
            print(f"  → {len(df):,} factor observations (alternative table)")
        except Exception:
            print("  → FF factors will be downloaded from Ken French website")
            df = pd.DataFrame(columns=['date', 'mktrf', 'smb', 'hml', 'rf', 'umd'])
    
    outpath = DATA_DIR / "fama_french_factors.parquet"
    df.to_parquet(outpath, index=False, engine='pyarrow')
    print(f"  → Saved to {outpath}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Download SPX option data from WRDS')
    parser.add_argument('--start', default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-31', help='End date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  WRDS Data Download Pipeline for SPX Options")
    print(f"  Sample period: {args.start} to {args.end}")
    print("=" * 70)
    
    db = connect_wrds()
    
    try:
        # List available OptionMetrics tables
        print("\n📋 Available OptionMetrics tables:")
        tables = db.list_tables(library='optionm')
        for t in tables[:20]:
            print(f"    - {t}")
        
        # Download all datasets
        opt_df = download_spx_options(db, args.start, args.end)
        und_df = download_spx_underlying(db, args.start, args.end)
        rf_df = download_risk_free_rates(db, args.start, args.end)
        vix_df = download_vix(db, args.start, args.end)
        ff_df = download_fama_french(db, args.start, args.end)
        
        print("\n" + "=" * 70)
        print("  ✅ Download complete!")
        print(f"  Options:     {len(opt_df):>10,} rows")
        print(f"  Underlying:  {len(und_df):>10,} rows")
        print(f"  Risk-free:   {len(rf_df):>10,} rows")
        print(f"  VIX:         {len(vix_df):>10,} rows")
        print(f"  FF factors:  {len(ff_df):>10,} rows")
        print("=" * 70)
        
    finally:
        db.close()
        print("\n🔌 WRDS connection closed.")


if __name__ == '__main__':
    main()
