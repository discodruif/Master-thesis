# WRDS Data Access Setup Guide

## For: Maurits van Eck (Tilburg University)

### Step 1: Get WRDS Account
1. Go to https://wrds-www.wharton.upenn.edu/
2. Click "Register" → select "Tilburg University" as your institution
3. Use your Tilburg email (ANR-based or @tilburguniversity.edu)
4. Complete Duo 2FA setup (see https://libguides.uvt.nl/financial-databases/access)

### Step 2: Configure Credentials

**Option A: .pgpass file (recommended)**
```bash
# Create the credentials file
echo "wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_WRDS_USERNAME:YOUR_WRDS_PASSWORD" > ~/.pgpass
chmod 600 ~/.pgpass
```

**Option B: Environment variable**
```bash
export WRDS_USERNAME=your_username
# You'll be prompted for password
```

### Step 3: Test Connection
```bash
cd master-thesis
source venv/bin/activate
python3 -c "import wrds; db = wrds.Connection(); print('Connected as:', db.username); db.close()"
```

### Step 4: Run Data Download
```bash
cd master-thesis
source venv/bin/activate
python3 scripts/01_download_wrds_data.py --start 2010-01-01 --end 2024-12-31
```

### Step 5: Run Preprocessing
```bash
python3 scripts/02_preprocess_data.py
python3 scripts/03_descriptive_statistics.py
```

### Data Tables Used
| Table | Description |
|---|---|
| `optionm.opprcd` | Option prices (daily) |
| `optionm.secnmd` | Security name mapping |
| `optionm.secprd` | Underlying security prices |
| `optionm.zerocd` | Zero-coupon yield curve |
| `cboe.cboe` or `cboe_exchange.cboe_vix` | VIX index |
| `ff.fivefactors_daily` | Fama-French factors |

### Expected Data Size
- SPX options 2010-2024: ~3-5 million rows (~500MB parquet)
- Download time: ~15-30 minutes per year
- Total: ~2-3 hours

### Troubleshooting
- **Connection refused**: Check if Tilburg VPN is needed (some WRDS access requires campus network)
- **Authentication error**: Verify .pgpass format (5 colon-separated fields, no spaces)
- **Table not found**: Run `db.list_tables(library='optionm')` to see available tables
- **Memory error**: The download script processes year-by-year to manage memory

### Alternative: WRDS Web Query
If Python access doesn't work, you can download data via the WRDS web interface:
1. Go to https://wrds-www.wharton.upenn.edu/pages/get-data/optionmetrics-ivy-db-us/
2. Select "Option Prices" → filter for SPX (secid 108105)
3. Download as CSV, save to `data/raw/`
4. The preprocessing script handles both Parquet and CSV input
