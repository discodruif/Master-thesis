#!/bin/bash
# Script to run NYT daily fetch and process data

# Navigate to the project directory
cd /home/claw/.openclaw/workspace/master-thesis

# Run the daily NYT fetch script
python3 scripts/fetch_nyt_articles.py

# Process articles (placeholder for data-analysis pipeline)
echo "NYT fetch script executed at $(date)" > logs/fetch_log.txt