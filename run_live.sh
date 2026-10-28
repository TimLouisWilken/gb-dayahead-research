#!/usr/bin/env bash
# Daily live forecast for tomorrow's GB day-ahead prices (see src/live.py).
# Intended schedule: ~08:50 UK time on D-1 - after the NDF full-day issue (08:45 UK) and the
# 07:30 UTC cutoff of the other inputs, before the gate closures (EPEX 09:20, N2EX 09:50 UK).
# Production runs on GitHub Actions (repo gb-dayahead-pipeline, workflow daily.yml), triggered by
# cron-job.org at 08:50 UK. This script is for local tests and manual backfills; the former
# LaunchAgent (com.dayahead.live.plist) is retired.
set -uo pipefail
cd "$(dirname "$0")/src"
mkdir -p ../outputs/logs
../.venv/bin/python -W ignore live.py "$@" >> ../outputs/logs/live.log 2>&1
# publish the new forecast to the public GitHub-Pages site (never fails the forecast)
../.venv/bin/python -W ignore publish_site.py >> ../outputs/logs/live.log 2>&1 || true
