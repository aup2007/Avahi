#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
LOG=eval/run.log
: > "$LOG"

echo "=== [$(date '+%H:%M:%S')] SMOKE: 2 claims, interleaved, 1 pass, no judge ===" >> "$LOG"
python3 eval/run_interleaved.py --limit 2 --runs 1 --no-judge --prefix smoke_ >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "=== SMOKE FAILED -- aborting before full run ===" >> "$LOG"
  exit 1
fi

echo "" >> "$LOG"
echo "=== [$(date '+%H:%M:%S')] FULL: all claims, interleaved, 2 passes + judge ===" >> "$LOG"
python3 eval/run_interleaved.py >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "=== FULL RUN FAILED -- partial rows saved in eval/arch*_results.json ===" >> "$LOG"
  exit 1
fi

echo "" >> "$LOG"
echo "=== [$(date '+%H:%M:%S')] COMPARE ===" >> "$LOG"
python3 eval/compare.py >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "=== COMPARE FAILED ===" >> "$LOG"
  exit 1
fi

echo "=== [$(date '+%H:%M:%S')] ALL DONE ===" >> "$LOG"
