#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --name controlled_validation_30 --repetitions 30 --seed 42 --semantic-backend tfidf
printf '\nFinished. Open experiments/controlled_validation_30/report/READ_RESULTS_FIRST.md\n'
