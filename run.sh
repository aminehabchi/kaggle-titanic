#!/usr/bin/env bash
# Run the training pipeline: cleaning -> feature engineering -> training -> evaluation
set -e
cd "$(dirname "$0")"
./titanic/bin/python scripts/main.py
