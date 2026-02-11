#!/bin/bash
# School Selection Advisor — Launch the app
# Run: bash run.sh

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "First time? Run setup first: bash setup.sh"
    exit 1
fi

source .venv/bin/activate
echo "Opening School Selection Advisor in your browser..."
streamlit run tools/school_advisor.py
