#!/bin/bash
# 启动中考志愿填报推荐 v2
set -e
cd "$(dirname "$0")"
exec .venv/bin/python3 -m streamlit run tools/school_advisor_v2.py
