"""Streamlit Cloud 入口文件 — 转发到 v2 主程序。

旧的 v1 已被新版 v2 取代（见 school_advisor_v2.py）。
保留此文件名是为了不破坏 Streamlit Cloud 上已部署 app 的 main file path 配置。
"""
from pathlib import Path
import sys

# 确保能 import 同目录下的 school_advisor_v2
sys.path.insert(0, str(Path(__file__).resolve().parent))

from school_advisor_v2 import main  # noqa: E402

main()
