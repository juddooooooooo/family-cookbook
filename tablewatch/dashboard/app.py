"""TableWatch dashboard.

    python -m tablewatch dashboard
    # or: streamlit run tablewatch/dashboard/app.py -- --db data/tablewatch.db

Two pages: Live monitoring (the table board for one run, live or replayed) and
Analytics (service history across all runs). Reads only the SQLite database
written by the pipeline: no video, frames or faces ever reach the dashboard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run without installing

import streamlit as st

from tablewatch.dashboard import analytics, monitor
from tablewatch.dashboard.common import get_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/tablewatch.db")
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    st.set_page_config(page_title="TableWatch", page_icon="🍽️", layout="wide")
    args = parse_args()
    with st.sidebar:
        st.header("TableWatch")
        db = st.text_input("Database", args.db)
        st.caption("Privacy: the dashboard reads only events and metrics. No video frames or faces are stored.")
    if not Path(db).exists():
        st.info("No database yet. Process a video (`python -m tablewatch run --source data/sample.mp4`) "
                "or generate demo data (`python -m tablewatch simulate --days 7`).")
        return
    store = get_store(db, Path(db).stat().st_ino)
    nav = st.navigation(
        [
            st.Page(lambda: monitor.page(store), title="Live monitoring", icon="🍽️", url_path="live", default=True),
            st.Page(lambda: analytics.page(store), title="Analytics", icon="📈", url_path="analytics"),
        ]
    )
    nav.run()


main()
