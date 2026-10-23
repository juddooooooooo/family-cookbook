"""Pieces shared by the dashboard pages: theme colours, the database handle, chart sizing."""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import streamlit as st

from tablewatch.palette import MEASURE_HEX, STATE_HEX
from tablewatch.state.table_state import TableState
from tablewatch.storage.db import EventStore

# Streamlit fits legend and axes inside a chart's height, so add room for them.
CHROME_AXIS = 70
CHROME_WITH_LEGEND = 100


@dataclass
class Theme:
    mode: str
    surface: str
    card: str
    border: str
    ink: str
    ink_2: str
    muted: str

    def state(self, state: TableState) -> str:
        return STATE_HEX[self.mode][state]

    def measure(self, name: str) -> str:
        return MEASURE_HEX[self.mode][name]


THEMES = {
    "light": Theme("light", "#ffffff", "#fcfcfb", "rgba(11,11,11,0.10)", "#0b0b0b", "#52514e", "#898781"),
    "dark": Theme("dark", "#0e1117", "#1a1a19", "rgba(255,255,255,0.10)", "#f0efec", "#c3c2b7", "#898781"),
}


def current_theme() -> Theme:
    try:
        mode = st.context.theme.type or "light"
    except AttributeError:
        mode = "light"
    return THEMES.get(mode, THEMES["light"])


@st.cache_resource
def get_store(path: str, file_id: int = 0) -> EventStore:
    """One read-only connection per database file; `file_id` (the inode) reopens it if the file is replaced."""
    return EventStore(path, read_only=True)


def legend() -> alt.Legend:
    return alt.Legend(orient="top", title=None, symbolType="square", labelFontSize=11)
