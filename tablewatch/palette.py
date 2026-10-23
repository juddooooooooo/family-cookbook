"""One colour per table state, shared by the annotated video and the dashboard.

States are categories (identity), so they take the first three categorical
slots, validated for colour-vision deficiency in light and dark mode; EMPTY is
neutral grey and alerts use the reserved "critical" status red. Colour is never
the only cue: every use also shows the state name or an icon.
"""

from __future__ import annotations

from tablewatch.state.table_state import TableState

STATE_HEX = {
    "light": {
        TableState.WAITING: "#2a78d6",  # slot 1 blue
        TableState.SEATED: "#eb6834",  # slot 2 orange
        TableState.ATTENDED: "#1baf7a",  # slot 3 aqua
        TableState.EMPTY: "#a3a29c",
    },
    "dark": {
        TableState.WAITING: "#3987e5",
        TableState.SEATED: "#d95926",
        TableState.ATTENDED: "#199e70",
        TableState.EMPTY: "#6b6a65",
    },
}
ALERT_HEX = "#d03b3b"  # status: critical

STATE_ICON = {
    TableState.EMPTY: "○",
    TableState.SEATED: "●",
    TableState.ATTENDED: "✔",
    TableState.WAITING: "◆",
}
ALERT_ICON = "⚠"

STATE_LABEL = {
    TableState.EMPTY: "no guests",
    TableState.SEATED: "guests seated, no staff visit yet",
    TableState.ATTENDED: "staff at the table",
    TableState.WAITING: "visited, waiting for the next visit",
}


def hex_to_bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)
