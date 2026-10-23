"""Analytics page: service history across every recorded run.

Answers the questions behind the live board: how long tables turn, how long
guests wait for staff, drinks and food, how that moves by day and through the
evening, and which tables lag.
"""

from __future__ import annotations

from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from tablewatch.analytics.history import by_day, by_hour, by_table, occupancy_frame, sittings_frame, summary
from tablewatch.core import fmt_duration
from tablewatch.dashboard.common import CHROME_AXIS, CHROME_WITH_LEGEND, Theme, current_theme, legend
from tablewatch.storage.db import EventStore

DEFAULT_DAYS = 14
MILESTONES = [("first_visit", "First staff visit", "avg_first_visit_min"),
              ("drinks", "Drinks on table", "avg_drinks_min"),
              ("food", "Food on table", "avg_food_min")]


def _fmt_min(minutes: float | None) -> str:
    return "-" if minutes is None or pd.isna(minutes) else fmt_duration(minutes * 60)


def milestone_chart(df: pd.DataFrame, x: str, x_title: str, order: list[str], theme: Theme,
                    measures: list[tuple[str, str, str]]) -> alt.Chart:
    """Average minutes from seating to each milestone, one line per milestone, labelled at its end."""
    long = df.melt(id_vars=[x], value_vars=[col for _, _, col in measures], var_name="col", value_name="minutes")
    names = {col: label for _, label, col in measures}
    long["milestone"] = long["col"].map(names)
    long = long.dropna(subset=["minutes"])
    long["time"] = long["minutes"].map(_fmt_min)
    scale = alt.Scale(domain=[label for _, label, _ in measures], range=[theme.measure(key) for key, _, _ in measures])
    base = alt.Chart(long).encode(
        x=alt.X(f"{x}:O", sort=order, title=x_title, axis=alt.Axis(labelAngle=0, labelOverlap="greedy")),
        y=alt.Y("minutes:Q", title="minutes after seating", scale=alt.Scale(zero=True)),
        color=alt.Color("milestone:N", scale=scale, legend=legend()),
        tooltip=[alt.Tooltip(f"{x}:O", title=x_title), "milestone:N", alt.Tooltip("time:N", title="average")],
    )
    lines = base.mark_line(strokeWidth=2).encode()
    points = base.mark_circle(size=64, opacity=1, stroke=theme.surface, strokeWidth=1.5)
    last = long[long[x] == next((v for v in reversed(order) if v in set(long[x])), None)]
    labels = alt.Chart(last).mark_text(align="left", dx=8, fontSize=11, color=theme.ink_2).encode(
        x=alt.X(f"{x}:O", sort=order), y="minutes:Q", text="milestone:N"
    )
    return alt.layer(lines, points, labels).properties(height=CHROME_WITH_LEGEND + 200, padding={"right": 110})


def bar_chart(df: pd.DataFrame, x: str, y: str, order: list[str], colour: str, x_title: str, y_title: str,
              tooltip: list) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, color=colour, width={"band": 0.5})
        .encode(
            x=alt.X(f"{x}:O", sort=order, title=x_title, axis=alt.Axis(labelAngle=0, labelOverlap="greedy")),
            y=alt.Y(f"{y}:Q", title=y_title),
            tooltip=tooltip,
        )
        .properties(height=CHROME_AXIS + 200)
    )


def page(store: EventStore) -> None:
    st.title("Service analytics")
    st.caption(
        "Every recorded service, one row per sitting. Drinks and food times are estimated from what the camera "
        "sees arrive on the table; treat them as trends rather than exact plate times."
    )
    everything = sittings_frame(store)
    if everything.empty:
        st.info("No sittings recorded yet. Process some video, or generate a demo week: "
                "`python -m tablewatch simulate --days 7`")
        return
    theme = current_theme()

    first_day, last_day = everything["service_date"].min(), everything["service_date"].max()
    f1, f2, f3 = st.columns([2, 1.5, 2.5])
    picked = f1.date_input(
        "Service dates", value=(max(first_day, last_day - timedelta(days=DEFAULT_DAYS - 1)), last_day),
        min_value=first_day, max_value=last_day,
    )
    start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (picked, picked))
    sites = sorted(everything["site"].unique())
    chosen_sites = f2.multiselect("Sites", sites, default=sites) if len(sites) > 1 else sites
    tables = sorted(everything.loc[everything["site"].isin(chosen_sites), "table_id"].unique())
    chosen_tables = f3.multiselect("Tables", tables, default=tables) or tables

    df = sittings_frame(store, chosen_sites, start, end)
    df = df[df["table_id"].isin(chosen_tables)]
    occ = occupancy_frame(store, chosen_sites, start, end)
    occ = occ[occ["table_id"].isin(chosen_tables)]
    if df.empty:
        st.info("No sittings in that selection.")
        return
    s = summary(df, occ)

    k = st.columns(7)
    k[0].metric("Sittings", s["sittings"], help=f"over {s['services']} service(s)")
    k[1].metric("Avg turn time", _fmt_min(s["avg_turn_time_min"]), help="Seated to guests gone, finished sittings only")
    k[2].metric("Avg to 1st visit", _fmt_min(s["avg_first_visit_min"]))
    k[3].metric("Avg to drinks", _fmt_min(s["avg_drinks_min"]), help="Estimated from cups and glasses arriving")
    k[4].metric("Avg to food", _fmt_min(s["avg_food_min"]), help="Estimated from bowls and food arriving")
    k[5].metric("Table utilisation", f"{s['utilisation_pct']:.0f}%" if s["utilisation_pct"] is not None else "-",
                help="Share of observed time tables were occupied")
    k[6].metric("Sittings with an alert", f"{s['sittings_with_alert_pct']:.0f}%")

    days = by_day(df)
    days["day"] = pd.to_datetime(days["service_date"]).dt.strftime("%a %d %b")
    day_order = list(days["day"])
    days["turn"] = days["avg_turn_time_min"].map(_fmt_min)

    st.subheader("Day by day")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**How long until the first visit, drinks and food**")
        st.altair_chart(milestone_chart(days, "day", "service", day_order, theme, MILESTONES), width="stretch")
    with c2:
        st.markdown("**Average table turn time**")
        st.altair_chart(
            bar_chart(days, "day", "avg_turn_time_min", day_order, theme.measure("volume"), "service", "minutes",
                      [alt.Tooltip("day:O", title="service"), alt.Tooltip("turn:N", title="avg turn time"),
                       alt.Tooltip("sittings:Q", title="sittings")]),
            width="stretch",
        )

    hours = by_hour(df)
    hours["seated"] = hours["hour"].map(lambda h: f"{h:02d}:00")
    hour_order = list(hours["seated"])
    st.subheader("Through the evening")
    st.caption("Grouped by the hour guests sat down: does service keep up when it gets busy?")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Sittings started**")
        st.altair_chart(
            bar_chart(hours, "seated", "sittings", hour_order, theme.measure("volume"), "hour seated", "sittings",
                      [alt.Tooltip("seated:O", title="hour seated"), alt.Tooltip("sittings:Q")]),
            width="stretch",
        )
    with c4:
        st.markdown("**Waits by hour seated**")
        st.altair_chart(
            milestone_chart(hours, "seated", "hour seated", hour_order, theme, [MILESTONES[0], MILESTONES[2]]),
            width="stretch",
        )

    st.subheader("By table")
    tables_df = by_table(df, occ)
    st.dataframe(
        tables_df.drop(columns=["site"] if len(chosen_sites) == 1 else []),
        hide_index=True, width="stretch",
        column_config={
            "site": "site", "table_id": "table",
            "avg_turn_time_min": st.column_config.NumberColumn("avg turn time", format="%.0f min"),
            "avg_first_visit_min": st.column_config.NumberColumn("to 1st visit", format="%.1f min"),
            "avg_drinks_min": st.column_config.NumberColumn("to drinks", format="%.1f min"),
            "avg_food_min": st.column_config.NumberColumn("to food", format="%.1f min"),
            "utilisation_pct": st.column_config.ProgressColumn("utilisation", min_value=0, max_value=100, format="%.0f%%"),
        },
    )

    st.subheader("Every sitting")
    st.download_button(
        "Download CSV", df.to_csv(index=False).encode(), file_name=f"tablewatch_sittings_{start}_{end}.csv",
        mime="text/csv",
    )
    minutes_cols = {c: st.column_config.NumberColumn(c.removesuffix("_min").replace("_", " "), format="%.1f min")
                    for c in df.columns if c.endswith("_min")}
    st.dataframe(
        df.sort_values("seated", ascending=False), hide_index=True, width="stretch",
        column_config={"seated": st.column_config.DatetimeColumn(format="ddd D MMM HH:mm"),
                       "left": st.column_config.DatetimeColumn(format="HH:mm"), **minutes_cols},
    )
