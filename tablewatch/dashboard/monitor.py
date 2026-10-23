"""Live monitoring page: the table board for one run, live or replayed.

"Live" follows a running pipeline; "Replay" scrubs through any run and
rebuilds table states from stored events.
"""

from __future__ import annotations

import html
import time
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from tablewatch.core import fmt_duration
from tablewatch.dashboard.common import CHROME_AXIS, CHROME_WITH_LEGEND, Theme, current_theme, legend
from tablewatch.palette import ALERT_HEX, ALERT_ICON, STATE_ICON, STATE_LABEL
from tablewatch.state.metrics import clip_records, sitting_metrics, snapshot_at, state_segments, table_summary
from tablewatch.state.table_state import Sitting, TableSnapshot, TableState, Visit
from tablewatch.storage.db import EventStore, RunInfo

STATE_ORDER = [TableState.EMPTY, TableState.SEATED, TableState.ATTENDED, TableState.WAITING]
STALE_AFTER_S = 15.0


# -- time helpers -------------------------------------------------------------


def clock(run: RunInfo, ts: float | None) -> datetime | None:
    """Naive local datetime for a run-relative timestamp (Vega-Lite shows it as is)."""
    return None if ts is None else datetime.fromtimestamp(run.media_start + ts)


def clock_str(run: RunInfo, ts: float | None) -> str:
    return "-" if ts is None else clock(run, ts).strftime("%H:%M:%S")


def minutes(seconds: float | None) -> float | None:
    return None if seconds is None else seconds / 60.0


# -- live snapshots -------------------------------------------------------------


def live_snapshots(run: RunInfo, rows: list[dict]) -> tuple[dict[str, TableSnapshot], float, float | None]:
    """Snapshots from the pipeline's status rows, with timers advanced to 'now'.

    Returns (snapshots, now_ts, seconds_since_last_update or None if finished).
    """
    by_table = {r["table_id"]: r for r in rows}
    now_ts, lag = run.last_ts, None
    if run.is_running and rows:
        last_update = max(r["updated_wall"] for r in rows)
        lag = max(0.0, time.time() - last_update)
        now_ts = max(r["ts"] for r in rows) + min(lag, STALE_AFTER_S)
    snapshots = {}
    for t in run.tables:
        tid = t["table_id"]
        r = by_table.get(tid)
        if r is None:
            snapshots[tid] = TableSnapshot(tid, now_ts, TableState.EMPTY, 0.0, empty_since=0.0)
            continue
        snapshots[tid] = TableSnapshot(
            table_id=tid, ts=now_ts, state=TableState(r["state"]), state_since=r["state_since"] or 0.0,
            guests=r["guests"] or 0, staff=r["staff"] or 0, sitting_key=r["sitting_key"], seated_at=r["seated_at"],
            visit_count=r["visit_count"] or 0, visit_started_at=r["visit_started_at"],
            last_visit_end=r["last_visit_end"], empty_since=r["empty_since"], food_served_at=r["food_served_at"],
        )
    return snapshots, now_ts, lag


# -- table grid -----------------------------------------------------------------


def primary_timer(snap: TableSnapshot) -> tuple[str, str]:
    """(big number, caption) for a table card."""
    if snap.state == TableState.EMPTY:
        idle = snap.ts - snap.empty_since if snap.empty_since is not None else None
        return fmt_duration(idle), "empty for"
    if snap.state == TableState.SEATED:
        return fmt_duration(snap.waiting_for_staff_s), "seated, no visit yet"
    if snap.state == TableState.ATTENDED:
        return fmt_duration(snap.ts - (snap.visit_started_at or snap.ts)), "staff visit in progress"
    return fmt_duration(snap.waiting_for_staff_s), "since last visit"


CARD_CSS = """
<style>
.tw-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; }}
.tw-card {{ background: {card}; border: 1px solid {border}; border-radius: 10px; padding: 12px 14px 12px;
           border-top: 4px solid var(--tw-state); color: {ink}; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
.tw-card.tw-alerting {{ box-shadow: 0 0 0 2px {alert}; }}
.tw-head {{ display: flex; justify-content: space-between; align-items: baseline; }}
.tw-id {{ font-weight: 650; font-size: 1.05rem; }}
.tw-seats, .tw-cap, .tw-meta {{ color: {ink2}; font-size: 0.78rem; }}
.tw-state {{ margin-top: 4px; font-size: 0.85rem; font-weight: 600; }}
.tw-dot {{ color: var(--tw-state); margin-right: 4px; }}
.tw-timer {{ font-size: 1.9rem; font-weight: 600; line-height: 1.1; margin-top: 6px; }}
.tw-meta {{ margin-top: 6px; }}
.tw-alert {{ margin-top: 8px; background: {alert}; color: #ffffff; border-radius: 6px; padding: 4px 8px; font-size: 0.78rem; font-weight: 600; }}
</style>
"""


def render_grid(snapshots: dict[str, TableSnapshot], seats: dict[str, int], active_alerts: list[dict], theme: Theme, live: bool) -> None:
    alerts_by_table: dict[str, list[dict]] = {}
    for a in active_alerts:
        alerts_by_table.setdefault(a["table_id"], []).append(a)
    cards = []
    for tid, snap in snapshots.items():
        timer, caption = primary_timer(snap)
        meta = []
        if snap.state != TableState.EMPTY:
            meta.append(f"seated {fmt_duration(snap.sitting_duration)} · visits {snap.visit_count}")
        if live and (snap.guests or snap.staff):
            meta.append(f"{snap.guests} guest{'s' if snap.guests != 1 else ''} · {snap.staff} staff in zone")
        alerts = alerts_by_table.get(tid, [])
        alert_html = "".join(
            f'<div class="tw-alert">{ALERT_ICON} {html.escape(a["rule"].replace("_", " "))}</div>' for a in alerts
        )
        cards.append(
            f'<div class="tw-card{" tw-alerting" if alerts else ""}" style="--tw-state:{theme.state(snap.state)}">'
            f'<div class="tw-head"><span class="tw-id">{html.escape(tid)}</span>'
            f'<span class="tw-seats">{seats.get(tid, "?")} seats</span></div>'
            f'<div class="tw-state"><span class="tw-dot">{STATE_ICON[snap.state]}</span>{snap.state.value.title()}</div>'
            f'<div class="tw-timer">{timer}</div><div class="tw-cap">{caption}</div>'
            f'<div class="tw-meta">{"<br>".join(meta) or "&nbsp;"}</div>{alert_html}</div>'
        )
    css = CARD_CSS.format(card=theme.card, border=theme.border, ink=theme.ink, ink2=theme.ink_2, alert=ALERT_HEX)
    st.markdown(css + f'<div class="tw-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


# -- alert feed -------------------------------------------------------------------


def render_alert_feed(run: RunInfo, alerts: list[dict], now_ts: float, theme: Theme) -> None:
    if not alerts:
        st.caption("No alerts so far.")
        return
    active = [a for a in alerts if a["resolved_ts"] is None or a["resolved_ts"] > now_ts]
    resolved = [a for a in alerts if a not in active]
    rows = []
    for a in active + resolved:
        is_active = a in active
        status = "active" if is_active else f"resolved {clock_str(run, a['resolved_ts'])}"
        colour = ALERT_HEX if is_active else theme.muted
        rows.append(
            f'<div style="padding:8px 0;border-bottom:1px solid {theme.border};font-size:0.85rem;color:{theme.ink}">'
            f'<span style="color:{colour}">{ALERT_ICON}</span> <b>{clock_str(run, a["ts"])}</b> '
            f'{html.escape(a["message"])}<br><span style="color:{theme.ink_2};font-size:0.75rem">'
            f'{html.escape(a["rule"].replace("_", " "))} · {a["severity"]} · {status}</span></div>'
        )
    with st.container(height=360, border=False):
        st.markdown("".join(rows), unsafe_allow_html=True)


# -- charts -------------------------------------------------------------------------


def state_scale(theme: Theme, states: list[TableState]) -> alt.Scale:
    return alt.Scale(domain=[s.value for s in states], range=[theme.state(s) for s in states])


def timeline_chart(run: RunInfo, tables: list[str], sittings, visits, alerts, now_ts: float, theme: Theme) -> alt.Chart | None:
    segs = state_segments(tables, sittings, visits, 0.0, now_ts)
    if not segs:
        return None
    df = pd.DataFrame(
        [
            {
                "table": s.table_id, "state": s.state.value, "start": clock(run, s.start), "end": clock(run, s.end),
                "from": clock_str(run, s.start), "to": clock_str(run, s.end), "duration": fmt_duration(s.end - s.start),
                "sitting": s.sitting_key or "-",
            }
            for s in segs
        ]
    )
    bars = (
        alt.Chart(df)
        .mark_bar(height={"band": 0.62}, cornerRadius=3, stroke=theme.surface, strokeWidth=1.5)
        .encode(
            x=alt.X("start:T", title=None, axis=alt.Axis(format="%H:%M", tickCount=8, labelOverlap="greedy", grid=False)),
            x2="end:T",
            y=alt.Y("table:N", sort=tables, title=None),
            color=alt.Color("state:N", scale=state_scale(theme, STATE_ORDER), legend=legend()),
            tooltip=["table", "state", "from", "to", "duration", "sitting"],
        )
    )
    layers = [bars]
    shown = [a for a in alerts if a["table_id"] in tables and a["ts"] <= now_ts]
    if shown:
        adf = pd.DataFrame(
            [
                {"table": a["table_id"], "at": clock(run, a["ts"]), "time": clock_str(run, a["ts"]), "alert": a["message"]}
                for a in shown
            ]
        )
        layers.append(
            alt.Chart(adf)
            .mark_point(shape="triangle-down", filled=True, size=150, color=ALERT_HEX, stroke=theme.surface, strokeWidth=1, opacity=1)
            .encode(x="at:T", y=alt.Y("table:N", sort=tables), tooltip=["table", "time", "alert"])
        )
    cursor = pd.DataFrame({"now": [clock(run, now_ts)]})
    layers.append(alt.Chart(cursor).mark_rule(color=theme.muted, strokeWidth=1).encode(x="now:T"))
    return alt.layer(*layers).properties(height=CHROME_WITH_LEGEND + 30 * len(tables))


def strip_chart(df: pd.DataFrame, x: str, x_title: str, colour: str, tables: list[str], tooltip: list[str],
                theme: Theme, threshold_min: float | None = None, integer: bool = False) -> alt.Chart:
    """One dot per sitting (or gap), one row per table: shows spread and outliers at any volume."""
    dots = (
        alt.Chart(df)
        .transform_calculate(jitter="(random() - 0.5) * 0.6")
        .mark_circle(size=80, color=colour, stroke=theme.surface, strokeWidth=1.5, opacity=0.9)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(zero=True),
                    axis=alt.Axis(tickMinStep=1, format="d") if integer else alt.Axis()),
            y=alt.Y("table:N", sort=tables, title=None, axis=alt.Axis(labelOverlap=False)),
            yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-0.5, 0.5])),
            tooltip=tooltip,
        )
    )
    layers = [dots]
    if threshold_min:
        rule = pd.DataFrame({"t": [threshold_min], "label": [f"alert at {fmt_duration(threshold_min * 60)}"]})
        layers.append(alt.Chart(rule).mark_rule(color=ALERT_HEX, strokeDash=[4, 3], strokeWidth=1.5).encode(x="t:Q", tooltip=["label"]))
        layers.append(
            alt.Chart(rule).mark_text(align="left", dx=4, dy=-2, color=ALERT_HEX, fontSize=11, baseline="bottom")
            .encode(x="t:Q", y=alt.value(0), text="label")
        )
    return alt.layer(*layers).properties(height=CHROME_AXIS + 30 * len(tables))


def stacked_chart(df: pd.DataFrame, y: str, theme: Theme, states: list[TableState], tooltip: list[str], x_title: str) -> alt.Chart:
    order = list(dict.fromkeys(df[y]))
    rank = {s.value: i for i, s in enumerate(states)}
    df = df.assign(order=df["state"].map(rank))
    return (
        alt.Chart(df)
        .mark_bar(size=16, stroke=theme.surface, strokeWidth=2)
        .encode(
            x=alt.X("minutes:Q", stack="zero", title=x_title),
            y=alt.Y(f"{y}:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)),
            color=alt.Color("state:N", scale=state_scale(theme, states), legend=legend()),
            order=alt.Order("order:Q"),
            tooltip=tooltip,
        )
        .properties(height=CHROME_WITH_LEGEND + 26 * len(order))
    )


BREAKDOWN_LIMIT = 15


def render_charts(run: RunInfo, tables: list[str], sittings: list[Sitting], visits: list[Visit], now_ts: float, theme: Theme) -> None:
    alerts_cfg = run.config.get("alerts", {})
    per = [sitting_metrics(s, visits, now_ts) for s in sittings if s.table_id in tables]
    if not per:
        st.caption("No sittings yet.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Time to first staff visit**")
        rows = [
            {"table": m.table_id, "sitting": m.sitting_key, "minutes": minutes(m.time_to_first_visit_s),
             "wait": fmt_duration(m.time_to_first_visit_s), "seated": clock_str(run, m.seated_at)}
            for m in per if m.time_to_first_visit_s is not None
        ]
        if rows:
            st.altair_chart(
                strip_chart(pd.DataFrame(rows), "minutes", "minutes after seating", theme.state(TableState.SEATED), tables,
                            ["sitting", "seated", "wait"], theme, minutes(alerts_cfg.get("no_first_visit_s"))),
                width="stretch",
            )
        else:
            st.caption("No first visits yet.")
    with c2:
        st.markdown("**Gaps between staff visits**")
        rows = [
            {"table": m.table_id, "sitting": m.sitting_key, "gap": f"gap {i + 1}", "minutes": minutes(g), "length": fmt_duration(g)}
            for m in per for i, g in enumerate(m.visit_gaps_s)
        ]
        if rows:
            st.altair_chart(
                strip_chart(pd.DataFrame(rows), "minutes", "minutes without staff", theme.state(TableState.WAITING), tables,
                            ["sitting", "gap", "length"], theme, minutes(alerts_cfg.get("visit_gap_s"))),
                width="stretch",
            )
        else:
            st.caption("No sitting has had two visits yet.")
    with c3:
        st.markdown("**Staff visits per sitting**")
        rows = [
            {"table": m.table_id, "sitting": m.sitting_key, "visits": m.visit_count,
             "status": "in progress" if m.is_open else "finished", "turn time": fmt_duration(m.duration_s)}
            for m in per
        ]
        st.altair_chart(
            strip_chart(pd.DataFrame(rows), "visits", "visits", theme.state(TableState.ATTENDED), tables,
                        ["sitting", "visits", "turn time", "status"], theme, integer=True),
            width="stretch",
        )
    st.caption("Each dot is one sitting (or one gap). Hover for details; the dashed line is the alert threshold.")

    segs = state_segments(tables, sittings, visits, 0.0, now_ts)
    c4, c5 = st.columns(2)
    with c4:
        recent = per[-BREAKDOWN_LIMIT:]
        title = "**Sitting breakdown** (bar length = turn time"
        title += f", latest {BREAKDOWN_LIMIT})" if len(per) > BREAKDOWN_LIMIT else ")"
        st.markdown(title)
        keys = {m.sitting_key: m for m in recent}
        rows = [
            {"sitting": s.sitting_key, "state": s.state.value, "minutes": minutes(s.end - s.start)}
            for s in segs if s.sitting_key in keys and s.state != TableState.EMPTY
        ]
        if rows:
            df = pd.DataFrame(rows).groupby(["sitting", "state"], as_index=False)["minutes"].sum()
            df["turn time"] = df["sitting"].map(lambda k: fmt_duration(keys[k].duration_s) + (" (open)" if keys[k].is_open else ""))
            df["visits"] = df["sitting"].map(lambda k: keys[k].visit_count)
            df["time"] = df["minutes"].map(lambda v: fmt_duration(v * 60))
            df["sitting"] = pd.Categorical(df["sitting"], list(keys))
            df = df.sort_values("sitting")
            df["sitting"] = df["sitting"].astype(str)
            st.altair_chart(
                stacked_chart(df, "sitting", theme, [TableState.SEATED, TableState.ATTENDED, TableState.WAITING],
                              ["sitting", "state", "time", "turn time", "visits"], "minutes"),
                width="stretch",
            )
    with c5:
        st.markdown("**Where each table's time went** (grey = idle)")
        rows = [{"table": s.table_id, "state": s.state.value, "minutes": minutes(s.end - s.start)} for s in segs]
        if rows:
            df = pd.DataFrame(rows).groupby(["table", "state"], as_index=False)["minutes"].sum()
            df["time"] = df["minutes"].map(lambda v: fmt_duration(v * 60))
            df["table"] = pd.Categorical(df["table"], tables)
            df = df.sort_values("table")
            df["table"] = df["table"].astype(str)
            st.altair_chart(
                stacked_chart(df, "table", theme, STATE_ORDER, ["table", "state", "time"], "minutes"),
                width="stretch",
            )


def render_tables(run: RunInfo, tables: list[str], sittings, visits, now_ts: float) -> None:
    summary = []
    for tid in tables:
        s = table_summary(tid, sittings, visits, 0.0, now_ts)
        summary.append(
            {
                "table": tid, "sittings": s.sittings, "visits": s.visits,
                "avg 1st visit": fmt_duration(s.avg_time_to_first_visit_s),
                "max 1st visit": fmt_duration(s.max_time_to_first_visit_s),
                "avg gap": fmt_duration(s.avg_visit_gap_s), "max gap": fmt_duration(s.max_visit_gap_s),
                "visits/sitting": round(s.avg_visits_per_sitting, 1) if s.avg_visits_per_sitting is not None else None,
                "avg turn time": fmt_duration(s.avg_turn_time_s), "idle": fmt_duration(s.idle_s),
                "utilisation": f"{s.utilisation:.0%}" if s.utilisation is not None else "-",
            }
        )
    st.markdown("**Per table**")
    st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
    per = [sitting_metrics(s, visits, now_ts) for s in sittings if s.table_id in tables]
    if per:
        st.markdown("**Per sitting**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "sitting": m.sitting_key, "table": m.table_id, "seated": clock_str(run, m.seated_at),
                        "left": clock_str(run, m.ended_at), "turn time": fmt_duration(m.duration_s),
                        "open": m.is_open, "1st visit after": fmt_duration(m.time_to_first_visit_s),
                        "visits": m.visit_count, "gaps": ", ".join(fmt_duration(g) for g in m.visit_gaps_s),
                        "food after": fmt_duration(m.time_to_food_s),
                    }
                    for m in per
                ]
            ),
            hide_index=True,
            width="stretch",
        )


# -- page ------------------------------------------------------------------------------


def render_body(store: EventStore, run_id: int, view: str, replay_ts: float | None, tables: list[str]) -> None:
    theme = current_theme()
    run = store.run(run_id)
    sittings_all, visits_all = store.sittings(run_id), store.visits(run_id)
    alerts = [a for a in store.alerts(run_id) if a["table_id"] in tables]
    seats = {t["table_id"]: t.get("seats", "?") for t in run.tables}

    if view == "Live":
        snapshots, now_ts, lag = live_snapshots(run, store.table_status(run_id))
        if lag is not None and lag > STALE_AFTER_S:
            st.warning(f"No updates from the pipeline for {fmt_duration(lag)}. Is it still running?")
    else:
        now_ts = replay_ts
        snapshots = {tid: snapshot_at(tid, now_ts, sittings_all, visits_all) for tid in seats}
        lag = None
    snapshots = {tid: snap for tid, snap in snapshots.items() if tid in tables}
    sittings, visits = clip_records(sittings_all, visits_all, now_ts)
    alerts = [a for a in alerts if a["ts"] <= now_ts]
    active = [a for a in alerts if a["resolved_ts"] is None or a["resolved_ts"] > now_ts]

    per = [sitting_metrics(s, visits, now_ts) for s in sittings if s.table_id in tables]
    firsts = [m.time_to_first_visit_s for m in per if m.time_to_first_visit_s is not None]
    gaps = [g for m in per for g in m.visit_gaps_s]
    turns = [m.duration_s for m in per if not m.is_open]
    occupied = sum(s.state != TableState.EMPTY for s in snapshots.values())
    k = st.columns(6)
    k[0].metric("Clock", clock_str(run, now_ts), help=f"{fmt_duration(now_ts)} into the run")
    k[1].metric("Occupied tables", f"{occupied} / {len(snapshots)}")
    k[2].metric(f"{ALERT_ICON} Active alerts", len(active))
    k[3].metric("Avg time to 1st visit", fmt_duration(sum(firsts) / len(firsts)) if firsts else "-")
    k[4].metric("Avg gap between visits", fmt_duration(sum(gaps) / len(gaps)) if gaps else "-")
    k[5].metric("Avg turn time", fmt_duration(sum(turns) / len(turns)) if turns else "-")

    render_grid(snapshots, seats, active, theme, live=view == "Live")
    st.write("")

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Alerts")
        render_alert_feed(run, alerts, now_ts, theme)
    with right:
        st.subheader("Timeline")
        chart = timeline_chart(run, tables, sittings, visits, alerts, now_ts, theme)
        if chart is None:
            st.caption("Nothing recorded yet.")
        else:
            st.altair_chart(chart, width="stretch")
            st.caption(f"{ALERT_ICON} marks when an alert was raised. Hover any bar for exact times.")

    st.subheader("Service metrics")
    render_charts(run, tables, sittings, visits, now_ts, theme)
    with st.expander("Table view (all numbers)"):
        render_tables(run, tables, sittings, visits, now_ts)


def run_label(r: RunInfo) -> str:
    status = "running" if r.is_running else "finished"
    started = datetime.fromtimestamp(r.started_wall).strftime("%d %b %H:%M")
    return f"#{r.run_id} · {Path(r.source).name} · {started} · {status} · {fmt_duration(r.last_ts)}"


def page(store: EventStore) -> None:
    runs = store.runs()
    if not runs:
        st.info("The database has no runs yet.")
        return
    with st.sidebar:
        by_id = {r.run_id: r for r in runs}
        run_id = st.selectbox("Run", list(by_id), format_func=lambda i: run_label(by_id[i]))
        refresh_s = st.slider("Live refresh (seconds)", 1, 10, 2)
        st.markdown("**States**")
        theme = current_theme()
        st.markdown(
            "".join(
                f'<div style="font-size:0.85rem;margin:2px 0"><span style="color:{theme.state(state)}">'
                f"{STATE_ICON[state]}</span> <b>{state.value.title()}</b>: {STATE_LABEL[state]}</div>"
                for state in STATE_ORDER
            )
            + f'<div style="font-size:0.85rem;margin:2px 0"><span style="color:{ALERT_HEX}">{ALERT_ICON}</span>'
            " <b>Alert</b>: a service threshold was crossed</div>",
            unsafe_allow_html=True,
        )

    run = by_id[run_id]
    table_ids = [t["table_id"] for t in run.tables]
    st.title("Table service")
    st.caption(f"Source: `{run.source}` · {'live' if run.is_live else 'recorded'} · {fmt_duration(run.last_ts)} processed")

    f1, f2, f3 = st.columns([1.2, 3, 2])
    view = f1.radio("View", ["Live", "Replay"], horizontal=True, index=0 if run.is_running else 1)
    replay_ts = None
    if view == "Replay":
        start = clock(run, 0.0).replace(microsecond=0)
        end = clock(run, max(run.last_ts, 1.0))
        picked = f2.slider("Replay time", min_value=start, max_value=end, value=end, step=timedelta(seconds=1), format="HH:mm:ss")
        replay_ts = min(run.last_ts, max(0.0, picked.timestamp() - run.media_start))
    selected = f3.multiselect("Tables", table_ids, default=table_ids) or table_ids

    if view == "Live" and run.is_running:

        @st.fragment(run_every=refresh_s)
        def live_body() -> None:
            render_body(store, run_id, view, replay_ts, selected)

        live_body()
    else:
        render_body(store, run_id, view, replay_ts, selected)
