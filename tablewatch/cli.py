"""Command line entry point.

    python -m tablewatch zones     --source data/sample.mp4          # draw table zones
    python -m tablewatch run       --source data/sample.mp4 --output outputs/annotated.mp4
    python -m tablewatch report                                       # metrics for the latest run
    python -m tablewatch dashboard                                    # Streamlit UI
    python -m tablewatch simulate  --hours 3                          # synthetic data, no video needed
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from tablewatch.config import load_config
from tablewatch.core import fmt_duration

DEFAULT_CONFIG = "config/config.yaml"


def _config(args: argparse.Namespace):
    path = args.config if Path(args.config).exists() else None
    if path is None and args.config != DEFAULT_CONFIG:
        sys.exit(f"Config file not found: {args.config}")
    overrides = {}
    for item in getattr(args, "set", None) or []:
        key, _, raw = item.partition("=")
        overrides[key] = yaml.safe_load(raw)
    return load_config(path, overrides)


def cmd_run(args: argparse.Namespace) -> None:
    from tablewatch.pipeline import run
    from tablewatch.zones.zones import load_zones

    cfg = _config(args)
    if args.source:
        cfg.input.source = args.source
    if args.zones:
        cfg.zones_path = args.zones
    if args.output:
        cfg.output.video_path = args.output
    if args.show:
        cfg.output.show_window = True
    if args.db:
        cfg.storage.db_path = args.db
    if args.realtime:
        cfg.input.realtime = True
    if not Path(cfg.zones_path).exists():
        sys.exit(f"No zones file at {cfg.zones_path}. Draw zones first: python -m tablewatch zones --source {cfg.input.source}")
    zone_map = load_zones(cfg.zones_path)
    if not zone_map.zones:
        sys.exit(f"{cfg.zones_path} has no tables")
    media_start = datetime.fromisoformat(args.start_time).timestamp() if args.start_time else None
    run_id = run(cfg, zone_map, args.max_seconds, media_start)
    if run_id is not None:
        print(f"\nRun {run_id} finished. Metrics: python -m tablewatch report --run {run_id} --db {cfg.storage.db_path}")


def cmd_report(args: argparse.Namespace) -> None:
    from tablewatch.state.metrics import sitting_metrics, table_summary
    from tablewatch.storage.db import EventStore

    db = args.db or _config(args).storage.db_path
    if not Path(db).exists():
        sys.exit(f"No database at {db}")
    store = EventStore(db, read_only=True)
    runs = store.runs()
    if not runs:
        sys.exit("No runs recorded yet")
    info = store.run(args.run) if args.run else runs[0]
    if info is None:
        sys.exit(f"Run {args.run} not found")
    sittings, visits = store.sittings(info.run_id), store.visits(info.run_id)
    now = info.last_ts
    print(f"Run {info.run_id}: {info.source}  ({fmt_duration(now)} processed)\n")
    header = f"{'table':<8}{'sittings':>9}{'visits':>8}{'1st visit':>11}{'avg gap':>9}{'max gap':>9}{'visits/sit':>11}{'turn time':>11}{'idle':>9}{'util':>7}"
    print(header)
    print("-" * len(header))
    for t in info.tables:
        s = table_summary(t["table_id"], sittings, visits, 0.0, now)
        util = f"{s.utilisation:.0%}" if s.utilisation is not None else "-"
        vps = f"{s.avg_visits_per_sitting:.1f}" if s.avg_visits_per_sitting is not None else "-"
        print(
            f"{s.table_id:<8}{s.sittings:>9}{s.visits:>8}{fmt_duration(s.avg_time_to_first_visit_s):>11}"
            f"{fmt_duration(s.avg_visit_gap_s):>9}{fmt_duration(s.max_visit_gap_s):>9}{vps:>11}"
            f"{fmt_duration(s.avg_turn_time_s):>11}{fmt_duration(s.idle_s):>9}{util:>7}"
        )
    if sittings:
        print("\nSittings")
        for st in sittings:
            m = sitting_metrics(st, visits, now)
            gaps = ", ".join(fmt_duration(g) for g in m.visit_gaps_s) or "-"
            print(
                f"  {m.sitting_key:<10} seated {fmt_duration(m.seated_at):>7}  duration {fmt_duration(m.duration_s):>7}"
                f"{' (open)' if m.is_open else '       '}  first visit {fmt_duration(m.time_to_first_visit_s):>6}"
                f"  visits {m.visit_count}  gaps {gaps}"
            )
    alerts = store.alerts(info.run_id)
    if alerts:
        print("\nAlerts")
        for a in reversed(alerts):
            resolved = f"resolved {fmt_duration(a['resolved_ts'])}" if a["resolved_ts"] is not None else "ACTIVE"
            print(f"  [{fmt_duration(a['ts'])}] {a['message']}  ({resolved})")


def cmd_simulate(args: argparse.Namespace) -> None:
    from tablewatch.simulate import simulate

    cfg = _config(args)
    if args.db:
        cfg.storage.db_path = args.db
    if not args.verbose:
        logging.getLogger("tablewatch.pipeline").setLevel(logging.WARNING)
    start = datetime.fromisoformat(args.start_time) if args.start_time else None
    run_id = simulate(cfg, args.tables, args.hours, args.seed, args.live, args.speed, start, notify=args.notify)
    print(f"Simulated run {run_id} written to {cfg.storage.db_path}. View it: python -m tablewatch dashboard --db {cfg.storage.db_path}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    app = Path(__file__).parent / "dashboard" / "app.py"
    db = args.db or _config(args).storage.db_path
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port), "--", "--db", db]
    sys.exit(subprocess.call(cmd))


def cmd_zones(args: argparse.Namespace) -> None:
    from tablewatch.zones.setup_tool import run_from_args

    run_from_args(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tablewatch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config (default: %(default)s)")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="process a video file or stream")
    p.add_argument("--source", help="video file, rtsp:// URL or webcam index (overrides config)")
    p.add_argument("--zones", help="zones.yaml (overrides config)")
    p.add_argument("--output", help="write an annotated video here (frames are not saved otherwise)")
    p.add_argument("--show", action="store_true", help="display the annotated video in a window")
    p.add_argument("--db", help="SQLite database (overrides config)")
    p.add_argument("--max-seconds", type=float, help="stop after this much video time")
    p.add_argument("--realtime", action="store_true", help="pace a file at real speed (simulates a live camera)")
    p.add_argument("--start-time", help="wall-clock time of the first frame, e.g. 2026-10-23T19:30 (files only)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any config value, e.g. state.seat_confirm_s=5")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="print metrics for a run")
    p.add_argument("--db", help="SQLite database (default: from config)")
    p.add_argument("--run", type=int, help="run id (default: latest)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("simulate", help="generate a synthetic service (no video needed) for demos")
    p.add_argument("--db", help="SQLite database (default: from config)")
    p.add_argument("--tables", type=int, default=8)
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--live", action="store_true", help="write in (accelerated) real time so the dashboard can follow")
    p.add_argument("--speed", type=float, default=30.0, help="with --live: simulated seconds per real second")
    p.add_argument("--start-time", help="clock time of the simulated service start, e.g. 2026-10-23T18:00")
    p.add_argument("--notify", action="store_true", help="send alerts to the configured notifiers")
    p.add_argument("--verbose", action="store_true", help="log every state change")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any config value")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("dashboard", help="start the Streamlit dashboard")
    p.add_argument("--db", help="SQLite database (default: from config)")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_dashboard)

    from tablewatch.zones import setup_tool

    p = sub.add_parser("zones", help="draw table zones on a frame", description=setup_tool.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    setup_tool.add_arguments(p)
    p.set_defaults(func=cmd_zones)

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    args.func(args)


if __name__ == "__main__":
    main()
