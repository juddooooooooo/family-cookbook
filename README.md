# TableWatch

A computer-vision pilot that watches restaurant video and measures table service: how long guests wait for their first staff visit, how long between visits, how many visits each sitting gets, how long tables are occupied and how long they sit empty. It raises alerts when a table is being neglected.

Behind the live table board, every sitting is recorded for **analytics**: table turn times, when drinks and food reach the table, waits by day and by hour, and per-table utilisation, across every service you record.

The pilot runs on recorded video (stock footage or phone recordings). The same code runs on a live RTSP camera: change `input.source` to an `rtsp://` URL and nothing else changes.

```
video file / RTSP ─▶ detect people (YOLO) ─▶ track IDs (ByteTrack) ─▶ assign to table zones
                                                                          │
                       annotated video ◀── overlay          staff / guest classifier
                       (opt-in)                                           │
                                            SQLite ◀── table state machines ──▶ alert rules ──▶ console / Slack / webhook / WhatsApp
                                              │
                                     Streamlit dashboard (live + replay)
```

**Privacy by default:** only events and metrics are stored. No frames, crops or face data are saved, and the dashboard never sees video. The annotated output video is opt-in, and it blurs heads unless you turn that off.

## Repository layout

```
tablewatch/
  input/           VideoSource: FileSource and StreamSource (RTSP/HTTP/webcam) behind one interface
  detection/       Detector interface + Ultralytics YOLO implementation
  tracking/        Tracker interface + ByteTrack (Roboflow `trackers`)
  classification/  RoleClassifier interface + heuristic staff/guest classifier
  zones/           table polygons, zone assignment, interactive zone-drawing tool
  state/           per-table state machine, metrics, replay helpers, drinks/food-served signals
  analytics/       service history across runs: one row per sitting, daily/hourly/per-table rollups
  alerts/          alert rules + engine, notifiers (console, webhook, Slack, WhatsApp stub)
  storage/         SQLite event store
  output/          annotated video renderer
  dashboard/       Streamlit app: Live monitoring and Analytics pages
  pipeline.py      wires it together (Pipeline = vision, ServiceMonitor = table logic)
  simulate.py      synthetic service for demos without video
  cli.py           `python -m tablewatch ...`
config/            config.yaml (all settings), zones.example.yaml, demo_video*.yaml (first-look demo)
.devcontainer/     GitHub Codespaces setup
scripts/           download_pexels.py
tests/             state machine, metrics, alerts, zones and role tests
```

## Setup

Requires Python 3.11 or newer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Optional, CPU-only machines: the CPU build of PyTorch is ~2 GB smaller
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
```

The YOLO weights (`yolo11n.pt`, about 6 MB) download automatically on the first run.

## First look (about 10 minutes)

**A. The dashboard, with no video at all.** A simulator feeds a realistic 3-hour service through the real state machine, alerts and database:

```bash
python -m tablewatch simulate --hours 3          # writes a synthetic service to data/tablewatch.db
python -m tablewatch dashboard                   # opens http://localhost:8501
```

The run has finished, so the dashboard opens in **Replay**. Drag the replay slider to scrub through the evening. To watch tables change live, run `python -m tablewatch simulate --hours 1 --live --speed 30` in one terminal and the dashboard in another, then pick the newest run in the sidebar.

For the **Analytics** page, simulate a week of evenings (Fridays and Saturdays are busier and slower), then open Analytics in the dashboard's sidebar:

```bash
python -m tablewatch simulate --days 7 --site "Demo bistro"
python -m tablewatch analytics          # the same numbers in the terminal; add --csv sittings.csv to export
```

**B. The vision pipeline, on a public test clip.** OpenCV's pedestrian clip isn't a restaurant, so `config/demo_video.yaml` draws three pretend "tables" on a walkway and shortens the timers. People walking through then trigger state changes and alerts:

```bash
curl -L -o data/vtest.avi https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi
python -m tablewatch --config config/demo_video.yaml run       # ~80 s of video; writes outputs/demo_video_annotated.mp4
python -m tablewatch --config config/demo_video.yaml report    # metrics table in the terminal
python -m tablewatch dashboard --db data/demo_video.db
```

Open `outputs/demo_video_annotated.mp4` in VLC. It's MPEG-4, which most browsers won't play. You'll see people boxed as staff or guest, zones coloured by state with their timers, and heads blurred. Walkers get tagged as staff because of the "distance walked" rule.

**C. In the browser, with nothing installed (GitHub Codespaces).** On the repo page, go to **Code → Codespaces → Create codespace on this branch**. The first build installs everything (about 5 minutes) and generates a simulated week of service. The dashboard then opens in a preview tab on port 8501, and step B works in the Codespace terminal too. The zone-drawing window needs a desktop, so draw zones for your own footage on your own machine.

**D. Your own restaurant footage.** Follow steps 1–4 below.

## 1. Get a sample restaurant clip from Pexels

Pexels videos are free to use under the [Pexels licence](https://www.pexels.com/license/).

**Manually (no account needed):**

1. Search Pexels videos, e.g. <https://www.pexels.com/search/videos/restaurant%20waiter/>, or try "cafe customers", "restaurant interior" or "waitress serving".
2. Choose a clip that has:
   - **a static camera.** This is essential, because zones are fixed polygons. Avoid pans, zooms and handheld shots.
   - an elevated or wide angle where several tables and their chairs are visible.
   - people actually sitting down, and staff coming to the tables.
   - ideally a minute or more of footage.
3. Click **Free download**, pick **HD 1280×720** or **Full HD**, and save it as `data/sample.mp4`.

**With the API helper** (free key from <https://www.pexels.com/api/>):

```bash
export PEXELS_API_KEY=your-key
python scripts/download_pexels.py --search "restaurant waiter" --list   # numbered candidates with their pages
python scripts/download_pexels.py --search "restaurant waiter" --pick 3 # saves data/sample.mp4
```

Your own phone recordings work too. Prop the phone up so it doesn't move, and get permission from the venue and the people in view.

Stock clips are usually short (10 to 60 s), much shorter than a real sitting. To see full state cycles on them, shorten the timers, for example `--set state.seat_confirm_s=3 --set state.empty_confirm_s=8 --set alerts.no_first_visit_s=20`.

## 2. Draw the table zones

```bash
python -m tablewatch zones --source data/sample.mp4 --out config/zones.yaml
```

A window shows the first frame (use `--at 12` to pick the frame at 12 s instead).

| Action | Control |
|---|---|
| add a corner | left click |
| undo last corner | right click or `u` |
| finish this table (asks for id and seats) | `Enter` or `n` |
| delete last table | `d` |
| clear current corners | `c` |
| save `zones.yaml` | `s` |
| quit | `q` / `Esc` |

**Draw each polygon over the floor area of the table and its chairs.** A person is assigned to a table by the **bottom-centre of their bounding box**, which for a seated guest lands on the chair or under the table. Keep neighbouring zones from touching, so that a guest near the edge doesn't flicker between tables. Re-running the tool on an existing file lets you add to it. You can also edit `zones.yaml` by hand (see `config/zones.example.yaml`). Zones are rescaled automatically if the camera resolution changes.

## 3. Run

```bash
# Process the clip, write events to SQLite, and save an annotated video for checking
python -m tablewatch run --source data/sample.mp4 --output outputs/annotated.mp4

# Print the metrics for the latest run
python -m tablewatch report

# Live camera: identical pipeline, just a different source
python -m tablewatch run --source "rtsp://user:pass@192.168.1.50:554/stream1"

# Show a window while processing; pace a file like a live camera
python -m tablewatch run --source data/sample.mp4 --show --realtime
```

Useful flags:
- `--set key=value` overrides any config value, e.g. `--set detection.model=yolo11s.pt`.
- `--max-seconds 60` stops after a minute of video.
- `--start-time 2026-10-23T19:30` stamps the recording with its real clock time.

On a laptop CPU, `yolo11n` at the default 10 fps processing rate runs about real time. A GPU (`detection.device: cuda:0`) or Apple Silicon (`mps`) gives headroom for larger models.

The annotated video shows each zone coloured by state, with its timers, and every person boxed as `staff #id` or `guest #id`. The dot on each box marks the anchor point used for zone assignment.

## 4. Dashboard

```bash
python -m tablewatch dashboard          # or: streamlit run tablewatch/dashboard/app.py -- --db data/tablewatch.db
```

- **Live** view follows a running pipeline and refreshes every few seconds. **Replay** view scrubs through any run, rebuilding every table's state at the chosen moment.
- The **table grid** shows each table's state, its main timer (time without staff, or visit in progress), the sitting's duration and visit count, plus any active alerts.
- The **alert feed** lists active alerts first, then resolved ones.
- The **timeline** shows every table's state over time, with alert markers.
- **Service metrics** covers time to first visit, gaps between visits and visits per sitting (one dot per sitting, with the alert threshold marked), a breakdown of each sitting, and where each table's time went (idle vs occupied).
- **Table view** lists every number behind the charts.

The **Analytics** page covers every recorded service, filtered by date range, site and table:
- headline numbers: sittings, average turn time, time to first visit, drinks and food, table utilisation, and the share of sittings that raised an alert
- **day by day**: time from seating to first visit, drinks and food, and average turn time
- **through the evening**: sittings started per hour, and waits by the hour guests sat down
- **by table**: turn time, waits, alerts and utilisation per table
- **every sitting**: one row per sitting with all its timings, downloadable as CSV

The same history is available from the command line: `python -m tablewatch analytics [--site NAME] [--from 2026-10-01] [--to 2026-10-31] [--csv out.csv]`. Sittings after midnight count toward the previous evening's service.

## How table state works

```
EMPTY ──guests present N s──▶ SEATED ──staff arrive──▶ ATTENDED ──staff leave──▶ WAITING
  ▲                                                        ▲                        │
  │                                                        └──────staff arrive──────┘
  └──────────────────── guests gone N s (from any occupied state) ────────────────────┘
```

| State | Meaning |
|---|---|
| EMPTY | no guests |
| SEATED | guests seated, no staff visit yet |
| ATTENDED | staff at the table |
| WAITING | visited at least once, staff currently away |

Each table runs two debounced lifecycles, and the state is derived from them:
- A **sitting** starts after guests have been present for `state.seat_confirm_s` and ends after they've been gone for `state.empty_confirm_s`.
- A **visit** starts after staff have been present for `state.visit_min_s` and ends after they've been away for `state.visit_end_grace_s`.

Start and end times are backdated to when presence really changed, so the debounce doesn't skew the metrics. If someone is only recognised as staff partway through a visit, the visit is backdated to when they reached the table.

What counts as a service visit:
- Staff who escort guests in before they sit don't count as the first visit.
- Staff who clear the table after the guests have left don't count either. Both are logged as visits with no sitting.

| Metric | Definition |
|---|---|
| time to first visit | first visit start − seated time |
| gap between visits | next visit start − previous visit end |
| visits per sitting | service visits during the sitting |
| turn time | guests left − guests seated |
| idle time | time with no sitting (from run start, between sittings, and until now) |
| utilisation | occupied time ÷ (occupied + idle) |
| time to drinks / food | drinks / food first seen on the table − seated time (estimated, see below) |
| order to food | food on the table − first staff visit (roughly order to plate) |
| reset time | first staff visit at the empty table − guests left |

**Drinks and food served** (`served:` in `config.yaml`, on by default) are estimated from object detections inside each table zone:
- drinks: cups, wine glasses and bottles
- food: bowls and the COCO food classes. COCO has no "plate" class, so bowls stand in for it.

When the count in a category rises above its level before service, within `visit_window_s` of a staff visit, the sitting is marked as served. Small objects seen from a ceiling camera are detected unreliably, so use these times for trends rather than for timing a single plate. Raising `detection.image_size` to 960 or 1280 helps. For exact times, wire in your POS: a `drinks_served` or `food_served` event in the `events` table is all the analytics need.

## Staff vs guest classification

`HeuristicRoleClassifier` labels a track as **staff** when any enabled rule fires (see `roles:` in `config.yaml`):

| Rule | Default | Rationale |
|---|---|---|
| lingered at `min_zones_for_staff` different tables | 2 | guests stay at one table |
| standing for `standing_staff_s` in total (box height/width > `standing_aspect_ratio`) | 30 s | guests sit |
| walked `travel_staff_body_heights` in total | 12 | staff cover the room |
| torso colour inside `uniform.hsv_ranges` (opt-in) | off | strongest signal if staff wear a uniform colour |

Everyone else is a guest, and once a track is staff it stays staff. Tune the aspect-ratio threshold to your camera angle; from directly overhead everyone looks squat. To measure a uniform colour, sample a few pixels of the uniform in any image editor and convert to OpenCV HSV (H 0–179, S and V 0–255).

To plug in a trained model, subclass `RoleClassifier`, implement `update(tracks, image, ts)` to set `track.role`, and set `roles.classifier: "my_package.my_module:MyClassifier"`.

## Alerts

| Rule | Config | Fires when |
|---|---|---|
| `no_first_visit` | `alerts.no_first_visit_s: 300` | seated 5 min with no staff visit |
| `visit_gap` | `alerts.visit_gap_s: 600` | 10 min since the last visit ended (once per gap) |
| `long_sitting` | `alerts.long_sitting_s: null` | optional, e.g. 5400 for a 90-min sitting |

Each alert is raised once, and is written to the `alerts` table and the console. If `repeat_every_s` is set, it is re-sent at that interval. It resolves automatically when the condition clears (a visit happens or the table empties).

Notifiers run on a background thread, so a slow webhook never stalls video processing. To enable one, flip `enabled: true` in `config.yaml`:

- `slack`: a Slack incoming-webhook URL.
- `webhook`: POSTs JSON (`event`, `text`, `alert`) to any URL (Zapier, n8n, your API).
- `whatsapp`: a stub that logs messages. Implement `WhatsAppNotifier.send` with Twilio or the Meta Cloud API (instructions are in its docstring).
- `my_package.module:MyNotifier`: your own `Notifier` subclass.

## Configuration

Every setting lives in `config/config.yaml`, with comments and defaults in `tablewatch/config.py`. Unknown keys are rejected, so typos fail loudly. Any value can be overridden per run with `--set section.key=value`.

## Tests

```bash
pytest
```

The tests cover:
- state machine transitions, debouncing, backdating, escort and clear-up handling
- metrics, idle time and replay reconstruction (checked against the live machine)
- alert raising, resolution, repetition and per-gap keys
- notifier isolation, and webhook/Slack payloads against a local HTTP server
- zone geometry, rescaling, the YAML round-trip and the zone editor
- the staff heuristics and uniform colour matching
- drinks/food detection, the analytics rollups (checked end to end against a scripted sitting), and migration of older databases

None of the tests needs a model or a video.

## Data model (SQLite)

All `ts` columns are seconds since the run started; `runs.media_start` maps them to clock time.

| Table | Contents |
|---|---|
| `runs` | one row per processing run: site, source, fps, tables, config |
| `events` | every state change, sitting start/end, visit start/end, drinks and food served |
| `sittings` / `visits` | one row each, with start and end times |
| `alerts` | raised, repeated and resolved alerts |
| `table_status` | latest snapshot per table (what the live dashboard reads) |

Set `site:` in `config.yaml` per camera or venue, so analytics from different cameras with the same table names stay apart. Databases from earlier versions are upgraded in place when the pipeline opens them.

## Known limitations and next steps

- **The staff heuristic is the weakest link.** A waiter's first table visit can be missed until their track has done enough to be classified, though visits are backdated once it is. ID switches in crowded scenes reset a track's history. Enabling the uniform colour rule, or training a small classifier on staff/guest crops from the target venue, is the biggest accuracy win.
- One camera per process. Cover more cameras by running one process per camera, each with its own zones, writing to the same database.
- Occlusion: guests hidden behind staff are tolerated by `empty_confirm_s`. Heavy occlusion from a low camera angle will undercount.
- Drinks and food times are estimates from object detection (see above). A POS integration would make them exact.
- **Licensing:** Ultralytics YOLO is AGPL-3.0. That's fine for a pilot, but a commercial product needs an Ultralytics Enterprise licence or a detector with a permissive licence behind the `Detector` interface.
