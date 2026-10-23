import numpy as np
import pytest

from tablewatch.config import RoleConfig, UniformConfig
from tablewatch.core import Role, Track
from tablewatch.classification.roles import HeuristicRoleClassifier, build_role_classifier, uniform_fraction
from tablewatch.zones.assigner import ZoneAssigner
from tablewatch.zones.setup_tool import ZoneEditor
from tablewatch.zones.zones import TableZone, ZoneMap, load_zones, save_zones

SQUARE = [(0, 0), (100, 0), (100, 100), (0, 100)]


def zone_map() -> ZoneMap:
    return ZoneMap(
        [TableZone("T1", SQUARE, 4), TableZone("T2", [(200, 0), (300, 0), (300, 100), (200, 100)], 2)],
        frame_size=(400, 200),
    )


def person(tid: int, x: float, y: float, w: float = 40, h: float = 60) -> Track:
    """A box whose bottom-centre (the anchor) is at (x, y)."""
    return Track(tid, (x - w / 2, y - h, x + w / 2, y), 0.9)


def test_point_in_polygon():
    z = TableZone("T1", SQUARE)
    assert z.contains((50, 50))
    assert not z.contains((150, 50))
    assert not z.contains((50, -1))


def test_assign_and_overlap_prefers_nearest_centroid():
    zm = ZoneMap([TableZone("A", [(0, 0), (100, 0), (100, 100), (0, 100)]), TableZone("B", [(80, 0), (200, 0), (200, 100), (80, 100)])])
    assert zm.assign((10, 50)) == "A"
    assert zm.assign((150, 50)) == "B"
    assert zm.assign((98, 50)) == "B"  # in both; 42 px from B's centroid (140, 50), 48 px from A's (50, 50)
    assert zm.assign((500, 500)) is None


def test_rescale_to_new_resolution():
    zm = zone_map().for_frame_size(800, 400)
    assert zm.get("T1").polygon[2] == (200.0, 200.0)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        ZoneMap([TableZone("T1", SQUARE), TableZone("T1", SQUARE)])


def test_zones_yaml_round_trip(tmp_path):
    path = tmp_path / "zones.yaml"
    save_zones(zone_map(), path)
    loaded = load_zones(path)
    assert loaded.table_ids == ["T1", "T2"]
    assert loaded.frame_size == (400, 200)
    assert loaded.get("T2").seats == 2


def test_zone_editor():
    ed = ZoneEditor(default_seats=6)
    for p in [(0, 0), (10, 0), (10, 10)]:
        ed.add_point(*p)
    ed.undo_point()
    assert not ed.can_finish()
    ed.add_point(5, 10)
    zone = ed.finish()
    assert zone.table_id == "T1" and zone.seats == 6
    ed.current = [(0, 0), (1, 0), (1, 1)]
    with pytest.raises(ValueError):
        ed.finish("T1")
    assert ed.finish("Window", 2).table_id == "Window"


def test_assigner_grace_period_and_entry_time():
    za = ZoneAssigner(zone_map(), exit_grace_s=1.0)
    t = person(1, 50, 50)
    za.update([t], 0.0)
    assert (t.zone_id, t.zone_entered_ts) == ("T1", 0.0)
    jitter = person(1, 105, 50)  # just outside the polygon
    za.update([jitter], 0.5)
    assert jitter.zone_id == "T1"  # within grace
    za.update([person(1, 105, 50)], 2.0)
    gone = person(1, 105, 50)
    za.update([gone], 2.1)
    assert gone.zone_id is None


def _walk(clf, tid, points, t0=0.0, dt=0.5, h=60):
    tracks = []
    for i, (x, y, zone) in enumerate(points):
        tr = person(tid, x, y, h=h)
        tr.zone_id = zone
        clf.update([tr], None, t0 + i * dt)
        tracks.append(tr)
    return tracks[-1]


def test_seated_guest_stays_guest():
    clf = HeuristicRoleClassifier(RoleConfig())
    # Squat box (aspect 1.0) with small jitter inside one zone for 10 minutes.
    pts = [(50 + (i % 3), 50 + (i % 2), "T1") for i in range(1200)]
    last = _walk(clf, 1, pts, h=40)
    assert last.role == Role.GUEST


def test_multi_zone_visitor_is_staff():
    clf = HeuristicRoleClassifier(RoleConfig(travel_staff_body_heights=None, standing_staff_s=None))
    pts = [(50, 50, "T1")] * 6 + [(250, 50, "T2")] * 6
    last = _walk(clf, 7, pts)
    assert last.role == Role.STAFF and last.role_reason == "multi-zone"


def test_long_standing_is_staff():
    clf = HeuristicRoleClassifier(RoleConfig(standing_staff_s=10, min_zones_for_staff=None, travel_staff_body_heights=None))
    last = _walk(clf, 3, [(50, 50, "T1")] * 30, h=120)  # tall narrow box, 15 s
    assert last.role == Role.STAFF and last.role_reason == "standing"


def test_walking_far_is_staff_but_jitter_is_not():
    clf = HeuristicRoleClassifier(RoleConfig(min_zones_for_staff=None, standing_staff_s=None, travel_staff_body_heights=5))
    walker = _walk(clf, 4, [(i * 30.0, 150, None) for i in range(20)])  # 30 px steps, 60 px tall
    assert walker.role == Role.STAFF and walker.role_reason == "travel"


def test_staff_is_sticky():
    clf = HeuristicRoleClassifier(RoleConfig(travel_staff_body_heights=None, standing_staff_s=None))
    _walk(clf, 9, [(50, 50, "T1")] * 6 + [(250, 50, "T2")] * 6)
    later = _walk(clf, 9, [(250, 50, "T2")] * 4, t0=10)
    assert later.role == Role.STAFF


def test_uniform_colour_rule():
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    image[0:200, 0:100] = (20, 20, 20)  # left half black (uniform), right half white
    black = (10.0, 0.0, 90.0, 200.0)
    white = (110.0, 0.0, 190.0, 200.0)
    ranges = [[[0, 0, 0], [179, 255, 60]]]
    assert uniform_fraction(image, black, ranges) == pytest.approx(1.0)
    assert uniform_fraction(image, white, ranges) == pytest.approx(0.0)

    cfg = RoleConfig(
        min_zones_for_staff=None, standing_staff_s=None, travel_staff_body_heights=None,
        uniform=UniformConfig(enabled=True, min_observations=3),
    )
    clf = HeuristicRoleClassifier(cfg)
    for i in range(4):
        a, b = Track(1, black, 0.9), Track(2, white, 0.9)
        clf.update([a, b], image, i * 0.5)
    assert a.role == Role.STAFF and a.role_reason == "uniform"
    assert b.role == Role.GUEST


def test_custom_classifier_by_import_path():
    clf = build_role_classifier(RoleConfig(classifier="tablewatch.classification.roles:HeuristicRoleClassifier"))
    assert isinstance(clf, HeuristicRoleClassifier)
    with pytest.raises(ValueError):
        build_role_classifier(RoleConfig(classifier="nonsense"))
