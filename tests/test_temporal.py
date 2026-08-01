"""Phase E (D + E): deterministic temporal operators and the IMU-only (non-visual) path.

These run through the real RobotMemory + FakeEmbedder. The fake keys its semantic vector on the
filename concept tokens, so a text query for a segment's concept aligns with that segment's
window -- the same discrimination the Modal demo asserts, proven here GPU-free.
"""

import numpy as np

from robomem import FakeEmbedder, RobotMemory
from robomem.ranking import RankWeights
from robomem.schema import Moment


def session_e():
    """A controlled multimodal timeline with known ground-truth events across modalities:

        t=1  image  calm wall            t=2  audio  quiet room
        t=3  motion calm (low)           t=5  audio  person shout   (#1)
        t=7  image  hallway  (right before the alarm)
        t=9  audio  alarm siren
        t=11 image  open doorway (after the alarm)
        t=13 audio  person shout   (#2)
        t=15 motion violent shake (high)
    """
    return [
        {"id": "img_calm", "t": 1.0, "modality": "image",
         "path_or_data": "cam0/calm_wall.png", "source": "cam0"},
        {"id": "aud_quiet", "t": 2.0, "duration": 1.0, "modality": "audio",
         "path_or_data": "mic0/quiet_room.wav", "source": "mic0"},
        {"id": "mot_calm", "t": 3.0, "duration": 1.0, "modality": "motion",
         "path_or_data": [[0.0, 0.0, 9.8], [0.01, 0.0, 9.8], [0.0, 0.01, 9.8]], "source": "imu0"},
        {"id": "aud_shout1", "t": 5.0, "duration": 1.0, "modality": "audio",
         "path_or_data": "mic0/person_shout.wav", "source": "mic0"},
        {"id": "img_hallway", "t": 7.0, "modality": "image",
         "path_or_data": "cam0/hallway_scene.png", "source": "cam0"},
        {"id": "aud_alarm", "t": 9.0, "duration": 1.0, "modality": "audio",
         "path_or_data": "mic0/alarm_siren.wav", "source": "mic0"},
        {"id": "img_door", "t": 11.0, "modality": "image",
         "path_or_data": "cam0/open_doorway.png", "source": "cam0"},
        {"id": "aud_shout2", "t": 13.0, "duration": 1.0, "modality": "audio",
         "path_or_data": "mic0/person_shout.wav", "source": "mic0"},
        {"id": "mot_shake", "t": 15.0, "duration": 1.0, "modality": "motion",
         "path_or_data": [[8.0, -7.5, 6.0], [-9.0, 8.0, -5.0], [7.0, -8.0, 9.0]], "source": "imu0"},
    ]


def _mem(tmp_path, segment=True):
    mem = RobotMemory.open(str(tmp_path / "sess_e"), embedder=FakeEmbedder())
    mem.index(session_e(), segment=segment)
    return mem


# ------------------------------------------------------------------ count (D)
def test_count_distinct_relevant_shout_episodes(tmp_path):
    mem = _mem(tmp_path)
    # two shouts at t=5 and t=13, non-contiguous -> two distinct episodes -> count 2
    assert mem.count("person shout", modality="audio") == 2
    # the table-size overload is preserved when called with no query
    assert mem.count() == len(session_e())


def test_count_ignores_irrelevant_audio(tmp_path):
    mem = _mem(tmp_path)
    # quiet room + alarm siren are audio but not shouts -> they do not inflate the count
    assert mem.count("alarm siren", modality="audio") == 1


# ------------------------------------------------------------------- last (D)
def test_last_returns_the_most_recent_relevant_episode(tmp_path):
    mem = _mem(tmp_path)
    hit = mem.last("person shout", modality="audio")
    assert hit is not None
    # the more recent of the two equally-relevant shouts (t=13), not the t=5 one
    assert hit.t_start == 13.0
    assert "aud_shout2" in hit.member_ids


def test_last_respects_modality(tmp_path):
    mem = _mem(tmp_path)
    hit = mem.last("shaking violently", modality="motion")
    assert hit is not None and hit.modality == "motion"
    assert all(mid.startswith("mot_") for mid in hit.member_ids)


# ----------------------------------------------------------- before / after (D)
def test_before_finds_the_visual_right_before_the_alarm(tmp_path):
    mem = _mem(tmp_path)
    hit = mem.before(anchor="alarm siren", target=None, modality="image",
                     anchor_modality="audio")
    assert hit is not None
    # alarm is at t=9; the nearest image before it is the hallway at t=7 (not the calm wall at t=1)
    assert hit.t_start == 7.0
    assert "img_hallway" in hit.member_ids
    assert hit.meta["anchor_time"] == 9.0


def test_after_finds_the_visual_following_the_alarm(tmp_path):
    mem = _mem(tmp_path)
    hit = mem.after(anchor="alarm siren", target=None, modality="image",
                    anchor_modality="audio")
    assert hit is not None
    assert hit.t_start == 11.0 and "img_door" in hit.member_ids


# ---------------------------------------------------------------- timeline (D)
def test_timeline_of_shouts_is_time_ordered(tmp_path):
    mem = _mem(tmp_path)
    tl = mem.timeline("person shout", modality="audio")
    assert [m.t_start for m in tl] == [5.0, 13.0]


def test_timeline_indexes_a_modality_without_a_query(tmp_path):
    mem = _mem(tmp_path)
    tl = mem.timeline(modality="image")
    assert [m.t_start for m in tl] == [1.0, 7.0, 11.0]
    # window bound trims the range
    tl2 = mem.timeline(modality="image", window=(6.0, 12.0))
    assert [m.t_start for m in tl2] == [7.0, 11.0]


# ---------------------------------------------------- IMU-only, non-visual (E)
def test_imu_only_query_returns_motion_with_no_video_or_image(tmp_path):
    mem = _mem(tmp_path)
    hits = mem.recall("moving vigorously", modality="motion", k=5)
    assert hits, "no motion hits"
    assert all(h.modality == "motion" for h in hits)
    # not a single visual row leaked into the result
    assert not any(h.modality in ("image", "video") for h in hits)


def test_last_motion_result_carries_no_visual_members(tmp_path):
    mem = _mem(tmp_path)
    hit = mem.last("moving vigorously", modality="motion")
    assert hit is not None and hit.modality == "motion"
    for mid in hit.member_ids:
        row = mem.store.get(mid)
        assert row["modality"] == "motion"


# ---------------------------------------------- three-signal beats naive cosine
def test_three_signal_ranking_beats_naive_on_a_recency_tie(tmp_path):
    # two equally-relevant moments; only recency distinguishes them.
    early = Moment(event_id="e", score=0.8, t_start=5.0, t_end=6.0, modality="audio",
                   meta={"episode_salience": 0.0})
    late = Moment(event_id="l", score=0.8, t_start=13.0, t_end=14.0, modality="audio",
                  meta={"episode_salience": 0.0})
    naive = RobotMemory.rerank(None, [early, late], now=14.0, weights=RankWeights.naive(),
                               halflife=10.0)
    three = RobotMemory.rerank(None, [early, late], now=14.0,
                               weights=RankWeights.three_signal(), halflife=10.0)
    # naive keeps input order on the tie; three-signal surfaces the most recent moment first
    assert [m.event_id for m in naive] == ["e", "l"]
    assert three[0].event_id == "l"


def test_three_signal_ranking_uses_salience(tmp_path):
    dull = Moment(event_id="dull", score=0.8, t_start=10.0, t_end=11.0, modality="motion",
                  meta={"episode_salience": 0.1})
    shake = Moment(event_id="shake", score=0.8, t_start=10.0, t_end=11.0, modality="motion",
                   meta={"episode_salience": 1.0})
    ranked = RobotMemory.rerank(None, [dull, shake], now=11.0,
                                weights=RankWeights(relevance=1.0, recency=0.0, salience=1.0),
                                halflife=10.0)
    assert ranked[0].event_id == "shake"
