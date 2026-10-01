import math

import pandas as pd

from vec_batchrank.ranking import aggregate_runs, metric_skill, official_score, pareto_front


def test_metric_skill_floor_and_ceiling():
    assert math.isclose(metric_skill(0.0, 0.0, 1.0, "higher"), 0.5)
    assert math.isclose(metric_skill(1.0, 0.0, 1.0, "higher"), 1.0)
    assert metric_skill(-10.0, 0.0, 1.0, "higher") < 0.1


def test_zero_metric_uses_absolute_value():
    a = metric_skill(0.2, -1.0, 0.0, "zero")
    b = metric_skill(-0.2, -1.0, 0.0, "zero")
    assert math.isclose(a, b)


def test_official_t1_floor_is_50():
    score, skills = official_score(
        {"de_score": 0.0, "de_direction": 0.0, "mmd_u": 0.08359, "variogram": 0.005219},
        "T1:val",
    )
    assert math.isclose(score, 50.0, rel_tol=1e-9)
    assert all(math.isclose(v, 0.5) for v in skills.values())


def test_official_t1_ceiling_is_100():
    score, _ = official_score(
        {"de_score": 0.8464, "de_direction": 0.7901, "mmd_u": 0.00406, "variogram": 0.000158},
        "T1:val",
    )
    assert math.isclose(score, 100.0, rel_tol=1e-9)


def test_pareto():
    df = pd.DataFrame(
        [
            {"candidate": "a", "de_score": 0.8, "de_direction": 0.8, "mmd_u": 0.02, "variogram": 0.002},
            {"candidate": "b", "de_score": 0.7, "de_direction": 0.7, "mmd_u": 0.03, "variogram": 0.003},
            {"candidate": "c", "de_score": 0.9, "de_direction": 0.6, "mmd_u": 0.01, "variogram": 0.001},
        ]
    ).set_index("candidate")
    p = pareto_front(df, "T1")
    assert p["a"]
    assert not p["b"]
    assert p["c"]


def test_aggregate_relative_ranking():
    rows = []
    for seed in [0, 1]:
        rows += [
            {"candidate": "good", "path": "/g", "seed": seed, "de_score": .8, "de_direction": .8, "mmd_u": .01, "variogram": .001},
            {"candidate": "bad", "path": "/b", "seed": seed, "de_score": .2, "de_direction": .2, "mmd_u": .08, "variogram": .01},
        ]
    s = aggregate_runs(pd.DataFrame(rows), "T1", None, 1.0)
    assert s.iloc[0]["candidate"] == "good"
    assert s.iloc[0]["rank_mean"] == 1


def test_all_published_boards_map_floor_to_50_ceiling_to_100():
    from vec_batchrank.ranking import BOARD_SPECS, TASK_SPECS
    for board, spec in BOARD_SPECS.items():
        floor = {m: a[0] for m, a in spec.anchors.items()}
        ceiling = {m: a[1] for m, a in spec.anchors.items()}
        sf, _ = official_score(floor, board)
        sc, _ = official_score(ceiling, board)
        assert math.isclose(sf, 50.0, abs_tol=1e-9)
        assert math.isclose(sc, 100.0, abs_tol=1e-9)
        assert math.isclose(sum(m.weight for m in TASK_SPECS[spec.task].values()), 1.0)
