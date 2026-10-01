import sys
import types

from vec_batchrank.runner import score_one


def test_score_one_and_cache(tmp_path, monkeypatch):
    calls = []
    fake = types.ModuleType("veckit")

    def score(**kwargs):
        calls.append(kwargs)
        return {
            "meta": {"task": kwargs["task"]},
            "metrics": {"de_score": 0.5},
        }

    fake.score = score
    monkeypatch.setitem(sys.modules, "veckit", fake)

    pred = tmp_path / "pred.h5ad"
    target = tmp_path / "target.h5ad"
    reference = tmp_path / "ref.h5ad"
    pred.write_bytes(b"p")
    target.write_bytes(b"t")
    reference.write_bytes(b"r")
    cache = tmp_path / "cache"

    first = score_one(
        candidate=pred,
        task="T1",
        target=target,
        reference=reference,
        wt=None,
        setting="heart",
        seed=1,
        allow_reorder=False,
        cache_dir=cache,
    )
    second = score_one(
        candidate=pred,
        task="T1",
        target=target,
        reference=reference,
        wt=None,
        setting="heart",
        seed=1,
        allow_reorder=False,
        cache_dir=cache,
    )

    assert first["_cache_hit"] is False
    assert second["_cache_hit"] is True
    assert len(calls) == 1
