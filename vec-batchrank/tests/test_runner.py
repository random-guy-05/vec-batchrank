import sys
import types

from vec_batchrank.runner import score_one


def test_score_one_and_cache(tmp_path, monkeypatch):
    calls = []
    fake = types.ModuleType("veckit")
    def score(**kwargs):
        calls.append(kwargs)
        return {"meta": {"task": kwargs["task"]}, "metrics": {"de_score": 0.5}}
    fake.score = score
    monkeypatch.setitem(sys.modules, "veckit", fake)

    pred = tmp_path / "pred.h5ad"; pred.write_bytes(b"p")
    target = tmp_path / "target.h5ad"; target.write_bytes(b"t")
    ref = tmp_path / "ref.h5ad"; ref.write_bytes(b"r")
    cache = tmp_path / "cache"

    a = score_one(candidate=pred, task="T1", target=target, reference=ref, wt=None,
                  setting="heart", seed=1, allow_reorder=False, cache_dir=cache)
    b = score_one(candidate=pred, task="T1", target=target, reference=ref, wt=None,
                  setting="heart", seed=1, allow_reorder=False, cache_dir=cache)
    assert a["_cache_hit"] is False
    assert b["_cache_hit"] is True
    assert len(calls) == 1
