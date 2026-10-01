from vec_batchrank.io import candidate_labels, discover_predictions


def test_discover_predictions(tmp_path):
    (tmp_path / "a.h5ad").write_bytes(b"x")
    (tmp_path / "b.h5ad").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_bytes(b"x")
    out = discover_predictions([str(tmp_path)])
    assert [p.name for p in out] == ["a.h5ad", "b.h5ad"]


def test_candidate_labels_disambiguate_same_basename(tmp_path):
    a = tmp_path / "run_a"; b = tmp_path / "run_b"
    a.mkdir(); b.mkdir()
    pa = (a / "prediction.h5ad").resolve(); pb = (b / "prediction.h5ad").resolve()
    pa.write_bytes(b"x"); pb.write_bytes(b"x")
    labels = candidate_labels([pa, pb])
    assert labels[pa] == "run_a/prediction.h5ad"
    assert labels[pb] == "run_b/prediction.h5ad"
