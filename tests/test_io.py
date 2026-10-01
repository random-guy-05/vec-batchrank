from vec_batchrank.io import candidate_labels, discover_predictions


def test_discover_predictions(tmp_path):
    (tmp_path / "a.h5ad").write_bytes(b"x")
    (tmp_path / "b.h5ad").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_bytes(b"x")
    out = discover_predictions([str(tmp_path)])
    assert [path.name for path in out] == ["a.h5ad", "b.h5ad"]


def test_candidate_labels_disambiguate_same_basename(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()

    path_a = (run_a / "prediction.h5ad").resolve()
    path_b = (run_b / "prediction.h5ad").resolve()
    path_a.write_bytes(b"x")
    path_b.write_bytes(b"x")

    labels = candidate_labels([path_a, path_b])
    assert labels[path_a] == "run_a/prediction.h5ad"
    assert labels[path_b] == "run_b/prediction.h5ad"
