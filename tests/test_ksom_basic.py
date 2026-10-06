import numpy as np
import pandas as pd
import pytest

from ksom.ksom_basic import BasicKSOM


@pytest.fixture
def X():
    return np.array(
        [
            [1, 0, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=float,
    )


@pytest.fixture
def W():
    return np.array(
        [
            [0.3, 0.5, 0.7, 0.2],
            [0.6, 0.5, 0.4, 0.2],
        ],
        dtype=float,
    )


def test_fit_weights_shape(X, W):
    model = BasicKSOM(W, learning_rate=0.1).fit(X, verbose=False)
    assert model.weights_.shape == (2, 4)


def test_history_recorded(X, W):
    model = BasicKSOM(W, learning_rate=0.1).fit(X, verbose=False)
    # one history row per input per iteration
    assert len(model.history_) == 4 * model.iterations_
    row = model.history_[0]
    for key in ("Iteration", "Input", "Winner", "Weight_Change", "Distance_C1"):
        assert key in row


def test_weights_move_toward_data(X, W):
    model = BasicKSOM(W, learning_rate=0.1).fit(X, verbose=False)
    assert not np.allclose(model.weights_, W)


def test_predict_clusters(X, W):
    model = BasicKSOM(W, learning_rate=0.1).fit(X, verbose=False)
    df = model.predict(X)
    assert len(df) == 4
    assert set(df["Cluster"]).issubset({"y1", "y2"})


def test_unfitted_raises(X, W):
    model = BasicKSOM(W)
    with pytest.raises(RuntimeError):
        model.predict(X)


def test_invalid_params(W):
    with pytest.raises(ValueError):
        BasicKSOM(W, learning_rate=0)
    with pytest.raises(ValueError):
        BasicKSOM(W, epsilon=0)
    with pytest.raises(ValueError):
        BasicKSOM(W, max_iterations=0)


def test_save_excel(X, W, tmp_path):
    model = BasicKSOM(W, learning_rate=0.1, max_iterations=5).fit(
        X, verbose=False
    )
    path = model.save_excel(X, str(tmp_path / "out.xlsx"))
    xl = pd.ExcelFile(path)
    assert set(xl.sheet_names) == {
        "Input Data",
        "Iterations",
        "Final Weights",
        "Clusters",
    }
