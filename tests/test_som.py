import numpy as np
import pytest

from ksom import SOM


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return rng.normal(size=(40, 3))


def test_fit_reduces_error(data):
    som = SOM(rows=4, cols=4, random_state=0).fit(data, epochs=15, verbose=False)
    errs = som.history_.quantization_error
    assert len(errs) == 15
    assert errs[-1] < errs[0]


def test_transform_shape(data):
    som = SOM(rows=3, cols=5, random_state=1).fit(data, epochs=3, verbose=False)
    coords = som.transform(data)
    assert coords.shape == (40, 2)
    assert (coords[:, 0] < 3).all() and (coords[:, 1] < 5).all()


def test_unfitted_raises(data):
    som = SOM(rows=2, cols=2)
    with pytest.raises(RuntimeError):
        som.transform(data)


def test_invalid_params():
    with pytest.raises(ValueError):
        SOM(rows=0, cols=3)
    with pytest.raises(ValueError):
        SOM(learning_rate=0)
    with pytest.raises(ValueError):
        SOM(neighborhood="weird")
