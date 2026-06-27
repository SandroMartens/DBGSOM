import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


@st.composite
def growable_dataset(draw):
    """Small dataset + seed, sized to trigger growth within a few epochs."""
    n_features = draw(st.integers(2, 5))
    n_samples = draw(st.integers(20, 60))
    X = draw(
        arrays(
            np.float64,
            (n_samples, n_features),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        )
    )
    seed = draw(st.integers(0, 10_000))
    return X, seed
