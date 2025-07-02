import numpy as np
import pytest
from delayedsw import DelayedSlidingWindow
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator


def test_delayed_sliding_window_basic():
    X = np.array([[1, 2, 3, 4, 5]])
                 
    scaler = DelayedSlidingWindow(window_size=2, delay_space=2)
    X_transformed = scaler.fit_transform(X)
    
    expected = np.array([[3, 1],[4, 2],[5, 3]]).transpose()
    np.testing.assert_allclose(X_transformed, expected, rtol=1e-5)


def test_delayed_sliding_window_multifeatures():
    # TODO: Implement a more complex test with multiple features
    pass


def test_pipeline_compatibility():
    # TODO: Implement a test to check if the transformer works within a sklearn pipeline
    pass


def test_sklearn_compatibility():
    """Run sklearn's built-in check on custom transformer."""
    # This test checks compliance with sklearn's API contracts
    check_estimator(DelayedSlidingWindow)
    pass