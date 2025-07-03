import numpy as np
import pytest
from delayedsw import DelayedSlidingWindow
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator


def test_delayed_sliding_window_basic():
    X = np.array([[1, 2, 3, 4, 5]]).transpose()
                 
    transformer = DelayedSlidingWindow(window_size=2, delay_space=2)
    X_transformed = transformer.fit_transform(X)
    
    expected = np.array([[3, 1],[4, 2],[5, 3]])
    np.testing.assert_allclose(X_transformed, expected, rtol=1e-5)


def test_delayed_sliding_window_multifeatures():
    X = np.array([[1, 2, 3, 4, 5],[10,20,30,40,50],[200,400,600,800,1000]]).transpose()
                 
    transformer = DelayedSlidingWindow(window_size=2, delay_space=2)
    X_transformed = transformer.fit_transform(X)

    expected = np.array([[3, 1, 30, 10, 600, 200],[4, 2, 40, 20, 800, 400],[5, 3, 50, 30, 1000, 600]])
    np.testing.assert_allclose(X_transformed, expected, rtol=1e-5)

def test_delayed_sliding_window_with_columns():
    X = np.array([[1, 2, 3, 4, 5],[10,20,30,40,50],[200,400,600,800,1000]]).transpose()
                 
    transformer = DelayedSlidingWindow(window_size=2, delay_space=2, columns_to_transform=[0, 2])
    X_transformed = transformer.fit_transform(X)

    expected = np.array([[3, 1, 600, 200],[4, 2, 800, 400],[5, 3, 1000, 600]])
    np.testing.assert_allclose(X_transformed, expected, rtol=1e-5)

def test_delayed_sliding_window_with_pandas():
    import pandas as pd
    from sklearn import set_config

    set_config(transform_output='pandas')

    X = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [200, 400, 600, 800, 1000]
    })

    transformer = DelayedSlidingWindow(window_size=2, delay_space=2, columns_to_transform=['B', 'C'])
    X_transformed = transformer.fit_transform(X)

    expected = pd.DataFrame({
        'B_0': [30, 40, 50],
        'B_2': [10, 20, 30],
        'C_0': [600, 800, 1000],
        'C_2': [200, 400, 600]
    })

    pd.testing.assert_frame_equal(X_transformed.reset_index(drop=True), expected.reset_index(drop=True))


def test_pipeline_compatibility():
    X = np.array([[1, 2, 3, 4, 5]]).transpose()
    transformer = DelayedSlidingWindow(window_size=2, delay_space=2)
    
    pipeline = Pipeline([
        ('delayed_sliding_window', transformer)
    ])

    X_transformed = pipeline.fit_transform(X)
    expected = np.array([[3, 1], [4, 2], [5, 3]])
    np.testing.assert_allclose(X_transformed, expected, rtol=1e-5)


def test_sklearn_compatibility():
    """Run sklearn's built-in check on custom transformer."""
    # This test checks compliance with sklearn's API contracts
    check_estimator(DelayedSlidingWindow)
    pass