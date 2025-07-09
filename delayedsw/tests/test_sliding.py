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

    X = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10.0, 20.0, 30.0, 40.0, 50.0],
        'C': [200, 400, 600, 800, 1000]
    })

    transformer = DelayedSlidingWindow(window_size=2, delay_space=2, columns_to_transform=['B', 'C'])
    X_transformed = transformer.fit_transform(X)

    expected = pd.DataFrame({
        'B_0': [30.0, 40.0, 50.0],
        'B_2': [10.0, 20.0, 30.0],
        'C_0': [600, 800, 1000],
        'C_2': [200, 400, 600]
    })

    pd.testing.assert_frame_equal(X_transformed.reset_index(drop=True), expected.reset_index(drop=True))


def test_delayed_sliding_window_with_pandas_ordered():
    import pandas as pd

    X = pd.DataFrame({
        'A': [2, 4, 1, 3, 5],
        'B': [20.0, 40.0, 10.0, 30.0, 50.0],
        'C': [400, 800, 200, 600, 1000],
        'order': [2, 4, 1, 3, 5]
    })
    order = X['order'].values
    X = X.drop(columns=['order'])  # Drop order column for transformation

    transformer = DelayedSlidingWindow(window_size=2, delay_space=2, 
                                       columns_to_transform=['B', 'C'], 
                                       order_by='order', include_order=False)
    X_transformed = transformer.fit_transform(X)

    # When ordered by 'order' column [2,4,1,3,5], the sorted sequence becomes:
    # order: [1,2,3,4,5] -> indices: [2,0,3,1,4]
    # B values in order: [10,20,30,40,50]
    # C values in order: [200,400,600,800,1000]
    # With window_size=2, delay_space=2, we get values at positions 0 and 2:
    expected = pd.DataFrame({
        'B_0': [30.0, 40.0, 50.0],  # B values at current position (indices 2,3,4 in ordered sequence)
        'B_2': [10.0, 20.0, 30.0], # B values at lag 2 (indices 0,1,2 in ordered sequence)
        'C_0': [600, 800, 1000],   # C values at current position
        'C_2': [200, 400, 600]     # C values at lag 2
    })

    pd.testing.assert_frame_equal(X_transformed.reset_index(drop=True), expected.reset_index(drop=True))


def test_delayed_sliding_window_with_pandas_split():
    import pandas as pd

    X = pd.DataFrame({
        'A': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
        'B': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
        'C': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000],
        'split1': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B'],
        'split2': [1,1,1,1,1,2,2,2,2,1,1,1,1,1,1]  # Fixed syntax error (missing comma)
    })

    X['split1'] = X['split1'].astype('category')

    transformer = DelayedSlidingWindow(window_size=2, delay_space=2, columns_to_transform=['B', 'C'], split_by=['split1','split2'])
    X_transformed = transformer.fit_transform(X)

    # When split by 'split1' and 'split2', we get separate groups:
    # Group A-1: indices [0,1,2,3,4] with order [1,2,3,4,5] -> B:[10,20,30,40,50], C:[200,400,600,800,1000]
    # Group A-2: indices [5,6,7,8] with order [1,2,3,4] -> B:[60,70,80,90], C:[1200,1400,1600,1800]
    # Group B-1: indices [9,10,11,12,13,14] with order [1,2,3,4,5,6] -> B:[100,110,120,130,140,150], C:[2000,2200,2400,2600,2800,3000]
    
    # With window_size=2, delay_space=2, each group contributes valid rows:
    expected = pd.DataFrame({
        'B_0': [30.0, 40.0, 50.0, 80.0, 90.0, 120.0, 130.0, 140.0, 150.0],  # Current values (lag 0)
        'B_2': [10.0, 20.0, 30.0, 60.0, 70.0, 100.0, 110.0, 120.0, 130.0], # Values at lag 2
        'C_0': [600, 800, 1000, 1600, 1800, 2400, 2600, 2800, 3000],       # Current C values
        'C_2': [200, 400, 600, 1200, 1400, 2000, 2200, 2400, 2600]         # C values at lag 2
    })

    pd.testing.assert_frame_equal(X_transformed.reset_index(drop=True), expected.reset_index(drop=True))


def test_delayed_sliding_window_with_pandas_ordered_split():
    import pandas as pd

    X = pd.DataFrame({
        'A': [8, 1, 12, 5, 14, 3, 9, 15, 7, 2, 11, 4, 6, 10, 13],
        'B': [80.0, 10.0, 120.0, 50.0, 140.0, 30.0, 90.0, 150.0, 70.0, 20.0, 110.0, 40.0, 60.0, 100.0, 130.0],
        'C': [1600, 200, 2400, 1000, 2800, 600, 1800, 3000, 1400, 400, 2200, 800, 1200, 2000, 2600],
        'order': [3, 1, 3, 5, 5, 3, 4, 6, 2, 2, 2, 4, 1, 1, 4],
        'split1': ['A', 'A', 'B', 'A', 'B', 'A', 'A', 'B', 'A', 'A', 'B', 'A', 'A', 'B', 'B'],
        'split2': [1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1]
    })

    X['split1'] = X['split1'].astype('category')

    transformer = DelayedSlidingWindow(window_size=2, delay_space=2, 
                                       columns_to_transform=['B', 'C'], 
                                       order_by='order', include_order=False, 
                                       split_by=['split1','split2'], include_split=False)
    X_transformed = transformer.fit_transform(X)

    # When split by 'split1' and 'split2', we get separate groups:
    # Group A-1: indices [0,1,2,3,4] with order [1,2,3,4,5] -> B:[10,20,30,40,50], C:[200,400,600,800,1000]
    # Group A-2: indices [5,6,7,8] with order [1,2,3,4] -> B:[60,70,80,90], C:[1200,1400,1600,1800]
    # Group B-1: indices [9,10,11,12,13,14] with order [1,2,3,4,5,6] -> B:[100,110,120,130,140,150], C:[2000,2200,2400,2600,2800,3000]
    
    # With window_size=2, delay_space=2, each group contributes valid rows:
    expected = pd.DataFrame({
        'B_0': [30.0, 40.0, 50.0, 80.0, 90.0, 120.0, 130.0, 140.0, 150.0],  # Current values (lag 0)
        'B_2': [10.0, 20.0, 30.0, 60.0, 70.0, 100.0, 110.0, 120.0, 130.0], # Values at lag 2
        'C_0': [600, 800, 1000, 1600, 1800, 2400, 2600, 2800, 3000],       # Current C values
        'C_2': [200, 400, 600, 1200, 1400, 2000, 2200, 2400, 2600]         # C values at lag 2
    })

    pd.testing.assert_frame_equal(X_transformed.reset_index(drop=True), expected.reset_index(drop=True))

def test_delayed_sliding_window_with_pandas_ordered_split_nodrop():
    import pandas as pd

    X = pd.DataFrame({
        'A': [8, 1, 12, 5, 14, 3, 9, 15, 7, 2, 11, 4, 6, 10, 13],
        'B': [80.0, 10.0, 120.0, 50.0, 140.0, 30.0, 90.0, 150.0, 70.0, 20.0, 110.0, 40.0, 60.0, 100.0, 130.0],
        'C': [1600, 200, 2400, 1000, 2800, 600, 1800, 3000, 1400, 400, 2200, 800, 1200, 2000, 2600],
        'order': [3, 1, 3, 5, 5, 3, 4, 6, 2, 2, 2, 4, 1, 1, 4],
        'split1': ['A', 'A', 'B', 'A', 'B', 'A', 'A', 'B', 'A', 'A', 'B', 'A', 'A', 'B', 'B'],
        'split2': [1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1]
    })

    X['split1'] = X['split1'].astype('category')

    transformer = DelayedSlidingWindow(window_size=2, delay_space=2, 
                                       columns_to_transform=['B', 'C'], 
                                       order_by='order', include_order=True, 
                                       split_by=['split1','split2'], include_split=True)
    X_transformed = transformer.fit_transform(X)

    # When split by 'split1' and 'split2', we get separate groups:
    # Group A-1: indices [0,1,2,3,4] with order [1,2,3,4,5] -> B:[10,20,30,40,50], C:[200,400,600,800,1000]
    # Group A-2: indices [5,6,7,8] with order [1,2,3,4] -> B:[60,70,80,90], C:[1200,1400,1600,1800]
    # Group B-1: indices [9,10,11,12,13,14] with order [1,2,3,4,5,6] -> B:[100,110,120,130,140,150], C:[2000,2200,2400,2600,2800,3000]
    
    # With window_size=2, delay_space=2, each group contributes valid rows:
    expected = pd.DataFrame({
        'B_0': [30.0, 40.0, 50.0, 80.0, 90.0, 120.0, 130.0, 140.0, 150.0],  # Current values (lag 0)
        'B_2': [10.0, 20.0, 30.0, 60.0, 70.0, 100.0, 110.0, 120.0, 130.0], # Values at lag 2
        'C_0': [600, 800, 1000, 1600, 1800, 2400, 2600, 2800, 3000],       # Current C values
        'C_2': [200, 400, 600, 1200, 1400, 2000, 2200, 2400, 2600],         # C values at lag 2
        'order': [3, 4, 5, 3, 4, 3, 4, 5, 6],  # Keep order column
        'split1': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],  # Keep split1 column
        'split2': [1, 1, 1, 2, 2, 1, 1, 1, 1]  # Keep split2 column
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
    check_estimator(DelayedSlidingWindow())