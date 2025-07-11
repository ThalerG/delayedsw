"""Tests for MovingAverageTransformer."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator

from delayedsw import MovingAverageTransformer


def test_basic_moving_average():
    """Test basic moving average functionality."""
    X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
    
    transformer = MovingAverageTransformer(window=3, min_periods=1)
    result = transformer.fit_transform(X)
    
    # First column: [1, 1.5, 2, 3, 4]
    # Second column: [10, 15, 20, 30, 40]
    expected = np.array([
        [1.0, 10.0],
        [1.5, 15.0], 
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0]
    ])
    
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_moving_average_with_min_periods():
    """Test moving average with min_periods."""
    X = np.array([[1], [2], [3], [4], [5]])
    
    transformer = MovingAverageTransformer(window=3, min_periods=3)
    result = transformer.fit_transform(X)
    
    # First two values should be NaN, then [2, 3, 4]
    expected = np.array([[np.nan], [np.nan], [2.0], [3.0], [4.0]])
    
    assert np.isnan(result[0, 0]) and np.isnan(result[1, 0])
    np.testing.assert_allclose(result[2:], expected[2:], rtol=1e-10)


def test_centered_moving_average():
    """Test centered moving average."""
    X = np.array([[1], [2], [3], [4], [5]])
    
    transformer = MovingAverageTransformer(window=3, center=True, min_periods=1)
    result = transformer.fit_transform(X)
    
    # Centered window: [1.5, 2, 3, 4, 4.5]
    expected = np.array([[1.5], [2.0], [3.0], [4.0], [4.5]])
    
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_pandas_dataframe():
    """Test with pandas DataFrame."""
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10.0, 20.0, 30.0, 40.0, 50.0],
        'C': ['x', 'y', 'z', 'w', 'v']  # Non-numeric column
    })
    
    transformer = MovingAverageTransformer(window=3, min_periods=1)
    result = transformer.fit_transform(df)
    
    # Should only transform numeric columns A and B
    expected = pd.DataFrame({
        'A': [1.0, 1.5, 2.0, 3.0, 4.0],
        'B': [10.0, 15.0, 20.0, 30.0, 40.0], 
        'C': ['x', 'y', 'z', 'w', 'v']  # Unchanged
    })
    
    pd.testing.assert_frame_equal(result, expected)


def test_specific_columns():
    """Test transforming only specific columns."""
    df = pd.DataFrame({
        'sensor1': [1, 2, 3, 4, 5],
        'sensor2': [10, 20, 30, 40, 50],
        'metadata': [100, 200, 300, 400, 500]
    })
    
    transformer = MovingAverageTransformer(
        window=3, 
        min_periods=1,
        columns_to_transform=['sensor1', 'sensor2']
    )
    result = transformer.fit_transform(df)
    
    expected = pd.DataFrame({
        'sensor1': [1.0, 1.5, 2.0, 3.0, 4.0],
        'sensor2': [10.0, 15.0, 20.0, 30.0, 40.0],
        'metadata': [100, 200, 300, 400, 500]  # Unchanged
    })
    
    pd.testing.assert_frame_equal(result, expected)


def test_with_ordering():
    """Test moving average with ordering."""
    df = pd.DataFrame({
        'value': [5, 1, 4, 2, 3],
        'timestamp': [5, 1, 4, 2, 3]
    })
    
    transformer = MovingAverageTransformer(
        window=3,
        min_periods=1, 
        order_by='timestamp',
        columns_to_transform=['value']
    )
    result = transformer.fit_transform(df)
    
    # After sorting by timestamp: values become [1, 2, 3, 4, 5]
    # Moving averages: [1, 1.5, 2, 3, 4]
    expected_values = [1.0, 1.5, 2.0, 3.0, 4.0]
    
    # Sort result by timestamp to check
    result_sorted = result.sort_values('timestamp')
    np.testing.assert_allclose(result_sorted['value'].values, expected_values, rtol=1e-10)


def test_with_grouping():
    """Test moving average with grouping."""
    df = pd.DataFrame({
        'value': [1, 2, 3, 1, 2, 3],
        'group': ['A', 'A', 'A', 'B', 'B', 'B']
    })
    
    transformer = MovingAverageTransformer(
        window=2,
        min_periods=1,
        split_by='group',
        columns_to_transform=['value']
    )
    result = transformer.fit_transform(df)
    
    # Group A: [1, 1.5, 2.5]
    # Group B: [1, 1.5, 2.5] 
    expected = pd.DataFrame({
        'value': [1.0, 1.5, 2.5, 1.0, 1.5, 2.5],
        'group': ['A', 'A', 'A', 'B', 'B', 'B']
    })
    
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_with_grouping_and_ordering():
    """Test moving average with both grouping and ordering."""
    df = pd.DataFrame({
        'value': [3, 1, 2, 3, 1, 2],
        'time': [3, 1, 2, 3, 1, 2],
        'group': ['A', 'A', 'A', 'B', 'B', 'B']
    })
    
    transformer = MovingAverageTransformer(
        window=2,
        min_periods=1,
        split_by='group',
        order_by='time',
        columns_to_transform=['value']
    )
    result = transformer.fit_transform(df)
    
    # Within each group, sort by time then apply moving average
    # Group A sorted: [1, 2, 3] -> MA: [1, 1.5, 2.5]
    # Group B sorted: [1, 2, 3] -> MA: [1, 1.5, 2.5]
    
    # Check by group
    group_a = result[result['group'] == 'A'].sort_values('time')
    group_b = result[result['group'] == 'B'].sort_values('time')
    
    expected_ma = [1.0, 1.5, 2.5]
    np.testing.assert_allclose(group_a['value'].values, expected_ma, rtol=1e-10)
    np.testing.assert_allclose(group_b['value'].values, expected_ma, rtol=1e-10)


def test_numpy_array_with_indices():
    """Test numpy array with column indices."""
    X = np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]])
    
    transformer = MovingAverageTransformer(
        window=2,
        min_periods=1,
        columns_to_transform=[0, 2]  # Transform first and third columns
    )
    result = transformer.fit_transform(X)
    
    # Column 0: [1, 1.5, 2.5, 3.5]
    # Column 1: [10, 20, 30, 40] (unchanged)
    # Column 2: [100, 150, 250, 350]
    expected = np.array([
        [1.0, 10, 100.0],
        [1.5, 20, 150.0],
        [2.5, 30, 250.0],
        [3.5, 40, 350.0]
    ])
    
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_pipeline_compatibility():
    """Test compatibility with sklearn pipelines."""
    X = np.array([[1], [2], [3], [4], [5]])
    
    pipeline = Pipeline([
        ('moving_avg', MovingAverageTransformer(window=3, min_periods=1))
    ])
    
    result = pipeline.fit_transform(X)
    expected = np.array([[1.0], [1.5], [2.0], [3.0], [4.0]])
    
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_sklearn_compatibility():
    """Test sklearn estimator compatibility."""
    check_estimator(MovingAverageTransformer())


def test_feature_names_out():
    """Test feature names output."""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    
    transformer = MovingAverageTransformer()
    transformer.fit(df)
    
    names = transformer.get_feature_names_out()
    expected = ['A', 'B']
    
    assert list(names) == expected