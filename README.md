# delayedsw

A Python library for preprocessing time series data using delayed space sliding window transformations. The `delayedsw` package provides a scikit-learn compatible transformer that creates lagged features from time series data with ordering and splitting capabilities.

## Installation

```bash
pip install git+https://github.com/ThalerG/delayedsw
```

## Requirements

- Python >= 3.10
- scikit-learn >= 1.6.1
- pandas (optional, for DataFrame support)

## Quick Start

```python
import numpy as np
from delayedsw import DelayedSlidingWindow

# Create sample time series data
X = np.array([[1, 2, 3, 4, 5]]).transpose()

# Initialize transformer
transformer = DelayedSlidingWindow(window_size=2, delay_space=2)

# Fit and transform
X_transformed = transformer.fit_transform(X)
print(X_transformed)
# Output: [[3, 1], [4, 2], [5, 3]]
```

## Features

### Core Functionality
- **Delayed Space Sliding Window**: Creates lagged features with configurable window size and delay spacing
- **Multi-feature Support**: Handles multiple time series columns simultaneously
- **Selective Column Transformation**: Transform only specific columns while preserving others
- **Pandas Integration**: Full support for pandas DataFrames with column name preservation
- **Ordering**: Sort data by specified columns before applying transformations
- **Data Splitting**: Process data in separate groups based on splitting columns
- **NaN Handling**: Configurable dropping of NaN values
- **Scikit-learn Compatibility**: Full compatibility with sklearn's transformer API

### Parameters

- `window_size` (int, default=1): Size of the sliding window
- `delay_space` (int, default=1): Space between delayed samples
- `columns_to_transform` (list, optional): Specific columns to transform (indices or names)
- `order_by` (int/str/list, optional): Column(s) to order by before transformation
- `split_by` (int/str/list, optional): Column(s) to split data into separate groups
- `drop_nan` (bool, default=True): Whether to drop NaN values from output
- `include_order` (bool, default=False): Whether to include order columns in output
- `include_split` (bool, default=False): Whether to include split columns in output

## Examples

### Basic Usage

```python
import numpy as np
from delayedsw import DelayedSlidingWindow

# Simple time series
X = np.array([[1, 2, 3, 4, 5]]).transpose()
transformer = DelayedSlidingWindow(window_size=3, delay_space=1)
X_transformed = transformer.fit_transform(X)
print(X_transformed)
# Creates 3 lagged features with delay of 1
```

### Multi-feature Time Series

```python
import numpy as np
from delayedsw import DelayedSlidingWindow

# Multiple time series
X = np.array([
    [1, 2, 3, 4, 5],
    [10, 20, 30, 40, 50],
    [100, 200, 300, 400, 500]
]).transpose()

transformer = DelayedSlidingWindow(window_size=2, delay_space=2)
X_transformed = transformer.fit_transform(X)
print(X_transformed.shape)  # (3, 6) - 2 windows × 3 features
```

### Pandas DataFrame Support

```python
import pandas as pd
from delayedsw import DelayedSlidingWindow

# Create DataFrame
df = pd.DataFrame({
    'sensor1': [1, 2, 3, 4, 5],
    'sensor2': [10.0, 20.0, 30.0, 40.0, 50.0],
    'sensor3': [100, 200, 300, 400, 500]
})

# Transform only specific columns
transformer = DelayedSlidingWindow(
    window_size=2, 
    delay_space=2, 
    columns_to_transform=['sensor1', 'sensor3']
)
result = transformer.fit_transform(df)
print(result.columns)
# Output: ['sensor1_0', 'sensor1_2', 'sensor3_0', 'sensor3_2']
```

### Ordering and Splitting

```python
import pandas as pd
from delayedsw import DelayedSlidingWindow

# Create DataFrame with ordering and splitting columns
df = pd.DataFrame({
    'value': [80, 10, 120, 50, 140, 30, 90, 150],
    'timestamp': [3, 1, 3, 5, 5, 3, 4, 6],
    'group': ['A', 'A', 'B', 'A', 'B', 'A', 'A', 'B']
})

# Transform with ordering and splitting
transformer = DelayedSlidingWindow(
    window_size=2, 
    delay_space=2,
    columns_to_transform=['value'],
    order_by='timestamp',    # Sort by timestamp within each group
    split_by='group',        # Process each group separately
    include_order=True,      # Include timestamp in output
    include_split=True       # Include group in output
)
result = transformer.fit_transform(df)
```

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from delayedsw import DelayedSlidingWindow

# Create pipeline
pipeline = Pipeline([
    ('delayed_window', DelayedSlidingWindow(window_size=3, delay_space=1)),
    ('scaler', StandardScaler())
])

X_processed = pipeline.fit_transform(X)
```

## How It Works

The delayed space sliding window transformer creates lagged features by:

1. **Data Splitting**: If `split_by` is specified, data is split into separate groups
2. **Ordering**: Within each group, data is sorted by `order_by` columns if specified
3. **Window Creation**: For each time point, creates a window of `window_size` previous values
4. **Delay Application**: Spaces the lagged values by `delay_space` time steps
5. **Feature Generation**: Generates new features with descriptive names (e.g., `feature_0`, `feature_2`, etc.)

For example, with `window_size=2` and `delay_space=2`:
- Original: `[1, 2, 3, 4, 5]`
- Transformed: `[[3, 1], [4, 2], [5, 3]]`
- Features represent: `[current, lag_2]`

## Scikit-learn Compatibility

✅ **Full scikit-learn API compatibility** - The transformer passes all sklearn estimator checks and can be used in:
- Pipelines
- Grid search
- Cross-validation
- Any sklearn workflow

## API Reference

### DelayedSlidingWindow

```python
class DelayedSlidingWindow(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=1, delay_space=1, columns_to_transform=None, 
                 order_by=None, split_by=None, drop_nan=True, 
                 include_order=False, include_split=False)
    
    def fit(self, X, y=None)
    def transform(self, X, y=None)
    def fit_transform(self, X, y=None)
    def get_feature_names_out(self, input_features=None)
```

#### Parameters

- **window_size** (int, default=1): Size of the sliding window. Must be >= 1.
- **delay_space** (int, default=1): Space between delayed samples. Must be >= 1.
- **columns_to_transform** (list, optional): List of column names or indices to transform. If None, all columns are transformed.
- **order_by** (str, int, or list, optional): Column(s) to sort by before transformation.
- **split_by** (str, int, or list, optional): Column(s) to split data into separate groups.
- **drop_nan** (bool, default=True): Whether to drop rows with NaN values.
- **include_order** (bool, default=False): Whether to include order columns in output.
- **include_split** (bool, default=False): Whether to include split columns in output.

## Testing

Run tests with pytest:

```bash
pytest delayedsw/tests/
```

The test suite includes:
- Basic functionality tests
- Multi-feature transformation tests
- Pandas DataFrame integration tests
- Ordering and splitting functionality tests
- Pipeline compatibility tests
- Full sklearn compatibility tests

## Installation for Development

```bash
git clone https://github.com/ThalerG/delayedsw
cd delayedsw
pip install -e .
pip install -e .[test]  # Install with test dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Gabriel Thaler

## Changelog

### v0.1.3

- Implementation of `order_by` and `split_by` parameters
- Support for `include_order` and `include_split` options
- Enhanced pandas DataFrame support with dtype preservation

### v0.1.2
- Full scikit-learn API compatibility

### v0.1.1
- Initial release with basic sliding window functionality
- Basic pandas support
- Limited sklearn compatibility