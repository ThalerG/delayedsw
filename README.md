# delayedsw

A Python library for preprocessing time series data using delayed space sliding window transformations. The `delayedsw` package provides a scikit-learn compatible transformer that creates lagged features from time series data.

## Installation

```bash
pip install git+https://github.com/ThalerG/delayedsw
```

## Requirements

- scikit-learn >= 1.0.0
- numpy
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
- **NaN Handling**: Configurable dropping of NaN values

### Parameters

- `window_size` (int, default=1): Size of the sliding window
- `delay_space` (int, default=1): Space between delayed samples
- `columns_to_transform` (list, optional): Specific columns to transform (indices or names)
- `drop_nan` (bool, default=True): Whether to drop NaN values from output
- `order_by` (int/str, optional): Column to order by (not implemented yet)
- `split_by` (int/str, optional): Column to split by (not implemented yet)

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

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from delayedsw import DelayedSlidingWindow

# Create pipeline
pipeline = Pipeline([
    ('delayed_window', DelayedSlidingWindow(window_size=3, delay_space=3)),
    ('scaler', StandardScaler())
])

X_processed = pipeline.fit_transform(X)
```

## How It Works

The delayed space sliding window transformer creates lagged features by:

1. **Window Creation**: For each time point, creates a window of `window_size` previous values
2. **Delay Application**: Spaces the lagged values by `delay_space` samples
3. **Feature Generation**: Generates new features with descriptive names (e.g., `feature_0`, `feature_1`, etc.)

For example, with `window_size=2` and `delay_space=2`:
- Original: `[1, 2, 3, 4, 5]`
- Transformed: `[[3, 1], [4, 2], [5, 3]]`
- Features represent: `[current, lag_2]`

## Current Limitations

### Known Issues:
- `order_by` and `split_by` parameters are not yet implemented

## API Reference

### DelayedSlidingWindow

```python
class DelayedSlidingWindow(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=1, delay_space=1, columns_to_transform=None, 
                 order_by=None, split_by=None, drop_nan=True)
    
    def fit(self, X, y=None)
    def transform(self, X, y=None)
    def fit_transform(self, X, y=None)
    def get_feature_names_out(self, input_features=None)
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

The test suite includes:
- Basic functionality tests
- Multi-feature transformation tests
- Pandas DataFrame integration tests
- Pipeline compatibility tests
- Basic sklearn compatibility tests

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Gabriel Thaler

## Roadmap

- [X] Full scikit-learn API compatibility
- [ ] Implementation of `order_by` and `split_by` parameters
- [ ] Performance optimizations
- [ ] Comprehensive documentation and examples