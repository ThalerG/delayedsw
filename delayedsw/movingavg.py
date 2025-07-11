"""Moving average transformer for time series data."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval
from numbers import Integral


class MovingAverageTransformer(TransformerMixin, BaseEstimator):
    """
    A transformer that applies moving averages to specified columns while preserving others.
    
    Similar to pandas.DataFrame.rolling() but works with sklearn pipelines and can handle
    grouped data with ordering.
    
    Parameters
    ----------
    window : int, default=3
        Size of the moving window.
    min_periods : int, default=None
        Minimum number of observations in window required to have a value
        (otherwise result is NaN). If None, defaults to window size.
    center : bool, default=False
        Whether to center the window around the current observation.
    columns_to_transform : list, default=None
        List of column names/indices to transform. If None, transforms all numeric columns.
    order_by : str, int, or list, default=None
        Column(s) to sort by before applying transformation.
    split_by : str, int, or list, default=None
        Column(s) to group by before applying transformation.
    """
    
    _parameter_constraints = {
        "window": [Interval(Integral, 1, None, closed="left")],
        "min_periods": [Interval(Integral, 1, None, closed="left"), None],
        "center": ["boolean"],
        "columns_to_transform": [list, None],
        "order_by": [str, int, list, None],
        "split_by": [str, int, list, None]
    }

    def __init__(self, window=1, min_periods=None, center=False, 
                 columns_to_transform=None, order_by=None, split_by=None):
        self.window = window
        self.min_periods = min_periods
        self.center = center
        self.columns_to_transform = columns_to_transform
        self.order_by = order_by
        self.split_by = split_by
        self._min_periods = min_periods if min_periods is not None else window

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the transformer to X."""
        # Check if X is a DataFrame:
        if (hasattr(X, 'columns') and 
            hasattr(X, 'index') and 
            hasattr(X, 'dtypes') and
            hasattr(X, 'iloc')):
            import pandas as pd
            self._is_dataframe = True
        else:
            self._is_dataframe = False

        if self._is_dataframe:
            X_val = X.copy()
        else:
            X_val = X

        if self._is_dataframe:
            # Convert string columns to category dtype for processing
            string_cols = [col for col in X.columns if pd.api.types.is_string_dtype(X[col])]
            for col in string_cols:
                X_val[col] = X_val[col].astype('category')
        
        # Use validate_data to set feature_names_in_ and n_features_in_
        X_validated = validate_data(self, X_val, reset=True)
        
        # Setup columns based on original input
        if self._is_dataframe:
            self._setup_dataframe_columns(X)
        else:
            self._setup_array_columns(X_validated)
            
        self._validate_parameters(X_validated)
        return self

    def _setup_dataframe_columns(self, X):
        """Setup column handling for pandas DataFrame."""
        self._column_names = list(X.columns)
        
        if self.columns_to_transform is None:
            # Transform all numeric columns
            self._transform_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self._transform_cols = self.columns_to_transform
            # Validate that specified columns exist
            missing_cols = set(self._transform_cols) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
            # Validate that specified columns are numeric
            for col in self._transform_cols:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    raise ValueError(f"Column '{col}' is not numeric and cannot be transformed")
            
        # Columns to preserve unchanged
        self._preserve_cols = [col for col in self._column_names 
                              if col not in self._transform_cols]

    def _setup_array_columns(self, X):
        """Setup column handling for numpy arrays."""
        n_features = X.shape[1]
        
        if self.columns_to_transform is None:
            self._transform_indices = list(range(n_features))
        else:
            self._transform_indices = self.columns_to_transform
            # Validate indices
            invalid_indices = [i for i in self._transform_indices if i >= n_features or i < 0]
            if invalid_indices:
                raise ValueError(f"Invalid column indices: {invalid_indices}")
            
        self._preserve_indices = [i for i in range(n_features) 
                                 if i not in self._transform_indices]

    def _validate_parameters(self, X):
        """Validate transformer parameters."""
        n_samples = X.shape[0]
        if self.window > n_samples:
            raise ValueError(f"Window size ({self.window}) cannot be larger than "
                           f"number of samples ({n_samples})")
        
        if self._min_periods > self.window:
            raise ValueError(f"min_periods ({self._min_periods}) cannot be larger than "
                           f"window ({self.window})")

    def transform(self, X, y=None):
        """Transform X using moving averages."""
        check_is_fitted(self)

        if self._is_dataframe:
            X_val = X.copy()
        else:
            X_val = X

        if self._is_dataframe:
            # Convert string columns to category dtype for processing
            string_cols = [col for col in X.columns if pd.api.types.is_string_dtype(X[col])]
            for col in string_cols:
                X_val[col] = X_val[col].astype('category')
        
        # Use validate_data to ensure sklearn compatibility
        X_validated = validate_data(self, X_val, reset=False)
        
        if self._is_dataframe:
            return self._transform_dataframe(X)
        else:
            return self._transform_array(X_validated)

    def _transform_dataframe(self, X):
        """Transform pandas DataFrame."""
        # Validate that transform columns exist and are numeric
        missing_cols = set(self._transform_cols) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Transform columns not found: {missing_cols}")
            
        # Check that transform columns are numeric
        for col in self._transform_cols:
            if not pd.api.types.is_numeric_dtype(X[col]):
                raise ValueError(f"Column '{col}' is not numeric and cannot be transformed")
        
        result = X.copy()
        
        if self.split_by is None:
            # No grouping - apply to entire DataFrame
            result = self._apply_moving_average_df(result)
        else:
            # Validate split_by columns exist
            split_cols = [self.split_by] if isinstance(self.split_by, str) else self.split_by
            missing_split = set(split_cols) - set(X.columns)
            if missing_split:
                raise ValueError(f"Split columns not found: {missing_split}")
                
            # Group by split_by columns and apply transformation
            result = result.groupby(self.split_by, group_keys=False, observed=False)[result.columns].apply(self._apply_moving_average_df)
            
        return result

    def _apply_moving_average_df(self, df):
        """Apply moving average to a DataFrame (possibly a group)."""
        if self.order_by is not None:
            # Validate order_by columns exist
            order_cols = [self.order_by] if isinstance(self.order_by, str) else self.order_by
            missing_order = set(order_cols) - set(df.columns)
            if missing_order:
                raise ValueError(f"Order columns not found: {missing_order}")
                
            df = df.sort_values(self.order_by)
            
        # Apply moving average only to specified columns
        for col in self._transform_cols:
            if col in df.columns:
                df[col] = df[col].rolling(
                    window=self.window,
                    min_periods=self._min_periods,
                    center=self.center
                ).mean()
                
        return df

    def _transform_array(self, X):
        """Transform numpy array."""
        result = X.copy().astype(float)
        
        if self.split_by is None:
            # No grouping - apply to entire array
            result = self._apply_moving_average_array(result)
        else:
            # Group by split_by columns
            split_indices = [self.split_by] if isinstance(self.split_by, int) else self.split_by
            
            # Validate split indices
            invalid_split = [i for i in split_indices if i >= X.shape[1] or i < 0]
            if invalid_split:
                raise ValueError(f"Invalid split column indices: {invalid_split}")
                
            # Get unique groups
            split_data = X[:, split_indices]
            unique_groups, group_indices = np.unique(split_data, axis=0, return_inverse=True)
            
            # Apply transformation to each group
            for group_id in range(len(unique_groups)):
                mask = group_indices == group_id
                group_data = result[mask]
                group_data = self._apply_moving_average_array(group_data)
                result[mask] = group_data
                
        return result

    def _apply_moving_average_array(self, X):
        """Apply moving average to numpy array."""
        result = X.copy()
        
        for col_idx in self._transform_indices:
            result[:, col_idx] = self._moving_average_1d(X[:, col_idx])
            
        return result

    def _moving_average_1d(self, x):
        """Compute moving average for 1D array."""
        n = len(x)
        result = np.full(n, np.nan)
        
        if self.center:
            # Centered window
            half_window = self.window // 2
            for i in range(n):
                start = max(0, i - half_window)
                end = min(n, i + half_window + 1)
                window_data = x[start:end]
                
                if len(window_data) >= self._min_periods:
                    result[i] = np.mean(window_data)
        else:
            # Trailing window (default)
            for i in range(n):
                start = max(0, i - self.window + 1)
                window_data = x[start:i + 1]
                
                if len(window_data) >= self._min_periods:
                    result[i] = np.mean(window_data)
                    
        return result

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        check_is_fitted(self)
        
        if self._is_dataframe:
            return np.array(self._column_names)
        else:
            return np.array([f"feature_{i}" for i in range(len(self._column_names))])

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        return tags