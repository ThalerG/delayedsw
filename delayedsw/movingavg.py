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

    def __init__(self, window=3, min_periods=None, center=False, 
                 columns_to_transform=None, order_by=None, split_by=None):
        """
        Initialize the MovingAverageTransformer.
        
        Parameters
        ----------
        window : int, default=3
            Size of the moving window. Must be >= 1.
            
        min_periods : int, optional
            Minimum number of observations in window required to have a value.
            If None, defaults to window size. Must be >= 1 if specified.
            
        center : bool, default=False
            Whether to center the window around the current observation.
            If True, the window is centered; if False, the window includes
            the current observation and (window-1) preceding observations.
            
        columns_to_transform : list of str or int, optional
            List of column names/indices to transform. If None, transforms
            all numeric columns. Non-numeric columns are always preserved.
            
        order_by : str, int, or list of str/int, optional
            Column(s) to sort by before applying transformation.
            
        split_by : str, int, or list of str/int, optional
            Column(s) to group by before applying transformation.
        """
        self.window = window
        self.min_periods = min_periods
        self.center = center
        self.columns_to_transform = columns_to_transform
        self.order_by = order_by
        self.split_by = split_by

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the transformer to the input data.
        
        Analyzes the input data structure, validates parameters, and prepares
        internal state for transformation. Determines which columns are numeric
        and can be transformed, and validates grouping/ordering parameters.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data. Can be a numpy array or pandas DataFrame.
            
        y : array-like of shape (n_samples,), optional
            Target values. Ignored, present for API consistency.
        
        Returns
        -------
        self : object
            Fitted transformer instance.
            
        Raises
        ------
        ValueError
            If order_by or split_by columns are not found in the data,
            or if columns_to_transform contains invalid column references.
        """
        # Preserve the original type of X
        self._is_dataframe = isinstance(X, pd.DataFrame)
        
        # Prepare data for validation - handle mixed types carefully
        if self._is_dataframe:
            X_for_validation = self._prepare_dataframe_for_validation(X)
        else:
            X_for_validation = X
        
        # Use validate_data to set feature_names_in_ and n_features_in_
        # Allow NaN values and handle mixed types
        X_validated = validate_data(
            self, 
            X_for_validation, 
            reset=True,
            ensure_all_finite='allow-nan',  # Allow NaN values
            dtype=None  # Allow mixed types
        )
        
        # Setup columns based on original input
        if self._is_dataframe:
            self._setup_dataframe_columns(X)
        else:
            self._setup_array_columns(X_validated)
            
        self._validate_parameters(X_validated)
        return self

    def _prepare_dataframe_for_validation(self, X):
        """Prepare DataFrame for sklearn validation by handling mixed types."""
        X_prep = X.copy()
        
        # Convert string/object columns to category for sklearn compatibility
        for col in X_prep.columns:
            if pd.api.types.is_object_dtype(X_prep[col]) or pd.api.types.is_string_dtype(X_prep[col]):
                # Convert to category with numeric codes
                X_prep[col] = pd.Categorical(X_prep[col]).codes
                # Replace -1 (missing category) with NaN
                X_prep[col] = X_prep[col].replace(-1, np.nan)
        
        return X_prep

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
        
        # Set _min_periods here to avoid parameter mutation
        self._min_periods = self.min_periods if self.min_periods is not None else self.window
        
        if self._min_periods > self.window:
            raise ValueError(f"min_periods ({self._min_periods}) cannot be larger than "
                           f"window ({self.window})")

    def transform(self, X, y=None):
        """
        Transform the input data using moving averages.
        
        Applies moving average calculations to the specified numeric columns
        while preserving all other columns unchanged. Respects grouping and
        ordering constraints if specified during initialization.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform. Must have the same structure as the
            data used during fit.
            
        y : array-like of shape (n_samples,), optional
            Target values. Ignored, present for API consistency.
        
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features)
            Transformed data with moving averages applied to specified columns.
            Non-numeric and non-selected columns are preserved unchanged.
            For pandas DataFrames, returns a DataFrame with same column names.
            For numpy arrays, returns an array with same structure.
        
        Notes
        -----
        - Output has the same number of rows and columns as input
        - NaN values are properly propagated through the moving average calculation
        - If center=True, the window is centered around each observation
        - If split_by is used, moving averages never cross group boundaries
        - Early/late observations may have NaN if insufficient data for the window
        """
        check_is_fitted(self)
        
        # Prepare data for validation if needed
        if self._is_dataframe:
            X_for_validation = self._prepare_dataframe_for_validation(X)
        else:
            X_for_validation = X
        
        # Use validate_data to ensure sklearn compatibility
        X_validated = validate_data(
            self, 
            X_for_validation, 
            reset=False,
            ensure_all_finite='allow-nan',  # Allow NaN values
            dtype=None  # Allow mixed types
        )
        
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
            grouped = result.groupby(self.split_by, group_keys=False)
            
            # Apply transformation with include_groups=False and then restore grouping columns
            try:
                # For newer pandas versions
                transformed_data = grouped.apply(self._apply_moving_average_df, include_groups=False)
                
                # Restore the grouping columns by merging back
                result = result[split_cols].join(transformed_data)
                
            except TypeError:
                # Fallback for older pandas versions 
                result = grouped.apply(self._apply_moving_average_df)
            
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
                # Use apply with a custom function that propagates NaN
                df[col] = df[col].rolling(
                    window=self.window,
                    min_periods=self._min_periods,
                    center=self.center
                ).apply(lambda x: np.nan if np.any(np.isnan(x)) else np.mean(x), raw=True)
                
        return df

    def _transform_array(self, X):
        """Transform numpy array."""
        result = X.copy()
        
        # Convert entire result to float if any transform columns are involved
        if self._transform_indices:
            result = result.astype(float)
        
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
        result = np.full(n, np.nan, dtype=float)
        
        # Convert to float if possible, otherwise skip
        try:
            x_float = x.astype(float)
        except (ValueError, TypeError):
            return result  # Return NaN array if can't convert to float
        
        if self.center:
            # Centered window
            half_window = self.window // 2
            for i in range(n):
                start = max(0, i - half_window)
                end = min(n, i + half_window + 1)
                window_data = x_float[start:end]
                
                # If any NaN values in window, result is NaN
                if np.any(np.isnan(window_data)):
                    result[i] = np.nan
                elif len(window_data) >= self._min_periods:
                    result[i] = np.mean(window_data)
        else:
            # Trailing window (default)
            for i in range(n):
                start = max(0, i - self.window + 1)
                window_data = x_float[start:i + 1]
                
                # If any NaN values in window, result is NaN
                if np.any(np.isnan(window_data)):
                    result[i] = np.nan
                elif len(window_data) >= self._min_periods:
                    result[i] = np.mean(window_data)
                    
        return result

    def get_feature_names_out(self, input_features=None):
        """
        Get the feature names for the transformed output.
        
        Returns the same feature names as the input since moving averages
        are applied in-place to the selected columns without creating new columns.
        
        Parameters
        ----------
        input_features : array-like of str, optional
            Input feature names. If None, uses the feature names from fit
            or generates default names for numpy arrays.
        
        Returns
        -------
        feature_names_out : ndarray of str
            Array of feature names for the transformed output. Same as input
            feature names since the transformation preserves column structure.
        
        Notes
        -----
        Must be called after fit. Unlike DelayedSlidingWindow, this transformer
        does not create new columns but modifies existing ones in-place.
        """
        check_is_fitted(self)
        
        if self._is_dataframe:
            return np.array(self._column_names)
        else:
            return np.array([f"feature_{i}" for i in range(len(self._column_names))])