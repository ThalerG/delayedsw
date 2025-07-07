"""Class for delayed sliding window transformers"""

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval
from numbers import Integral
import numpy as np

class DelayedSlidingWindow(TransformerMixin, BaseEstimator):


    _parameter_constraints = {
        "window_size": [Interval(Integral, 1, None, closed="left")],  # Must be a positive integer
        "delay_space": [Interval(Integral, 1, None, closed="left")],  # Must be a positive integer
        "columns_to_transform": [list, None],  # Must be a list of column names or indices or None
        "order_by": [str, int, None],  # Must be a string or int
        "split_by": [str, int, None],  # Must be a string or int
        "drop_nan": ["boolean"],  # Must be a boolean
    }

    def __init__(self, window_size:int =1, delay_space: int=1, columns_to_transform: list[str|int]|None =None, 
                 order_by: int|str|None = None, split_by: int|str|None = None, drop_nan: bool = True):
        """Initialize the DelayedSlidingWindow transformer."""

        self.window_size = window_size
        self.delay_space = delay_space
        self.columns_to_transform = columns_to_transform
        self.order_by = order_by
        self.split_by = split_by
        self.drop_nan = drop_nan

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Get column names and number of features from the input data."""

        # Store original input information for output formatting
        self.input_type_ = type(X)
        
        if y is not None:
            (X, y) = validate_data(self,X, y, accept_sparse=True, reset = True)
        else:
            X = validate_data(self, X, accept_sparse=True, reset = True)

        self._check_input(X)
            
        return self
    
    def _check_input(self, X):
        """Check and set input data attributes."""
        
        n_rows = X.shape[0]
        if n_rows < self.window_size:
            raise ValueError("Input array length must be at least as large as window_size.")
        
        if self.columns_to_transform is not None: # Check if specific columns are provided
            if all(isinstance(col, int) for col in self.columns_to_transform):  # If all columns are indices
                if not all(0 <= col < self.n_features_in_ for col in self.columns_to_transform):
                    raise ValueError("All indices in columns_to_transform must be less than the number of features in the input data.")
                else:
                    # If columns_to_transform contains valid indices, use them directly
                    self.columns_indices_ = self.columns_to_transform
                    self.columns_in_str = [str(col) for col in self.columns_to_transform]
            else:
                # If columns_to_transform contains strings, check if they are valid column names
                if not all(isinstance(col, str) for col in self.columns_to_transform):
                    raise ValueError("All elements in columns_to_transform must be either valid indices or valid column names of the input data.")
                else:
                    # If columns_to_transform contains strings, check if they are valid column names
                    if hasattr(self, 'feature_names_in_'):
                        if not all(col in self.feature_names_in_ for col in self.columns_to_transform):
                            raise ValueError("All column names in columns_to_transform must be valid names of the input data.")
                        else:
                            # Convert column names to indices
                            temp_columns = list(self.feature_names_in_)
                            self.columns_indices_ = [temp_columns.index(col) for col in self.columns_to_transform]
                            self.columns_in_str = self.columns_to_transform
                    else:
                        raise ValueError("If columns_to_transform contains strings, X must provide feature names.")
        else:
            # If no specific columns are provided, use all columns
            self.columns_indices_ = list(range(self.n_features_in_))
            self.columns_in_str = [str(col) for col in self.columns_indices_]

        self.feature_names_out_ = []
        for column_in in self.columns_in_str:
            delay_index = [k * self.delay_space for k in range(self.window_size)]
            feature_name = [f"{column_in}_{d}" for d in delay_index]
            self.feature_names_out_.extend(feature_name)
    
    def transform(self, X, y=None):
        """Transform the data using a sliding window with delay."""

        check_is_fitted(self, 'columns_indices_')
        
        original_index = X.index if hasattr(X, 'index') else np.arange(X.shape[0])
        if hasattr(X, 'columns'):
            input_dtypes = X.dtypes.values[self.columns_indices_]

        X = validate_data(self, X, accept_sparse=True, reset=False)

        X = X[:, self.columns_indices_]

        for i in range(X.shape[1]):
            matrix_sliding = self.sliding_1d(X[:, i])

            # TODO: implement sorting by order_by and split_by if specified

            if i == 0:
                delayedData = matrix_sliding
            else:
                delayedData = np.concatenate((delayedData, matrix_sliding), axis=1)

        # Get the index of the remaining rows before dropping NaNs
        if self.drop_nan:
            valid_rows = ~np.isnan(delayedData).any(axis=1)
            delayedData = delayedData[valid_rows]
        else:
            valid_rows = np.arange(delayedData.shape[0])

        if self.input_type_ == np.ndarray:
            # If input was a numpy array, return a numpy array
            return delayedData
        else:
            import pandas as pd
            # If input was a pandas DataFrame, return a DataFrame with appropriate column names and index
            
            output_dtypes = input_dtypes.repeat(self.window_size)
            output_dtypes = {name: dtype for name, dtype in zip(self.feature_names_out_, output_dtypes)}
            delayedData =  pd.DataFrame(delayedData, columns=self.feature_names_out_, index=original_index[valid_rows])
            return delayedData.astype(output_dtypes)
    
    def sliding_1d(self, X):
        """Apply delayed space sliding window to a 1D array."""

        if X.ndim > 1:
            # Only one dimension should have more than one element
            shape = X.shape
            non_singleton_dims = [i for i, s in enumerate(shape) if s > 1]
            if len(non_singleton_dims) != 1:
                raise ValueError("Input array must have more than one element in only one dimension.")
            X = X.flatten()

        n = X.shape[0]
        if n < self.window_size:
            raise ValueError("Input array length must be at least as large as window_size.")

        # Create the sliding window with delay
        windows = []
        
        # Efficient implementation using numpy stride tricks
        pad_width = (self.window_size - 1) * self.delay_space
        X_padded = np.concatenate([np.full(pad_width, np.nan), X])
        indices = np.arange(n) + pad_width
        window_indices = np.array([indices - k * self.delay_space for k in range(self.window_size)])
        windows = X_padded[window_indices].T

        return windows
    
    def get_feature_names_out(self, input_features=None):
        """Get the feature names after transformation."""
        if self.feature_names_in_ is None:
            raise ValueError("The transformer has not been fitted yet. Call 'fit' before 'get_feature_names_out'.")
        
        return self.feature_names_out_