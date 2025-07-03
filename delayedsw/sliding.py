"""Class for delayed sliding window transformers"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class DelayedSlidingWindow(BaseEstimator, TransformerMixin):
    def __init__(self, window_size:int =1, delay_space: int=1, columns_to_transform: list[str|int]|None =None, 
                 feature_names_in: list[str]|None = None, n_features_in: int|None = None, 
                 order_by: int|str|None = None, split_by: int|str|None = None, drop_nan: bool = True):
        """Initialize the DelayedSlidingWindow transformer."""

        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if not isinstance(delay_space, int) or delay_space < 0:
            raise ValueError("delay_space must be a non-negative integer.")
        if columns_to_transform is not None and not isinstance(columns_to_transform, list):
            raise ValueError("columns_to_transform must be a list or None.")
        if columns_to_transform is not None and not all(isinstance(col, (int, str)) for col in columns_to_transform):
            raise ValueError("All elements in columns_to_transform must be either integers or strings.")
        if feature_names_in is not None and not isinstance(feature_names_in, list):
            raise ValueError("feature_names_in must be a list or None.")
        if n_features_in is not None and (not isinstance(n_features_in, int) or n_features_in <= 0):
            raise ValueError("n_features_in must be a positive integer or None.")
        if order_by is not None and not isinstance(order_by, (int, str)):
            raise ValueError("order_by must be an integer, string, or None.")
        if split_by is not None and not isinstance(split_by, (int, str)):
            raise ValueError("split_by must be an integer, string, or None.")

        
        self.window_size = window_size
        self.delay_space = delay_space
        self.feature_names_in = feature_names_in
        self.n_features_in = n_features_in
        self.columns_to_transform = columns_to_transform
        self.order_by = order_by
        self.split_by = split_by
        self.drop_nan = drop_nan

    def __sklearn_tags__(self):
        """Additional tags for sklearn compatibility."""
        return {
            'requires_fit': True,
            'allow_nan': True,
            'requires_columns': False,
            'preserves_dtype': [],
            '_xfail_checks': {
                'check_estimators_dtypes',
                'check_fit2d_predict1d',
            }
        }

    def fit(self, X, y=None):
        """Get column names and number of features from the input data."""

        # Store original input information for output formatting
        self._check_input(X)

        self.input_type_ = type(X)
            
        return self
    
    def _check_input(self, X):
        """Check and set input data attributes."""
        
        if hasattr(X, 'shape'):
            self.n_features_in = X.shape[1]
        else:
            raise ValueError("Input data must have a shape attribute (e.g., a numpy array or pandas DataFrame).")
        
        n = X.shape[0]
        if n < self.window_size:
            raise ValueError("Input array length must be at least as large as window_size.")
        
        if self.feature_names_in is None:
            if hasattr(X, 'columns'):
                self.feature_names_in = X.columns.tolist()
            else:
                self.feature_names_in = [f"feature_{i}" for i in range(X.shape[1])]

        if self.columns_to_transform is not None:
            if not all(isinstance(col, int) and 0 <= col < self.n_features_in for col in self.columns_to_transform): # Column names are not integer indices
                if hasattr(X, 'columns'):
                    # X is a pandas DataFrame, columns_to_transform should be column names (strings)
                    if not all(col in X.columns for col in self.columns_to_transform):
                        # If any column name is not in the DataFrame's columns, raise an error
                        raise ValueError("All columns in columns_to_transform must be either valid indices or valid column names of the input data.")
                    else:
                        # Convert column names to indices
                        self.columns_indices_ = [X.columns.get_loc(col) for col in self.columns_to_transform]
                else:
                    # X is a numpy array or similar, columns_to_transform should be indices
                    raise ValueError("All columns in columns_to_transform must be valid indices of the input data.")
            else:
                # If columns_to_transform are valid indices, use them directly
                self.columns_indices_ = self.columns_to_transform

            self.feature_names_in = [self.feature_names_in[i] for i in self.columns_indices_]

        else:
            # If no specific columns are provided, transform all columns
            self.columns_indices_ = list(range(self.n_features_in))
    
    def transform(self, X, y=None):
        """Transform the data using a sliding window with delay."""


        if not hasattr(self, 'columns_indices_'):
            # If columns_indices_ is not set, it means fit was not called or columns_to_transform was not specified
            raise ValueError("The transformer has not been fitted yet. Call 'fit' before 'transform'.")
        
        original_index = X.index if hasattr(X, 'index') else np.arange(X.shape[0])
        if hasattr(X, 'columns'):
            input_dtypes = X.dtypes.values[self.columns_indices_]
            X = X.values  # Convert DataFrame to numpy array if necessary
            

        X = X[:, self.columns_indices_]
        self.feature_names_out_ = []

        for i in range(X.shape[1]):
            matrix_sliding = self.sliding_1d(X[:, i])

            # TODO: implement sorting by order_by and split_by if specified

            if i == 0:
                delayedData = matrix_sliding
            else:
                delayedData = np.concatenate((delayedData, matrix_sliding), axis=1)

            # Generate feature names
            delay_index = [k * self.delay_space for k in range(self.window_size)]
            feature_name = [f"{self.feature_names_in[i]}_{d}" for d in delay_index]
            self.feature_names_out_.extend(feature_name)

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
        if self.feature_names_in is None:
            raise ValueError("The transformer has not been fitted yet. Call 'fit' before 'get_feature_names_out'.")
        
        
        return self.feature_names_out_