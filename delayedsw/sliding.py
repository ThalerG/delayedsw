"""Class for delayed sliding window transformers"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class DelayedSlidingWindow(BaseEstimator, TransformerMixin):
    def __init__(self, window_size:int =1, delay_space: int=1, columns_to_transform: list[str|int]|None =None, 
                 feature_names_in: list[str]|None = None, n_features_in: int|None = None, 
                 order_by: int|str|None = None, split_by: int|str|None = None):
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
        self.feature_names_in_ = feature_names_in
        self.n_features_in_ = n_features_in
        self.columns_to_transform = columns_to_transform
        self.order_by = order_by
        self.split_by = split_by

    def fit(self, X, y=None):
        """Get column names and number of features from the input data."""
        if hasattr(X, 'shape'):
            self.n_features_in_ = X.shape[1]
        else:
            raise ValueError("Input data must have a shape attribute (e.g., a numpy array or pandas DataFrame).")
        
        n = X.shape[0]
        if n < self.window_size:
            raise ValueError("Input array length must be at least as large as window_size.")
        
        if self.feature_names_in_ is None:
            if hasattr(X, 'columns'):
                self.feature_names_in_ = X.columns.tolist()
            else:
                self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

        if self.columns_to_transform is not None:
            if not all(isinstance(col, int) and 0 <= col < self.n_features_in_ for col in self.columns_to_transform): # Column names are not integer indices
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

            self.feature_names_in_ = [self.feature_names_in_[i] for i in self.columns_indices_]

        else:
            # If no specific columns are provided, transform all columns
            self.columns_indices_ = list(range(self.n_features_in_))
            
        return self
    
    def transform(self, X, y=None):
        """Transform the data using a sliding window with delay."""

        self.feature_names_out_ = []

        if not hasattr(self, 'columns_indices_'):
            # If columns_indices_ is not set, it means fit was not called or columns_to_transform was not specified
            raise ValueError("The transformer has not been fitted yet. Call 'fit' before 'transform'.")
        
        if hasattr(X, 'columns'):
            X = X.values  # Convert DataFrame to numpy array if necessary
        
        X = X[:, self.columns_indices_]

        for i in range(X.shape[1]):
            matrix_sliding = self.sliding_1d(X[:, i])
            if i == 0:
                delayedData = matrix_sliding
            else:
                delayedData = np.concatenate((delayedData, matrix_sliding), axis=1)

            # Generate feature names
            delay_index = [k * self.delay_space for k in range(self.window_size)]
            feature_name = [f"{self.feature_names_in_[i]}_{d}" for d in delay_index]
            self.feature_names_out_.extend(feature_name)
        
        return delayedData
    
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
        for i in range(n - self.window_size * self.delay_space + self.delay_space, 0, -self.delay_space):
            start = i - self.window_size * self.delay_space
            indices = range(start, i, self.delay_space)
            window = X[list(indices)]
            windows.append(window)

        return np.array(windows[::-1])
    
    def get_feature_names_out(self, input_features=None):
        """Get the feature names after transformation."""
        if self.feature_names_in_ is None:
            raise ValueError("The transformer has not been fitted yet. Call 'fit' before 'get_feature_names_out'.")
        
        
        return self.feature_names_out_