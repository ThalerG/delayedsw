"""Class for delayed sliding window transformers"""

from sklearn.base import BaseEstimator, TransformerMixin

class DelayedSlidingWindow(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=1, delay_space=1, columns_to_transform=None, feature_names_in=None, n_features_in=None, suffix_template="decreasing_space"):
        """Initialize the DelayedSlidingWindow transformer."""

        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if not isinstance(delay_space, int) or delay_space < 0:
            raise ValueError("delay_space must be a non-negative integer.")
        if columns_to_transform is not None and not isinstance(columns_to_transform, list):
            raise ValueError("columns_to_transform must be a list or None.")
        if feature_names_in is not None and not isinstance(feature_names_in, list):
            raise ValueError("feature_names_in must be a list or None.")
        if n_features_in is not None and (not isinstance(n_features_in, int) or n_features_in <= 0):
            raise ValueError("n_features_in must be a positive integer or None.")
        if suffix_template not in ["decreasing_space", "increasing_space", "increasing", "decreasing"]:
            raise ValueError("suffix_template must be one of 'decreasing_space', 'increasing_space', 'increasing', or 'decreasing'.")
        
        self.window_size = window_size
        self.delay_space = delay_space
        self.feature_names_in_ = feature_names_in
        self.n_features_in_ = n_features_in
        self.columns_to_transform = columns_to_transform
        self.suffix_template = suffix_template

    def fit(self, X, y=None):
        """Get column names and number of features from the input data."""
        if hasattr(X, 'shape'):
            self.n_features_in_ = X.shape[1]
        else:
            raise ValueError("Input data must have a shape attribute (e.g., a numpy array or pandas DataFrame).")
        
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

        else:
            # If no specific columns are provided, transform all columns
            self.columns_to_transform = list(range(self.n_features_in_))
            
        if self.feature_names_in_ is None:
            if hasattr(X, 'columns'):
                self.feature_names_in_ = X.columns.tolist()
            else:
                self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]
        return self
    
    def transform(self, X):
        """Transform the data using a sliding window with delay."""
        import numpy as np

        # TODO: Implement the sliding window transformation with delay

        # Placeholder for delayed data
        delayedData = np.zeros((X.shape[0] - self.window_size + 1, self.window_size))

        return delayedData
    
    def get_feature_names_out(self, input_features=None):
        # TODO: Implement the method to return feature names after transformation
        pass