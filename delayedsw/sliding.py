"""Class for delayed sliding window transformers"""

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval
from numbers import Integral
import numpy as np

class DelayedSlidingWindow(TransformerMixin, BaseEstimator, auto_wrap_output_keys=None):


    _parameter_constraints = {
        "window_size": [Interval(Integral, 1, None, closed="left")],  # Must be a positive integer
        "delay_space": [Interval(Integral, 1, None, closed="left")],  # Must be a positive integer
        "columns_to_transform": [list, None],  # Must be a list of column names or indices or None
        "order_by": [str, int, list, None],  # Must be a string or int or list or None
        "split_by": [str, int, list, None],  # Must be a string or int or list or None
        "drop_nan": ["boolean"],  # Must be a boolean
        "include_order": ["boolean"],  # Must be a boolean
        "include_split": ["boolean"]  # Must be a boolean
    }

    def __init__(self, window_size:int =1, delay_space: int=1, columns_to_transform: list[str|int]|None =None, 
                 order_by: int|str|list[int|str]|None = None, split_by: int|str|list[int|str]|None = None, 
                 drop_nan: bool = True, include_order: bool = False, include_split: bool = False):
        """Initialize the DelayedSlidingWindow transformer."""

        self.window_size = window_size
        self.delay_space = delay_space
        self.columns_to_transform = columns_to_transform
        self.drop_nan = drop_nan
        self.include_order = include_order
        self.include_split = include_split
        self.order_by = order_by
        self.split_by = split_by        

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Get column names and number of features from the input data."""

        # Store original input information for output formatting
        self.input_type_ = type(X)
        
        if y is not None:
            (X,y) = validate_data(self,X, y, reset = True)
        else:
            X = validate_data(self, X, reset=True)

        self._check_order(X)
        self._check_split(X)
        self._check_input(X)
            
        return self
    
    def _check_order(self, X):
        """Check and set order_by attribute."""
        # Check if order_by is a single value or a list
        if isinstance(self.order_by, list):
            if not all(isinstance(col, (str, int)) for col in self.order_by):
                raise ValueError("If order_by is a list, it must contain only strings or integers.")
        elif self.order_by is None:
            self._order_by = []
            self._order_by_array = np.arange(X.shape[0]).reshape(-1, 1)  # Default: original order
            return
        elif isinstance(self.order_by, (str, int)):
            self._order_by = [self.order_by]
        else:
            raise ValueError("order_by must be a string, integer, list of strings or integers, or None.")

        if all(isinstance(col, int) for col in self.order_by):
            # If all columns are indices, check if they are valid
            if not all(0 <= col < X.shape[1] for col in self.order_by):
                raise ValueError("All indices in order_by must be less than the number of features in the input data.")
            else:
                self._order_by = self.order_by
                self._order_by_array = np.asarray(X[:, self._order_by])  # Use as is for sorting
        else:
            # If order_by contains strings, check if they are valid column names
            if hasattr(self, 'feature_names_in_'):
                if not all(col in self.feature_names_in_ for col in self.order_by):
                    raise ValueError("All column names in order_by must be valid names of the input data.")
                else:
                    self._order_by = self.order_by
                    temp_columns = list(self.feature_names_in_)
                    order_by_index = [temp_columns.index(col) for col in self._order_by]
                    self._order_by_array = np.asarray(X[:, order_by_index])
            else:
                raise ValueError("If order_by contains strings, X must provide feature names.")


    def _check_split(self, X):
        """Check and set split_by attribute."""
        
        # Check if split_by is a single value or a list
        if isinstance(self.split_by, list):
            if not all(isinstance(col, (str, int)) for col in self.split_by):
                raise ValueError("If split_by is a list, it must contain only strings or integers.")
        elif self.split_by is None:
            self._split_by = []
            self._split_by_array = np.full((X.shape[0], 1), ' ', dtype=str)  # Array for no split
            return
        elif isinstance(self.split_by, (str, int)):
            self._split_by = [self.split_by]
        else:
            raise ValueError("split_by must be a string, integer, list of strings or integers, or None.")

        if all(isinstance(col, int) for col in self.split_by):
            # If all columns are indices, check if they are valid
            if not all(0 <= col < X.shape[1] for col in self.split_by):
                raise ValueError("All indices in split_by must be less than the number of features in the input data.")
            else:
                self._split_by = self.split_by
                self._split_by_array = np.asarray(X[:, self._split_by], dtype=str)  # Save as string array for mixed types
        else:
            # If split_by contains strings, check if they are valid column names
            if hasattr(self, 'feature_names_in_'):
                if not all(col in self.feature_names_in_ for col in self.split_by):
                    raise ValueError("All column names in split_by must be valid names of the input data.")
                else:
                    self._split_by = self.split_by
                    temp_columns = list(self.feature_names_in_)
                    split_by_index = [temp_columns.index(col) for col in self._split_by]
                    self._split_by_array = np.asarray(X[:, split_by_index], dtype=str) # Save as string array for mixed types
            else:
                raise ValueError("If split_by contains strings, X must provide feature names.")



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
                    self._columns_indices = self.columns_to_transform
                    self._columns_in_str = [str(col) for col in self.columns_to_transform]
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
                            self._columns_indices = [temp_columns.index(col) for col in self.columns_to_transform]
                            self._columns_in_str = self.columns_to_transform
                    else:
                        raise ValueError("If columns_to_transform contains strings, X must provide feature names.")
        else:
            # If no specific columns are provided, use all columns
            self._columns_indices = list(range(self.n_features_in_))
            self._columns_in_str = [str(col) for col in self._columns_indices]

        self.feature_names_out_ = []
        for column_in in self._columns_in_str:
            delay_index = [k * self.delay_space for k in range(self.window_size)]
            feature_name = [f"{column_in}_{d}" for d in delay_index]
            self.feature_names_out_.extend(feature_name)

        
    def transform(self, X, y=None):
        """Transform the data using a sliding window with delay."""

        # Save the original index of the input data for output formatting if pandas DataFrame
        original_index = X.index if hasattr(X, 'index') else None

        # Save the input dtypes for output formatting if pandas DataFrame
        if hasattr(self, 'feature_names_in_'):
            if (hasattr(X, 'columns') and 
                hasattr(X, 'index') and 
                hasattr(X, 'dtypes') and
                hasattr(X, 'iloc') and
                type(X).__name__ == 'DataFrame'):
                is_pandas = True
                input_dtypes = X.dtypes.values[self._columns_indices]
            else:
                is_pandas = False
                input_dtypes = None
        else:
            is_pandas = False
            input_dtypes = None

        X = validate_data(self, X, reset=False)

        original_index = original_index if original_index is not None else np.arange(X.shape[0])

        check_is_fitted(self, '_columns_indices')

        # If columns_to_transform is specified, select only those columns
        X = X[:, self._columns_indices]

        # Split based on split_by:
        unique_splits = np.unique(self._split_by_array, axis=0)


        delayedData = []
        for split in unique_splits:
            # Get the indices of the current split
            split_indices = np.all(self._split_by_array == split, axis=1)
            for i in range(X.shape[1]):
                # Apply sliding window to each column in the current split
                matrix_sliding = self.sliding_1d(X[split_indices, i])

                # If this is the first column, initialize delayedData
                if i == 0:
                    delayedData_split = matrix_sliding
                else:
                    delayedData_split = np.concatenate((delayedData_split, matrix_sliding), axis=1)
            # Append the delayed data for the current split to the main delayedData
            delayedData.append(delayedData_split)

        # Concatenate all splits into a single array
        delayedData = np.concatenate(delayedData, axis=0)

        # for i in range(X.shape[1]):
        #     matrix_sliding = self.sliding_1d(X[:, i])

        #     # TODO: implement sorting by order_by and split_by if specified

        #     if i == 0:
        #         delayedData = matrix_sliding
        #     else:
        #         delayedData = np.concatenate((delayedData, matrix_sliding), axis=1)

        # Get the index of the remaining rows before dropping NaNs
        if self.drop_nan:
            delayedData = delayedData.astype(float) # TODO: might lead to trouble if input is not numeric
            valid_rows = ~np.isnan(delayedData).any(axis=1)
            delayedData = delayedData[valid_rows]
        else:
            valid_rows = np.arange(delayedData.shape[0])

        if is_pandas:
            import pandas as pd
            # If input was a pandas DataFrame, return a DataFrame with appropriate column names and index
            
            output_dtypes = input_dtypes.repeat(self.window_size)
            output_dtypes = {name: dtype for name, dtype in zip(self.feature_names_out_, output_dtypes)}
            delayedData =  pd.DataFrame(delayedData, columns=self.feature_names_out_, index=original_index[valid_rows])
            return delayedData.astype(output_dtypes)
        else:
            # If input was a numpy array, return a numpy array
            return np.array(delayedData)
    
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
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False  # This transformer can handle sparse input
        return tags