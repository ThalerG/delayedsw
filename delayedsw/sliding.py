"""Class for delayed sliding window transformers"""

from sklearn.base import BaseEstimator, TransformerMixin, _fit_context
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils._param_validation import Interval
from numbers import Integral
import numpy as np
import pandas as pd

class DelayedSlidingWindow(TransformerMixin, BaseEstimator):
    """
    A transformer that creates lagged features using delayed space sliding windows.
    
    This transformer generates lagged versions of input features by applying a sliding window
    with configurable window size and delay spacing. It supports selective column transformation,
    data ordering, and group-wise processing while preserving all original features.
    
    The key difference from traditional sliding windows is the "delay space" parameter,
    which controls the spacing between lagged values rather than creating consecutive lags.
    
    Parameters
    ----------
    window_size : int, default=1
        Size of the sliding window. Must be >= 1. This determines how many lagged
        features will be created for each selected column.
        
    delay_space : int, default=1
        Space between delayed samples. Must be >= 1. This controls the lag spacing
        between consecutive window elements. For example, delay_space=2 creates
        lags at positions [0, 2, 4, ...] relative to the current position.
        
    columns_to_transform : list of str or int, optional
        Specific columns to transform (by name or index). If None, all columns
        are transformed. All original features are always preserved in the output.
        
    order_by : str, int, or list of str/int, optional
        Column(s) to sort by before applying the transformation. Useful for
        time series data to ensure proper temporal ordering.
        
    split_by : str, int, or list of str/int, optional
        Column(s) to group by before applying the transformation. Each group
        is processed separately, preventing lag values from crossing group boundaries.
    
    Attributes
    ----------
    input_type_ : type
        The type of the input data (numpy.ndarray or pandas.DataFrame).
        
    n_features_in_ : int
        Number of features in the input data.
        
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit (only if input is DataFrame).
    
    Examples
    --------
    Basic usage with numpy arrays:
    
    >>> import numpy as np
    >>> from delayedsw import DelayedSlidingWindow
    >>> X = np.array([[1, 2, 3, 4, 5]]).T
    >>> transformer = DelayedSlidingWindow(window_size=2, delay_space=2)
    >>> X_transformed = transformer.fit_transform(X)
    >>> print(X_transformed)
    [[  1.  nan]
     [  2.  nan]
     [  3.   1.]
     [  4.   2.]
     [  5.   3.]]
    
    With pandas DataFrames and selective columns:
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
    >>> transformer = DelayedSlidingWindow(
    ...     window_size=2, 
    ...     delay_space=1,
    ...     columns_to_transform=['A']
    ... )
    >>> result = transformer.fit_transform(df)
    >>> print(result.columns.tolist())
    ['A_0', 'A_1']
    
    With grouping and ordering:
    
    >>> df = pd.DataFrame({
    ...     'value': [3, 1, 4, 2],
    ...     'time': [3, 1, 4, 2], 
    ...     'group': ['A', 'A', 'B', 'B']
    ... })
    >>> transformer = DelayedSlidingWindow(
    ...     window_size=2,
    ...     order_by='time',
    ...     split_by='group',
    ...     columns_to_transform=['value']
    ... )
    >>> result = transformer.fit_transform(df)
    
    Notes
    -----
    - The output always has the same number of rows as the input
    - Early rows contain NaN where lagged values are not available
    - When using split_by, lag values never cross group boundaries
    - All original features are preserved regardless of columns_to_transform
    - The transformer is compatible with sklearn pipelines
    
    See Also
    --------
    MovingAverageTransformer : For computing moving averages instead of lags
    """


    _parameter_constraints = {
        "window_size": [Interval(Integral, 1, None, closed="left")],  # Must be a positive integer
        "delay_space": [Interval(Integral, 1, None, closed="left")],  # Must be a positive integer
        "columns_to_transform": [list, None],  # Must be a list of column names or indices or None
        "order_by": [str, int, list, None],  # Must be a string or int or list or None
        "split_by": [str, int, list, None]  # Must be a string or int or list or None
    }

    def __init__(self, window_size:int =1, delay_space: int=1, columns_to_transform: list[str|int]|None =None, 
                 order_by: int|str|list[int|str]|None = None, split_by: int|str|list[int|str]|None = None):
        """
        Initialize the DelayedSlidingWindow transformer.
        
        Parameters
        ----------
        window_size : int, default=1
            Size of the sliding window. Must be >= 1.
            
        delay_space : int, default=1
            Space between delayed samples. Must be >= 1.
            
        columns_to_transform : list of str or int, optional
            Specific columns to transform. If None, all columns are transformed.
            All original features are preserved regardless of this setting.
            
        order_by : str, int, or list of str/int, optional
            Column(s) to sort by before transformation.
            
        split_by : str, int, or list of str/int, optional
            Column(s) to group by before transformation.
        
        Notes
        -----
        All input columns are preserved in the output. The columns_to_transform parameter
        specifies which columns will have sliding window transformations applied as NEW columns.
        """

        self.window_size = window_size
        self.delay_space = delay_space
        self.columns_to_transform = columns_to_transform
        self.order_by = order_by
        self.split_by = split_by        

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """
        Fit the transformer to the input data.
        
        This method analyzes the input data structure, validates parameters,
        and prepares internal state for transformation. It determines column
        types, validates ordering and splitting parameters, and sets up
        the transformation configuration.
        
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

        # Store original input information for output formatting
        self.input_type_ = type(X)
        
        # Store original types and structure before validation
        self._get_types(X)
        
        # Prepare data for validation - handle mixed types carefully
        if self._is_pandas:
            X_for_validation = self._prepare_dataframe_for_validation(X)
        else:
            X_for_validation = X
        
        if y is not None:
            (X_validated, y) = validate_data(self, X_for_validation, y, reset=True, 
                                           ensure_all_finite='allow-nan', dtype=None)
        else:
            X_validated = validate_data(self, X_for_validation, reset=True,
                                      ensure_all_finite='allow-nan', dtype=None)

        # Now use original X for checks since we need original column names/types
        self._check_order(X)
        self._check_split(X)
        self._check_input(X)
            
        return self
    
    def _get_types(self, X):
        """
        Analyze and store the input data type information.
        
        Determines whether the input is a pandas DataFrame and stores
        relevant type information for output formatting.
        
        Parameters
        ----------
        X : array-like
            Input data to analyze.
        
        Notes
        -----
        Sets self._is_pandas and self._input_dtypes attributes for use
        during transformation and output formatting.
        """

        # Save the input dtypes for output formatting if pandas DataFrame
        if (hasattr(X, 'columns') and 
            hasattr(X, 'index') and 
            hasattr(X, 'dtypes') and
            hasattr(X, 'iloc') and
            type(X).__name__ == 'DataFrame'):
            self._is_pandas = True
            self._input_dtypes = X.dtypes.values
        else: # Output is not a pandas DataFrame, no need to save dtypes
            self._is_pandas = False
            self._input_dtypes = None

    
    def _check_order(self, X):
        """Check and set order_by attribute."""
        # Check if order_by is a single value or a list

        # TODO: I'm not happy with this implementation, it should be more robust

        if isinstance(self.order_by, list):
            if not all(isinstance(col, (str, int)) for col in self.order_by):
                raise ValueError("If order_by is a list, it must contain only strings or integers.")
            else:
                self._order_by = self.order_by
        elif self.order_by is None:
            self._order_by = None
            self._order_by_array = np.arange(X.shape[0]).reshape(-1, 1)
            self._order_by_dtype = None
            return
        elif isinstance(self.order_by, (str, int)):
            self._order_by = [self.order_by]
        else:
            raise ValueError("order_by must be a string, integer, list of strings or integers, or None.")

        if all(isinstance(col, int) for col in self._order_by):
            # If all columns are indices, check if they are valid
            if not all(0 <= col < X.shape[1] for col in self._order_by):
                raise ValueError("All indices in order_by must be less than the number of features in the input data.")
            else:
                self._order_by_array = np.asarray(X[:, self._order_by])  # Use as is for sorting
                if self._is_pandas:
                    self._order_by_dtype = self._input_dtypes[self._order_by]
                else:
                    self._order_by_dtype = X[:, self._order_by].dtype
        else:
            # If order_by contains strings, check if they are valid column names
            if hasattr(self, 'feature_names_in_'):
                if not all(col in self.feature_names_in_ for col in self._order_by):
                    raise ValueError("All column names in order_by must be valid names of the input data.")
                else:
                    temp_columns = list(self.feature_names_in_)
                    order_by_index = [temp_columns.index(col) for col in self._order_by]
                    if self._is_pandas:
                        self._order_by_array = np.asarray(X.iloc[:, order_by_index])
                        self._order_by_dtype = self._input_dtypes[order_by_index]
                    else:
                        self._order_by_array = np.asarray(X[:, order_by_index])
                        self._order_by_dtype = X[:, order_by_index].dtype
            else:
                raise ValueError("If order_by contains strings, X must provide feature names.")


    def _check_split(self, X):
        """Check and set split_by attribute."""
        # Check if split_by is a single value or a list

        # TODO: I'm not happy with this implementation, it should be more robust

        if isinstance(self.split_by, list):
            if not all(isinstance(col, (str, int)) for col in self.split_by):
                raise ValueError("If split_by is a list, it must contain only strings or integers.")
            else:
                self._split_by = self.split_by
        elif self.split_by is None:
            self._split_by = None
            self._split_by_array = np.full((X.shape[0], 1), ' ', dtype=str)  # Array for no split
            self._split_by_dtype = None
            return
        elif isinstance(self.split_by, (str, int)):
            self._split_by = [self.split_by]
        else:
            raise ValueError("split_by must be a string, integer, list of strings or integers, or None.")

        if all(isinstance(col, int) for col in self._split_by):
            # If all columns are indices, check if they are valid
            if not all(0 <= col < X.shape[1] for col in self._split_by):
                raise ValueError("All indices in split_by must be less than the number of features in the input data.")
            else:
                self._split_by_array = np.asarray(X[:, self._split_by], dtype=str)  # Save as string array for mixed types
                if self._is_pandas:
                    self._split_by_dtype = self._input_dtypes[self._split_by]
                else:
                    self._split_by_dtype = X[:, self._split_by].dtype
        else:
            # If split_by contains strings, check if they are valid column names
            if hasattr(self, 'feature_names_in_'):
                if not all(col in self.feature_names_in_ for col in self._split_by):
                    raise ValueError("All column names in split_by must be valid names of the input data.")
                else:
                    temp_columns = list(self.feature_names_in_)
                    split_by_index = [temp_columns.index(col) for col in self._split_by]
                    if self._is_pandas:
                        self._split_by_array = np.asarray(X.iloc[:, split_by_index], dtype=str) # Save as string array for mixed types
                        self._split_by_dtype = self._input_dtypes[split_by_index]
                    else:
                        self._split_by_array = np.asarray(X[:, split_by_index], dtype=str) # Save as string array for mixed types
                        self._split_by_dtype = X[:, split_by_index].dtype
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
                    # Use actual feature names if available, otherwise fall back to indices as strings
                    if hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None:
                        self._columns_in_str = [self.feature_names_in_[col] for col in self.columns_to_transform]
                    else:
                        self._columns_in_str = [f"feature_{col}" for col in self.columns_to_transform]
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
            # Use actual feature names if available, otherwise use default feature names
            if hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None:
                self._columns_in_str = list(self.feature_names_in_)
            else:
                self._columns_in_str = [f"feature_{col}" for col in self._columns_indices]

        self.feature_names_out_ = []
        for column_in in self._columns_in_str:
            # Skip lag=0 since original features are already preserved
            delay_index = [k * self.delay_space for k in range(1, self.window_size)]
            feature_name = [f"{column_in}_{d}" for d in delay_index]
            self.feature_names_out_.extend(feature_name)

        
    def transform(self, X, y=None):
        """
        Transform the input data using delayed space sliding windows.
        
        Creates lagged features for the specified columns while preserving all
        original features. The transformation respects grouping and ordering
        constraints if specified during initialization.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform. Must have the same structure as the
            data used during fit.
            
        y : array-like of shape (n_samples,), optional
            Target values. Ignored, present for API consistency.
        
        Returns
        -------
        X_transformed : array-like of shape (n_samples, n_features_out)
            Transformed data with original features plus lagged features.
            For pandas DataFrames, returns a DataFrame with descriptive
            column names (e.g., 'feature_0_2' for feature with lag 2).
            For numpy arrays, returns an array with lagged columns appended.
        
        Notes
        -----
        - Output has the same number of rows as input
        - Early rows contain NaN where lagged values are not available
        - If split_by is used, lag values never cross group boundaries
        - Column order in output: selected columns with their lags, then
          any remaining non-transformed columns
        """

        # Save the original index of the input data for output formatting if pandas DataFrame
        original_index = X.index if self._is_pandas else None
        original_X = X.copy() if self._is_pandas else X.copy()  # Preserve original data

        # Save the input dtypes for output formatting if pandas DataFrame
        if self._is_pandas:
            transform_dtypes = self._input_dtypes[self._columns_indices]        

        # Store original X for output formatting
        X_original = X
        
        # Prepare data for validation if needed  
        if self._is_pandas:
            X_for_validation = self._prepare_dataframe_for_validation(X)
        else:
            X_for_validation = X

        X = validate_data(self, X_for_validation, reset=False, ensure_all_finite='allow-nan')

        if self.split_by is None:
            split_by_array = np.full((X.shape[0], 1), ' ', dtype=str)  # Array for no split
            split_by_dtype = None
        else:
            split_by_array = self._split_by_array
            split_by_dtype = self._split_by_dtype
        
        if self.order_by is None:
            order_by_array = np.arange(X.shape[0]).reshape(-1, 1)
            order_by_dtype = None
        else:
            order_by_array = self._order_by_array
            order_by_dtype = self._order_by_dtype
            

        original_index = original_index if original_index is not None else np.arange(X.shape[0])

        check_is_fitted(self, '_columns_indices')

        # Extract only the columns to transform for sliding window processing
        X_to_transform = X[:, self._columns_indices]

        # Split based on split_by:
        unique_splits = np.unique(split_by_array, axis=0)

        delayedData = []
        index_after_split = []
        
        for split in unique_splits:
            # Get the indices of the current split
            split_indices = np.all(split_by_array == split, axis=1)
            order_by_split = order_by_array[split_indices]

            X_split = X_to_transform[split_indices, :]
            split_indexes = original_index[split_indices]

            # Sort the current split by order_by
            order_indices = np.argsort(order_by_split, axis=0).flatten()
            X_split = X_split[order_indices]
            split_indexes = split_indexes[order_indices]

            # Store the index of the current split for later use
            index_after_split.append(split_indexes)

            for i in range(X_split.shape[1]):
                # Apply sliding window to each column in the current split
                matrix_sliding = self.sliding_1d(X_split[:,i])

                # If this is the first column, initialize delayedData
                if i == 0:
                    delayedData_split = matrix_sliding
                else:
                    delayedData_split = np.concatenate((delayedData_split, matrix_sliding), axis=1)
            # Append the delayed data for the current split to the main delayedData
            delayedData.append(delayedData_split)

        # Concatenate all splits into a single array
        delayedData = np.concatenate(delayedData, axis=0)
        index_after_split = np.concatenate(index_after_split, axis=0)

        # No NaN dropping - preserve all computed sliding window values
        final_delayedData = delayedData
        final_index = index_after_split

        # Now we need to preserve the original columns and add the new sliding window features
        if self._is_pandas:
            # Get the original data for the final indices
            original_data_subset = X_original.iloc[final_index]
            
            # Create DataFrame with sliding features
            sliding_features_df = pd.DataFrame(final_delayedData, columns=self.feature_names_out_, index=final_index)
            
            # Combine original data with sliding window features
            result = pd.concat([original_data_subset.reset_index(drop=True), 
                               sliding_features_df.reset_index(drop=True)], axis=1)
            result.index = final_index
            
            return result
        else:
            # For numpy arrays, preserve all original columns and add sliding window features
            original_data_subset = X_original[final_index, :]
            
            # Combine original data with sliding window features  
            result = np.concatenate([original_data_subset, final_delayedData], axis=1)
            
            return result
    
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

        # Special case: if window_size=1, no lagged features to create
        if self.window_size == 1:
            return np.empty((n, 0))  # Return empty 2D array with n rows and 0 columns
        
        # Efficient implementation using numpy stride tricks
        # We start from k=1 to skip lag=0 (current value) since original features are already preserved
        pad_width = (self.window_size - 1) * self.delay_space
        X_padded = np.concatenate([np.full(pad_width, np.nan), X])
        indices = np.arange(n, dtype=int) + pad_width
        window_indices = np.array([indices - k * self.delay_space for k in range(1, self.window_size)], dtype=int)
        windows = X_padded[window_indices].T

        return windows
    
    def get_feature_names_out(self, input_features=None):
        """
        Get the feature names for the transformed output.
        
        Returns descriptive names for all features in the transformed output,
        including both original features and newly created lagged features.
        
        Parameters
        ----------
        input_features : array-like of str, optional
            Input feature names. If None, uses the feature names from fit
            or generates default names.
        
        Returns
        -------
        feature_names_out : ndarray of str
            Array of feature names for the transformed output. Includes
            original feature names plus lagged feature names in the format
            'feature_lag' (e.g., 'sensor1_2' for sensor1 with lag 2).
        
        Notes
        -----
        Must be called after fit. The order matches the column order
        of the transform output.
        """
        check_is_fitted(self, '_columns_indices')
        
        # Start with all original feature names
        if hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None:
            original_names = list(self.feature_names_in_)
        else:
            # If no feature names available, use default naming
            original_names = [f"feature_{i}" for i in range(self.n_features_in_)]
        
        # Add sliding window feature names
        all_names = original_names + self.feature_names_out_
        
        return all_names
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False  # This transformer can handle sparse input
        return tags
    
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