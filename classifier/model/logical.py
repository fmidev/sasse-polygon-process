import sys
import numpy as np
import multiprocessing as mp
import concurrent.futures

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
#from memory_profiler import profile

class Logical(BaseEstimator):
    """
    Logical model for SASSE ERA5 polygon classification
    """

    def __init__(self, n_jobs=-1):
        """
        ...
        """

        self.n_jobs=n_jobs


    def fit(self, X, y):
        """
        Empty function, logical model doesn't need any training
        """
        return self

    def partial_fit(self, X, y):
        """
        Empty
        """
        return self

    def predict(self, X):
        """
        Predict based on heuristical rules.

        Note! X need to be a pandas dataframe
        """

        def c(row):
            """ classification logic """
            
            if row.low_limit > 20:
                return 3
            if row.low_limit > 15:
                return 2
            if row["AVG CAPE"] > 50:
                return 1

            return 0

        y_pred = X.apply(lambda x: c(x), axis=1)
        # print(y_pred)
        #ret =  np.zeros((X.shape[0], 1)).astype(np.int)

        return y_pred.values.ravel()
