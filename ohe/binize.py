
import numbers
import numpy as np
import warnings
import numpy

from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES

def CheckForLess(list1, val):
    return(all(x < val for x in list1))

class VL_Binizer( ):

    def __init__(self, n_bins=5, encode='onehot', strategy='quantile', edge_array=[]):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.edge_array = edge_array

    def fit(self, X, y=None):

        X = check_array(X, dtype='numeric')

        valid_encode = ('onehot', 'onehot-dense', 'ordinal')
        if self.encode not in valid_encode:
            raise ValueError("Valid options for 'encode' are {}. "
                             "Got encode={!r} instead."
                             .format(valid_encode, self.encode))
        valid_strategy = ('uniform', 'quantile', 'analyst_supervised')
        if self.strategy not in valid_strategy:
            raise ValueError("Valid options for 'strategy' are {}. "
                             "Got strategy={!r} instead."
                             .format(valid_strategy, self.strategy))

        n_features = X.shape[1]

        #FEATURES ARE COLUMS

        print("n_features " + str(n_features))
        n_bins = self._validate_n_bins(n_features)
        print("n_bins " + str(n_bins))
        bin_edges = np.zeros(n_features, dtype=object)
        print("bin_edges " + str(bin_edges))

        for jj in range(n_features):



            column = X[:, jj]

            print("column " + str(column))
            col_min, col_max = column.min(), column.max()

            print("col_min " + str(col_min))
            print("col_max " + str(col_max))

            if col_min == col_max:
                warnings.warn("Feature %d is constant and will be "
                              "replaced with 0." % jj)
                n_bins[jj] = 1
                bin_edges[jj] = np.array([-np.inf, np.inf])
                continue

            if self.strategy == 'uniform':

                print("n_bins[jj] " + str(n_bins[jj]))

                #Return evenly spaced numbers over a specified interval.
                bin_edges[jj] = np.linspace(col_min, col_max, n_bins[jj] + 1)
                print("bin_edges[jj] " + str(bin_edges[jj]))

            elif self.strategy == 'quantile':
                quantiles = np.linspace(0, 100, n_bins[jj] + 1)
                bin_edges[jj] = np.asarray(np.percentile(column, quantiles))

            elif self.strategy == 'analyst_supervised':

                if self.edge_array == []:
                    raise ValueError("Must have edges ")

                if CheckForLess(self.edge_array,col_max) == False:
                    raise ValueError("No edge bigger than number in list ")

                bin_edge_manual = self.edge_array
                arr = numpy.array(bin_edge_manual)
                bin_edges[jj] = arr

            # Remove bins whose width are too small (i.e., <= 1e-8)
            if self.strategy in ('quantile'):
                mask = np.ediff1d(bin_edges[jj], to_begin=np.inf) > 1e-8
                bin_edges[jj] = bin_edges[jj][mask]
                if len(bin_edges[jj]) - 1 != n_bins[jj]:
                    warnings.warn('Bins whose width are too small (i.e., <= '
                                  '1e-8) in feature %d are removed. Consider '
                                  'decreasing the number of bins.' % jj)
                    n_bins[jj] = len(bin_edges[jj]) - 1

        self.bin_edges_ = bin_edges
        self.n_bins_ = n_bins


        return self

    def _validate_n_bins(self, n_features):

        orig_bins = self.n_bins
        if isinstance(orig_bins, numbers.Number):
            if not isinstance(orig_bins, numbers.Integral):
                raise ValueError("{} received an invalid n_bins type. "
                                 "Received {}, expected int."
                                 .format(VL_Binizer.__name__,
                                         type(orig_bins).__name__))
            if orig_bins < 2:
                raise ValueError("{} received an invalid number "
                                 "of bins. Received {}, expected at least 2."
                                 .format(VL_Binizer.__name__, orig_bins))
            return np.full(n_features, orig_bins, dtype=np.int)

        n_bins = check_array(orig_bins, dtype=np.int, copy=True,
                             ensure_2d=False)

        if n_bins.ndim > 1 or n_bins.shape[0] != n_features:
            raise ValueError("n_bins must be a scalar or array "
                             "of shape (n_features,).")

        bad_nbins_value = (n_bins < 2) | (n_bins != orig_bins)

        violating_indices = np.where(bad_nbins_value)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            raise ValueError("{} received an invalid number "
                             "of bins at indices {}. Number of bins "
                             "must be at least 2, and must be an int."
                             .format(VL_Binizer.__name__, indices))
        return n_bins

    def transform(self, X):

        check_is_fitted(self)

        Xt = check_array(X, copy=True, dtype=FLOAT_DTYPES)
        n_features = self.n_bins_.shape[0]
        if Xt.shape[1] != n_features:
            raise ValueError("Incorrect number of features. Expecting {}, "
                             "received {}.".format(n_features, Xt.shape[1]))

        bin_edges = self.bin_edges_
        for jj in range(Xt.shape[1]):

            rtol = 1.e-5
            atol = 1.e-8
            eps = atol + rtol * np.abs(Xt[:, jj])
            Xt[:, jj] = np.digitize(Xt[:, jj] + eps, bin_edges[jj][1:])
        np.clip(Xt, 0, self.n_bins_ - 1, out=Xt)

        if self.encode == 'ordinal':
            return Xt

        return self._encoder.transform(Xt)

    def inverse_transform(self, Xt):

        check_is_fitted(self)

        if 'onehot' in self.encode:
            Xt = self._encoder.inverse_transform(Xt)

        Xinv = check_array(Xt, copy=True, dtype=FLOAT_DTYPES)
        n_features = self.n_bins_.shape[0]
        if Xinv.shape[1] != n_features:
            raise ValueError("Incorrect number of features. Expecting {}, "
                             "received {}.".format(n_features, Xinv.shape[1]))

        for jj in range(n_features):
            bin_edges = self.bin_edges_[jj]
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            Xinv[:, jj] = bin_centers[np.int_(Xinv[:, jj])]

        return Xinv
