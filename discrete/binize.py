"""
To make ml models more powerful on continuous data
VL uses discretization (also known as binning).
We discretize the feature and one-hot encode the transformed data.
Note that if the bins are not reasonably wide,
there would appear to be a substantially increased risk of overfitting,
so the discretizer parameters should usually be tuned under cross validation.
After discretization, linear regression and decision tree make exactly the same prediction.
As features are constant within each bin, any model must
predict the same value for all points within a bin.
Compared with the result before discretization,
linear model become much more flexible while decision tree gets much less flexible.
Note that binning features generally has no
beneficial effect for tree-based models,
as these models can learn to split up the data anywhere.

Bin continuous data into intervals.
Parameters
----------
n_bins : int or array-like, shape (n_features,) (default=5)
    The number of bins to produce. Raises ValueError if ``n_bins < 2``.

encode : {'onehot', 'onehot-dense', 'ordinal'}, (default='onehot')
    Method used to encode the transformed result.

    onehot
        Encode the transformed result with one-hot encoding
        and return a sparse matrix. Ignored features are always
        stacked to the right.
    onehot-dense
        Encode the transformed result with one-hot encoding
        and return a dense array. Ignored features are always
        stacked to the right.
    ordinal
        Return the bin identifier encoded as an integer value.

strategy : {'uniform', 'quantile', 'kmeans'}, (default='quantile')
    Strategy used to define the widths of the bins.

    uniform
        All bins in each feature have identical widths.
    quantile
        All bins in each feature have the same number of points.
    kmeans
        Values in each bin have the same nearest center of a 1D k-means
        cluster.


n_bins_ : int array, shape (n_features,)
    Number of bins per feature. Bins whose width are too small
    (i.e., <= 1e-8) are removed with a warning.

bin_edges_ : array of arrays, shape (n_features, )
    The edges of each bin. Contain arrays of varying shapes ``(n_bins_, )``
    Ignored features will have empty arrays.



Sometimes it may be useful to convert the data back into the original
feature space. The ``inverse_transform`` function converts the binned
data into the original feature space. Each value will be equal to the mean
of the two bin edges.

DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
Finds core samples of high density and expands clusters from them.
Good for data which contains clusters of similar density.


The maximum distance between two samples for one to be considered as in the neighborhood of the other.
This is not a maximum bound on the distances of points within a cluster. This is the most important


eps: Two points are considered neighbors if the distance between the two points is below the threshold epsilon.
min_samples: The minimum number of neighbors a given point should have in order to be classified as a core point.
It’s important to note that the point itself is included in the minimum number of samples.
metric: The metric to use when calculating distance between instances in a feature array (i.e. euclidean distance).

The algorithm works by computing the distance between every point and all other points.
We then place the points into one of three categories.

Core point: A point with at least min_samples points whose distance
with respect to the point is below the threshold defined by epsilon.

Border point: A point that isn’t in close proximity to at least min_samples points but is close enough to one or more core point.
Border points are included in the cluster of the closest core point.

Noise point: Points that aren’t close enough to core points to be considered border points. Noise points are ignored.
That is to say, they aren’t part of any cluster.


"""
import warnings
import numbers
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES


def check_for_less(list1, val):
    """
    check for least in list
    """
    return all(x < val for x in list1)

class VlBinizer():
    """
    To make linear model more powerful on continuous data is to use discretization (also known as binning).
    We discretize the feature and one-hot encode the transformed data. Note that if the bins are not reasonably wide,
    there would appear to be a substantially increased risk of overfitting,
    so the discretizer parameters should usually be tuned under cross validation.
    After discretization, linear regression and decision tree make exactly the same prediction.
    As features are constant within each bin, any model must predict the same value for all points within a bin.
    Compared with the result before discretization, linear model become much more flexible while decision tree gets much less flexible.
    Note that binning features generally has no beneficial effect for tree-based models, as these models can learn to split up the data anywhere.

    Bin continuous data into intervals.
    Parameters
    ----------
    n_bins : int or array-like, shape (n_features,) (default=5)
        The number of bins to produce. Raises ValueError if ``n_bins < 2``.

    encode : {'onehot', 'onehot-dense', 'ordinal'}, (default='onehot')
        Method used to encode the transformed result.

        onehot
            Encode the transformed result with one-hot encoding
            and return a sparse matrix. Ignored features are always
            stacked to the right.
        onehot-dense
            Encode the transformed result with one-hot encoding
            and return a dense array. Ignored features are always
            stacked to the right.
        ordinal
            Return the bin identifier encoded as an integer value.

    strategy : {'uniform', 'quantile', 'kmeans'}, (default='quantile')
        Strategy used to define the widths of the bins.

        uniform
            All bins in each feature have identical widths.
        quantile
            All bins in each feature have the same number of points.
        kmeans
            Values in each bin have the same nearest center of a 1D k-means
            cluster.


    n_bins_ : int array, shape (n_features,)
        Number of bins per feature. Bins whose width are too small
        (i.e., <= 1e-8) are removed with a warning.

    bin_edges_ : array of arrays, shape (n_features, )
        The edges of each bin. Contain arrays of varying shapes ``(n_bins_, )``
        Ignored features will have empty arrays.



    Sometimes it may be useful to convert the data back into the original
    feature space. The ``inverse_transform`` function converts the binned
    data into the original feature space. Each value will be equal to the mean
    of the two bin edges.

    DBSCAN - Density-Based Spatial Clustering of Applications with Noise.
    Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density.


The maximum distance between two samples for one to be considered as in the neighborhood of the other.
This is not a maximum bound on the distances of points within a cluster. This is the most important


    eps: Two points are considered neighbors if the distance between the two points is below the threshold epsilon.
    min_samples: The minimum number of neighbors a given point should have in order to be classified as a core point.
    It’s important to note that the point itself is included in the minimum number of samples.
    metric: The metric to use when calculating distance between instances in a feature array (i.e. euclidean distance).

The algorithm works by computing the distance between every point and all other points. We then place the points into one of three categories.

Core point: A point with at least min_samples points whose distance with respect to the point is below the threshold defined by epsilon.

Border point: A point that isn’t in close proximity to at least min_samples points but is close enough to one or more core point.
Border points are included in the cluster of the closest core point.

Noise point: Points that aren’t close enough to core points to be considered border points. Noise points are ignored.
That is to say, they aren’t part of any cluster.

    """
    # pylint: disable=dangerous-default-value
    # pylint: disable=too-many-locals
    # pylint: disable=invalid-name
    # pylint: disable=singleton-comparison
    # pylint: disable=no-member
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=duplicate-code


    def __init__(
            self,
            n_bins=5,
            encode='onehot',
            strategy='quantile',
            edge_array=[]):
        self.n_bins = n_bins
        self.encode = encode
        self.strategy = strategy
        self.edge_array = edge_array

    def fit(self, x_my):
        """
        Fit the estimator.

        Parameters
        ----------
        X : numeric array-like, shape (n_samples, n_features)
            Data to be discretized.

        y : None
            Ignored. This parameter exists only for compatibility with
            :class:`sklearn.pipeline.Pipeline`.

        Returns
        -------
        self
        """

        x_my = check_array(x_my, dtype='numeric')

        valid_encode = ('onehot', 'onehot-dense', 'ordinal')
        if self.encode not in valid_encode:
            raise ValueError("Valid options for 'encode' are {}. "
                             "Got encode={!r} instead."
                             .format(valid_encode, self.encode))
        valid_strategy = ('uniform', 'quantile', 'analyst_supervised', '')
        if self.strategy not in valid_strategy:
            raise ValueError("Valid options for 'strategy' are {}. "
                             "Got strategy={!r} instead."
                             .format(valid_strategy, self.strategy))

        n_features = x_my.shape[1]

        # FEATURES ARE COLUMS

        print("n_features " + str(n_features))
        n_bins = self._validate_n_bins(n_features)
        print("n_bins " + str(n_bins))
        bin_edges = np.zeros(n_features, dtype=object)
        print("bin_edges " + str(bin_edges))

        for jj in range(n_features):

            column = x_my[:, jj]

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

                # Return evenly spaced numbers over a specified interval.
                bin_edges[jj] = np.linspace(col_min, col_max, n_bins[jj] + 1)
                print("bin_edges[jj] " + str(bin_edges[jj]))

            elif self.strategy == 'quantile':
                quantiles = np.linspace(0, 100, n_bins[jj] + 1)
                bin_edges[jj] = np.asarray(np.percentile(column, quantiles))

            elif self.strategy == 'analyst_supervised':

                if self.edge_array == []:
                    raise ValueError("Must have edges ")

                if check_for_less(self.edge_array, col_max) == False:
                    raise ValueError("No edge bigger than number in list ")

                bin_edge_manual = self.edge_array
                arr = np.array(bin_edge_manual)
                bin_edges[jj] = arr

            # Remove bins whose width are too small (i.e., <= 1e-8)
            if self.strategy in 'quantile':
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
                                 .format(VlBinizer.__name__,
                                         type(orig_bins).__name__))
            if orig_bins < 2:
                raise ValueError("{} received an invalid number "
                                 "of bins. Received {}, expected at least 2."
                                 .format(VlBinizer.__name__, orig_bins))
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
                             .format(VlBinizer.__name__, indices))
        return n_bins

    def transform(self, x_my):
        """
        Discretize the data.

        Parameters
        ----------
        X : numeric array-like, shape (n_samples, n_features)
            Data to be discretized.

        Returns
        -------
        Xt : numeric array-like or sparse matrix
            Data in the binned space.
        """
        check_is_fitted(self)
        x_myt = check_array(x_my, copy=True, dtype=FLOAT_DTYPES)
        n_features = self.n_bins_.shape[0]
        if x_myt.shape[1] != n_features:
            raise ValueError("Incorrect number of {}, "
                             "received {}.".format(n_features, x_myt.shape[1]))
        bin_edges = self.bin_edges_
        for jj in range(x_myt.shape[1]):

            rtol = 1.e-5
            atol = 1.e-8
            eps = atol + rtol * np.abs(x_myt[:, jj])
            x_myt[:, jj] = np.digitize(x_myt[:, jj] + eps, bin_edges[jj][1:])
        np.clip(x_myt, 0, self.n_bins_ - 1, out=x_myt)

        if self.encode == 'ordinal':
            return x_myt

        return self._encoder.transform(x_myt)

    def inverse_transform(self, x_myt):
        """
        Transform discretized data back to original feature space.

        Note that this function does not regenerate the original data
        due to discretization rounding.

        Parameters
        ----------
        Xt : numeric array-like, shape (n_sample, n_features)
            Transformed data in the binned space.

        Returns
        -------
        Xinv : numeric array-like
            Data in the original feature space.
        """
        check_is_fitted(self)

        if 'onehot' in self.encode:
            x_myt = self._encoder.inverse_transform(x_myt)

        x_myinv = check_array(x_myt, copy=True, dtype=FLOAT_DTYPES)
        n_features = self.n_bins_.shape[0]
        if x_myinv.shape[1] != n_features:
            raise ValueError("Incorrect number of features. Expecting {}, "
                             "received {}.".format(n_features, x_myinv.shape[1]))

        for jj in range(n_features):
            bin_edges = self.bin_edges_[jj]
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) * 0.5
            x_myinv[:, jj] = bin_centers[np.int_(x_myinv[:, jj])]

        return x_myinv
