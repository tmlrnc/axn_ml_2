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

import csv


def main():
    """
transform ts

      """
    ######################################################################
    #
    # read run commands
    #
    # pylint: disable=consider-iterating-dictionary
    # pylint: disable=too-many-locals


    print("Discretize --- START ")
    filename = 'covid_joe7.csv'
    outfile = 'covid_joe7_pivot.csv'
    headers = []
    headerset = set()
    regions = {}
    with open(filename) as io_my:
        reader = csv.DictReader(io_my)
        for r_my in reader:
            print(r_my)
            region = r_my['Country']
            date = r_my['\ufeffDate']
            if region not in regions:
                regions[region] = {}
            regions[region][date] = {'confirmed': r_my['confirmed'],
                                     'deaths': r_my['deaths'],
                                     'recovered': r_my['recovered']
                                     }
    records = []
    for region in regions.keys():
        r_my = {}
        r_my['Country'] = region
        for date, data in regions[region].items():
            confirmed = date + '_confirmed'
            deaths = date + '_deaths'
            recovered = date + '_recovered'
            r_my[confirmed] = data['confirmed']
            r_my[deaths] = data['deaths']
            r_my[recovered] = data['recovered']
            for h_my in [confirmed, deaths, recovered]:
                if h_my in headerset:
                    continue
                headers.append(h_my)
                headerset.add(h_my)
        records.append(r_my)
    headers.sort(reverse=True)
    headers = ['Country'] + headers
    with open(outfile, 'w') as io_my:
        writer = csv.DictWriter(io_my, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)

    print("Discretize --- END ")


if __name__ == '__main__':
    main()
