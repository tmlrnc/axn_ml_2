from sklearn.cluster import KMeans
from ohe.predictor import OneHotPredictor, Commandline
from ohe.config import get_ohe_config
import numpy
import logging
import numpy
import random
from collections import defaultdict


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

class KMedian(object):
    """
    Calculations associated with K-Means clustering on a set of n-dimensional data points to find clusters - closely
    located groups - of dataset points.
    """

    def __init__(self, dataset_numpy_array, k_number_of_clusters, number_of_centroid_initializations,
                 max_number_of_iterations=30):
        """
        Attributes associated with all K-Means clustering of data points
        :param dataset_numpy_array: numpy array of n-dimensional points you'd like to cluster
        :param k_number_of_clusters: number of clusters to create
        :param max_number_of_iterations: maximum number of possible iterations to run K-Means
        """
        self.dataset = dataset_numpy_array
        self.k_number_of_clusters = k_number_of_clusters
        self.number_of_instances, self.number_of_features = self.dataset.shape
        self.number_of_centroid_initializations = number_of_centroid_initializations
        self.inertia_values = []
        self.max_number_of_iterations = max_number_of_iterations
        self.clusters_all_iterations_record = []  # all centroids and clustered dataset points

    @staticmethod
    def get_euclidean_distance(n_dimensional_numpy_array_0, n_dimensional_numpy_array_1):
        """
        Static method to calculate the normalized Euclidean distance between any n-dimensional numpy arrays
        :param n_dimensional_numpy_array_0: one n-dimensional numpy array (aka a point in space)
        :param n_dimensional_numpy_array_1: another n-dimensional numpy array (aka a point in space)
        :return: magnitude of Euclidean distance between two n-dimensional numpy arrays; scalar value
        """
        return numpy.linalg.norm(n_dimensional_numpy_array_0 - n_dimensional_numpy_array_1)

    def create_random_initial_centroids(self):
        """
        Create random initial centroids based on dataset; creates # of centroids to match # of clusters
        :return:
        """
        random_dataset_indices = random.sample(range(0, self.number_of_instances), self.k_number_of_clusters)
        random_initial_centroids = self.dataset[random_dataset_indices]
        return random_initial_centroids

    def assign_dataset_points_to_closest_centroid(self, centroids):
        """
        Given any number of centroid values, assign each point to its closest centroid based on the Euclidean distance
        metric. Use data structure cluster_iteration_record to keep track of the centroid and associated points in a
        single iteration.
        :param centroids: numpy array of centroid values
        :return: record of centroid and associated dataset points in its cluster for a single K-Means iteration
        """
        cluster_single_iteration_record = defaultdict(list)
        for dataset_point in self.dataset:
            euclidean_distances_between_dataset_point_and_centroids = []
            for centroid in centroids:
                distance_between_centroid_and_dataset_point = self.get_euclidean_distance(centroid, dataset_point)
                euclidean_distances_between_dataset_point_and_centroids.append(
                    distance_between_centroid_and_dataset_point)
            index_of_closest_centroid = numpy.argmin(euclidean_distances_between_dataset_point_and_centroids)
            closest_centroid = tuple(centroids[index_of_closest_centroid])
            cluster_single_iteration_record[closest_centroid].append(dataset_point)
        return cluster_single_iteration_record

    def run_kmeans_initialized_centroid(self, initialization_number):
        """
        Assign dataset points to clusters based on nearest centroid; update centroids based on mean of cluster points.
        Repeat steps above until centroids don't move or we've reached max_number_of_iterations.

        :return: None
        """
        centroids = self.create_random_initial_centroids()

        self.clusters_all_iterations_record.append([])  # list of record of iteration centroids and clustered points

        for iteration in range(1, self.max_number_of_iterations + 1):
            cluster_single_iteration_record = self.assign_dataset_points_to_closest_centroid(centroids=centroids)
            self.clusters_all_iterations_record[initialization_number].append(cluster_single_iteration_record)
            updated_centroids = []
            for centroid in cluster_single_iteration_record:
                cluster_dataset_points = cluster_single_iteration_record[centroid]

                updated_centroid = numpy.mean(cluster_dataset_points, axis=0)
                updated_centroids.append(updated_centroid)
            if self.get_euclidean_distance(numpy.array(updated_centroids), centroids) == 0:

                break
            centroids = updated_centroids
        return None

    def fit(self):
        """
        Implements K-Means the max number_of_centroid_initializations times; each time, there's new initial centroids.
        :return: None
        """

        for initialization_number in range(self.number_of_centroid_initializations):
            self.run_kmeans_initialized_centroid(initialization_number=initialization_number)

            # index of -1 is for the last cluster assignment of the iteration
            inertia_of_last_cluster_record = self.inertia(
                self.clusters_all_iterations_record[initialization_number][-1])
            self.inertia_values.append(inertia_of_last_cluster_record)
        return None

    def inertia(self, clusters):
        """
        Get the sum of squared distances of dataset points to their cluster centers for all clusters - defined as inertia
        :return: cluster_sum_of_squares_points_to_clusters
        """
        cluster_sum_of_squares_points_to_clusters = 0

        for centroid, cluster_points in clusters.items():

            for cluster_point in cluster_points:
                euclidean_norm_distance = self.get_euclidean_distance(cluster_point, centroid)
                euclidean_norm_distance_squared = euclidean_norm_distance ** 2

                cluster_sum_of_squares_points_to_clusters += euclidean_norm_distance_squared
        return cluster_sum_of_squares_points_to_clusters

    def index_lowest_inertia_cluster(self):
        """
        In our list of inertia_values, finds the index of the minimum inertia
        :return: index_lowest_inertia
        """
        minimum_inertia_value = min(self.inertia_values)
        index_lowest_inertia = self.inertia_values.index(minimum_inertia_value)
        return index_lowest_inertia

    def final_iteration_optimal_cluster(self):
        """
        Get results of optimal cluster assignment based  on the lowest inertia value
        :return: dictionary with keys as centroids and values as list of dataset points in the clusters
        """
        # -1 gets us the final iteration from a centroid initialization of running K-Means
        return self.clusters_all_iterations_record[self.index_lowest_inertia_cluster()][-1]

    def final_iteration_optimal_cluster_centroids(self):
        """
        Get centroids of the optimal cluster assignment based on the lowest inertia value
        :return: list of tuples with tuples holding centroid locations
        """
        return list(self.final_iteration_optimal_cluster().keys())



    def predict(self, n_dimensional_numpy_array):
        """
        Predict which cluster a new point belongs to; calculates euclidean distance from point to all centroids
        :param n_dimensional_numpy_array: new observation that has same n-dimensions as dataset points
        :return: closest_centroid
        """
        # initially assign closest_centroid as large value; we'll reassign it later
        closest_centroid = numpy.inf
        for centroid in self.final_iteration_optimal_cluster_centroids():
            distance = self.get_euclidean_distance(centroid, n_dimensional_numpy_array)
            if distance < closest_centroid:
                closest_centroid = centroid
        return closest_centroid



@Commandline("K_Means_VL")
class K_Means_VL(OneHotPredictor):

    def __init__(self, target, X_test, X_train, y_test, y_train):
        """
        initializes the training and testing features and labels

        :param target: string - label to be predicted or classified
        :param X_test: array(float) - testing features
        :param X_train: array(float) - training features
        :param y_test: array(float) - testing label
        :param y_train: array(float) - testing label
        """
        super().__init__(target, X_test, X_train, y_test, y_train)
        self.model_name = 'K_Means_VL'

    def predict(self):
        """
        trains the scikit-learn  python machine learning algorithm library function
        https://scikit-learn.org

        then passes the trained algorithm the features set and returns the
        predicted y test values form, the function

        then compares the y_test values from scikit-learn predicted to
        y_test values passed in

        then returns the accuracy
        """
        algorithm = KMedian(n_clusters=2, random_state=0)
        algorithm.fit(self.X_train.toarray(), self.y_train)
        y_pred = list(algorithm.predict(self.X_test.toarray()))
        self.acc = OneHotPredictor.get_accuracy(y_pred, self.y_test)
        return self.acc