"""
Listory recommendation engine
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import defaultdict
from surprise import SVD
from surprise import Dataset
import json
import pandas as pd
from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
import argparse
from collections import namedtuple
import numbers

import numpy as np

def baseline_als(self):
    """Optimize biases using ALS.

    Args:
        self: The algorithm that needs to compute baselines.

    Returns:
        A tuple ``(bu, bi)``, which are users and items baselines.
    """

    bu = []
    bi = []
    reg_pu = []
    lr_pu = []
    reg_bi = []
    lr_bi = []
    reg_bu = []
    reg_qi = []
    lr_bu = []
    lr_qi = []

    u, i = 0
    r, err, dev_i, dev_u = 0
    global_mean = self.trainset.global_mean

    n_epochs = self.bsl_options.get('n_epochs', 10)
    reg_u = self.bsl_options.get('reg_u', 15)
    reg_i = self.bsl_options.get('reg_i', 10)

    for dummy in range(n_epochs):
        for i in self.trainset.all_items():
            dev_i = 0
            for (u, r) in self.trainset.ir[i]:
                dev_i += r - global_mean - bu[u]

            bi[i] = dev_i / (reg_i + len(self.trainset.ir[i]))

        for u in self.trainset.all_users():
            dev_u = 0
            for (i, r) in self.trainset.ur[u]:
                dev_u += r - global_mean - bi[i]
            bu[u] = dev_u / (reg_u + len(self.trainset.ur[u]))

    return bu, bi


def baseline_sgd(self):
    """Optimize biases using SGD.

    Args:
        self: The algorithm that needs to compute baselines.

    Returns:
        A tuple ``(bu, bi)``, which are users and items baselines.
    """
    bu = []
    bi = []
    reg_pu = []
    lr_pu = []
    reg_bi = []
    lr_bi = []
    reg_bu = []
    reg_qi = []
    lr_bu = []
    lr_qi = []
    bu = np.zeros(self.trainset.n_users)
    bi = np.zeros(self.trainset.n_items)

    u, i = 0
    r, err, dev_i, dev_u = 0
    global_mean = self.trainset.global_mean

    n_epochs = self.bsl_options.get('n_epochs', 20)
    reg = self.bsl_options.get('reg', 0.02)
    lr = self.bsl_options.get('learning_rate', 0.005)

    for dummy in range(n_epochs):
        for u, i, r in self.trainset.all_ratings():
            err = (r - (global_mean + bu[u] + bi[i]))
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])

    return bu, bi


def get_rng(random_state):
    """Return a 'validated' RNG.

    If random_state is None, use RandomState singleton from numpy.  Else if
    it's an integer, consider it's a seed and initialized an rng with that
    seed. If it's already an rng, return it.
    """
    if random_state is None:
        return np.random.mtrand._rand
    elif isinstance(random_state, (numbers.Integral, np.integer)):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError('Wrong random state. Expecting None, an int or a numpy '
                     'RandomState instance, got a '
                     '{}'.format(type(random_state)))


class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible.

    When raised, the estimation :math:`\hat{r}_{ui}` is set to the global mean
    of all ratings :math:`\mu`.
    """

    pass


class Prediction(namedtuple('Prediction',
                            ['uid', 'iid', 'r_ui', 'est', 'details'])):
    """A named tuple for storing the results of a prediction.

    It's wrapped in a class, but only for documentation and printing purposes.

    Args:
        uid: The (raw) user id. See :ref:`this note<raw_inner_note>`.
        iid: The (raw) item id. See :ref:`this note<raw_inner_note>`.
        r_ui(float): The true rating :math:`r_{ui}`.
        est(float): The estimated rating :math:`\\hat{r}_{ui}`.
        details (dict): Stores additional details about the prediction that
            might be useful for later analysis.
    """

    __slots__ = ()  # for memory saving purpose.

    def __str__(self):
        s = 'user: {uid:<10} '.format(uid=self.uid)
        s += 'item: {iid:<10} '.format(iid=self.iid)
        if self.r_ui is not None:
            s += 'r_ui = {r_ui:1.2f}   '.format(r_ui=self.r_ui)
        else:
            s += 'r_ui = None   '
        s += 'est = {est:1.2f}   '.format(est=self.est)
        s += str(self.details)

        return s

class AlgoBase(object):
    """Abstract class where is defined the basic behavior of a prediction
    algorithm.

    Keyword Args:
        baseline_options(dict, optional): If the algorithm needs to compute a
            baseline estimate, the ``baseline_options`` parameter is used to
            configure how they are computed. See
            :ref:`baseline_estimates_configuration` for usage.
    """

    def __init__(self, **kwargs):

        self.bsl_options = kwargs.get('bsl_options', {})
        self.sim_options = kwargs.get('sim_options', {})
        if 'user_based' not in self.sim_options:
            self.sim_options['user_based'] = True

    def fit(self, trainset):
        """Train an algorithm on a given training set.

        This method is called by every derived class as the first basic step
        for training an algorithm. It basically just initializes some internal
        structures and set the self.trainset attribute.

        Args:
            trainset(:obj:`Trainset <surprise.Trainset>`) : A training
                set, as returned by the :meth:`folds
                <surprise.dataset.Dataset.folds>` method.

        Returns:
            self
        """

        self.trainset = trainset

        # (re) Initialise baselines
        self.bu = self.bi = None

        return self

    def predict(self, uid, iid, r_ui=None, clip=True, verbose=False):
        """Compute the rating prediction for given user and item.

        The ``predict`` method converts raw ids to inner ids and then calls the
        ``estimate`` method which is defined in every derived class. If the
        prediction is impossible (e.g. because the user and/or the item is
        unknown), the prediction is set according to
        :meth:`default_prediction()
        <surprise.prediction_algorithms.algo_base.AlgoBase.default_prediction>`.

        Args:
            uid: (Raw) id of the user. See :ref:`this note<raw_inner_note>`.
            iid: (Raw) id of the item. See :ref:`this note<raw_inner_note>`.
            r_ui(float): The true rating :math:`r_{ui}`. Optional, default is
                ``None``.
            clip(bool): Whether to clip the estimation into the rating scale.
                For example, if :math:`\\hat{r}_{ui}` is :math:`5.5` while the
                rating scale is :math:`[1, 5]`, then :math:`\\hat{r}_{ui}` is
                set to :math:`5`. Same goes if :math:`\\hat{r}_{ui} < 1`.
                Default is ``True``.
            verbose(bool): Whether to print details of the prediction.  Default
                is False.

        Returns:
            A :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` object
            containing:

            - The (raw) user id ``uid``.
            - The (raw) item id ``iid``.
            - The true rating ``r_ui`` (:math:`r_{ui}`).
            - The estimated rating (:math:`\\hat{r}_{ui}`).
            - Some additional details about the prediction that might be useful
              for later analysis.
        """

        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        details = {}
        try:
            est = self.estimate(iuid, iiid)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.default_prediction()
            details['was_impossible'] = True
            details['reason'] = str(e)

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    def default_prediction(self):
        """Used when the ``PredictionImpossible`` exception is raised during a
        call to :meth:`predict()
        <surprise.prediction_algorithms.algo_base.AlgoBase.predict>`. By
        default, return the global mean of all ratings (can be overridden in
        child classes).

        Returns:
            (float): The mean of all ratings in the trainset.
        """

        return self.trainset.global_mean

    def test(self, testset, verbose=False):
        """Test the algorithm on given testset, i.e. estimate all the ratings
        in the given testset.

        Args:
            testset: A test set, as returned by a :ref:`cross-validation
                itertor<use_cross_validation_iterators>` or by the
                :meth:`build_testset() <surprise.Trainset.build_testset>`
                method.
            verbose(bool): Whether to print details for each predictions.
                Default is False.

        Returns:
            A list of :class:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` objects
            that contains all the estimated ratings.
        """

        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]
        return predictions

    def compute_baselines(self):
        """Compute users and items baselines.

        The way baselines are computed depends on the ``bsl_options`` parameter
        passed at the creation of the algorithm (see
        :ref:`baseline_estimates_configuration`).

        This method is only relevant for algorithms using :func:`Pearson
        baseline similarty<surprise.similarities.pearson_baseline>` or the
        :class:`BaselineOnly
        <surprise.prediction_algorithms.baseline_only.BaselineOnly>` algorithm.

        Returns:
            A tuple ``(bu, bi)``, which are users and items baselines."""

        # Firt of, if this method has already been called before on the same
        # trainset, then just return. Indeed, compute_baselines may be called
        # more than one time, for example when a similarity metric (e.g.
        # pearson_baseline) uses baseline estimates.
        if self.bu is not None:
            return self.bu, self.bi

        method = dict(als=baseline_als,
                      sgd=baseline_sgd)

        method_name = self.bsl_options.get('method', 'als')

        try:
            if getattr(self, 'verbose', False):
                print('Estimating biases using', method_name + '...')
            self.bu, self.bi = method[method_name](self)
            return self.bu, self.bi
        except KeyError:
            raise ValueError('Invalid method ' + method_name +
                             ' for baseline computation.' +
                             ' Available methods are als and sgd.')

    def compute_similarities(self):
        """Build the similarity matrix.

        The way the similarity matrix is computed depends on the
        ``sim_options`` parameter passed at the creation of the algorithm (see
        :ref:`similarity_measures_configuration`).

        This method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

        Returns:
            The similarity matrix."""

        construction_func = {'cosine': sims.cosine,
                             'msd': sims.msd,
                             'pearson': sims.pearson,
                             'pearson_baseline': sims.pearson_baseline}

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        min_support = self.sim_options.get('min_support', 1)

        args = [n_x, yr, min_support]

        name = self.sim_options.get('name', 'msd').lower()
        if name == 'pearson_baseline':
            shrinkage = self.sim_options.get('shrinkage', 100)
            bu, bi = self.compute_baselines()
            if self.sim_options['user_based']:
                bx, by = bu, bi
            else:
                bx, by = bi, bu

            args += [self.trainset.global_mean, bx, by, shrinkage]

        try:
            if getattr(self, 'verbose', False):
                print('Computing the {0} similarity matrix...'.format(name))
            sim = construction_func[name](*args)
            if getattr(self, 'verbose', False):
                print('Done computing similarity matrix.')
            return sim
        except KeyError:
            raise NameError('Wrong sim name ' + name + '. Allowed values ' +
                            'are ' + ', '.join(construction_func.keys()) + '.')

    def get_neighbors(self, iid, k):
        """Return the ``k`` nearest neighbors of ``iid``, which is the inner id
        of a user or an item, depending on the ``user_based`` field of
        ``sim_options`` (see :ref:`similarity_measures_configuration`).

        As the similarities are computed on the basis of a similarity measure,
        this method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

        For a usage example, see the :ref:`FAQ <get_k_nearest_neighbors>`.

        Args:
            iid(int): The (inner) id of the user (or item) for which we want
                the nearest neighbors. See :ref:`this note<raw_inner_note>`.

            k(int): The number of neighbors to retrieve.

        Returns:
            The list of the ``k`` (inner) ids of the closest users (or items)
            to ``iid``.
        """

        if self.sim_options['user_based']:
            all_instances = self.trainset.all_users
        else:
            all_instances = self.trainset.all_items

        others = [(x, self.sim[iid, x]) for x in all_instances() if x != iid]
        others.sort(key=lambda tple: tple[1], reverse=True)
        k_nearest_neighbors = [j for (j, _) in others[:k]]

        return k_nearest_neighbors


class SVD_LIST(AlgoBase):
    """
    THIS IS THE LHISTOTY IMPLEMENTATION
    The minimization is performed by a stochastic gradient descent:
    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u

    If user :math:`u` is unknown, then the bias :math:`b_u` and the factors
    :math:`p_u` are assumed to be zero. The same applies for item :math:`i`
    with :math:`b_i` and :math:`q_i`.


    To estimate all the unknown, we minimize the following regularized squared
    error:

    .. math::
        \sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \\right)^2 +
        \lambda\\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\\right)


    .. math::
        b_u &\\leftarrow b_u &+ \gamma (e_{ui} - \lambda b_u)\\\\
        b_i &\\leftarrow b_i &+ \gamma (e_{ui} - \lambda b_i)\\\\
        p_u &\\leftarrow p_u &+ \gamma (e_{ui} \\cdot q_i - \lambda p_u)\\\\
        q_i &\\leftarrow q_i &+ \gamma (e_{ui} \\cdot p_u - \lambda q_i)

    where :math:`e_{ui} = r_{ui} - \\hat{r}_{ui}`. These steps are performed
    over all the ratings of the trainset and repeated ``n_epochs`` times.
    Baselines are initialized to ``0``. User and item factors are randomly
    initialized according to a normal distribution, which can be tuned using
    the ``init_mean`` and ``init_std_dev`` parameters.

    You also have control over the learning rate :math:`\gamma` and the
    regularization term :math:`\lambda`. Both can be different for each
    kind of parameter (see below). By default, learning rates are set to
    ``0.005`` and regularization terms are set to ``0.02``.

    .. _unbiased_note:

    .. note::
        You can choose to use an unbiased version of this algorithm, simply
        predicting:

        .. math::
            \hat{r}_{ui} = q_i^Tp_u

        This is equivalent to Probabilistic Matrix Factorization
        (:cite:`salakhutdinov2008a`, section 2) and can be achieved by setting
        the ``biased`` parameter to ``False``.


    Args:
        n_factors: The number of factors. Default is ``100``.
        n_epochs: The number of iteration of the SGD procedure. Default is
            ``20``.
        biased(bool): Whether to use baselines (or biases). See :ref:`note
            <unbiased_note>` above.  Default is ``True``.
        init_mean: The mean of the normal distribution for factor vectors
            initialization. Default is ``0``.
        init_std_dev: The standard deviation of the normal distribution for
            factor vectors initialization. Default is ``0.1``.
        lr_all: The learning rate for all parameters. Default is ``0.005``.
        reg_all: The regularization term for all parameters. Default is
            ``0.02``.
        lr_bu: The learning rate for :math:`b_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_bi: The learning rate for :math:`b_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_pu: The learning rate for :math:`p_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_qi: The learning rate for :math:`q_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        reg_bu: The regularization term for :math:`b_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_bi: The regularization term for :math:`b_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_pu: The regularization term for :math:`p_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_qi: The regularization term for :math:`q_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for initialization. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same initialization over multiple calls to
            ``fit()``.  If RandomState instance, this same instance is used as
            RNG. If ``None``, the current RNG from numpy is used.  Default is
            ``None``.
        verbose: If ``True``, prints the current epoch. Default is ``False``.

    Attributes:
        pu(numpy array of size (n_users, n_factors)): The user factors (only
            exists if ``fit()`` has been called)
        qi(numpy array of size (n_items, n_factors)): The item factors (only
            exists if ``fit()`` has been called)
        bu(numpy array of size (n_users)): The user biases (only
            exists if ``fit()`` has been called)
        bi(numpy array of size (n_items)): The item biases (only
            exists if ``fit()`` has been called)
    """

    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose

        AlgoBase.__init__(self)

    def fit(self, trainset):

        # @LISTORY IMPLEMENTATION
        AlgoBase.fit(self, trainset)
        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        # @LISTORY IMPLEMENTATION


        bu = []
        bi = []
        reg_pu = []
        lr_pu = []
        reg_bi = []
        lr_bi = []
        reg_bu = []
        reg_qi = []
        lr_bu = []
        lr_qi = []

        #cdef np.ndarray[np.double_t, ndim=2] pu
        # item factors
        #cdef np.ndarray[np.double_t, ndim=2] qi

        #cdef int u, i, f
        #cdef double r, err, dot, puf, qif
        #cdef double global_mean = self.trainset.global_mean

        #cdef double lr_bu = self.lr_bu
        #cdef double lr_bi = self.lr_bi
        #cdef double lr_pu = self.lr_pu
        #cdef double lr_qi = self.lr_qi

        #cdef double reg_bu = self.reg_bu
        #cdef double reg_bi = self.reg_bi
        #cdef double reg_pu = self.reg_pu
       #cdef double reg_qi = self.reg_qi

        rng = get_rng(self.random_state)

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))

        if not self.biased:
            global_mean = 0

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                # compute current error
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                # @LISTORY IMPLEMENTATION

                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):
        # Should we cythonize this as well?

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unknown.')

        return est


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


import pymongo

def get_data_mongodb():
    user_str =  "tom"
    password_str =  "wCrb7Jg0mzWVEYky"
    mongo_str = "mongodb+srv://listory.vmew8.gcp.mongodb.net/Listory"
    #myclient = pymongo.MongoClient(mongo_str)

    mongo_str2 = "mongodb+srv://tom:wCrb7Jg0mzWVEYky@listory.vmew8.gcp.mongodb.net/Listory"
    myclient = pymongo.MongoClient(mongo_str2)


    mydb = myclient["Listory"]
    mycol = mydb["Stories"]

    #tags: "Newsletter"
    myquery = '{tags:"Newsletter"}'
    print("$$$$$$$$$$$")
    result = mycol.find({"tags": "Newsletter"})
    #mydoc = mycol.find(myquery)

    for x in result:
      print(x)

def get_recommendations_JSON_TEST(file_in_name,file_out_name):

    # TEST DATA

    file_in_name1 = "/Users/tomlorenc/Sites/VL_standard/ml/axn/ml/recommend/data/results-20201218-075350.json"
    file_in_name2 = "/Users/tomlorenc/Sites/VL_standard/ml/axn/ml/recommend/data/results-20201218-091546.json"
    file_in_name3 = "/Users/tomlorenc/Sites/VL_standard/ml/axn/ml/recommend/data/results-20201218-091903.json"
    file_in_name4 = "/Users/tomlorenc/Sites/VL_standard/ml/axn/ml/recommend/data/results-20201218-093345.json"
    file_in_name6 = "/Users/tomlorenc/Sites/VL_standard/ml/axn/ml/recommend/data/results-20201218-101736.json"
    file_in_name5 = "/Users/tomlorenc/Sites/VL_standard/ml/axn/ml/recommend/data/results-20201218-094845.json"
    file_in_name7 = "/Users/tomlorenc/Sites/VL_standard/ml/axn/ml/recommend/data/results-20201218-115515.json"


    # read file
    with open(file_in_name2, 'r') as myfile:
        data_json2=myfile.read()

    # parse file
    obj_json2 = json.loads(data_json2)

    print(obj_json2)



    # read file
    with open(file_in_name1, 'r') as myfile:
        data_json1=myfile.read()

    # parse file
    obj_json1 = json.loads(data_json1)

    print(obj_json1)




    # read file
    with open(file_in_name3, 'r') as myfile:
        data_json3=myfile.read()

    # parse file
    obj_json3 = json.loads(data_json3)

    print(obj_json3)


    with open(file_in_name4, 'r') as myfile:
        data_json4=myfile.read()

    # parse file
    obj_json4 = json.loads(data_json4)


    with open(file_in_name5, 'r') as myfile:
       data_json5=myfile.read()

    # parse file
    obj_json5 = json.loads(data_json5)


    with open(file_in_name6, 'r') as myfile:
       data_json6=myfile.read()

    # parse file
    obj_json6 = json.loads(data_json6)

    obj_json = obj_json1 + obj_json2 + obj_json3 + obj_json4 + obj_json5 + obj_json5


    ratings_dict = {}

    itemID_list = []
    userID_list = []
    rating_list = []

    ratings_dict = {'itemID': [], 'userID': [], 'rating': []}
    ratings_dict_5 = {'itemID': [], 'userID': [], 'rating': []}

    # show values
    row_count = 0
    for row in obj_json:
        print ("*****" )
        row_count = row_count + 1
        print ("row " + str(row) )
        print ("*****" )

        itemID = row['sid']

        print(type(itemID))
        print("sid " + str(itemID))

        if itemID is None:
            print("SKIP")
            continue

        print ("*****" )

        userID = row['uid']
        print("userID " + str(userID))

        if userID is None:
            print("SKIP")
            continue

        itemID_list.append(itemID)
        userID_list.append(userID)
        rating_list.append(1)

        ratings_dict_5['itemID'].append(itemID)
        ratings_dict_5['userID'].append(userID)
        ratings_dict_5['rating'].append(1)

    # rememebr to read MOST POPULAR FOR COLD START
    ratings_dict['itemID'].append(itemID_list)
    ratings_dict['userID'].append(userID_list)
    ratings_dict['rating'].append(rating_list)


    df = pd.DataFrame(ratings_dict_5)

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 1))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)

    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    top_n = get_top_n(predictions, n=10)
    print ("PRINT TOP 10 STORY IDs per USERID" )

    # Print the recommended items for each user
    for uid, user_ratings in top_n.items():
        print("******************")
        print(uid, [iid for (iid, _) in user_ratings])
        # convert to json outout


    print ("ROW COUNT *****" + str(row_count))


def calc_listory_score(storyID = 1,
                       userID = 1,
                       current_listory_score = 1,
                       date_published = 1,
                       date_read = 1,
                       time_on_story = 1,
                       story_category= 1):

    #get_data_mongodb()


    return 1


def get_recommendations(file_in_name,file_out_name):

    # TEST DATA
    file_in_name2 = file_in_name

    #file_in_name2 = "/Users/tomlorenc/Sites/VL_standard/ml/axn/ml/recommend/data/results-20201218-115515.json"


    # read file
    with open(file_in_name2, 'r') as myfile:
        data_json2=myfile.read()

    # parse file
    obj_json2 = json.loads(data_json2)

    obj_json = obj_json2
    itemID_list = []
    userID_list = []
    rating_list = []

    ratings_dict = {'itemID': [], 'userID': [], 'rating': []}
    # show values
    row_count = 0
    for row in obj_json:
        print ("*****" )
        row_count = row_count + 1
        print ("row " + str(row) )

        event_params = row['event_params']
        print ("event_params " + str(event_params) )

        for ele in event_params:
            print("#################")
            print("ele " + str(ele))

            print("key " + str(ele['key']))

            if ele['key'] == 'story_sid':
                value = ele['value']
                print(".................................")
                print("value " + str(value))
                itemID = value['string_value']
                print("*****")
                print("itemID " + str(itemID))

            if ele['key'] == 'userdetails_uid':
                value = ele['value']
                print(".................................")
                print("value " + str(value))
                userID = value['string_value']
                print("*****")
                print("userID " + str(userID))

        if itemID is None:
            print("SKIP")
            continue

        if userID is None:
            print("SKIP")
            continue

        listory_score = calc_listory_score()
        print("listory_score " + str(listory_score))

        itemID_list.append(itemID)
        userID_list.append(userID)
        rating_list.append(1)

        ratings_dict['itemID'].append(itemID)
        ratings_dict['userID'].append(userID)
        ratings_dict['rating'].append(1)

    # rememebr to read MOST POPULAR FOR COLD START


    df = pd.DataFrame(ratings_dict)

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 1))

    # The columns must correspond to user id, item id and ratings (in that order).
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)

    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    top_n = get_top_n(predictions, n=10)
    print ("PRINT TOP 10 STORY IDs per USERID" )

    # Print the recommended items for each user
    print_dict = {'userID': [], 'storyID': []}
    userID_list = []
    with open(file_out_name, "a") as outfile:
        for uid, user_ratings in top_n.items():
            print("******************")
            #print(uid, [iid for (iid, _) in user_ratings])
            # convert to json outout
            userID_list.append(uid)
            storyID_list = [iid for (iid, _) in user_ratings]
            storyID_list_flip = []
            storyID_list_flip.append(storyID_list[9])
            storyID_list_flip.append(storyID_list[8])
            storyID_list_flip.append(storyID_list[7])
            storyID_list_flip.append(storyID_list[6])
            storyID_list_flip.append(storyID_list[5])
            storyID_list_flip.append(storyID_list[4])
            storyID_list_flip.append(storyID_list[3])
            storyID_list_flip.append(storyID_list[2])
            storyID_list_flip.append(storyID_list[1])
            storyID_list_flip.append(storyID_list[0])
            print_dict['userID'] = uid
            print_dict['storyID'] = storyID_list_flip
            json.dump(print_dict, outfile)
            print(str(print_dict))





    print ("ROW COUNT *****" + str(row_count))

def parse_command_line():
    """
    reads the command line args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_in')
    parser.add_argument(
        '--file_out',
        help='csv file output encoded using one-hot one-of-K encoding scheme')
    args = parser.parse_args()
    return args


def main():
    """
Factorization. Matrix Factorization algorithm using Singular Value Decomposition
(SVD) for LHISORY with gradient decent
This CF model is developed using machine learning Singular Value Decomposition
(SVD) to predict user’s rating of unrated items - THAT THEY HAVE NOT SEEN BEFORE.
The idea behind this model is that attitudes or preferences of a user can be determined by a number of hidden features.
We can call these embedded features. Matrix Factorization is implemented as an optimization problem with loss functions and constraints.
The constraints are chosen based on the property of our model Non

      """

    args = parse_command_line()
    file_in_name = args.file_in
    file_out_name = args.file_out
    get_recommendations(file_in_name,file_out_name)



if __name__ == '__main__':
    main()
