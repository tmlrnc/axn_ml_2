import yaml

class OHE_Config():
    """
  reads config yaml file into config dictionary data object
  to drive the scikit learn machine learning algorithm
      """
    ohe_config = None

    def __init__(self, config_yaml_file="./ohe_config.yaml"):
        if OHE_Config.ohe_config is not None:
            raise Exception("OHE Config is a singleton, it should not be initialized more than once.")

        with open(config_yaml_file, 'r') as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        self.rf_estimators = config_dict["rf_estimators"]
        self.rf_max_depth = config_dict["rf_max_depth"]
        self.r_n_estimators = config_dict["r_n_estimators"]
        self.r_max_depth = config_dict["r_max_depth"]
        self.s_random_state = config_dict["s_random_state"]
        self.s_max_iterations = config_dict["s_max_iterations"]
        self.s_multi_class = config_dict["s_multi_class"]
        self.rfc_max_features = config_dict["rfc_max_features"]
        self.perceptron_class_weight = config_dict["perceptron_class_weight"]
        self.lr_random_state = config_dict["lr_random_state"]
        self.logisticregression_solver = config_dict["logisticregression_solver"]
        self.dtc_random_state = config_dict["dtc_random_state"]
        self.rnr_radius = config_dict["rnr_radius"]
        self.kmeans_bins = config_dict["kmeans_bins"]
        self.kneighbors_classifiernn = config_dict["kneighbors_classifiernn"]
        self.rnn_window_size = config_dict["rnn_window_size"]
        self.rnn_n = config_dict["rnn_n"]
        self.rnn_epochs = config_dict["rnn_epochs"]
        self.rnn_learning_rate = config_dict["rnn_learning_rate"]
        self.mlp_solver = config_dict["mlp_solver"]
        self.MLP_random_state = config_dict["mlp_random_state"]
        self.MLP_layers = config_dict["mlp_layers"]
        self.MLP_neurons = config_dict["mlp_neurons"]
        self.MLP_alpha = config_dict["mlp_alpha"]
        OHE_Config.ohe_config = self

def get_ohe_config():
    if OHE_Config.ohe_config is None:
        raise Exception("OHE Config has not been initialized, call init_ohe_config first.")
    return OHE_Config.ohe_config

def init_ohe_config(config_yaml_file):
    return OHE_Config(config_yaml_file)
