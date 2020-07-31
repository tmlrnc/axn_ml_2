"""
reads config yaml file into config dictionary data object
to drive the scikit learn machine learning algorithm
"""

import yaml

class OHEConfig():
    """
  reads config yaml file into config dictionary data object
  to drive the scikit learn machine learning algorithm
      """
    # pylint: disable=too-many-instance-attributes
    # We are using this class to store ML hyperparameters
    # pylint: disable=too-few-public-methods
    # We are using this class to store ML hyperparameters
    ohe_config = None
    def __init__(self, config_yaml_file="./ohe_config.yaml"):
        if OHEConfig.ohe_config is not None:
            raise Exception("OHE Config is a singleton, it should not be initialized more than once.")
        with open(config_yaml_file, 'r') as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        self.rf_estimators = config_dict["rf_estimators"]
        self.rf_max_depth = config_dict["rf_max_depth"]
        self.s_random_state = config_dict["s_random_state"]
        self.lr_random_state = config_dict["lr_random_state"]
        self.dtc_random_state = config_dict["dtc_random_state"]
        self.rnr_radius = config_dict["rnr_radius"]
        self.rnn_window_size = config_dict["rnn_window_size"]
        self.rnn_n = config_dict["rnn_n"]
        self.rnn_epochs = config_dict["rnn_epochs"]
        self.rnn_learning_rate = config_dict["rnn_learning_rate"]
        self.mlp_solver = config_dict["mlp_solver"]
        self.mlp_random_state = config_dict["mlp_random_state"]
        self.mlp_layers = config_dict["mlp_layers"]
        self.mlp_neurons = config_dict["mlp_neurons"]
        self.mlp_alpha = config_dict["mlp_alpha"]
        OHEConfig.ohe_config = self

def get_ohe_config():
    """
  reads config yaml file into config dictionary data object
  to drive the scikit learn machine learning algorithm
      """
    if OHEConfig.ohe_config is None:
        raise Exception("OHE Config has not been initialized, call init_ohe_config first.")
    return OHEConfig.ohe_config
def init_ohe_config(config_yaml_file):
    """
  reads config yaml file into config dictionary data object
  to drive the scikit learn machine learning algorithm
      """
    return OHEConfig(config_yaml_file)
