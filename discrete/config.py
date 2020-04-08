import yaml

class OHE_Config(object):
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

        self.RF_n_estimators = config_dict["RF_n_estimators"]
        self.RF_max_depth = config_dict["RF_max_depth"]
        self.SVM_random_state = config_dict["SVM_random_state"]
        self.LR_random_state = config_dict["LR_random_state"]
        self.DTC_random_state = config_dict["DTC_random_state"]
        self.RNR_radius = config_dict["RNR_radius"]


        self.RNN_window_size = config_dict["RNN_window_size"]
        self.RNN_n = config_dict["RNN_n"]
        self.RNN_epochs = config_dict["RNN_epochs"]
        self.RNN_learning_rate = config_dict["RNN_learning_rate"]

        self.MLP_solver = config_dict["MLP_solver"]
        self.MLP_random_state = config_dict["MLP_random_state"]
        self.MLP_layers = config_dict["MLP_layers"]
        self.MLP_neurons = config_dict["MLP_neurons"]
        self.MLP_alpha = config_dict["MLP_alpha"]

        OHE_Config.ohe_config = self

def get_ohe_config():
    if OHE_Config.ohe_config == None:
        raise Exception("OHE Config has not been initialized, call init_ohe_config first.")
    return OHE_Config.ohe_config

def init_ohe_config(config_yaml_file):
    return OHE_Config(config_yaml_file)