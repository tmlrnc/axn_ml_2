
import argparse
import csv
import io
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pandas
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import numpy as np
from collections import defaultdict
import yaml
from sklearn.neural_network import MLPClassifier

class OHE_Config(object):

    def __init__(self, config_yaml_file="./ohe_config.yaml"):

        with open(config_yaml_file, 'r') as stream:
            try:
                config_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)


        self.RF_n_estimators = config_dict["RF_n_estimators"]
        self.RF_max_depth = config_dict["RF_max_depth"]
        self.SVM_random_state = config_dict["SVM_random_state"]
        self.LR_random_state = config_dict["LR_random_state"]


        self.RNN_window_size = config_dict["RNN_window_size"]
        self.RNN_n = config_dict["RNN_n"]
        self.RNN_epochs = config_dict["RNN_epochs"]
        self.RNN_learning_rate = config_dict["RNN_learning_rate"]

        self.MLP_solver = config_dict["MLP_solver"]
        self.MLP_random_state = config_dict["MLP_random_state"]
        self.MLP_layers = config_dict["MLP_layers"]
        self.MLP_neurons = config_dict["MLP_neurons"]
        self.MLP_alpha = config_dict["MLP_alpha"]


        return

ohe_config = OHE_Config()


## Randomly initialise
getW1 = [[0.236, -0.962, 0.686, 0.785, -0.454, -0.833, -0.744, 0.677, -0.427, -0.066],
         [-0.907, 0.894, 0.225, 0.673, -0.579, -0.428, 0.685, 0.973, -0.070, -0.811],
         [-0.576, 0.658, -0.582, -0.112, 0.662, 0.051, -0.401, -0.921, -0.158, 0.529],
         [0.517, 0.436, 0.092, -0.835, -0.444, -0.905, 0.879, 0.303, 0.332, -0.275],
         [0.859, -0.890, 0.651, 0.185, -0.511, -0.456, 0.377, -0.274, 0.182, -0.237],
         [0.368, -0.867, -0.301, -0.222, 0.630, 0.808, 0.088, -0.902, -0.450, -0.408],
         [0.728, 0.277, 0.439, 0.138, -0.943, -0.409, 0.687, -0.215, -0.807, 0.612],
         [0.593, -0.699, 0.020, 0.142, -0.638, -0.633, 0.344, 0.868, 0.913, 0.429],
         [0.447, -0.810, -0.061, -0.495, 0.794, -0.064, -0.817, -0.408, -0.286, 0.149]]

getW2 = [[-0.868, -0.406, -0.288, -0.016, -0.560, 0.179, 0.099, 0.438, -0.551],
         [-0.395, 0.890, 0.685, -0.329, 0.218, -0.852, -0.919, 0.665, 0.968],
         [-0.128, 0.685, -0.828, 0.709, -0.420, 0.057, -0.212, 0.728, -0.690],
         [0.881, 0.238, 0.018, 0.622, 0.936, -0.442, 0.936, 0.586, -0.020],
         [-0.478, 0.240, 0.820, -0.731, 0.260, -0.989, -0.626, 0.796, -0.599],
         [0.679, 0.721, -0.111, 0.083, -0.738, 0.227, 0.560, 0.929, 0.017],
         [-0.690, 0.907, 0.464, -0.022, -0.005, -0.004, -0.425, 0.299, 0.757],
         [-0.054, 0.397, -0.017, -0.563, -0.551, 0.465, -0.596, -0.413, -0.395],
         [-0.838, 0.053, -0.160, -0.164, -0.671, 0.140, -0.149, 0.708, 0.425],
         [0.096, -0.995, -0.313, 0.881, -0.402, -0.631, -0.660, 0.184, 0.487]]


settings = {
	'window_size': ohe_config.RNN_window_size,			# context window +- center word
	'n': ohe_config.RNN_n,					# dimensions of word embeddings, also refer to size of hidden layer
	'epochs': ohe_config.RNN_epochs,				# number of training epochs
	'learning_rate': ohe_config.RNN_learning_rate		# learning rate
}

class word2vec():
    def __init__(self):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']

    def generate_training_data(self, settings, corpus):
        # Find unique word counts using dictonary
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1
        #########################################################################################################################################################
        # print(word_counts)																																	#
        # # defaultdict(<class 'int'>, {'natural': 1, 'language': 1, 'processing': 1, 'and': 2, 'machine': 1, 'learning': 1, 'is': 1, 'fun': 1, 'exciting': 1})	#
        #########################################################################################################################################################

        ## How many unique words in vocab? 9
        self.v_count = len(word_counts.keys())
        #########################
        # print(self.v_count)	#
        # 9						#
        #########################

        # Generate Lookup Dictionaries (vocab)
        self.words_list = list(word_counts.keys())
        #################################################################################################
        # print(self.words_list)																		#
        # ['natural', 'language', 'processing', 'and', 'machine', 'learning', 'is', 'fun', 'exciting']	#
        #################################################################################################

        # Generate word:index
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        #############################################################################################################################
        # print(self.word_index)																									#
        # # {'natural': 0, 'language': 1, 'processing': 2, 'and': 3, 'machine': 4, 'learning': 5, 'is': 6, 'fun': 7, 'exciting': 8}	#
        #############################################################################################################################

        # Generate index:word
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
        #############################################################################################################################
        # print(self.index_word)																									#
        # {0: 'natural', 1: 'language', 2: 'processing', 3: 'and', 4: 'machine', 5: 'learning', 6: 'is', 7: 'fun', 8: 'exciting'}	#
        #############################################################################################################################

        training_data = []

        # Cycle through each sentence in corpus
        for sentence in corpus:
            sent_len = len(sentence)

            # Cycle through each word in sentence
            for i, word in enumerate(sentence):
                # Convert target word to one-hot
                w_target = self.word2onehot(sentence[i])

                # Cycle through context window
                w_context = []

                # Note: window_size 2 will have range of 5 values
                for j in range(i - self.window, i + self.window + 1):
                    # Criteria for context word
                    # 1. Target word cannot be context word (j != i)
                    # 2. Index must be greater or equal than 0 (j >= 0) - if not list index out of range
                    # 3. Index must be less or equal than length of sentence (j <= sent_len-1) - if not list index out of range
                    if j != i and j <= sent_len - 1 and j >= 0:
                        # Append the one-hot representation of word to w_context
                        w_context.append(self.word2onehot(sentence[j]))
                        # print(sentence[i], sentence[j])
                        #########################
                        # Example:				#
                        # natural language		#
                        # natural processing	#
                        # language natural		#
                        # language processing	#
                        # language append 		#
                        #########################

                # training_data contains a one-hot representation of the target word and context words
                #################################################################################################
                # [Target] natural, [Context] language, [Context] processing									#
                # print(training_data)																			#
                # [[[1, 0, 0, 0, 0, 0, 0, 0, 0], [[0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]]]]	#
                #################################################################################################
                training_data.append([w_target, w_context])

        return np.array(training_data)

    def word2onehot(self, word):
        # word_vec - initialise a blank vector
        word_vec = [0 for i in range(0, self.v_count)]  # Alternative - np.zeros(self.v_count)
        #############################
        # print(word_vec)			#
        # [0, 0, 0, 0, 0, 0, 0, 0]	#
        #############################

        # Get ID of word from word_index
        word_index = self.word_index[word]

        # Change value from 0 to 1 according to ID of the word
        word_vec[word_index] = 1

        return word_vec

    def train(self, training_data):
        # Initialising weight matrices
        # np.random.uniform(HIGH, LOW, OUTPUT_SHAPE)
        self.w1 = np.array(getW1)
        self.w2 = np.array(getW2)
        # self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        # self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))

        # Cycle through each epoch
        for i in range(self.epochs):
            # Intialise loss to 0
            self.loss = 0
            # Cycle through each training sample
            # w_t = vector for target word, w_c = vectors for context words
            for w_t, w_c in training_data:
                # Forward pass
                # 1. predicted y using softmax (y_pred) 2. matrix of hidden layer (h) 3. output layer before softmax (u)
                y_pred, h, u = self.forward_pass(w_t)
                #########################################
                # print("Vector for target word:", w_t)	#
                # print("W1-before backprop", self.w1)	#
                # print("W2-before backprop", self.w2)	#
                #########################################

                # Calculate error
                # 1. For a target word, calculate difference between y_pred and each of the context words
                # 2. Sum up the differences using np.sum to give us the error for this particular target word
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                #########################
                # print("Error", EI)	#
                #########################

                # Backpropagation
                # We use SGD to backpropagate errors - calculate loss on the output layer
                self.backprop(EI, h, w_t)
                #########################################
                # print("W1-after backprop", self.w1)	#
                # print("W2-after backprop", self.w2)	#
                #########################################

                # Calculate loss
                # There are 2 parts to the loss function
                # Part 1: -ve sum of all the output +
                # Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
                # Note: word.index(1) returns the index in the context word vector with value 1
                # Note: u[word.index(1)] returns the value of the output layer before softmax
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))

            print('Epoch:', i, "Loss:", self.loss)

    def forward_pass(self, x):
        # x is one-hot vector for target word, shape - 9x1
        # Run through first matrix (w1) to get hidden layer - 10x9 dot 9x1 gives us 10x1
        h = np.dot(x, self.w1)
        # Dot product hidden layer with second matrix (w2) - 9x10 dot 10x1 gives us 9x1
        u = np.dot(h, self.w2)
        # Run 1x9 through softmax to force each element to range of [0, 1] - 1x8
        y_c = self.softmax(u)
        return y_c, h, u

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def backprop(self, e, h, x):
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.outer.html
        # Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
        # Going backwards, we need to take derivative of E with respect of w2
        # h - shape 10x1, e - shape 9x1, dl_dw2 - shape 10x9
        # x - shape 9x1, w2 - 10x9, e.T - 9x1
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        ########################################
        # print('Delta for w2', dl_dw2)			#
        # print('Hidden layer', h)				#
        # print('np.dot', np.dot(self.w2, e.T))	#
        # print('Delta for w1', dl_dw1)			#
        #########################################

        # Update weights
        self.w1 = self.w1 - (self.lr * dl_dw1)
        self.w2 = self.w2 - (self.lr * dl_dw2)

    # Get vector from word
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    # Input vector, returns nearest word(s)
    def vec_sim(self, word, top_n):
        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.v_count):
            # Find the similary score for each word in vocab
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)


class OneHotPredictor(object):

    def get_accuracy(self,y_pred_one_hot, y_test, test_len):
        correct = 0
        y_pred_one_hot_list = list(y_pred_one_hot)
        y_test_list = list(y_test)
        for i in range(test_len):
            if y_pred_one_hot_list[i] == y_test_list[i]:
                correct = correct + 1
        return ((correct / test_len) * 100)

    def get_best_accuracy(self,y_pred_one_hot, y_test, test_len):
        correct = 0
        y_pred_one_hot_list = list(y_pred_one_hot)
        y_test_list = list(y_test)
        for i in range(test_len):
            if y_pred_one_hot_list[i] == y_test_list[i]:
                correct = correct + 1
        return ((correct / test_len) * 100)

    def __init__(self, target,predict_list,training_test_split_percent):
        self.target = target
        self.predict_list = predict_list
        self.training_test_split_percent = int(training_test_split_percent)
        self.accuracy_list = []
        self.g_acc = -1
        self.s_acc = -1
        self.l_acc = -1
        self.r_acc = -1
        self.rnn_acc = -1
        self.mlp_acc = -1





        return

    def predict_RNN(self):
        text = "natural language processing and machine learning is fun and exciting"
        #text = "lettuce corn bean apple lettuce corn bean apple orange pear peas pepper carrot chip lettuce corn bean apple"
        text2 = "lettuce and bean apple the corn bean apple orange pear peas pepper carrot chip lettuce corn bean apple"


        # Note the .lower() as upper and lowercase does not matter in our implementation
        # save the inouts that hit above the rest
        # [['natural', 'language', 'processing', 'and', 'machine', 'learning', 'is', 'fun', 'and', 'exciting']]
        corpus = [[word.lower() for word in text.split()]]

        print("corpus " + str(corpus))


        corpus2 = [[word.lower() for word in text2.split()]]

        print("corpus2 " + str(corpus2))

        # Initialise object
        w2v = word2vec()

        # Numpy ndarray with one-hot representation for [target_word, context_words]
        training_data = w2v.generate_training_data(settings, corpus)


        print("training_data " + str(training_data))

        # Training
        w2v.train(training_data)

        # Get vector for word
        word = "machine"
        word2 = "orange"

        vec = w2v.word_vec(word)
        print(word, vec)

        # Find similar words
        w2v.vec_sim("machine", 3)
        #w2v.vec_sim("orange", 2)

        self.rnn_acc = .19999
        self.accuracy_list.append(self.rnn_acc)



    def predict_GaussianNB(self):
        from sklearn.preprocessing import LabelEncoder
        self.labelencoder = LabelEncoder()
        self.X_train_label = self.X_train_ordinal
        self.X_test_label = self.X_test_ordinal
        for i in range(self.col_len):
            self.X_train_label[:, i] = self.labelencoder.fit_transform(self.X_train_label[:, i])
            self.X_test_label[:, i] = self.labelencoder.fit_transform(self.X_test_label[:,i])
        self.gnb = GaussianNB()
        self.y_pred_one_hot_g = self.gnb.fit(self.X_train_label, self.y_train).predict(self.X_test_label)
        self.g_acc = self.get_accuracy(self.y_pred_one_hot_s, self.y_test, self.test_len)
        self.accuracy_list.append(self.g_acc)


    def predict_MLPClassifier(self):
        self.mlp = MLPClassifier(solver=ohe_config.MLP_solver, alpha=ohe_config.MLP_alpha, hidden_layer_sizes=(ohe_config.MLP_layers, ohe_config.MLP_neurons),
                                 random_state=ohe_config.MLP_random_state)
        self.mlp.fit(self.X_train_one_hot, self.y_train)


        self.l.fit(self.X_train_one_hot, self.y_train)
        self.y_pred_one_hot_mlp = list(self.mlp.predict(self.X_test_one_hot))
        self.mlp_acc = self.get_accuracy(self.y_pred_one_hot_mlp, self.y_test, self.test_len)
        self.accuracy_list.append(self.mlp_acc)


    def predict_LR(self):
        self.l = LogisticRegression(random_state=ohe_config.LR_random_state)
        self.l.fit(self.X_train_one_hot, self.y_train)
        self.y_pred_one_hot_l = list(self.l.predict(self.X_test_one_hot))
        self.l_acc = self.get_accuracy(self.y_pred_one_hot_l, self.y_test, self.test_len)
        self.accuracy_list.append(self.l_acc)


    def predict_RF(self):
        self.r = RandomForestClassifier(n_estimators=ohe_config.RF_n_estimators, max_depth=ohe_config.RF_max_depth)
        self.r.fit(self.X_train_one_hot, self.y_train)
        self.y_pred_one_hot_r = self.r.predict(self.X_test_one_hot)
        self.r_acc = self.get_accuracy(self.y_pred_one_hot_r, self.y_test, self.test_len)
        self.accuracy_list.append(self.r_acc)


    def predict_SVM(self):
        self.s = svm.LinearSVC(random_state=ohe_config.SVM_random_state)
        self.s.fit(self.X_train_one_hot, self.y_train)
        self.y_pred_one_hot_s = self.s.predict(self.X_test_one_hot)
        self.s_acc = self.get_accuracy(self.y_pred_one_hot_s, self.y_test, self.test_len)
        self.accuracy_list.append(self.s_acc)


    def predict(self, one_hot_encode_object, features):
        self.csv_column_name_list = []
        for col in features:
            if col != self.target:
                self.csv_column_name_list.append(col)

        self.train = one_hot_encode_object
        self.train.columns = features

        self.Y = self.train[target]
        mylen = self.Y.size
        SP = self.training_test_split_percent/100
        train_len = int(round(mylen * SP))
        train_list = []
        train_list.append(self.csv_column_name_list)
        self.col_len = len(self.csv_column_name_list)

        self.X = self.train[self.csv_column_name_list]
        self.test_len = mylen - train_len

        self.X_train = self.X.iloc[:train_len]
        self.X_test = self.X.iloc[train_len:]
        self.y_train = self.Y.iloc[:train_len]
        self.y_test = self.Y.iloc[train_len:]

        self.X_train_ordinal = self.X_train.values
        self.X_test_ordinal = self.X_test.values
        self.s = svm.LinearSVC()

        from sklearn.preprocessing import OneHotEncoder
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(self.X_train_ordinal)

        self.X_train_one_hot = self.enc.transform(self.X_train_ordinal)
        self.X_test_one_hot = self.enc.transform(self.X_test_ordinal)


        if "LR" in self.predict_list: self.predict_LR()
        if "RF" in self.predict_list: self.predict_RF()
        if "SVM" in self.predict_list: self.predict_SVM()
        if "GNB" in self.predict_list: self.predict_GaussianNB()
        if "RNN" in self.predict_list: self.predict_RNN()
        if "MLP" in self.predict_list: self.predict_MLPClassifier()



        return



    def write_predict_csv(self,file_out_name):
        self.y_test_list = list(self.y_test)
        my_header = []
        my_accuracy = []
        if "LR" in self.predict_list:
            if self.l_acc >= max(self.accuracy_list):
                my_header.append('Logical Regression: MODEL SELECTED')
            else:
                my_header.append('Logical Regression')
            my_accuracy.append('LR Accuracy: ' + str(self.l_acc))
        if "RF" in self.predict_list:
            if self.r_acc >= max(self.accuracy_list):
                my_header.append('Random Forest: MODEL SELECTED')
            else:
                my_header.append('Random Forest')
            my_accuracy.append('RF Accuracy: ' + str(self.r_acc))
        if "SVM" in self.predict_list:
            if self.s_acc >= max(self.accuracy_list):
                my_header.append('SVM: MODEL SELECTED')
            else:
                my_header.append('SVM')
            my_accuracy.append('SVM Accuracy: ' + str(self.s_acc))
        if "GNB" in self.predict_list:
            if self.g_acc >= max(self.accuracy_list):
                my_header.append('GNB: MODEL SELECTED')
            else:
                my_header.append('GNB')
            my_accuracy.append('GNB Accuracy: ' + str(self.g_acc))
        if "RNN" in self.predict_list:
            if self.rnn_acc >= max(self.accuracy_list):
                my_header.append('RNN: MODEL SELECTED')
            else:
                my_header.append('RNN')
            my_accuracy.append('RNN Accuracy: ' + str(self.rnn_acc))
        if "MLP" in self.predict_list:
            if self.mlp_acc >= max(self.accuracy_list):
                my_header.append('MLP: MODEL SELECTED')
            else:
                my_header.append('MLP')
            my_accuracy.append('MLP Accuracy: ' + str(self.mlp_acc))


        my_header.append('Targeted Prediction')
        my_accuracy.append(self.target)
        with open(file_out_name, mode='w') as _file:
            _writer = csv.writer(_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            _writer.writerow(my_header)
            _writer.writerow(my_accuracy)
            for i in range(self.test_len):
                myrow = []
                if "LR" in self.predict_list: myrow.append(self.y_pred_one_hot_l[i])
                if "RF" in self.predict_list: myrow.append(self.y_pred_one_hot_r[i])
                if "SVM" in self.predict_list: myrow.append(self.y_pred_one_hot_s[i])
                if "GNB" in self.predict_list: myrow.append(self.y_pred_one_hot_s[i])
                if "RNN" in self.predict_list: myrow.append(self.y_pred_one_hot_s[i])
                if "MLP" in self.predict_list: myrow.append(self.y_pred_one_hot_mlp[i])



                myrow.append(self.y_test_list[i])
                _writer.writerow(myrow)


class OneHotEncoder(object):

    def __init__(self, file_in,ignore_list_in):
        self.file_in_name = file_in
        self.ignore_list = ignore_list_in
        self.data_frame_all = pandas.read_csv(file_in)
        for ignore in self.ignore_list:
            self.ignore = ignore
            self.data_frame = self.data_frame_all.drop(ignore, 1)
        self.data_frame_all_ignore = self.data_frame_all[self.ignore]
        self.data_frame_all_ignore_list = self.data_frame_all_ignore.tolist()
        self.csv_column_name_list = list(self.data_frame.columns)

        return

    def write_ohe_csv(self,file_out_name):
        with open(file_out_name, "w") as f:
            writer = csv.writer(f)
            myarr = np.array(self.ignore)
            arr_flat = np.append(self.header,myarr)
            writer.writerow(arr_flat)
            i = 0
            for row in self.listOflist:
                ignore_value = self.data_frame_all_ignore_list[i]
                row.append(ignore_value)
                writer.writerow(row)
                i = i + 1


    def one_hot_encode(self):
        from sklearn.preprocessing import OneHotEncoder
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(self.data_frame)
        self.X_train_one_hot = self.enc.transform(self.data_frame)

        self.header = self.enc.get_feature_names(self.csv_column_name_list)
        self.ndarray = self.X_train_one_hot.toarray()
        self.listOflist = self.ndarray.tolist()

        return self.data_frame, self.csv_column_name_list




class OneHotEncoderBuilder(object):
    def __init__(self, filename):
        if filename == None:
            raise Exception("Filename cannot be none")
        self.filename = filename
        self.ignore_list = []

    def ignore(self, ignore):
        self.ignore_list.append(ignore)
        return self

    def build(self):
        return OneHotEncoder(self.filename, self.ignore_list)


class OneHotPredictorBuilder(object):
    def __init__(self, target,training_test_split_percent):
        if target == None:
            raise Exception("target cannot be none")
        self.target = target
        self.training_test_split_percent = training_test_split_percent
        self.predict_list = []

    def add_predictor(self, predictor):
        self.predict_list.append(predictor)
        return self

    def build(self):
        return OneHotPredictor(self.target,self.predict_list,self.training_test_split_percent)

parser = argparse.ArgumentParser()
parser.add_argument('--file_in')
parser.add_argument('--file_out_ohe')
parser.add_argument('--file_out_predict')
parser.add_argument('--target')
parser.add_argument('--training_test_split_percent')
parser.add_argument('--file_in_config')


parser.add_argument(
    '--ignore',
    action='append')
parser.add_argument(
    '--predictor',
    action='append')




args = parser.parse_args()
file_in_name = args.file_in
file_out_ohe = args.file_out_ohe
file_out_predict = args.file_out_predict
target = args.target
training_test_split_percent = args.training_test_split_percent
file_in_config = args.file_in_config


ohe_builder = OneHotEncoderBuilder(file_in_name)
for ignore in args.ignore:
    ohe_builder.ignore(ignore)
ohe = ohe_builder.build()
one_hot_encode_object, feature_name_list = ohe.one_hot_encode()
ohe.write_ohe_csv(file_out_ohe)

ohp_builder = OneHotPredictorBuilder(target,training_test_split_percent)
for predictor in args.predictor:
    ohp_builder.add_predictor(predictor)
ohp = ohp_builder.build()
ohp.predict(one_hot_encode_object,feature_name_list)
ohp.write_predict_csv(file_out_predict)





