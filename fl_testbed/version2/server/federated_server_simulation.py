"""
    This script launches the federated version  simulation of a RUL, it has both RUL regressor model for the
    RUL predictor. The script uses its specific small dataset version: ex. if this machine is labeled
    as 0 then it will use data_set_0.csv therefore it will only process data that belongs to this label.
    It fillows the same execution steps and model as the centralized learning.
     

"""
import os
import math

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf


NUM_CLIENTS = 5
ROUNDS=40


class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train) -> None:
        super().__init__()
        self.model = model
        split_idx = math.floor(len(x_train) * 0.9)  # Use 10% of x_train for validation
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=2, verbose=2)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": acc}


def client_fn(cid: str) -> fl.client.Client:
    print("CHEKING CID",str(cid))
    # Load model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    partition_size = math.floor(len(x_train) / NUM_CLIENTS)
    idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
    print("INDEXES", str(idx_from), str(idx_to))
    x_train_cid = x_train[idx_from:idx_to] / 255.0
    y_train_cid = y_train[idx_from:idx_to]

    # Create and return client
    return FlwrClient(model, x_train_cid, y_train_cid)


def main() -> None:
    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": 4},
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=5,
            min_evaluate_clients=5,
            min_available_clients=NUM_CLIENTS,
        ),
    )


if __name__ == "__main__":
    main()


# #GENERAL LIBRARIES
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.random import set_seed
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import (
#     mean_squared_error,
#     mean_absolute_error,
#     explained_variance_score,
#     accuracy_score,
#     r2_score,
# )
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     roc_auc_score,
#     recall_score,
#     cohen_kappa_score,
#     matthews_corrcoef,
#     precision_recall_fscore_support,
# )

# from tensorflow.keras.layers import Dropout,Conv1D,MaxPooling1D,Flatten, Activation, Dense
# # from tensorflow.keras.regularizers import L1L2
# from tensorflow.keras.optimizers import RMSprop
# import tensorflow as tf
# from datetime import datetime
# from tensorflow.keras.layers import Dropout,Conv1D,MaxPooling1D,Flatten, Activation
# # from tensorflow.keras.regularizers import L1L2
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.models import Sequential
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# # from autosklearn.regression import AutoSklearnRegressor
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.regularizers import L2,L1,L1L2
# import tensorflow as tf
# #LSTM
# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
# from tensorflow.keras.regularizers import L1L2
# from tensorflow.keras.layers import Dropout,Conv1D,MaxPooling1D,Flatten, Activation, Dense
# from tensorflow.keras.regularizers import L1L2
# from tensorflow.keras.optimizers import RMSprop
# import tensorflow as tf
# from datetime import datetime
# from tensorflow.keras.layers import Dropout,Conv1D,MaxPooling1D,Flatten, Activation
# from tensorflow.keras.regularizers import L1L2
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.models import Sequential
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# # from autosklearn.regression import AutoSklearnRegressor
# from tensorflow.keras.callbacks import EarlyStopping
# import tensorflow as tf
# from tensorflow.keras.regularizers import L2,L1,L1L2
# import pandas as pd
# import os
# import numpy as np
# RNDSEED = np.random.seed(39)
# PRECISION = 4 # 3 of digits to keep after the decimal point
# from sklearn import tree
# from sklearn import metrics
# from sklearn.metrics import f1_score, cohen_kappa_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_regression
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# RUNNING_ON_COLAB = False # we assume running on CoLab! Change to False if running locally.
# from keras.regularizers import l2, l1
# from tensorflow.keras.layers import Dropout
# from keras.layers import LeakyReLU
# import json
# from sklearn.metrics import r2_score, mean_squared_error
# from sklearn.metrics import f1_score, cohen_kappa_score
# import tensorflow as tf
# from sklearn.model_selection import KFold
# import sys
# import argparse
# import matplotlib.pyplot as plt
# import numpy as np
# from collections.abc import Iterable
# import pandas as pd
# import os
# import itertools
# import pickle
# import random as rn
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.layers import Dense
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import (
#     mean_squared_error,
#     mean_absolute_error,
#     explained_variance_score,
#     accuracy_score,
#     r2_score,
#     mean_absolute_percentage_error,
#     max_error,
#     mean_squared_log_error,
#     median_absolute_error,
# )
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     roc_auc_score,
#     recall_score,
#     cohen_kappa_score,
#     matthews_corrcoef,
#     precision_recall_fscore_support,
# )
# from sklearn import preprocessing
# from sklearn.model_selection import StratifiedKFold
# from sklearn import preprocessing
# from sklearn.model_selection import StratifiedKFold
# from abc import ABC
# import pickle
# from typing import Dict, Tuple
# from tensorflow.keras.models import Sequential, save_model, load_model
# from flwr.common import Config, NDArrays, Scalar
# from sklearn.model_selection import KFold
# from sklearn import preprocessing
# import numpy as np
# import pandas as pd
# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import os
# from sklearn.model_selection import train_test_split
# # from autosklearn.regression import AutoSklearnRegressor
# from tensorflow.keras.regularizers import L2,L1,L1L2
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import os
# import numpy as np
# RNDSEED = np.random.seed(39)
# PRECISION = 4 # 3 of digits to keep after the decimal point
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# RUNNING_ON_COLAB = False # we assume running on CoLab! Change to False if running locally.
# import tensorflow as tf
# from sklearn.model_selection import KFold
# import numpy as np
# import pandas as pd
# import os
# import pickle
# import random as rn
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split
# from sklearn import preprocessing


# #EX LIBRARIES
# import flwr as fl
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from typing import Tuple, List
# import sys
# import CustomNumpyClient_OFFSET
# import argparse
# import pickle




# def load_data(TRANSFORMED_FOLDER,data_file_name,dfn_test_x,dfn_test_y):
        
#     print("INI")

#     #TRAINING
#     with open(
#             TRANSFORMED_FOLDER + data_file_name,
#             # self.concatenated_identifier +
#             # self.FILE_NAME + "y_train.pkl",
#             "rb",
#     ) as file:
#         # self.df = pickle.load(file)
#         df=pd.read_pickle(file)
    

#     #TEST
#     with open(
#             TRANSFORMED_FOLDER + dfn_test_x.replace('\n', ''),
#             # self.concatenated_identifier +
#             # self.FILE_NAME + "y_train.pkl",
#             "rb",
#     ) as file:
#         X_test = pickle.load(file)

#     with open(
#             TRANSFORMED_FOLDER + dfn_test_y.replace('\n', ''),
#             # self.concatenated_identifier +
#             # self.FILE_NAME + "y_train.pkl",
#             "rb",
#     ) as file:
#         y_test = pickle.load(file)

#     return df,X_test,y_test

    
    
    
    
    

# def model_definition(df,test_inputs,test_out,RNDSEED):

#     print("#ASSIGNAMEMNT")
    



#     train_inputs=np.array(df.X.apply(lambda x: np.array(x)).tolist())
#     # train_inputs=train_inputs
#     # self.df.X.apply(lambda row: row[:][:-1])
#     train_out=np.array(df.y.apply(lambda x: np.array(x)).tolist())
#     # np.array(self.df.y).reshape(-1,1)
    

#     # print(train_inputs)
#     # print(train_out)

#     print(train_inputs.shape)
#     print(train_out.shape)
#     # input()
#     #TESTS PKL
#     # JT CHANGES!!!
#     # test_inputs=X_test
#     # test_out=y_test
    


    

#     # Use the same function above for the validation set WE JUST SPLIT IT IN 0.25 and 0.75 OF THE PREVIOUS SPLIT
#     train_inputs, vals_inputs, train_out, vals_out = train_test_split(train_inputs, train_out, 
#         test_size=0.25,shuffle=False, random_state= RNDSEED) # 0.25 x 0.8 = 0.2




#     #DROPPING STATUS

#     print(train_inputs.shape)
#     train_inputs=train_inputs[:,:,:-1]
#     test_inputs=test_inputs[:,:,:-1]
#     vals_inputs=vals_inputs[:,:,:-1]
#     print(train_inputs.shape)
#     # input()

#     print("train_out")
#     print(train_out.shape)
#     # polyline = np.array(np.linspace(0,len(train_out),len(train_out) ) )
#     scaler=MinMaxScaler(feature_range=(0, 1))
#     train_out=scaler.fit_transform(train_out)
#     # train_out = train_out.apply(NormalizeData)
#     # NormalizeData(polyline)
#     train_out=train_out.reshape(-1,1)
    

#     print("test_out")
#     print(test_out.shape)
#     # polyline = np.array(np.linspace(0,len(test_out),len(test_out) ) )
#     scaler=MinMaxScaler(feature_range=(0, 1))
#     test_out=scaler.fit_transform(test_out)
#     # test_out = test_out.apply(NormalizeData)
#     # test_out = NormalizeData(polyline)
#     test_out=test_out.reshape(-1,1)


#     print("vals_out")
#     # print(vals_out)
#     print(vals_out.shape)
#     # polyline = np.array(np.linspace(0,len(vals_out),len(vals_out) ) )
#     scaler=MinMaxScaler(feature_range=(0, 1))
#     vals_out=scaler.fit_transform(vals_out)
#     # vals_out = vals_out.apply(NormalizeData)
#     # vals_out = NormalizeData(polyline)
#     vals_out=vals_out.reshape(-1,1)
#     # print(vals_out)
#     print(vals_out.shape)
    

#     #NOW THE SEQUENCES
#     "train_inputs"
#     for seq in range(train_inputs.shape[0]):
#         scaler=StandardScaler()
        
#         train_inputs[seq]=scaler.fit_transform(train_inputs[seq])

#     "test_inputs"
#     for seq in range(test_inputs.shape[0]):
#         scaler=StandardScaler()
        
#         test_inputs[seq]=scaler.fit_transform(test_inputs[seq])



#     "vals_inputs"
#     for seq in range(vals_inputs.shape[0]):
#         scaler=StandardScaler()
        
#         vals_inputs[seq]=scaler.fit_transform(vals_inputs[seq])


#     print(train_inputs.shape)
#     print(test_inputs.shape)
#     print(vals_inputs.shape)

    

    
#     tf.keras.backend.clear_session()

#     nb_features = train_inputs.shape[2]
#     sequence_length  = train_inputs.shape[1]
#     nb_out = train_out.shape[1]

#     model = tf.keras.models.Sequential([
#         tf.keras.layers.LSTM(64, input_shape = (sequence_length, nb_features), return_sequences = True),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.LSTM(32),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(nb_out, activation = 'relu')
#     ])

    

#     model.compile(loss=tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(lr = 10**-7), metrics =['mse','mae'])

#     # JT CHANGES
#     # X_train=train_inputs
#     # y_train=train_out
#     # X_vals=vals_inputs
#     # y_vals=vals_out
#     # X_test=test_inputs
#     # y_test=test_out

    
#     return model, train_inputs,train_out,vals_inputs,vals_out,test_inputs,test_out




#     # # X_train=df.X.values
#     # # y_train=df.y.values
#     # X_train=np.array(df.X.apply(lambda x: np.array(x)).tolist())#
#     # # train_inputs=train_inputs
#     # # self.df.X.apply(lambda row: row[:][:-1])
#     # y_train=np.array(df.y.apply(lambda x: np.array(x)).tolist())#.


#     # print(X_train)
#     # print(y_train)

#     # print(X_train.shape)
#     # print(y_train.shape)
    

#     # # Use the same function above for the validation set
#     # X_train, X_vals, y_train, y_vals = train_test_split(X_train, y_train, 
#     #     test_size=0.25, random_state= RNDSEED,shuffle=True) # 0.25 x 0.8 = 0.2


#     # from sklearn.preprocessing import StandardScaler,LabelBinarizer
#     # lbz = LabelBinarizer()

#     # print("MODELSLSSDLLDSLSD")
#     # print(X_train.shape)
#     # print(X_train.dtype)
#     # print(X_train)

#     # scaler=StandardScaler()
#     # X_train=scaler.fit_transform(X_train)


#     # y_train=lbz.fit_transform(y_train)



#     # scaler=StandardScaler()
#     # X_vals=scaler.fit_transform(X_vals)
#     # y_vals=lbz.fit_transform(y_vals)


#     # scaler=StandardScaler()
#     # X_test=scaler.fit_transform(X_test)
#     # y_test=lbz.fit_transform(y_test)


#     # print(X_train.shape)
#     # print(X_vals.shape)
#     # print(X_test.shape)


#     # from  datetime import datetime
#     # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#     # print(log_dir)
#     # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#     # lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))


#     # #TRAINING
#     # tf.keras.backend.clear_session()


#     # # self.model
#     # model = tf.keras.models.Sequential([
#     # tf.keras.layers.Dense(60, activation="relu", input_dim=X_train.shape[1],kernel_regularizer=L1L2(l2=0.001,l1=0.001)), #Better

#     # tf.keras.layers.Dropout(0.5),
#     # tf.keras.layers.Dense(30, activation="relu",kernel_regularizer=L2(l2=0.001)),  #Better
#     # tf.keras.layers.Dropout(0.5),

#     # # model.add(Dense(40, activation="relu"))
#     # # model.add(Dropout(0.2))
#     # # model.add(Dense(10, activation="relu",kernel_regularizer=L2(l2=0.001)))  #Better
#     # # output layer
#     # tf.keras.layers.Dense(5, activation="softmax")])
#     #             # softmax for probability, #values are sigmoid

#     # # Configure the model and start training
#     # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    
#     # X_train=np.array(X_train.astype('float64'))
#     # y_train=np.array(y_train.astype('float64'))
    
#     # X_vals=np.array(X_vals.astype('float64'))
#     # y_vals=np.array(y_vals.astype('float64'))
    
#     # X_test=np.array(X_test.astype('float64'))
#     # y_test=np.array(y_test.astype('float64'))
    


#     # return model, X_train,y_train,X_vals,y_vals,X_test,y_test


# def modeling(model, train_inputs,train_out,vals_inputs,vals_out,batch_size,epochs):

    
#     # JT CHJANGES!!! 
#     # train_inputs=X_train
#     # train_out=y_train
#     # vals_inputs=X_vals
#     # vals_out=y_vals




#     lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))

#     # model.compile(loss=tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(lr = 10**-7), metrics =['mse','mae'])


#     #FAST AI SEE IF TRIANING IMPROVES !

#     es = EarlyStopping(monitor="val_loss",
#             mode="auto",
#             verbose=0,
#             patience=1,
#             min_delta=0.0001,
#             restore_best_weights=True)
#     # 1420492
#     # history = model.fit(train_inputs, train_out, epochs = 20, callbacks = [lr])
#     lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))

#     from  datetime import datetime

#     # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#     # print(log_dir)
#     # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


#     history=model.fit(train_inputs,train_out,epochs=epochs,validation_data= (vals_inputs,vals_out) ,verbose=0,) #tensorboard_callback


    
#     # lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))

#     # # model.compile(loss=tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(lr = 10**-7), metrics =['mse','mae'])


#     # #FAST AI SEE IF TRIANING IMPROVES !

#     # es = EarlyStopping(monitor="val_loss",
#     #         mode="auto",
#     #         verbose=0,
#     #         patience=1,
#     #         min_delta=0.0001,
#     #         restore_best_weights=True)
#     # # 1420492
#     # # history = model.fit(train_inputs, train_out, epochs = 20, callbacks = [lr])
#     # lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))

#     # from  datetime import datetime

#     # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#     # print(log_dir)
#     # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#     # history=model.fit(X_train,y_train,validation_data= (X_vals,y_vals) ,epochs=epochs ,verbose=0,callbacks=[tensorboard_callback,lr,es])
#     # return history
#     return history




# # def testing(self):
    
# #     self.lst_accu_stratified_a=max(self.history.history['val_accuracy'])
# #     #https://datascience.stackexchange.com/questions/116692/accuracy-vs-categorical-accuracy
# #     # metrics: List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. See tf.keras.metrics. Typically you will use metrics=['accuracy']. A function is any callable with the signature result = fn(y_true, y_pred). To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a':'accuracy', 'output_b':['accuracy', 'mse']}. You can also pass a list to specify a metric or a list of metrics for each output, such as metrics=[['accuracy'], ['accuracy', 'mse']] or metrics=['accuracy', ['accuracy', 'mse']]. When you pass the strings 'accuracy' or 'acc', we convert this to one of tf.keras.metrics.BinaryAccuracy, tf.keras.metrics.CategoricalAccuracy, tf.keras.metrics.SparseCategoricalAccuracy based on the shapes of the targets and of the model output. We do a similar conversion for the strings 'crossentropy' and 'ce' as well. The metrics passed here are evaluated without sample weighting; if you would like sample weighting to apply, you can specify your metrics via the weighted_metrics argument instead.

# #     self.lst_accu_stratified_l=min(self.history.history['val_loss'])



# # Make TensorFlow logs less verbose
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# # XY = Tuple[np.ndarray, np.ndarray]
# # XYList = List[XY]


# # Define Flower client
# class FlowerClient(fl.client.NumPyClient):

#     def __init__(self, model, X_train, y_train, X_test, y_test,X_vals,y_vals,epochs):
#         #LOAD DATA!
#         # self.df,self.X_test,self.y_test
#         #LOAD MODEL "MODEL DEFINITION FUNCTION" UP TO COMPILE
        
#         self.model = model
#         self.X_train, self.y_train = X_train, y_train
#         self.X_test, self.y_test = X_test, y_test
#         self.X_vals, self.y_vals = X_vals, y_vals
#         self.epochs=epochs

        

#     def get_properties(self, config): #
#         """Get properties of client."""
#         raise Exception("Not implemented")

#     def get_parameters(self, config):#, config
#         """Get parameters of the local model."""
#         raise Exception("Not implemented (server-side parameter initialization)")


#     def fit(self, parameters, config):
#         """Train parameters on the locally held training set."""

#         # Update local model parameters
#         self.model.set_weights(parameters)

#         # Get hyperparameters for this round
#         batch_size: int = config["batch_size"]
#         epochs: int = config["local_epochs"]

#         # Train the model using hyperparameters from config
#         #WE FIT THE MODEL

#         #JT CHANGED TO FOLLOW TEMP
#         history=modeling(self.model, self.X_train,self.y_train,self.X_vals,self.y_vals,batch_size,epochs)
#         # history = self.model.fit(
#         #     self.x_train,
#         #     self.y_train,
#         #     batch_size,
#         #     epochs,
#         #     validation_split=0.1,
#         # )

#         # Return updated model parameters and results
#         parameters_prime = self.model.get_weights()
#         num_examples_train = len(self.X_train)
#         results = {
#             "loss": history.history["loss"][0],
#             "mse": history.history["mse"][0],
#             "mae": history.history["mae"][0],

#             "val_loss": history.history["val_loss"][0],
#             "val_mse": history.history["val_mse"][0],
#             "val_mae": history.history["val_mae"][0],
#         }
#         return parameters_prime, num_examples_train, results

#     # def get_parameters(self, config):

#     #     self.load_data()
#     #     self.model_definition()
#     #     # self.train_cut_split()
#     #     # self.pre_modeling()  # TO DO CHANGES HERE
#     #     # self.modeling()

#     #     return self.model.get_weights()

#     # def fit(self, parameters, config):

#     #     self.load_data()
#     #     self.model_definition()
#     #     # self.train_cut_split()
#     #     # self.pre_modeling()  # TO DO CHANGES HERE
#     #     # self.modeling()

#     #     self.model.set_weights(parameters)

#     #     self.modeling()  # CHANGES HERE separated same seed

#     #     return self.model.get_weights(), len(self.X_train), {}

#     # def evaluate(self, parameters, config):

#     #     self.model.set_weights(parameters)

#     #     self.testing()  # accuracy calculator separated same seed

#     #     return (
#     #         self.lst_accu_stratified_l,
#     #         len(self.y_train),
#     #         {
#     #             "accuracy": self.lst_accu_stratified_a
#     #         },
#     #     )
#     def evaluate(self, parameters, config):
#         #WE THIS IS NEW FUNCTION!
#         """Evaluate parameters on the locally held test set."""

#         # Update local model with global parameters
#         self.model.set_weights(parameters)

#         # Get config values
#         steps: int = config["val_steps"]
#         #JT ACTIVATED NOW FOLOWOS EXAMPLE

#         # Evaluate global model parameters on the local test data and return results
#         loss, mse, mae = self.model.evaluate(self.X_test, self.y_test,32,verbose=0)# #JT 32 is changed to 256
#         num_examples_test = len(self.X_test)





       
#         # JT CHANGES _test_inputs=self.X_test[0:]
#         # _test_out=self.y_test[0:]


#         y_pred = self.model.predict(self.X_test[0:],verbose=0) ## using the untinted dataset!
    
#         print('R^2:', metrics.r2_score(self.y_test[0:], y_pred))
#         print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(self.y_test[0:], y_pred))
#         print('Mean Squared Error (MSE):', metrics.mean_squared_error(self.y_test[0:], y_pred))
#         print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(self.y_test[0:], y_pred))
#         print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(self.y_test[0:], y_pred))) # np.sqrt

#         print('Explained Variance Score:', metrics.explained_variance_score(self.y_test[0:], y_pred))
#         print('Max Error:', metrics.max_error(self.y_test[0:], y_pred))
#         print('Mean Squared Log Error:', metrics.mean_squared_log_error(self.y_test[0:], y_pred))
#         print('Median Absolute Error:', metrics.median_absolute_error(self.y_test[0:], y_pred))

#         return loss, num_examples_test, {"mse": mse}


# def main() -> None:

#     parser = argparse.ArgumentParser()

#     parser.add_argument("-dfn",
#                         "--data_file_name",
#                         help="Data file name",
#                         type=str)
#     parser.add_argument(
#         "-cm",
#         "--clients_max",
#         help="Maximun number of clients",
#         type=int,
#         required=True,
#     )

#     parser.add_argument(
#         "-e",
#         "--epochs",
#         help="epochs",
#         type=int,
#         required=True,
#         default=0,
#     )

#     parser.add_argument("-ip", "--ip", help="IP address", type=str)

#     parser.add_argument(
#         "-cn",
#         "--clients_number",
#         help="Number of a specific client <= maximun number of clients",
#         type=int,
#         required=True,
#     )

#     parser.add_argument("-dfn_test_x",
#                         "--dfn_test_x",
#                         help="dfn_test_x",
#                         type=str)

#     parser.add_argument("-dfn_test_y",
#                         "--dfn_test_y",
#                         help="dfn_test_y",
#                         type=str)

#     args = parser.parse_args()

#     clients_max = int(args.clients_max)
#     epochs = int(args.epochs)
#     clients_number = int(args.clients_number)
#     ip = str(args.ip)
#     data_file_name = str(args.data_file_name)
#     dfn_test_x = str(args.dfn_test_x)
#     dfn_test_y = str(args.dfn_test_y)








#     # Configuration
#     root_path = os.path.dirname(os.path.abspath("__file__"))
#     os.chdir(root_path)

#     RNDSEED = np.random.seed(39)
#     np.random.seed(RNDSEED)

#     os.environ["PYTHONHASHSEED"] = str(RNDSEED)


#     #WE LOAD THE DATA
#     TRANSFORMED_FOLDER= "fl_testbed/version2/data/transformed/"
#     df,X_test,y_test=load_data(TRANSFORMED_FOLDER,data_file_name,dfn_test_x,dfn_test_y)
#     #WE LOAD THE MODEL UP TO MODEL COMPILE
#     #JT CHANGES
#     del data_file_name,dfn_test_x,dfn_test_y
#     model, X_train, y_train, X_test, y_test,X_vals,y_vals=model_definition(df,X_test,y_test,RNDSEED)
#     #JT CHANGES
#     del df

    

    

#     client=FlowerClient(

#         model=model, 
#         X_train=X_train,
#         y_train=y_train,
#         X_test=X_test,
#         y_test=y_test,
#         X_vals=X_vals,
#         y_vals=y_vals,
#         epochs=epochs,
#             # data_file_name=data_file_name,
#             # dfn_test_x=dfn_test_x,
#             # dfn_test_y=dfn_test_y,
#             # clients_number=clients_number,
#             # rndseed=RNDSEED,
#             # epochs=epochs,
           
           

#         )

#     # Start Flower client
#     fl.client.start_numpy_client(
#         server_address="10.144.187.117:8080",
#         client=client,
        
#     )


# if __name__ == "__main__":
#     main()