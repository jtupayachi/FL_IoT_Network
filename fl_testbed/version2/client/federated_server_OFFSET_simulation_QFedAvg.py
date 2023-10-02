"""
    This script launches the federated version  simulation of a RUL, it has both RUL regressor model for the
    RUL predictor. The script uses its specific small dataset version: ex. if this machine is labeled
    as 0 then it will use data_set_0.csv therefore it will only process data that belongs to this label.
    It fillows the same execution steps and model as the centralized learning.
     

"""
import os
import math

# Make TensorFlow logs less verbose
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



#GENERAL LIBRARIES
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    accuracy_score,
    r2_score,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from tensorflow.keras.layers import Dropout,Conv1D,MaxPooling1D,Flatten, Activation, Dense
# from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.layers import Dropout,Conv1D,MaxPooling1D,Flatten, Activation
# from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from autosklearn.regression import AutoSklearnRegressor
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
#LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

from tensorflow.keras.layers import Dropout,Conv1D,MaxPooling1D,Flatten, Activation, Dense

from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.layers import Dropout,Conv1D,MaxPooling1D,Flatten, Activation
from tensorflow.keras.regularizers import L1L2, L2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from autosklearn.regression import AutoSklearnRegressor
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import pandas as pd
import os
import numpy as np
RNDSEED = np.random.seed(39)
PRECISION = 4 # 3 of digits to keep after the decimal point
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
RUNNING_ON_COLAB = False # we assume running on CoLab! Change to False if running locally.
# from keras.regularizers import l2, l1
from tensorflow.keras.layers import Dropout
# from keras.layers import LeakyReLU
import json
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, cohen_kappa_score
import tensorflow as tf
from sklearn.model_selection import KFold
import sys
import argparse

import numpy as np
from collections.abc import Iterable
import pandas as pd
import os
import itertools
import pickle
import random as rn
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    accuracy_score,
    r2_score,
    mean_absolute_percentage_error,
    max_error,
    mean_squared_log_error,
    median_absolute_error,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from abc import ABC
import pickle
from typing import Dict, Tuple
from tensorflow.keras.models import Sequential, save_model, load_model
from flwr.common import Config, NDArrays, Scalar
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import os
from sklearn.model_selection import train_test_split
# from autosklearn.regression import AutoSklearnRegressor
# from tensorflow.keras.regularizers import L2,L1,L1L2
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
RNDSEED = np.random.seed(39)
PRECISION = 4 # 3 of digits to keep after the decimal point
from sklearn.preprocessing import StandardScaler, MinMaxScaler
RUNNING_ON_COLAB = False # we assume running on CoLab! Change to False if running locally.
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
import pickle
import random as rn
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


#EX LIBRARIES
import flwr as fl
import numpy as np
import os

from typing import Tuple, List
import sys
# import CustomNumpyClient_OFFSET
import argparse
import pickle
# import gc
# import psutil



def load_data(TRANSFORMED_FOLDER,data_file_name,dfn_test_x,dfn_test_y):
        
    print("INI")
    # gc.collect()


    #TRAINING
    with open(
            TRANSFORMED_FOLDER + data_file_name,
            # self.concatenated_identifier +
            # self.FILE_NAME + "y_train.pkl",
            "rb",
    ) as file:
        # self.df = pickle.load(file)
        df=pd.read_pickle(file)
    
    
    #TEST
    with open(
            TRANSFORMED_FOLDER + dfn_test_x.replace('\n', ''),
            # self.concatenated_identifier +
            # self.FILE_NAME + "y_train.pkl",
            "rb",
    ) as file:
        X_test = pickle.load(file)

    with open(
            TRANSFORMED_FOLDER + dfn_test_y.replace('\n', ''),
            # self.concatenated_identifier +
            # self.FILE_NAME + "y_train.pkl",
            "rb",
    ) as file:
        y_test = pickle.load(file)

    del file

    #DISABLED!! JT
    # mem_usage = psutil.virtual_memory()
    # print(f"Free: {mem_usage.percent}%")
    # print(f"Total: {mem_usage.total/(1024**3):.2f}G")
    # print(f"Used: {mem_usage.used/(1024**3):.2f}G")

    return df,X_test,y_test

    


def model_definition(df,X_test,y_test,RNDSEED):

    print("#ASSIGNAMEMNT")
    

    # X_train=df.X.values
    # y_train=df.y.values
    X_train=np.array(df.X.apply(lambda x: np.array(x)).tolist())#
    # train_inputs=train_inputs
    # self.df.X.apply(lambda row: row[:][:-1])
    y_train=np.array(df.y.apply(lambda x: np.array(x)).tolist())#.

    

    # Use the same function above for the validation set
    X_train, X_vals, y_train, y_vals = train_test_split(X_train, y_train, 
        test_size=0.25, random_state= RNDSEED,shuffle=True) # 0.25 x 0.8 = 0.2


    from sklearn.preprocessing import StandardScaler,LabelBinarizer
    lbz = LabelBinarizer()


    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)


    y_train=lbz.fit_transform(y_train)



    scaler=StandardScaler()
    X_vals=scaler.fit_transform(X_vals)
    y_vals=lbz.fit_transform(y_vals)


    scaler=StandardScaler()
    X_test=scaler.fit_transform(X_test)
    y_test=lbz.fit_transform(y_test)


    print(X_train.shape)
    print(X_vals.shape)
    print(X_test.shape)


    from  datetime import datetime
    # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # print(log_dir)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))


    #TRAINING
    tf.keras.backend.clear_session()
    conf = tf.compat.v1.ConfigProto()
    conf.gpu_options.allow_growth=True
    session = tf.compat.v1.Session(config=conf)
    # self.model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation="relu", input_dim=X_train.shape[1],kernel_regularizer=L1L2(l2=0.001,l1=0.001)), #Better

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation="relu",kernel_regularizer=L2(l2=0.001)),  #Better
    tf.keras.layers.Dropout(0.5),

    # model.add(Dense(40, activation="relu"))
    # model.add(Dropout(0.2))
    # model.add(Dense(10, activation="relu",kernel_regularizer=L2(l2=0.001)))  #Better
    # output layer
    tf.keras.layers.Dense(5, activation="softmax")])
                # softmax for probability, #values are sigmoid

    # Configure the model and start training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    
    X_train=np.array(X_train.astype('float64'))
    y_train=np.array(y_train.astype('float64'))
    
    X_vals=np.array(X_vals.astype('float64'))
    y_vals=np.array(y_vals.astype('float64'))
    
    X_test=np.array(X_test.astype('float64'))
    y_test=np.array(y_test.astype('float64'))
    


    return model, X_train,y_train,X_vals,y_vals,X_test,y_test
    






def modeling(model, X_train,y_train,X_vals,y_vals,batch_size,epochs):

    
    lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))

    # model.compile(loss=tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(lr = 10**-7), metrics =['mse','mae'])


    #FAST AI SEE IF TRIANING IMPROVES !

    es = EarlyStopping(monitor="val_loss",
            mode="auto",
            verbose=2,
            patience=1,
            min_delta=0.0001,
            restore_best_weights=True)
    # 1420492
    # history = model.fit(train_inputs, train_out, epochs = 20, callbacks = [lr])
    lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))

    from  datetime import datetime

    # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # print(log_dir)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history=model.fit(X_train,y_train,validation_data= (X_vals,y_vals) ,epochs=epochs ,verbose=2)#tensorboard_callback
    return history


NUM_CLIENTS = 5 #REMEMBER WE HAVE CID 0,1,2,3,4
ROUNDS=4000 
CONST_EPOCHS=2
DATA_FILE_NAME_FIRST='M3_4_'#4_ddf_LSTM.pkl'
DATA_FILE_NAME_SECOND='_ddf_MLP.pkl'#M3_4_4
DFN_TEST_X='30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl'
DFN_TEST_Y='30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'
NUM_CPUS=60 # 14 cores per clietn
RATIO_GPU=0.80 #0.2 of the GPU per client

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test,X_vals,y_vals,epochs) -> None:
        super().__init__()
        self.model = model
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.X_vals, self.y_vals = X_vals, y_vals
        self.epochs=epochs

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        # self.model.set_weights(parameters)
        # self.model.fit(self.x_train, self.y_train, epochs=2, verbose=2)
        # return self.model.get_weights(), len(self.x_train), {}
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = 32#config["batch_size"] JT INPUTTED MANUALLY
        epochs: int = 1#config["local_epochs"] JT INPUTTED MANUALLY

        # Train the model using hyperparameters from config
        #WE FIT THE MODEL

        #JT CHANGED TO FOLLOW TEMP
        history=modeling(self.model, self.X_train,self.y_train,self.X_vals,self.y_vals,batch_size,epochs)
        # history = self.model.fit[(
        #     self.x_train,
        #     self.y_train,
        #     batch_size,
        #     epochs,
        #     validation_split=0.1,
        # )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.X_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        
        

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        # self.model.set_weights(parameters)
        # loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        # return loss, len(self.x_val), {"accuracy": acc}
        
        #WE THIS IS NEW FUNCTION!
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = 10#config["val_steps"] #MANUALLY INPUTTED JT
        #JT ACTIVATED NOW FOLOWOS EXAMPLE

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test,32,verbose=0)# #JT 32 is changed to 256
        num_examples_test = len(self.X_test)




        y_test=self.y_test #JT
        y_prob = self.model.predict(self.X_test,verbose=2) #JT
        y_prob_am=np.argmax(y_prob, axis=1)
        y_test_am=np.argmax(y_test, axis=1)
        macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
        weighted_roc_auc_ovo = roc_auc_score(
            y_test, y_prob, multi_class="ovo", average="weighted"
        )
        macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        weighted_roc_auc_ovr = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="weighted"
        )
        print(
            "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(ovo weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
        )
        print(
            "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(ovr weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
        )
        print(classification_report(y_test_am,y_prob_am ))
        print("MCS", matthews_corrcoef(y_test_am,y_prob_am ))

        return loss, num_examples_test, {"accuracy": accuracy}


def client_fn(cid: str) -> fl.client.Client:
    print("CHEKING CID",str(cid))
    # Load model



    #VARIABLES AUTOGENERATED BY CID AND SCRIPT SIMULATION
    data_file_name=DATA_FILE_NAME_FIRST+str(cid)+DATA_FILE_NAME_SECOND
    dfn_test_x=DFN_TEST_X
    dfn_test_y=DFN_TEST_Y
    epochs=CONST_EPOCHS

    print("IN MODEL: ", str(data_file_name)+' '+str(dfn_test_x)+' '+str(dfn_test_y)+' '+str(epochs))




    # Configuration
    root_path = os.path.dirname(os.path.abspath("__file__"))
    os.chdir(root_path)

    RNDSEED = np.random.seed(39)
    np.random.seed(RNDSEED)

    os.environ["PYTHONHASHSEED"] = str(RNDSEED)


    #WE LOAD THE DATA
    TRANSFORMED_FOLDER= "fl_testbed/version2/data/transformed/"
    df,X_test,y_test=load_data(TRANSFORMED_FOLDER,data_file_name,dfn_test_x,dfn_test_y)
    #WE LOAD THE MODEL UP TO MODEL COMPILE
    #JT CHANGES
    del data_file_name,dfn_test_x,dfn_test_y
    model, X_train, y_train, X_test, y_test,X_vals,y_vals=model_definition(df,X_test,y_test,RNDSEED)
    #JT CHANGES
    del df

    # Create and return client
    return FlwrClient( model, X_train, y_train, X_test, y_test,X_vals,y_vals,epochs)


def main() -> None:
    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": NUM_CPUS,"num_gpus":RATIO_GPU},
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=fl.server.strategy.QFedAvg(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            q_param= 0.2,	
            qffl_learning_rate = 0.1,
            min_fit_clients=NUM_CLIENTS,
            min_evaluate_clients=NUM_CLIENTS,
            min_available_clients=NUM_CLIENTS,
        ),
    )


if __name__ == "__main__":
    main()

