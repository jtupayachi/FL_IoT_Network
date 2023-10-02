"""

Main python file for running the federated client, parameter and the Strategy class are initialized.

"""

from ast import Dict
from inspect import Parameter
import flwr as fl
import tensorflow as tf
# from tensorflow import keras
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
import pandas as pd
import os
import itertools
import calendar
import time
import matplotlib.pyplot as plt

# scaling and train test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# creating a model
# from tensorflow.keras.models import Sequential, save_model, load_model
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam

# evaluation on test data
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)
from sklearn.metrics import classification_report, confusion_matrix
from typing import Optional, Tuple, Union, List
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn import preprocessing
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import pickle
import numpy
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
)
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import argparse
import sys

import json

import tensorflow as tf
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
import pandas as pd
import os
import itertools
import pickle
import random as rn
from tensorflow.keras.models import Sequential, save_model, load_model
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
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

import CustomStrategy_OFFSET


class SaveModelStrategy(CustomStrategy_OFFSET.CustomFedAvg):

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy,
                            fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameter], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[
                np.ndarray] = fl.common.parameters_to_ndarrays(
                    aggregated_parameters)

            self.load_data()

            self.modeling()

            self.model.set_weights(aggregated_ndarrays)

            self.testing()

            # model.evaluate(X_test, clf.transform(y_test))

        return aggregated_parameters, aggregated_metrics


if __name__ == "__main__":

    # Arguments
    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "-cm",
        "--clients_max",
        help="Maximun number of clients",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-e",
        "--epochs",
        help="epochs",
        type=int,
        required=True,
        default=0,
    )

    parser.add_argument("-ip", "--ip", help="IP address", type=str)

    parser.add_argument("-dfn_test_x",
                        "--dfn_test_x",
                        help="dfn_test_x",
                        required=True,
                        type=str)
    

    parser.add_argument("-dfn_test_y",
                    "--dfn_test_y",
                    help="dfn_test_y",
                    required=True,
                    type=str)
    
    # parser.add_argument("-fq",
    #                 "--fq",
    #                 help="fq",
    #                 required=True,
    #                 default=None)


    parser.add_argument("-dfn",
                        "--dfn",
                        help="dfn",
                        required=True,
                        default=None)

    parser.add_argument("-r",
                        "--rounds",
                        help="federated rounds",
                        required=True,
                        default=None)

    args = parser.parse_args()

    
    clients_max = int(args.clients_max)
    epochs = int(args.epochs)
    

    ip = str(args.ip)
    dfn_test_y = str(args.dfn_test_y)
    dfn_test_x = str(args.dfn_test_x)

    dfn = str(args.dfn)
    
    # fq=str(args.fq)




    rounds = int(args.rounds)

    # Configuration
    root_path = os.path.dirname(os.path.abspath("__file__"))
    os.chdir(root_path)

    RNDSEED = np.random.seed(39)
    np.random.seed(RNDSEED)

    os.environ["PYTHONHASHSEED"] = str(RNDSEED)

    # Create strategy and run server

    strategy = SaveModelStrategy(
        rndseed=RNDSEED,
        epochs=epochs,
        dfn=dfn,
        
        
        clients_max=clients_max,
        ip=ip,
        dfn_test_y=dfn_test_y,
        dfn_test_x=dfn_test_x,
        # fq=float(fq),


        weights=None,
        min_available_clients=clients_max,
        min_evaluate_clients=clients_max,
        min_fit_clients=clients_max,
        df=None,
    )  # ,evaluate_fn=evaluate

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address=str(ip) + ":8080",
        config=fl.server.ServerConfig(num_rounds=rounds),
        grpc_max_message_length=1024 * 1024 * 1024,
        strategy=strategy,
        
    )

    exit()
