"""
    This script launches the federated version of a RUL, it has both RUL regressor model for the
    RUL predictor. The script uses its specific small dataset version: ex. if this machine is labeled
    as 0 then it will use data_set_0.csv therefore it will only process data that belongs to this label.
    It fillows the same execution steps and model as the centralized learning.
     

"""
import flwr as fl
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple, List
import sys
import CustomNumpyClient_OFFSET
import argparse
import pickle

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]


# Define Flower client
class FlowerClient(CustomNumpyClient_OFFSET.NumPyClient):

    def get_parameters(self, config):

        self.load_data()
        self.model_definition()
        # self.train_cut_split()
        # self.pre_modeling()  # TO DO CHANGES HERE
        # self.modeling()

        return self.model.get_weights()

    def fit(self, parameters, config):

        self.load_data()
        self.model_definition()
        # self.train_cut_split()
        # self.pre_modeling()  # TO DO CHANGES HERE
        # self.modeling()

        self.model.set_weights(parameters)

        self.modeling()  # CHANGES HERE separated same seed

        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):

        self.model.set_weights(parameters)

        self.testing()  # accuracy calculator separated same seed

        return (
            self.lst_accu_stratified_l,
            len(self.y_train),
            {
                "mse": self.lst_accu_stratified_a
            },
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-dfn",
                        "--data_file_name",
                        help="Data file name",
                        type=str)
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

    parser.add_argument(
        "-cn",
        "--clients_number",
        help="Number of a specific client <= maximun number of clients",
        type=int,
        required=True,
    )

    parser.add_argument("-dfn_test_x",
                        "--dfn_test_x",
                        help="dfn_test_x",
                        type=str)

    parser.add_argument("-dfn_test_y",
                        "--dfn_test_y",
                        help="dfn_test_y",
                        type=str)

    args = parser.parse_args()
    # comparative_path_y_test=str(args.comparative_path_y_test)
    # comparative_path_X_test=str(args.comparative_path_X_test)

    clients_max = int(args.clients_max)
    epochs = int(args.epochs)

    clients_number = int(args.clients_number)
    ip = str(args.ip)

    data_file_name = str(args.data_file_name)

    dfn_test_x = str(args.dfn_test_x)
    dfn_test_y = str(args.dfn_test_y)
    # rounds = args.rounds

    # Configuration
    root_path = os.path.dirname(os.path.abspath("__file__"))
    os.chdir(root_path)

    RNDSEED = np.random.seed(39)
    np.random.seed(RNDSEED)

    os.environ["PYTHONHASHSEED"] = str(RNDSEED)

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="172.19.0.8:8080",
        client=FlowerClient(
            data_file_name=data_file_name,
            dfn_test_x=dfn_test_x,
            dfn_test_y=dfn_test_y,
            clients_number=clients_number,
            rndseed=RNDSEED,
            epochs=epochs,

            clients_max=clients_max,
            ip=ip,

        ),
        grpc_max_message_length=1024 * 1024 * 1024,
    )
