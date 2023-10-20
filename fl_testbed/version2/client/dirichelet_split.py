#SIMILAR IMPLEMENTATION AS: arXiv:1909.06335


"""
    NEW IMPLEMENTATION!

    MLP M3: python3 fl_testbed/version2/client/dirichelet_split.py -data_X_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_train.pkl -data_X_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_vals.pkl -data_y_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_train.pkl -data_y_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_vals.pkl -cm 4 -alpha 0.2 -beta 0.2 -motor 3 -type MLP 

"""
import sys
import pickle
import numpy as np
from collections.abc import Iterable
import pandas as pd
import os
import itertools
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from scipy.stats import dirichlet
import matplotlib.pyplot as plt



class DataSplit:

    def __init__(
        self,
        data_X_train,
        data_X_vals,
        data_y_train,
        data_y_vals,
        clients_max, #NUMBER FROM 0 to n clients, IDENTIFIER: 0,1,2,3,4 ....
        alpha, #DIRICHELET ALPHA
        beta, #PARAMETER SPLIT SIZE PER CLIENT/CLASS
        motor, #IDENTIFIER M1,M2,M3
        type,
    ):

        self.data_X_train=data_X_train #PATH
        self.data_X_vals=data_X_vals #PATH
        self.data_y_train=data_y_train #PATH
        self.data_y_vals=data_y_vals #PATH
        self.clients_max = clients_max[0]
        self.alpha=alpha[0]
        self.beta=beta[0]
        self.motor = motor[0]
        self.type = type[0]

        
        self.concatenated_identifier = (
                                        str(self.motor) + "_" +

                                        str(self.clients_max) + "_")
        

        self.DATA_FOLDER = "fl_testbed/version2/data/transformed/"
        self.TRANSFORMED_FOLDER = "fl_testbed/version2/data/transformed/"
        self.FILE_NAME = "dirichelet_split"
        self.PRECISION = 4

        self.X=None
        self.y=None


    def load_data_1(self):

        with open(self.DATA_FOLDER+self.data_X_train[0], 'rb') as f: 
            X_train = pd.read_pickle(f)

        #X_VALS
        with open(self.DATA_FOLDER+self.data_X_vals[0], 'rb') as f: 
            X_vals = pd.read_pickle(f)

        #y_TRAIN
        with open(self.DATA_FOLDER+self.data_y_train[0], 'rb') as f: 
            y_train = pd.read_pickle(f)

        #y_VALS
        with open(self.DATA_FOLDER+self.data_y_vals[0], 'rb') as f: 
            y_vals = pd.read_pickle(f)




        y_train=y_train.values.reshape(-1,1)
        y_vals=y_vals.values.reshape(-1,1)

        print(X_train.shape)
        print(X_vals.shape)
        print(y_train.shape)
        print(y_vals.shape)

        self.X=np.vstack((X_train,X_vals))
        self.y=np.vstack((y_train,y_vals))
        print(self.X.shape)
        print(self.y.shape)


        df_X = pd.DataFrame.from_records(zip(self.X),columns=['X'])
        df_y = pd.DataFrame.from_records(zip(self.y),columns=['y'])


        self.df=df_X.merge(df_y,left_index=True,right_index=True)
        

    def load_data_2(self):

        with open(self.DATA_FOLDER+self.data_X_train[0], 'rb') as f: 
            X_train = pickle.load(f)

        #X_VALS
        with open(self.DATA_FOLDER+self.data_X_vals[0], 'rb') as f: 
            X_vals = pickle.load(f)

        #y_TRAIN
        with open(self.DATA_FOLDER+self.data_y_train[0], 'rb') as f: 
            y_train = pickle.load(f)

        #y_VALS
        with open(self.DATA_FOLDER+self.data_y_vals[0], 'rb') as f: 
            y_vals = pickle.load(f)




        # y_train=y_train.values.reshape(-1,1)
        # y_vals=y_vals.values.reshape(-1,1)

        print(X_train.shape)
        print(X_vals.shape)
        print(y_train.shape)
        print(y_vals.shape)

        self.X=np.vstack((X_train,X_vals))
        self.y=np.vstack((y_train,y_vals))
        # print(self.X)
        # print(self.y)
        print(self.X.shape)
        print(self.y.shape)


        df_X = pd.DataFrame.from_records(zip(self.X),columns=['X'])
        df_z = pd.DataFrame.from_records(zip(self.y),columns=['z'])
        
        ##NEED TO WORK


        self.df=df_X.merge(df_z,left_index=True,right_index=True)
        self.df['y'] = int(self.df.X[0][0][-1])
        print(self.df)
        # print(self.df.y.values)
        
    def splitdata_1(self):

        #INPUTS TO USE

        #DATA_FILE 
        # print(self.df.y.value_counts()[0]) CAN BE ACCESSED
        print(self.df.y.value_counts())

        
        #ALPHA
        print(self.alpha)
        #BETA
        print(self.beta)
        #CM
        print(self.clients_max)

        print("END OF READING!")





        # Example usage and visualization:
        N = 5  # Number of classes
        alpha = 0.2  # Concentration parameter
        num_clients = 5
        num_samples_per_client = 5
        TOTAL_N_DATAPOINTS=1000







        # DATA_FOLDER = "/FL_AM_Defect-Detection/fl_testbed/version2/client/"
        # os.chdir(root_path+DATA_FOLDER)
        print(os.getcwd)

        def generate_non_identical_clients(N, alpha, num_clients, num_samples_per_client, p):
            clients = []

            for _ in range(num_clients):
                q = dirichlet.rvs(alpha * np.array(p))
                print(q)
                q = np.maximum(q, 0)
                q /= q.sum()

                data = []
                for _ in range(num_samples_per_client):
                    # Generate class labels following a categorical distribution (q)
                    sample = np.random.choice(N, p=np.squeeze(q))
                    data.append(sample)

                client_info = {
                    'q': q,
                    'data': data
                }
                clients.append(client_info)

            return clients

        def plot_class_distribution(client,name):
            q = client['q']
            plt.bar(range(len(q)), np.squeeze(q))
            plt.xticks(range(len(q)), range(1, len(q) + 1))
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('Class Distribution (q) for a Client')
            plt.savefig("Client: " + str(name)+ "_("+ str(self.alpha)+")_("+str(self.beta) +").png" )








        print(np.ones(5)*TOTAL_N_DATAPOINTS)
        # Another useful fact about the Dirichlet distribution is that you naturally get it, if you generate a Gamma-distributed set of random variables and then divide them by their sum.
        p=np.random.dirichlet(np.ones(num_clients)*1000.,size=1).reshape((-1,))
        # st.dirichlet.pdf(list_of_xs.T, alpha)
        print(p) #TO BE USED FOR THE REMAINIG PART OF DATA NOW EVERYTHING IS SPLITTED PROPERLY
        # print(p.dtype)


        # p = [0.8, 0.2, 0.2, 0.2, 0.2]  # Prior class distribution

        non_identical_clients = generate_non_identical_clients(N, alpha, num_clients, num_samples_per_client, p)

        # Visualize the class distribution for the first client
        for i, client in enumerate(non_identical_clients):
            print(f"Client {i + 1} Class Distribution (q):")
            plot_class_distribution(client,i)










if __name__ == "__main__":



    # Arguments
    parser = argparse.ArgumentParser()





    parser.add_argument(
        "-data_X_train",
        "--data_X_train",
        help="data_X_train",
        type=str,
        required=True,
    ),
    parser.add_argument(
        "-data_X_vals",
        "--data_X_vals",
        help="data_X_vals",
        type=str,
        required=True,
    ),
    parser.add_argument(
        "-data_y_train",
        "--data_y_train",
        help="data_y_train",
        type=str,
        required=True,
    ),
    parser.add_argument(
        "-data_y_vals",
        "--data_y_vals",
        help="data_y_vals",
        type=str,
        required=True,
    )
   
    parser.add_argument(
        "-cm",
        "--clients_max",
        help="Maximun number of clients",
        type=int,
        required=True,
    )


    parser.add_argument("-alpha",
                    "--alpha",
                    help="alpha for dirchelet",
                    type=float,
                    required=True,
                    default=None)

    parser.add_argument("-beta",
                    "--beta",
                    help="beta fraction distribution",
                    type=float,
                    required=True,
                    default=None)

    parser.add_argument("-motor",
                    "--motor",
                    help="motor number",
                    type=int,
                    required=True,
                    default=None)
    
    parser.add_argument("-type",
                "--type",
                help="type",
                type=str,
                required=True,
                default=None)










    args = parser.parse_args()




    data_X_train=str(args.data_X_train),
    data_X_vals=str(args.data_X_vals),
    data_y_train=str(args.data_y_train),
    data_y_vals=str(args.data_y_vals),
    
    clients_max=int(args.clients_max), #NUMBER FROM 0 to n clients
    alpha=float(args.alpha),
    beta=float(args.beta),
    motor=args.motor, #IDENTIFIER M1,M2,M3
    type=args.type,
    
   

  

    # Configuration
    root_path = os.path.dirname(os.path.abspath("__file__"))
    os.chdir(root_path)

    RNDSEED = np.random.seed(39)
    np.random.seed(RNDSEED)

    os.environ["PYTHONHASHSEED"] = str(RNDSEED)
    
    datasplit = DataSplit(

        data_X_train,
        data_X_vals,
        data_y_train,
        data_y_vals,

        clients_max, #NUMBER FROM 0 to n clients
        alpha,
        beta,
        motor, #IDENTIFIER M1,M2,M3
        type,
 
    )

    
    if type[0] == 'MLP':
        print("ENTERED!")
        datasplit.load_data_1()
        print(type[0])
        datasplit.splitdata_1()
    if type[0] == 'LSTM':
        print("ENTERED2!")
        datasplit.load_data_2()
        print(type[0])
        datasplit.splitdata_2()
    
    exit()
