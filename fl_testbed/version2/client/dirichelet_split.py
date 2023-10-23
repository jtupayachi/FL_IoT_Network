#SIMILAR IMPLEMENTATION AS: arXiv:1909.06335
#FOR client_max use 5 for 5 cleints it iwll create 0,1,2,3,4 , use 4 for 4 clients it wil create 0,1,2,3  


"""
    NEW IMPLEMENTATION!

    MLP M3: python3 fl_testbed/version2/client/dirichelet_split.py -data_X_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_train.pkl -data_X_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_vals.pkl -data_y_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_train.pkl -data_y_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_vals.pkl -cm 4 -alpha 0.2 -beta 0.2 -motor 3 -type MLP 

"""

import random
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
from collections import Counter


np.random.seed(2)



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






        # Example usage:
        N = self.clients_max # Number of classes
        alpha = self.alpha  # Concentration parameter (a scalar)
        num_clients = self.clients_max
        num_samples_per_client = 100000000 #UYSE A HIGH NUMBER FOR USE ALL

        p = [self.beta] * N  # Prior class distribution (1-dimensional) STARTS WITH SAME NUMBER OF DATAPOITNS PER PARTITION
        
        
        total = sum(p)
        # Standardize the numbers to sum to 1
        p = [x / total for x in p]

        print("STANDARIZED : ",p)




        # DATA_FOLDER = "/FL_AM_Defect-Detection/fl_testbed/version2/client/"
        # os.chdir(root_path+DATA_FOLDER)
        print(os.getcwd)

        def split_dataframe_to_non_identical_clients(df, N, alpha, num_clients, num_samples_per_client, p, output_dir):



            def sample_group(group, frac):
                return group.sample(frac=frac)

            def unwrap_array(arr):
                return arr[0] 


            if alpha <= 0:
                raise ValueError("Alpha must be a positive number.")
            if len(p) != N or not all(0 <= qi <= 1 for qi in p):
                raise ValueError("The prior class distribution 'p' must be a 1-dimensional array with values between 0 and 1.")
            
            clients = []
            # df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
            
            # print(p)

            flattened_y = np.concatenate(df['y'])
            # Get unique classes
            unique_classes = np.unique(flattened_y)
            # Convert the unique classes to a list if needed
            unique_classes_list = unique_classes.tolist()
            

            value_counts = df['y'].value_counts()
            min_count_class = value_counts.idxmin()[0]
            min_count_quantity = value_counts.min()

            
            counter=0
            #FOR EACH OF MY 0.2 SPLITS OF DATA!
            for i in p:

                print("This is split: "+str(counter))
                
                #DATAFRAME CREATION!

                #THIS SECTION IS FOR CREATING THE EUQLA DATA PARTITIIONS
                # Group the DataFrame by a custom key, e.g., a constant, since we want to sample without grouping
                grouped = self.df.groupby(lambda x: 0)

                # Use apply to sample rows within each group
                sampled_df = grouped.apply(sample_group, frac=i)

    
                #DISTRIBUTION GENENRATION!
                num_samples = min(num_samples_per_client, len(df))
                data = df.iloc[:num_samples, :].copy()
                df = df.iloc[num_samples:, :]

                q = dirichlet.rvs([alpha] * N)  # Ensure alpha is a scalar
                q = np.maximum(q, 0)
                q /= q.sum()

                #ROUNDING P

                # smallest_number = min(q)
                
                
                # q=np.array(q)
        
                # # Replace values smaller than 0.01 with the smallest number using np.where
                # q = np.squeeze(np.where(q < 0.1, smallest_number+0.00000001, q).tolist())


                # aggressiveness_factor = 9999999999999999

                # print(q)
                # q=[ float(aggressiveness_factor) ** float(w) for w in q]




                # arr = np.array(q)

                # # Replace elements less than 0.000000001 with 0.0001 element-wise
                # modified_arr = np.where(arr < 0.0001, 0.0001, arr)

                # # Convert the modified array back to a list (if needed)
                # q = modified_arr.tolist()

                

                print(q)




                #MY MAPPER! FOR EACH DATASPLIT !
                # class_probabilities=zip(unique_classes_list,np.squeeze(q))
                result_dict = [{'map': item[0], 'weights': item[1]} for item in zip(unique_classes_list,np.squeeze(q))]

                # If you want to convert the result_dict into a DataFrame:
                class_probabilities= pd.DataFrame(result_dict)
                class_probabilities['map'] = class_probabilities['map'].apply(lambda x: np.array([x]))

                # Map the DataFrames using the custom function
                sampled_df['mapper'] = sampled_df['y'].apply(unwrap_array)
                class_probabilities['mapper'] = class_probabilities['map'].apply(unwrap_array)

                merged_df = pd.merge(sampled_df, class_probabilities, on='mapper',how='inner')
                # print("FIRST")
                # print(merged_df.mapper.value_counts().sort_index())

                


                #CREATE ANOTHER DATAFRAME JUST FOR THE MINIMUN OF 30 DATAPOITNS TO ENSURE ALL CLASSES ARE REPRESENTED
                min_data_points_per_class = 30




                # Create a DataFrame to store the selected data points for each class
                selected_data = pd.DataFrame()


                # Iterate through unique classes
                for class_label in merged_df['mapper'].unique():
                    class_data = merged_df[merged_df['mapper'] == class_label]
                    
                    # If there are fewer than 10 data points in the class, select them all
                    if len(class_data) < min_data_points_per_class:
                        selected_data = pd.concat([selected_data, class_data])
                    else:
                        # Randomly select 10 data points from the class
                        selected_data = pd.concat([selected_data, class_data.sample(n=min_data_points_per_class, replace=False)])


                # Randomly sample the remaining data points if necessary
                merged_df = merged_df[~merged_df.index.isin(selected_data.index)]

                # Calculate the sum of the specified column
                column_sum = merged_df['weights'].sum()

                # Normalize the values in the column so that they sum up to 1
                merged_df['weights'] = merged_df['weights'] / column_sum



                # Extract the weights as a NumPy array
                weights = merged_df['weights'].values

                # Perform a random choice based on the weights
                # merged_df_sampled = random.choices(items, weights=weights, k=n_samples)
                # Perform random sampling without replacement
                sampled_index = random.choices(merged_df.index, weights=weights, k=min_count_quantity)

                # Get the selected row(s) from the DataFrame
                merged_df_sampled = merged_df.loc[sampled_index]
                
                # merged_df.sample(n=min_count_quantity, weights=weights, replace=False)



                # Combine the selected data from each class with the sampled remaining data
                merged_df_sampled = pd.concat([selected_data, merged_df_sampled])





                # print("second")
                print(merged_df_sampled.mapper.value_counts().sort_index())






                
                



                

                

                counter=counter+1




            return clients

        def plot_class_distribution_and_save(client, output_dir, client_index):
            q = client['q']
            plt.bar(range(len(q)), np.squeeze(q))
            plt.xticks(range(len(q)), range(1, len(q) + 1))
            plt.xlabel('Class')
            plt.ylabel('Probability')
            plt.title('Class Distribution (q) for Client {}'.format(client_index))
            
            # Save the plot as an image file
            output_file = os.path.join(output_dir, 'client_{}_class_distribution.png'.format(client_index))
            plt.savefig(output_file)
            plt.close()  # Close the plot to avoid displaying it





        # print(np.ones(5)*TOTAL_N_DATAPOINTS)
        # # Another useful fact about the Dirichlet distribution is that you naturally get it, if you generate a Gamma-distributed set of random variables and then divide them by their sum.
        # p=np.random.dirichlet(np.ones(num_clients)*1000.,size=1).reshape((-1,))
        # print(p) 
        print("SPLITS!!!")




        # data = {'x': np.random.randn(500), 'y': np.random.randn(500)}
        # df = pd.DataFrame(data)






        # Output directory for saving plots
        output_directory = "client_plots"
        os.makedirs(output_directory, exist_ok=True)

        non_identical_clients = split_dataframe_to_non_identical_clients(self.df, N, alpha, num_clients, num_samples_per_client, p, output_directory)
        print(non_identical_clients)

        # Generate and save plots for each client
        # for i, client in enumerate(non_identical_clients):
        #     print(f"Client {i + 1} Class Distribution (q):")
        #     print(client['q'])
        #     plot_class_distribution_and_save(client, output_directory, i)

        # print("Plots have been saved to the '{}' directory.".format(output_directory))


        # print(self.df.head())

        # for _split in p:
        #     # print(_split)
        #     # sample_df = self.df.groupby(['y'][0]).apply(lambda x: x.sample(frac=_split))
            

        #     def sample_group(group, frac):
        #         return group.sample(frac=frac)


        #     # Group the DataFrame by a custom key, e.g., a constant, since we want to sample without grouping
        #     grouped = self.df.groupby(lambda x: 0)

        #     # Use apply to sample rows within each group
        #     sampled_df = grouped.apply(sample_group, frac=_split)

        #     print(sampled_df.y.value_counts())
        #     print(sampled_df.X.tail())





            # st.dirichlet.pdf(list_of_xs.T, alpha)
            #TO BE USED FOR THE REMAINIG PART OF DATA NOW EVERYTHING IS SPLITTED PROPERLY
            # print(p.dtype)


            # p = [0.8, 0.2, 0.2, 0.2, 0.2]  # Prior class distribution

        # non_identical_clients = generate_non_identical_clients(N, alpha, num_clients, num_samples_per_client, p)
        # print(non_identical_clients)

            # Visualize the class distribution for the first client
            # for i, client in enumerate(non_identical_clients):
            #     print(f"Client {i + 1} Class Distribution (q):")
            #     plot_class_distribution(client,i)










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
