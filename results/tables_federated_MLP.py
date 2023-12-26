"""TO USE

python3 results/tables_federated_MLP.py
"""



import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np
from scipy.signal import savgol_filter, medfilt
from sklearn import preprocessing 
import re

dir_path = os.path.dirname(os.path.realpath(__file__))

ROUNDS=100
FOLDER_PATH=dir_path+"/"+'final'
PATH_FIND_PART1='MLP_TESLA_'
# PATH_FIND_PART2='_SEQ20_'

# DATA_TYPE=['TYPE1','TYPE2']
METRICS=['accuracy','weighted_avg','MCS']
# COLORS=['b','g','r','c']
names=[]
plots_dict=[]
# plt.rcParams.update({'font.size': 16})

os.chdir(FOLDER_PATH)
print(FOLDER_PATH)
# files = glob(PATH_FIND_PART1+'*')
files = [file for file in glob(PATH_FIND_PART1 + '*') if not (file.endswith('.pdf') or file.endswith('.csv'))]

# files=[k for k in files if PATH_FIND_PART2 in k]

print(files)

dataframes=[]

"""MLP_TESLA_FedAvgM_0.1_slr_0.001_0.9.txt"""
"""MLP_TESLA_FedAvg_0.1.txt"""



def round_float_or_keep_string(value):
    if isinstance(value, (float, int)):  # Check if it's a float or int
        return f"{round(value, 3):.3f}"  # Round to 2 decimal places for floats
    else:
        return value  # Keep the string as is


for name in files:
    #PARAMETERS LIST INI
    type_model=[]
    type_machine=[]
    type_algo=[]
    type_alpha=[]
    type_slr=[]
    type_sparam=[]
    time_end=None


    accuracy_list=[]
    weightedavg_list=[]
    mcs_list=[]
    loss_list=[]


    max_value = -999999
    max_index = None
    max_value = 0.0
    max_index = -1  # Initialize to an invalid index

    consecutive_count = 0  # Counter for consecutive elements


    path_name=name
    name=name.split(".tx")[0]
    


    if  "_FedAvg_" in name:
        
        type_model.append(name.split("_")[0])
        type_machine.append(name.split("_")[1])
        type_algo.append(name.split("_")[2])
        type_alpha.append(name.split("_")[3])
        type_slr.append("-")
        type_sparam.append("-")



        data=open(path_name)

        for line in data:
            if line.startswith("    accuracy"):
                # print(line.split("      "))
                # input()
                accuracy_list.append(float(line.split("     ")[-2]))
                # pair_counter=pair_counter+1
            elif line.startswith("weighted avg"):
                weightedavg_list.append(float(line.split("     ")[-2]))
            elif line.startswith("MCS "):

                mcs_list.append(float(line.split(" ")[-1].replace("\n","")))
            elif "| fit progress:" in line :
                # print(line)
                # print(line.split(","))
                # print(len(line.split(",")))
                # print(float(line.split(",")[-6]))
                loss_list.append(float(line.split(",")[-6]))
            if "FL finished in" in line:
                print(line)
                print("enter")
                time_end=round(float(line.split("in ")[-1]),3)#.split(" ")[2:]
                print(time_end,"end")

        
        #GET POSITION WHERE THE GHIGHEST ELEMNT LIEST ON
        accuracy_list=accuracy_list[:99]
        weightedavg_list=weightedavg_list[:99]
        mcs_list=mcs_list[:99]
        loss_list=loss_list[:99]
        # Iterate through the list

        for i, value in enumerate(mcs_list):
            if value > max_value and value < 1.0:
                consecutive_count += 1
                if consecutive_count == 10:
                    max_index = i
                    break
            else:
                consecutive_count = 0

        if max_index != -1:
            print(f"Sequence of 10 continuous elements ends at index {max_index}.")

        


        # for i, value in enumerate(r2_list):
        #     if (value > max_value and value <1.0):
        #         max_value = value
        #         max_index = i
        


        
        if int(int(max_index)+1)>0:


            dataframes.append(pd.DataFrame({
            # 'type_model':type_model,
            # 'type_machine':type_machine,
            'type_algo':type_algo,
            'type_alpha':type_alpha,
            'type_slr':type_slr,
            'type_sparam':type_sparam,



            'Epoch':int(int(max_index)+1),
            'Accuracy':accuracy_list[max_index],
            'F1-Weighted':weightedavg_list[max_index],
            'MCS':mcs_list[max_index],
            'Loss':loss_list[max_index],
            'time_end':time_end
            
            }))
        else:
            dataframes.append(pd.DataFrame({
            # 'type_model':type_model,
            # 'type_machine':type_machine,
            'type_algo':type_algo,
            'type_alpha':type_alpha,
            'type_slr':type_slr,
            'type_sparam':type_sparam,



            'Epoch':str(ROUNDS)+"+",
            'Accuracy':"-",
            'F1-Weighted':"-",
            'MCS':"-",
            'Loss':"-",
            'time_end':time_end
            
            }))

    
    else:
        type_model.append(name.split("_")[0])
        type_machine.append(name.split("_")[1])
        type_algo.append(name.split("_")[2])
        type_alpha.append(name.split("_")[3])
        type_slr.append(name.split("_")[5])
        type_sparam.append(name.split("_")[6])



        data=open(path_name)
        print(path_name)

        for line in data:
            if line.startswith("    accuracy"):
                # print(line.split("      "))
                # input()
                accuracy_list.append(float(line.split("     ")[-2]))
                # pair_counter=pair_counter+1
            elif line.startswith("weighted avg"):
                weightedavg_list.append(float(line.split("     ")[-2]))
            elif line.startswith("MCS "):

                mcs_list.append(float(line.split(" ")[-1].replace("\n","")))
            elif "| fit progress:" in line :
                # print(line)
                # print(line.split(","))
                # print(len(line.split(",")))
                # print(float(line.split(",")[-6]))
                loss_list.append(float(line.split(",")[-6]))
            if "FL finished in" in line:
                print(line)
                print("enter")
                time_end=round(float(line.split("in ")[-1]),3)#.split(" ")[2:]
                print(time_end,"end")
        
        #GET POSITION WHERE THE GHIGHEST ELEMNT LIEST ON

        accuracy_list=accuracy_list[:99]
        weightedavg_list=weightedavg_list[:99]
        mcs_list=mcs_list[:99]
        loss_list=loss_list[:99]

        # Iterate through the list

        for i, value in enumerate(mcs_list):
            if value > max_value and value < 1.0:
                consecutive_count += 1
                if consecutive_count == 10:
                    max_index = i
                    break
            else:
                consecutive_count = 0

        if max_index != -1:
            print(f"Sequence of 10 continuous elements ends at index {max_index}.")


        


        

        # print(data_)
        # input()
        # print(max_index)

        if int(int(max_index)+1)>0:


            dataframes.append(pd.DataFrame({
            # 'type_model':type_model,
            # 'type_machine':type_machine,
            'type_algo':type_algo,
            'type_alpha':type_alpha,
            'type_slr':type_slr,
            'type_sparam':type_sparam,



            'Epoch':int(int(max_index)+1),
            'Accuracy':accuracy_list[max_index],
            'F1-Weighted':weightedavg_list[max_index],
            'MCS':mcs_list[max_index],
            'Loss':loss_list[max_index],
            'time_end':time_end,
            
            }))
        else:
            dataframes.append(pd.DataFrame({
            # 'type_model':type_model,
            # 'type_machine':type_machine,
            'type_algo':type_algo,
            'type_alpha':type_alpha,
            'type_slr':type_slr,
            'type_sparam':type_sparam,



            'Epoch':str(ROUNDS)+"+",
            'Accuracy':"-",
            'F1-Weighted':"-",
            'MCS':"-",
            'Loss':"-",
            'time_end':time_end,
            
            }))
        

result = pd.concat(dataframes, axis=0)

# Reset the index, if needed
result = result.reset_index(drop=True).sort_values(['type_alpha','type_slr','type_algo','type_sparam'],ascending=[True,True,True,True])


result['Accuracy'] = result['Accuracy'].apply(round_float_or_keep_string)
result['F1-Weighted'] = result['F1-Weighted'].apply(round_float_or_keep_string)
result['MCS'] = result['MCS'].apply(round_float_or_keep_string)
result['Loss'] = result['Loss'].apply(round_float_or_keep_string)

# result.to_csv('FEDERATED_MLP.csv',index=False)

result['type_alpha'] = result['type_alpha'].apply(lambda x: '{:.0f}'.format(float(x)) if float(x).is_integer() else '{:g}'.format(float(x)))
# Convert DataFrame to LaTeX table
latex_table = result.drop(['Epoch'], axis=1).to_latex(index=False)#,'time_end'#.drop(['epoch'], axis=1)

# Print the LaTeX table or save it to a .tex file
print(latex_table)

# result.to_csv('data.csv', index=False)
# print(result)
