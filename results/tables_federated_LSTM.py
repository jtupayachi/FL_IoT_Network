"""TO USE

python3 results/tables_federated_LSTM.py
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

ROUNDS=1000
FOLDER_PATH=dir_path+"/"+'final'
PATH_FIND_PART1='LSTM_TESLA_'
# PATH_FIND_PART2='_SEQ20_'

# DATA_TYPE=['TYPE1','TYPE2']
METRICS=['r2','mse','mae']
# COLORS=['b','g','r','c']
names=[]
plots_dict=[]
# plt.rcParams.update({'font.size': 16})

os.chdir(FOLDER_PATH)
print(FOLDER_PATH)
files = glob(PATH_FIND_PART1+'*')
# files=[k for k in files if PATH_FIND_PART2 in k]

print(files)

dataframes=[]

"""LSTM_TESLA_FedAvgM_0.001_slr_0.001_0.0.txt"""
"""LSTM_TESLA_FedAvg_0.02.txt"""



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
    time_ini=None
    time_end=None


    r2_list=[]
    mse_list=[]
    mae_list=[]
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
        #ONLY HAPPENS ONE TIME!


        




        data=open(path_name)

        for line in data:
            if "| fit progress:" in line :
                # prine.split(",")[-6]))
                r2_list.append(float(re.findall("\d+\.\d+", line.split(",")[-2])[0]) )
                mse_list.append(float(re.findall("\d+\.\d+",line.split(",")[-4])[0]))
                mae_list.append(float(re.findall("\d+\.\d+",line.split(",")[-3])[0]))
                loss_list.append(float(line.split(",")[-5]))

            # print(line)
            # input()
            if "| app.py:185 | Disconnect and shut down" in line.strip():
                # print(line)
                print("enter")
                time_end=line.split(",")[0]#.split(" ")[2:]
                print(time_end,"end")
                input()
            if "tensorflow/core/platform/cpu_feature_guard.cc:182]" in line:
                time_ini=line.split(".")[0]
                print(time_ini,"ini")
                # input()



        
        #GET POSITION WHERE THE GHIGHEST ELEMNT LIEST ON

        # Iterate through the list

        for i, value in enumerate(r2_list):
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
        


        
        if r2_list[max_index]<1.0:


            dataframes.append(pd.DataFrame({
            # 'type_model':type_model,
            # 'type_machine':type_machine,
            'type_algo':type_algo,
            'type_alpha':type_alpha,
            'type_slr':type_slr,
            'type_sparam':type_sparam,



            'epoch':int(int(max_index)+1),
            'r2':r2_list[max_index],
            'mse':mse_list[max_index],
            'mae':mae_list[max_index],
            'loss':loss_list[max_index],
            
            }))
        else:
            dataframes.append(pd.DataFrame({
            # 'type_model':type_model,
            # 'type_machine':type_machine,
            'type_algo':type_algo,
            'type_alpha':type_alpha,
            'type_slr':type_slr,
            'type_sparam':type_sparam,



            'epoch':str(ROUNDS)+"+",
            'r2':"-",
            'mse':"-",
            'mae':"-",
            'loss':"-",
            
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
            if "| fit progress:" in line :
                # prine.split(",")[-6]))
                r2_list.append(float(re.findall("\d+\.\d+", line.split(",")[-2])[0]) )
                mse_list.append(float(re.findall("\d+\.\d+",line.split(",")[-4])[0]))
                mae_list.append(float(re.findall("\d+\.\d+",line.split(",")[-3])[0]))
                loss_list.append(float(line.split(",")[-5]))
            # print(line)
            # input()
            if "| app.py:185 | Disconnect and shut down".strip().lower() in line.strip().lower():
                # print(line)
                print("enter")
                time_end=line.split(",")[0]#.split(" ")[2:]
                print(time_end,"end")
                input()
            if "tensorflow/core/platform/cpu_feature_guard.cc:182]" in line:
                time_ini=line.split(".")[0]
                print(time_ini,"ini")
                # input()


        
        #GET POSITION WHERE THE GHIGHEST ELEMNT LIEST ON



        # Iterate through the list

        for i, value in enumerate(r2_list):
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

        if r2_list[max_index]<1.0:


            dataframes.append(pd.DataFrame({
            # 'type_model':type_model,
            # 'type_machine':type_machine,
            'type_algo':type_algo,
            'type_alpha':type_alpha,
            'type_slr':type_slr,
            'type_sparam':type_sparam,



            'epoch':int(int(max_index)+1),
            'r2':r2_list[max_index],
            'mse':mse_list[max_index],
            'mae':mae_list[max_index],
            'loss':loss_list[max_index],
            
            }))
        else:
            dataframes.append(pd.DataFrame({
            # 'type_model':type_model,
            # 'type_machine':type_machine,
            'type_algo':type_algo,
            'type_alpha':type_alpha,
            'type_slr':type_slr,
            'type_sparam':type_sparam,



            'epoch':str(ROUNDS)+"+",
            'r2':"-",
            'mse':"-",
            'mae':"-",
            'loss':"-",
            
            }))
        

    input()
result = pd.concat(dataframes, axis=0)

# Reset the index, if needed
result = result.reset_index(drop=True).sort_values(['type_alpha','type_slr','type_algo','type_sparam'],ascending=[True,True,True,True])


result['r2'] = result['r2'].apply(round_float_or_keep_string)
result['mse'] = result['mse'].apply(round_float_or_keep_string)
result['mae'] = result['mae'].apply(round_float_or_keep_string)
result['loss'] = result['loss'].apply(round_float_or_keep_string)

result.to_csv('FEDERATED_LSTM.csv',index=False)



result['type_alpha'] = result['type_alpha'].apply(lambda x: '{:.0f}'.format(float(x)) if float(x).is_integer() else '{:g}'.format(float(x)))
# Convert DataFrame to LaTeX table
latex_table = result.drop('epoch', axis=1).to_latex(index=False)

# Print the LaTeX table or save it to a .tex file
print(latex_table)

# result.to_csv('data.csv', index=False)
# print(result)

