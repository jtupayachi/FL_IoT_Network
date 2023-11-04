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

"""LSTM_TESLA_FedAvgM_0.001_slr_0.001_0.0.txt"""
"""LSTM_TESLA_FedAvg_0.02.txt"""



for name in files:

    #PARAMETERS LIST INI
    type_model=[]
    type_machine=[]
    type_algo=[]
    type_alpha=[]
    type_slr=[]
    type_sparam=[]


    r2_list=[]
    mse_list=[]
    mae_list=[]
    loss_list=[]


    if  "_FedAvg_" in name:
        
        type_model=type_model.append(name.split("_")[0])
        type_machine=type_machine.append(name.split("_")[1])
        type_algo=type_algo.append(name.split("_")[2])
        type_alpha=type_alpha.append(name.split("_")[3])
        type_slr=type_slr.append("-")
        type_sparam=type_sparam.append("-")





        
        # print("this")
        #CASE WITH FED AVERAGE
    
    else:
        type_model=type_model.append(name.split("_")[0])
        type_machine=type_machine.append(name.split("_")[1])
        type_algo=type_algo.append(name.split("_")[2])
        type_alpha=type_alpha.append(name.split("_")[3])
        type_slr=type_slr.append(name.split("_")[5])
        type_sparam=type_sparam.append(name.split("_")[6])

        #OTHER CASES
    #THIS IS THE FED AVERAGE





# input()
# for type_data in DATA_TYPE:
#     # I GET THE FILE TYPE I WANT
#     files_filtered=filter(lambda c: type_data in c, files)
    
#     for element in files_filtered:
        
#         data=open(element)
        
#         name_fed=element.split("_")[-3]
#         name_full=element
#         names.append(name_fed)


#         r2_list=[]
#         mse_list=[]
#         mae_list=[]
#         loss_list=[]

#         for line in data:
#             # if line.startswith("    accuracy"):
#             #     accuracy_list.append(float(line.split("      ")[-2]))
#             #     # pair_counter=pair_counter+1
#             # elif line.startswith("weighted avg"):
#             #     weightedavg_list.append(float(line.split(" ")[-7]))
#             # elif line.startswith("MCS "):
#             #     mcs_list.append(float(line.split(" ")[-1].replace("\n","")))
#             if "| fit progress:" in line :
#                 # print(line)
#                 # print(line.split(","))
#                 # print(len(line.split(",")))
#                 # print(float(line.split(",")[-6]))
#                 r2_list.append(float(re.findall("\d+\.\d+", line.split(",")[-2])[0]) )
#                 mse_list.append(float(re.findall("\d+\.\d+",line.split(",")[-4])[0]))
#                 mae_list.append(float(re.findall("\d+\.\d+",line.split(",")[-3])[0]))
#                 loss_list.append(float(line.split(",")[-5]))
        
                


                

                
        
        
#         print(len(r2_list))
#         print(len(mse_list))
#         print(len(mae_list))
#         print(len(loss_list))

#         # min_lenght=min(len(accuracy_list),len(weightedavg_list),len(mcs_list))

#         # accuracy_list=accuracy_list[:min_lenght]
#         # weightedavg_list=weightedavg_list[:min_lenght]
#         # mcs_list=mcs_list[:min_lenght]
        




        
#         #CHECK FOR DUPLICATE OUTPUT!!!
#         # try:
#         inner_pandas=pd.DataFrame.from_dict({'r2':r2_list,'mse':mse_list,'mae':mae_list,'loss':loss_list})
#         # t = inner_pandas[['accuracy', 'weighted_avg', 'MCS']]     
#         # inner_pandas=inner_pandas[(t.ne(t.shift())).any(axis=1)]
        


#         # min_lenght=min(inner_pandas.shape[0],len(loss_list))
        
#         # inner_pandas=inner_pandas.iloc[:min_lenght,:]

#         # loss_list=loss_list[:min_lenght]

#         # print(inner_pandas.shape)
#         # print(len(loss_list))
#         # inner_pandas['loss'] = loss_list
#         # ,'loss':loss_list


#         # print(len(loss_list))


#         data_frame=pd.DataFrame({'name_full':name_full,'name_fed':name_fed,'type':type_data,'data':[inner_pandas]})
#         # print(data_frame['data'][0].head())
#         plots_dict.append(data_frame)
#         # except Exception as e:
#         #     print(e)
    
#     for i in DATA_TYPE:
#         plot_r2=[]
#         plot_mse=[]
#         plot_mae=[]
#         plot_loss=[]
        


#         #CREAte A DATAFRAME FOR EACH DATA_TYPE
#         # master=pd.DataFrame(index=np.arange(4000), columns=['columns']).reset_index()
        
#         for x in plots_dict:
#             # print(master)
#             if x['type'][0] ==i:
#                 plot_r2.append(x['data'][0]['r2'].values)
#                 plot_mse.append(x['data'][0]['mse'].values)
#                 plot_mae.append(x['data'][0]['mae'].values)
#                 plot_loss.append(x['data'][0]['loss'].values)

        
#         fig, ax = plt.subplots(1)
        
#         print(plot_r2)
#         print(plot_mse)
        


#         # print(len(plot_accuracy))
        

#         scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        

#         for name,sub,color in zip(list(set(names)),plot_r2,COLORS):
#             print(name)
#             try:

#                 #MAKE SURE 0 and 1
#                 sub=[x if 0 <= x <= 1 else 0 for x in sub]
#                 print("OK")
#                 ax.plot(np.arange(len(sub)), sub,color ,label=name)
#                 # medfilt(scaler.fit_transform(.reshape(-1, 1)).reshape(-1),51)
#                 ax.set_xlim([0, ROUNDS])
#                 ax.set_xbound(lower=-3, upper=ROUNDS)
#                 ax.set_ylim([0, 1])
#                 ax.set_ybound(lower=0, upper=1)
#                 ax.set_xlabel('Rounds')
#                 ax.set_ylabel('R2')
#                 ax.legend(loc="lower right")

#             except Exception as e:
#                 print(e)
        
#         plt.tight_layout()
#         plt.savefig(FOLDER_PATH+"/"+i+PATH_FIND_PART2+"r2"+"LSTM.pdf")

#         fig, ax = plt.subplots(1)

    

#         # ax.set_title("plot_weighted_avg")
#         # print(len(plot_weighted_avg))
#         for name,sub,color in zip(list(set(names)),plot_mse,COLORS):
#             try:
#                 print("OK")
#                 ax.plot(np.arange(len(sub)), sub,color,label=name)
#                 ax.set_xlim([0, ROUNDS])
#                 ax.set_xbound(lower=-3, upper=ROUNDS)
# #                ax.set_ylim([0, 1])
# #                ax.set_ybound(lower=0, upper=1)
#                 ax.set_xlabel('Rounds')
#                 ax.set_ylabel('MSE')
#                 ax.legend(loc="upper right")
#             except Exception as e:
#                 print(e)
        
#         plt.tight_layout()
#         plt.savefig(FOLDER_PATH+"/"+i+PATH_FIND_PART2+"mse"+"LSTM.pdf")
        
#         fig, ax = plt.subplots(1)
        
 


#         # ax.set_title("plot_MCS")  
#         # print(len(plot_MCS))
#         for name,sub,color in zip(list(set(names)),plot_mae,COLORS):
#             try:
                
#                 ax.plot(np.arange(len(sub)), sub,color,label=name)
#                 ax.set_xlim([0, ROUNDS])
#                 ax.set_xbound(lower=-3, upper=ROUNDS)
# #                ax.set_ylim([0, 1])
# #                ax.set_ybound(lower=0, upper=1)
#                 ax.set_xlabel('Rounds')
#                 ax.set_ylabel('MAE')
#                 ax.legend(loc="upper right")
#             except Exception as e:
#                 print(e)
#         plt.tight_layout()
#         plt.savefig(FOLDER_PATH+"/"+i+PATH_FIND_PART2+"mae"+"LSTM.pdf")

#         fig, ax = plt.subplots(1)


#                 # ax.set_title("plot_MCS")  
#         print(len(plot_loss))
#         for name,sub,color in zip(list(set(names)),plot_loss,COLORS):
#             # try:
#             ax.plot(np.arange(len(sub)),sub, color,label=name)
#             # ax.set_aspect('auto')
#             ax.set_xlim([0, ROUNDS])
#             ax.set_xbound(lower=-3, upper=ROUNDS)
#             # ax.set_ylim([0, 1])
#             # ax.set_ybound(lower=0, upper=1)
#             ax.set_xlabel('Rounds')
#             ax.set_ylabel('Loss')
#             ax.legend(loc="upper right")
#             # except Exception as e:
#             #     print(e)
#         plt.tight_layout()
#         plt.savefig(FOLDER_PATH+"/"+i+PATH_FIND_PART2+"loss"+"LSTM.pdf")

#         fig, ax = plt.subplots(1)

  
 

#     print("")    

