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
FOLDER_PATH=dir_path+"/"+'tuesep26'
PATH_FIND='MLP_TESLA_'
DATA_TYPE=['TYPE1','TYPE2']
METRICS=['accuracy','weighted_avg','MCS']
COLORS=['b','g','r','c']
names=[]
plots_dict=[]



#(epoch,loss,accuracy)
HISTORY_MLP_M3C=[(1,1.8707,0.1341),(2,1.8519,0.1479),(3,1.8114,0.1883),(4,1.7277,0.3251),(5,1.5653,0.5774),(6,1.2772,0.7503),(7,0.9096,0.7932),(8,0.6379,0.8813),(9,0.4300,0.9297),(10,0.3273,0.9434),(11,0.2525,0.9670),(12,0.1906,0.9814),(13,0.1552,0.9849)] #TAKEN FROM THE VALIDATION INSIDE 

HISTORY_LSTM_SEQ20_M3=[]

HISTORY_LSTM_SEQ40_M3=[]

HISTORY_LSTM_SEQ80_M3=[]

#FONT SIZE
plt.rcParams.update({'font.size': 16})





#FOR LSTM EXRACTOR:
PATHS=['out_server_14_RULM3_SEQ20_TYPE1.txt2','out_server_14_RULM3_SEQ40_TYPE1.txt2','out_server_14_RULM3_SEQ80_TYPE1.txt2']




r2_list=[]
mse_list=[]
mae_list=[]
loss_list=[]

for element in PATHS:
        
    data=open(dir_path+"/"+'tuesep26'+"/"+element)
    for line in data:
        # if line.startswith("    accuracy"):
        #     accuracy_list.append(float(line.split("      ")[-2]))
        #     # pair_counter=pair_counter+1
        # elif line.startswith("weighted avg"):
        #     weightedavg_list.append(float(line.split(" ")[-7]))
        # elif line.startswith("MCS "):
        #     mcs_list.append(float(line.split(" ")[-1].replace("\n","")))
        if " - loss: " in line :
            # print(line)
            # print(line.split(","))
            # print(len(line.split(",")))
            # print(float(line.split(",")[-6]))
            r2_list.append(float(re.findall("\d+\.\d+", line.split(",")[-2])[0]) )
            mse_list.append(float(re.findall("\d+\.\d+",line.split(",")[-4])[0]))
            mae_list.append(float(re.findall("\d+\.\d+",line.split(",")[-3])[0]))
            loss_list.append(float(line.split(",")[-5]))
        elif "Restoring model weights from the end of the best epoch: " in line :
            print(line)




#ACCURACY


fig, ax = plt.subplots(1)




ax.plot(np.arange(len(sub)), medfilt(scaler.fit_transform(sub.reshape(-1, 1)).reshape(-1),51),color,label=name)
ax.set_xlim([0, ROUNDS])
ax.set_xbound(lower=-3, upper=ROUNDS)
ax.set_xlabel('Rounds')
ax.set_ylabel('MSE')
ax.legend(loc="upper right")
    # except Exception as e:
    #     print(e)


plt.tight_layout()
plt.savefig(FOLDER_PATH+"/"+i+PATH_FIND_PART2+"mse"+"LSTM.pdf")




#LOSS
fig, ax = plt.subplots(1)


ax.plot(np.arange(len(sub)),sub, color,label=name)
# ax.set_aspect('auto')
ax.set_xlim([0, ROUNDS])
ax.set_xbound(lower=-3, upper=ROUNDS)
ax.set_xlabel('Rounds')
ax.set_ylabel('Loss')
ax.legend(loc="upper right")
    # except Exception as e:
    #     print(e)
plt.tight_layout()
plt.savefig(FOLDER_PATH+"/"+i+"loss"+"MLP.pdf")


