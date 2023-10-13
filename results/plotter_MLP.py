import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np
from scipy.signal import savgol_filter, medfilt
from sklearn import preprocessing 


dir_path = os.path.dirname(os.path.realpath(__file__))

ROUNDS=100
FOLDER_PATH=dir_path+"/"+'tuesep26'
PATH_FIND='MLP_TESLA_'
DATA_TYPE=['TYPE1','TYPE2']
METRICS=['accuracy','weighted_avg','MCS']
COLORS=['b','g','r','c']
names=[]
plots_dict=[]


os.chdir(FOLDER_PATH)
print(FOLDER_PATH)
files = glob(PATH_FIND+'*')
print(files)

for type_data in DATA_TYPE:
    # I GET THE FILE TYPE I WANT
    files_filtered=filter(lambda c: type_data in c, files)
    
    for element in files_filtered:
        
        data=open(element)
        
        name_fed=element.split("_")[-2]
        name_full=element
        names.append(name_fed)


        accuracy_list=[]
        weightedavg_list=[]
        mcs_list=[]
        loss_list=[]

        for line in data:
            if line.startswith("    accuracy"):
                accuracy_list.append(float(line.split("      ")[-2]))
                # pair_counter=pair_counter+1
            elif line.startswith("weighted avg"):
                weightedavg_list.append(float(line.split(" ")[-7]))
            elif line.startswith("MCS "):
                mcs_list.append(float(line.split(" ")[-1].replace("\n","")))
            elif "| fit progress:" in line :
                # print(line)
                # print(line.split(","))
                # print(len(line.split(",")))
                # print(float(line.split(",")[-6]))
                loss_list.append(float(line.split(",")[-6]))
        
                


                

                
        
        
        print(len(accuracy_list))
        print(len(weightedavg_list))
        print(len(mcs_list))

        min_lenght=min(len(accuracy_list),len(weightedavg_list),len(mcs_list))

        accuracy_list=accuracy_list[:min_lenght]
        weightedavg_list=weightedavg_list[:min_lenght]
        mcs_list=mcs_list[:min_lenght]
        




        
        #CHECK FOR DUPLICATE OUTPUT!!!
        # try:
        inner_pandas=pd.DataFrame.from_dict({'accuracy':accuracy_list,'weighted_avg':weightedavg_list,'MCS':mcs_list})
        t = inner_pandas[['accuracy', 'weighted_avg', 'MCS']]     
        inner_pandas=inner_pandas[(t.ne(t.shift())).any(axis=1)]
        


        min_lenght=min(inner_pandas.shape[0],len(loss_list))
        
        inner_pandas=inner_pandas.iloc[:min_lenght,:]

        loss_list=loss_list[:min_lenght]

        # print(inner_pandas.shape)
        # print(len(loss_list))
        inner_pandas['loss'] = loss_list
        # ,'loss':loss_list


        # print(len(loss_list))


        data_frame=pd.DataFrame({'name_full':name_full,'name_fed':name_fed,'type':type_data,'data':[inner_pandas]})
        # print(data_frame['data'][0].head())
        plots_dict.append(data_frame)
        # except Exception as e:
        #     print(e)
    
    for i in DATA_TYPE:
        plot_accuracy=[]
        plot_weighted_avg=[]
        plot_MCS=[]
        plot_loss=[]
        


        #CREAte A DATAFRAME FOR EACH DATA_TYPE
        # master=pd.DataFrame(index=np.arange(4000), columns=['columns']).reset_index()
        
        for x in plots_dict:
            # print(master)
            if x['type'][0] ==i:
                plot_accuracy.append(x['data'][0]['accuracy'].values)
                plot_weighted_avg.append(x['data'][0]['weighted_avg'].values)
                plot_MCS.append(x['data'][0]['MCS'].values)
                plot_loss.append(x['data'][0]['loss'].values)

        
        fig, ax = plt.subplots(1)
        
        


        # print(len(plot_accuracy))
        

        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        

        for name,sub,color in zip(list(set(names)),plot_accuracy,COLORS):
            print(name)
            try:
                print("OK")
                ax.plot(np.arange(len(sub)), medfilt(scaler.fit_transform(sub.reshape(-1, 1)).reshape(-1),51),color ,label=name)
                ax.set_xlim([0, ROUNDS])
                ax.set_xbound(lower=-3, upper=ROUNDS)
                ax.set_xlabel('Rounds')
                ax.set_ylabel('Accuracy')
                ax.legend(loc="lower right")

            except Exception as e:
                print(e)
        
        plt.savefig(FOLDER_PATH+"/"+i+"accuracy"+"MLP.pdf")

        fig, ax = plt.subplots(1)

    

        # ax.set_title("plot_weighted_avg")
        # print(len(plot_weighted_avg))
        for name,sub,color in zip(list(set(names)),plot_weighted_avg,COLORS):
            try:
                print("OK")
                ax.plot(np.arange(len(sub)), medfilt(scaler.fit_transform(sub.reshape(-1, 1)).reshape(-1),51),color,label=name)
                ax.set_xlim([0, ROUNDS])
                ax.set_xbound(lower=-3, upper=ROUNDS)
                ax.set_xlabel('Rounds')
                ax.set_ylabel('Weighted Avg')
                ax.legend(loc="lower right")
            except Exception as e:
                print(e)
        
        plt.savefig(FOLDER_PATH+"/"+i+"weighted_avg"+"MLP.pdf")
        
        fig, ax = plt.subplots(1)
        
 


        # ax.set_title("plot_MCS")  
        # print(len(plot_MCS))
        for name,sub,color in zip(list(set(names)),plot_MCS,COLORS):
            try:
                
                ax.plot(np.arange(len(sub)), medfilt(scaler.fit_transform(sub.reshape(-1, 1)).reshape(-1),51),color,label=name)
                
                # ax.set_aspect('auto')
                ax.set_xlim([0, ROUNDS])
                ax.set_xbound(lower=-3, upper=ROUNDS)
                ax.set_xlabel('Rounds')
                ax.set_ylabel('MCS')
                ax.legend(loc="lower right")
            except Exception as e:
                print(e)
        plt.savefig(FOLDER_PATH+"/"+i+"MCS"+"MLP.pdf")

        fig, ax = plt.subplots(1)


                # ax.set_title("plot_MCS")  
        print(len(plot_loss))
        for name,sub,color in zip(list(set(names)),plot_loss,COLORS):
            # try:
            ax.plot(np.arange(len(sub)),sub, color,label=name)
            # ax.set_aspect('auto')
            ax.set_xlim([0, ROUNDS])
            ax.set_xbound(lower=-3, upper=ROUNDS)
            ax.set_xlabel('Rounds')
            ax.set_ylabel('Loss')
            ax.legend(loc="lower right")
            # except Exception as e:
            #     print(e)
        plt.savefig(FOLDER_PATH+"/"+i+"loss"+"MLP.pdf")

        fig, ax = plt.subplots(1)

  
      



                

           

    print("")    

