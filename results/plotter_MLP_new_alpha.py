import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np
from scipy.signal import savgol_filter, medfilt
from sklearn import preprocessing 


dir_path = os.path.dirname(os.path.realpath(__file__))

ROUNDS=100
FOLDER_PATH=dir_path+"/"+'final'
PATH_FIND_PART1='MLP_TESLA_'
DATA_TYPE=['0.001'] #,'0.01','0.1','0.02','0.2','0.005','0.05','0.5','0.075','1.0','1000000.0'
METRICS=['accuracy','weighted avg','MCS']
COLORS=['b','g','r','c']
names=[]
plots_dict=[]



#FONT SIZE
plt.rcParams.update({'font.size': 16})

os.chdir(FOLDER_PATH)
print(FOLDER_PATH)
files = glob(PATH_FIND_PART1+'*')# Filter lines based on the content between the second and third underscores
files = [k for k in files if k.split('_')[3] == DATA_TYPE[0]]


# files=[k for k in files if PATH_FIND_PART2 in k]


# for type_data in DATA_TYPE:
#     # I GET THE FILE TYPE I WANT
#     files_filtered=filter(lambda c: type_data in c, files)
    
for element in files:

    try:
        
        data=open(element)
        print(element)
            
        name_fed=element.split("_")[-3]
        name_full=element
        names.append(name_fed)


        accuracy_list=[]
        weightedavg_list=[]
        mcs_list=[]
        loss_list=[]

        for line in data:
            if line.startswith("    accuracy"):
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
        
        # loss_list=loss_list[1:]
            

        min_lenght=min(len(accuracy_list),len(weightedavg_list),len(mcs_list),len(loss_list))

        accuracy_list=accuracy_list[:min_lenght]
        weightedavg_list=weightedavg_list[:min_lenght]
        mcs_list=mcs_list[:min_lenght]
        loss_list=loss_list[:min_lenght]

        print(element)
        print("accuracy_list")
        print(accuracy_list)
        print(len(accuracy_list))
        print("weightedavg_list")
        print(weightedavg_list)
        print(len(weightedavg_list))
        print("mcs_list")
        print(mcs_list)
        print(len(mcs_list))
        print("loss_list")
        print(loss_list)
        print(len(loss_list))
        # input()
        #CHECK
    

                    
            
            
        # print(accuracy_list)
        # print(weightedavg_list)
        # print(mcs_list)
        # input()


            




            
        #CHECK FOR DUPLICATE OUTPUT!!!
        # try:
        inner_pandas=pd.DataFrame.from_dict({'accuracy':accuracy_list,'weighted_avg':weightedavg_list,'MCS':mcs_list,'Loss':loss_list})
        
        # t = inner_pandas[['accuracy', 'weighted_avg', 'MCS','Loss']]     
        # inner_pandas=inner_pandas[(t.ne(t.shift())).any(axis=1)]

        #DATA IS BAD HEre
        print(inner_pandas)
        df=inner_pandas
        # input()

        # Create a plot with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(element)

        # Plot lines for each metric in separate subplots
        for i, column in enumerate(df.columns):
            row = i // 2
            col = i % 2
            axs[row, col].plot(df.index, df[column], label=column, marker='o')

            # Add trend line
            z = np.polyfit(df.index, df[column], 1)
            p = np.poly1d(z)
            axs[row, col].plot(df.index, p(df.index), 'r--', label='Trend Line')

            axs[row, col].set_xlabel('Rounds')
            axs[row, col].set_ylabel(column)
            axs[row, col].set_xlim([0, ROUNDS])
            axs[row, col].set_xbound(lower=-3, upper=ROUNDS)
            axs[row, col].legend()
            

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


        # Save the plot to a file (uncomment the lines below if needed)
        # FOLDER_PATH = "path/to/your/folder"
        # element = "your_element_name"
        # plt.savefig(FOLDER_PATH + "/" + element + "_metrics_MLP.pdf")

        # Show the plot
        plt.show()
        plt.savefig(FOLDER_PATH+ "/" + element +"loss"+"MLP.pdf")

                
        


        # min_lenght=min(inner_pandas.shape[0],len(loss_list))


        
        # inner_pandas=inner_pandas.iloc[:min_lenght,:]



        # loss_list=loss_list[:min_lenght]


        # print(inner_pandas.shape)
        # print(len(loss_list))
        # inner_pandas['loss'] = loss_list
        # ,'loss':loss_list

        
    #     print(inner_pandas)
    #     print(inner_pandas.describe())


    #     # print(len(loss_list))

    #     data_frame=pd.DataFrame({'name_full':name_full,'name_fed':name_fed,'type':DATA_TYPE[0],'data':[inner_pandas]})

    #     plots_dict.append(data_frame)

    #     plot_accuracy=[]
    #     plot_weighted_avg=[]
    #     plot_MCS=[]
    #     plot_loss=[]
            


    #         #CREAte A DATAFRAME FOR EACH DATA_TYPE
    #         # master=pd.DataFrame(index=np.arange(4000), columns=['columns']).reset_index()
            
    #     for x in plots_dict:
    #         # print(master)
    #         if x['type'][0] ==DATA_TYPE[0]:
    #             plot_accuracy.append(x['data'][0]['accuracy'].values)
    #             plot_weighted_avg.append(x['data'][0]['weighted_avg'].values)
    #             plot_MCS.append(x['data'][0]['MCS'].values)
    #             plot_loss.append(x['data'][0]['loss'].values)


    #     # print(len(plot_accuracy))
    #     # print(len(plot_weighted_avg))
    #     # print(len(plot_MCS))
    #     # print(len(plot_loss))
    #     print(element)
    #     print("plot_accuracy")
    #     print(plot_accuracy)
    #     print("plot_weighted_avg")
    #     print(plot_weighted_avg)
    #     print("plot_MCS")
    #     print(plot_MCS)
    #     print("plot_loss")
    #     print(plot_loss)
    #     print("#####################################")
    #     # input()


    #     # Extract information from the element variable
    #     split_element = element.split('_')
    #     # model_name = split_element[0]
    #     # data_name = split_element[1]
    #     learning_rate = split_element[3]
    #     slr = split_element[5]

    #     # Define specific colors for each metric
    #     colors_accuracy = 'blue'
    #     colors_weighted_avg = 'green'
    #     colors_MCS = 'orange'
    #     colors_loss = 'red'

    #     # Create a single plot for all metrics
    #     fig, ax = plt.subplots(figsize=(12, 8))

    #     # Define different line styles for each metric
    #     line_styles_accuracy = ['-', '--', '-.', ':']
    #     line_styles_weighted_avg = ['--', ':', '-', '-.']
    #     line_styles_MCS = ['-.', ':', '--', '-']
    #     line_styles_loss = [':', '-', '--', '-.']





        

    #     # Create a single plot for all metrics
    #     fig, ax = plt.subplots(figsize=(12, 8))

    #     scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    #     for metric, ylabel, color, linestyle in zip([plot_accuracy, plot_weighted_avg, plot_MCS, plot_loss],
    #                                             ['Accuracy', 'F1-Macro W. Avg', 'MCS', 'Loss'],
    #                                             ['blue', 'green', 'orange', 'red'],
    #                                             ['-', '--', '-.', ':']):
    #         for name, sub in zip(list(set(names)), metric):
    #             try:
    #                 # sub = medfilt(sub, 51)  # Apply median filter
    #                 ax.plot(np.arange(len(sub)), sub, linestyle, color=color, label=f"{name} - {ylabel}")
    #                 ax.set_xlim([0, ROUNDS])
    #                 ax.set_xbound(lower=-3, upper=ROUNDS)
    #                 ax.set_xlabel('Rounds')
    #                 ax.set_ylabel("Metrics")
    #                 ax.legend(loc="lower right")

    #             except Exception as e:
    #                 print(e)

    #     fig.suptitle(f"Performance Metrics - LR {learning_rate} - SLR {slr}")

    #     # Adjust layout
    #     plt.tight_layout()

    #     # Save the combined figure
    #     plt.savefig(FOLDER_PATH + "/" + element + "_combined_metrics_MLP.pdf", bbox_inches='tight')

    #     # Show the combined figure
    #     plt.show()

    #     # Clear the current figure for the next iteration
    #     plt.clf()
        
    #     # fig, ax = plt.subplots(1)
            
            


            

    #     # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        

    #     # for name,sub,color in zip(list(set(names)),plot_accuracy,COLORS):
    #     #     print(name)
    #     #     try:
    #     #         print("OK")
    #     #         ax.plot(np.arange(len(sub)), sub,color ,label=name)
    #     #         # medfilt(scaler.fit_transform(.reshape(-1, 1)).reshape(-1),51)
    #     #         ax.set_xlim([0, ROUNDS])
    #     #         ax.set_xbound(lower=-3, upper=ROUNDS)
    #     #         ax.set_xlabel('Rounds')
    #     #         ax.set_ylabel('Accuracy')
    #     #         ax.legend(loc="lower right")

    #     #     except Exception as e:
    #     #         print(e)
        
    #     # plt.tight_layout()
    #     # plt.savefig(FOLDER_PATH+ "/" + element +"accuracy"+"MLP.pdf")

    #     # fig, ax = plt.subplots(1)



    #     #     # ax.set_title("plot_weighted_avg")
    #     #     # print(len(plot_weighted_avg))
    #     # for name,sub,color in zip(list(set(names)),plot_weighted_avg,COLORS):
    #     #     try:
    #     #         print("OK")
    #     #         ax.plot(np.arange(len(sub)), sub,color,label=name)
    #     #         # medfilt(scaler.fit_transform(.reshape(-1, 1)).reshape(-1),51)
    #     #         ax.set_xlim([0, ROUNDS])
    #     #         ax.set_xbound(lower=-3, upper=ROUNDS)
    #     #         ax.set_xlabel('Rounds')
    #     #         ax.set_ylabel('F1-Macro W. Avg')
    #     #         ax.legend(loc="lower right")
    #     #     except Exception as e:
    #     #         print(e)
        
    #     # plt.tight_layout()
    #     # plt.savefig(FOLDER_PATH+ "/" + element +"weighted_avg"+"MLP.pdf")
        
    #     # fig, ax = plt.subplots(1)
            



    #     # # ax.set_title("plot_MCS")  
    #     # # print(len(plot_MCS))
    #     # for name,sub,color in zip(list(set(names)),plot_MCS,COLORS):
    #     #     try:
                
    #     #         ax.plot(np.arange(len(sub)), sub,color,label=name)
    #     #         # medfilt(scaler.fit_transform(.reshape(-1, 1)).reshape(-1),51)
    #     #         # ax.set_aspect('auto')
    #     #         ax.set_xlim([0, ROUNDS])
    #     #         ax.set_xbound(lower=-3, upper=ROUNDS)
    #     #         ax.set_xlabel('Rounds')
    #     #         ax.set_ylabel('MCS')
    #     #         ax.legend(loc="lower right")
    #     #     except Exception as e:
    #     #         print(e)
    #     # plt.tight_layout()
    #     # plt.savefig(FOLDER_PATH+"/"+ element +"MCS"+"MLP.pdf")

    #     # fig, ax = plt.subplots(1)


    #     #     # ax.set_title("plot_MCS")  
    #     # print(len(plot_loss))
    #     # for name,sub,color in zip(list(set(names)),plot_loss,COLORS):
    #     #     # try:
    #     #     ax.plot(np.arange(len(sub)),sub, color,label=name)
    #     #     # ax.set_aspect('auto')
    #     #     ax.set_xlim([0, ROUNDS])
    #     #     ax.set_xbound(lower=-3, upper=ROUNDS)
    #     #     ax.set_xlabel('Rounds')
    #     #     ax.set_ylabel('Loss')
    #     #     ax.legend(loc="upper right")
    #     #     # except Exception as e:
    #     #     #     print(e)

    #     # fig, ax = plt.subplots(1)


    #     # print("")    
    except:
        print("Error")
        print(element)
        
