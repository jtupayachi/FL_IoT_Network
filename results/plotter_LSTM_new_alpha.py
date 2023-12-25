import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
import numpy as np
from scipy.signal import savgol_filter, medfilt
from sklearn import preprocessing 
import re


"""
THIS SCRIPT ONLY ALLOWS 1 alpha
"""
dir_path = os.path.dirname(os.path.realpath(__file__))

ROUNDS=1000
FOLDER_PATH=dir_path+"/"+'final'
PATH_FIND_PART1='LSTM_TESLA_'
# PATH_FIND_PART2='0.001'

DATA_TYPE=['0.01'] #,'0.01','0.1','0.02','0.2','0.005','0.05','0.5','0.075','1.0','1000000.0'
METRICS=['r2','mse','mae']
COLORS=['b','g','r','c']
names=[]
plots_dict=[]
plt.rcParams.update({'font.size': 16})

os.chdir(FOLDER_PATH)
print(FOLDER_PATH)
files = glob(PATH_FIND_PART1+'*')# Filter lines based on the content between the second and third underscores
files = [k for k in files if k.split('_')[3] == DATA_TYPE[0]]


# files=[k for k in files if PATH_FIND_PART2 in k]

print(files)
# for type_data in DATA_TYPE:
#     # I GET THE FILE TYPE I WANT
#     files_filtered=filter(lambda c: type_data in c, files)
    
for element in files:

    try:
        
        data=open(element)
        
        name_fed=element.split("_")[-3]
        name_full=element
        names.append(name_fed)


        r2_list=[]
        mse_list=[]
        mae_list=[]
        loss_list=[]

        for line in data:
            # if line.startswith("    accuracy"):
            #     accuracy_list.append(float(line.split("      ")[-2]))
            #     # pair_counter=pair_counter+1
            # elif line.startswith("weighted avg"):
            #     weightedavg_list.append(float(line.split(" ")[-7]))
            # elif line.startswith("MCS "):
            #     mcs_list.append(float(line.split(" ")[-1].replace("\n","")))
            if "| fit progress:" in line :
                # print(line)
                # print(line.split(","))
                # print(len(line.split(",")))
                # print(float(line.split(",")[-6]))
                r2_list.append(float(re.findall("\d+\.\d+", line.split(",")[-2])[0]) )
                mse_list.append(float(re.findall("\d+\.\d+",line.split(",")[-4])[0]))
                mae_list.append(float(re.findall("\d+\.\d+",line.split(",")[-3])[0]))
                loss_list.append(float(line.split(",")[-5]))
        
                


                

                
        
        


        min_lenght=min(len(r2_list),len(mse_list),len(mae_list),len(loss_list))

        r2_list=r2_list[:min_lenght]
        mse_list=mse_list[:min_lenght]
        mae_list=mae_list[:min_lenght]
        loss_list=loss_list[:min_lenght]
        




        
        #CHECK FOR DUPLICATE OUTPUT!!! 
        inner_pandas=pd.DataFrame.from_dict({'r2':r2_list,'mse':mse_list,'mae':mae_list,'loss':loss_list})



     
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
            axs[row, col].legend()

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


        # Save the plot to a file (uncomment the lines below if needed)
        # FOLDER_PATH = "path/to/your/folder"
        # element = "your_element_name"
        # plt.savefig(FOLDER_PATH + "/" + element + "_metrics_MLP.pdf")

        # Show the plot
        plt.show()
        plt.savefig(FOLDER_PATH+ "/" + element +"loss"+"LSTM.pdf")

        # data_frame=pd.DataFrame({'name_full':name_full,'name_fed':name_fed,'type':DATA_TYPE[0],'data':[inner_pandas]})
        # plots_dict.append(data_frame)
        
        # plot_r2=[]
        # plot_mse=[]
        # plot_mae=[]
        # plot_loss=[]



        # for x in plots_dict:
        #     # print(master)
        #     if x['type'][0] ==DATA_TYPE[0]:
        #         plot_r2.append(x['data'][0]['r2'].values)
        #         plot_mse.append(x['data'][0]['mse'].values)
        #         plot_mae.append(x['data'][0]['mae'].values)
        #         plot_loss.append(x['data'][0]['loss'].values)


        # print(plot_r2[:-1])
        # print(plot_mse[:-1])
        # print(plot_mae[:-1])
        # print(plot_loss[:-1])
        # input()

        # input()


        # OLD SECTION!!!!
        
        # fig, ax = plt.subplots(1)
        # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        # for name,sub,color in zip(list(set(names)),plot_r2,COLORS):
        #     print(name)
        #     try:
        #         sub=[x if 0 <= x <= 1 else 0 for x in sub]
        #         print("OK")
        #         ax.plot(np.arange(len(sub)), sub,color ,label=name)
        #         # medfilt(scaler.fit_transform(.reshape(-1, 1)).reshape(-1),51)
        #         ax.set_xlim([0, ROUNDS])
        #         ax.set_xbound(lower=-3, upper=ROUNDS)
        #         ax.set_ylim([0, 1])
        #         ax.set_ybound(lower=0, upper=1)
        #         ax.set_xlabel('Rounds')
        #         ax.set_ylabel('R2')
        #         ax.legend(loc="lower right")

        #     except Exception as e:
        #         print(e)
        
        # plt.tight_layout()
        # plt.savefig(FOLDER_PATH+"/"+element+"_"+"r2"+"LSTM.pdf")

        # fig, ax = plt.subplots(1)
        # for name,sub,color in zip(list(set(names)),plot_mse,COLORS):
        #     try:
        #         print("OK")
        #         ax.plot(np.arange(len(sub)), sub,color,label=name)
        #         ax.set_xlim([0, ROUNDS])
        #         ax.set_xbound(lower=-3, upper=ROUNDS)
        #         # ax.set_ylim([0, 1])
        #         # ax.set_ybound(lower=0, upper=1)
        #         ax.set_xlabel('Rounds')
        #         ax.set_ylabel('MSE')
        #         ax.legend(loc="upper right")
        #     except Exception as e:
        #         print(e)
        
        # plt.tight_layout()
        # plt.savefig(FOLDER_PATH+"/"+element+"_"+"mse"+"LSTM.pdf")
        


        # fig, ax = plt.subplots(1)
        # for name,sub,color in zip(list(set(names)),plot_mae,COLORS):
        #     try:
        #         ax.plot(np.arange(len(sub)), sub,color,label=name)
        #         ax.set_xlim([0, ROUNDS])
        #         ax.set_xbound(lower=-3, upper=ROUNDS)
        #         # ax.set_ylim([0, 1])
        #         # ax.set_ybound(lower=0, upper=1)
        #         ax.set_xlabel('Rounds')
        #         ax.set_ylabel('MAE')
        #         ax.legend(loc="upper right")
        #     except Exception as e:
        #         print(e)
        # plt.tight_layout()
        # plt.savefig(FOLDER_PATH+"/"+element+"_"+"mae"+"LSTM.pdf")

        # fig, ax = plt.subplots(1)
        # for name,sub,color in zip(list(set(names)),plot_loss,COLORS):
        #     # try:
        #     ax.plot(np.arange(len(sub)),sub, color,label=name)
        #     # ax.set_aspect('auto')
        #     ax.set_xlim([0, ROUNDS])
        #     ax.set_xbound(lower=-3, upper=ROUNDS)
        #     # ax.set_ylim([0, 1])
        #     # ax.set_ybound(lower=0, upper=1)
        #     ax.set_xlabel('Rounds')
        #     ax.set_ylabel('Loss')
        #     ax.legend(loc="upper right")
        #     # except Exception as e:
        #     #     print(e)
        # plt.tight_layout()
        # plt.savefig(FOLDER_PATH+"/"+element+"_"+"loss"+"LSTM.pdf")

        # fig, ax = plt.subplots(1)

        # print("FINISH LOOP")    



        # # Extract information from the element variable
        # split_element = element.split('_')
        # # model_name = split_element[0]
        # # data_name = split_element[1]
        # learning_rate = split_element[3]
        # slr = split_element[5]

        # # Define specific colors for each metric
        # colors_r2 = 'blue'
        # colors_mse = 'green'
        # colors_mae = 'orange'
        # colors_loss = 'red'

        # # Create a single plot for all metrics
        # fig, ax = plt.subplots(figsize=(12, 8))

        # # Define different line styles for each metric
        # line_styles_r2 = ['-', '--', '-.', ':']
        # line_styles_mse = ['--', ':', '-', '-.']
        # line_styles_mae = ['-.', ':', '--', '-']
        # line_styles_loss = [':', '-', '--', '-.']

        # # Plot R2
        # for i, (name, sub) in enumerate(zip(list(set(names)), plot_r2)):
        #     try:
        #         sub = [x if 0 <= x <= 1 else 0 for x in sub]
        #         ax.plot(np.arange(len(sub)), sub, linestyle=line_styles_r2[i % len(line_styles_r2)], color=colors_r2, label=f"{name} - R2")

        #     except Exception as e:
        #         print(e)

        # # Plot MSE
        # for i, (name, sub) in enumerate(zip(list(set(names)), plot_mse)):
        #     try:
        #         ax.plot(np.arange(len(sub)), sub, linestyle=line_styles_mse[i % len(line_styles_mse)], color=colors_mse, label=f"{name} - MSE")

        #     except Exception as e:
        #         print(e)

        # # Plot MAE
        # for i, (name, sub) in enumerate(zip(list(set(names)), plot_mae)):
        #     try:
        #         ax.plot(np.arange(len(sub)), sub, linestyle=line_styles_mae[i % len(line_styles_mae)], color=colors_mae, label=f"{name} - MAE")

        #     except Exception as e:
        #         print(e)

        # # Plot Loss
        # for i, (name, sub) in enumerate(zip(list(set(names)), plot_loss)):
        #     try:
        #         ax.plot(np.arange(len(sub)), sub, linestyle=line_styles_loss[i % len(line_styles_loss)], color=colors_loss, label=f"{name} - Loss")

        #     except Exception as e:
        #         print(e)

        # # Set the super title
        # fig.suptitle(f"Performance Metrics - LR {learning_rate} - SLR {slr}")

        # # Set labels and limits
        # ax.set_xlim([0, ROUNDS])
        # ax.set_xbound(lower=-3, upper=ROUNDS)
        # ax.set_xlabel('Rounds')
        # ax.set_ylabel('Metrics')

        # # Display a single legend for all metrics with different line styles and colors
        # ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1), frameon=False)

        # # Adjust layout
        # plt.tight_layout()

        # # Save the combined figure
        # plt.savefig(FOLDER_PATH + "/" + element + f"1_combined_plots_LSTM_{element}.pdf", bbox_inches='tight')

        # # Show the combined figure
        # plt.show()

        # # Clear the current figure for the next iteration
        # plt.clf()


        # print("FINISHHH")

    except:
        print("Error")
        print(element)
        
