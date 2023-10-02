#GENERATOR FOR OFFSET ! DATA

import numpy as np
import random
import math
import datetime
import calendar


#random.seed(0)


N_CLIENTS=15 #CHANGE ACCORDING TO EXPERIMENT::MAXIMUN NUMBER OF CLIENTS
IP_SERVER='172.19.0.5' # FIXED IP SERVER
IP_CLIENT_INI=6 #CLIENTS MUST HAVE IP ADDRESSES IN CONSECUTIVE ORDER
CLASSES=5
EPOCHS=11
EPOCHS_FL=1
ROUNDS=200
MIN_THRESHOLD=60





print("############Experiment Generation#############")
date = datetime.datetime.utcnow()
utc_time = calendar.timegm(date.utctimetuple())
print("UTC TIME: ",utc_time)

num_samples = CLASSES*N_CLIENTS
desired_mean = random.randint(50, 100)
desired_std_dev = random.randint(10, 15) # VALUE WE INCREASE!! initial 20,60 --  100,150


samples = np.random.normal(loc=0.0, scale=desired_std_dev, size=num_samples)

actual_mean = np.mean(samples)
actual_std = np.std(samples)
print("Initial samples stats   : mean = {:.4f} stdv = {:.4f}".format(actual_mean, actual_std))
zero_mean_samples = samples - (actual_mean)
zero_mean_mean = np.mean(zero_mean_samples)
zero_mean_std = np.std(zero_mean_samples)
print("True zero samples stats : mean = {:.4f} stdv = {:.4f}".format(zero_mean_mean, zero_mean_std))
scaled_samples = zero_mean_samples * (desired_std_dev/zero_mean_std)
scaled_mean = np.mean(scaled_samples)
scaled_std = np.std(scaled_samples)
print("Scaled samples stats    : mean = {:.4f} stdv = {:.4f}".format(scaled_mean, scaled_std))
l=[math.ceil(abs(ele+MIN_THRESHOLD)) for ele in scaled_samples] # number assures no less than ten datapoints


random.shuffle(l)






print("SERVER SENTENCE")

print()
#GENERATE -l #distribution

# #l=[x for x in random.randint(2, 200)]

# l = [ random.randint(2, 200) for _ in range(CLASSES*N_CLIENTS)]

l=str(l).replace(",","").replace("[", " ").replace("]", " ")



#GENERATE -fq # factions




fq=values = [0.0, 1.0] + [random.random() for _ in range(N_CLIENTS - 1)]
fq=values.sort()
fq=[values[i+1] - values[i] for i in range(N_CLIENTS)]


random.shuffle(fq)

# fq= [ random.random(0, 1) for _ in range(N_CLIENTS -1)]# " 0.1 0.6 0.3 1 "


fq=fq[:-1]
fq.append(1)
fq=str(fq).replace(",","").replace("[", " ").replace("]", " ")





# python3 fl_testbed/version2/client/datasplit.py -ml 2 -cm " + str(N_CLIENTS) + " -dfn combined_offset_misalignment.csv -ip " + str(IP_SERVER) + " -l " +  str(l)+ " -fq " +  str(fq) +" > out_server_"+str(N_CLIENTS)+".txt1"  + " && " + 
name = "python3 fl_testbed/version2/server/federated_server.py  -ml 1"                        + " -cm " + str(N_CLIENTS)  + " -e " + str(EPOCHS) + " --rounds " + str(ROUNDS)  + " -ip " + str(IP_SERVER)  + " --comparative_path_y_test " + str(EPOCHS)+"_1_"+ str(N_CLIENTS)+"_"+str(N_CLIENTS)+"_combined_offset_misalignment.csv__client_centralizedy_test.pkl" + " --comparative_path_X_test " +  str(EPOCHS)+"_1_"+ str(N_CLIENTS)+"_"+str(N_CLIENTS)+"_combined_offset_misalignment.csv__client_centralizedX_test.pkl" + " > out_server_"+str(N_CLIENTS)+".txt3"

print(name)

print("")

for i in range(N_CLIENTS):
    print()
    
    print("CLIENT SENTENCE: "+ str(i))
    print("ALLINONE")


    #python3 fl_testbed/version2/client/centralized.py -ml 1"  + " -cn " + str(i) + " -cm " + str(N_CLIENTS) + " -e " + str(EPOCHS) +  " -dfn   'combined_offset_misalignment.csv' " + " -ip " + "'172.19.0." + str(IP_CLIENT_INI +i)+ "'" +" > out_client_"+str(i)+".txt1"  + " && " +
    client_name= "python3 fl_testbed/version2/client/independent.py  -ml 1 "  + " -e " + str(EPOCHS) + " -cm " + str(N_CLIENTS) + " -cn " + str(i) +   "  -dfn  DATASET_"+ str(i)+".csv" + " -ip " + "'172.19.0." + str(IP_CLIENT_INI +i)+ "'" + " --comparative_path_y_test " + str(EPOCHS)+"_1_"+ str(N_CLIENTS)+"_"+str(i)+"_combined_offset_misalignment.csv__client_centralizedy_test.pkl" + " --comparative_path_X_test " +  str(EPOCHS)+"_1_"+ str(N_CLIENTS)+"_"+str(i)+"_combined_offset_misalignment.csv__client_centralizedX_test.pkl " +" > out_client_"+str(i)+".txt2" + " && " + "python3 fl_testbed/version2/client/federated_client.py  -ml 1 "  + " -e " + str(EPOCHS_FL) + " -cm " + str(N_CLIENTS) + " -cn " + str(i) +   " -dfn  'DATASET_"+ str(i)+".csv'"  + " -ip " + "'172.19.0." + str(IP_CLIENT_INI + i) + "'" + " > out_client_"+str(i)+".txt3" 
    
    print(client_name)





