## FL_AM_Defect-Detection
Paper for FL AM 


<img align="center" src="https://federated.withgoogle.com/assets/comic/panel046.png">
</img>

Federated Learning code is contained inside the fl_testbed folder. The file structure follows the shown schema.

<pre>
📦fl_testbed
 ┣ 📂version2
 ┃ ┣ 📂client
 ┃ ┃ ┣ 📜CustomNumpyClient.py
 ┃ ┃ ┣ 📜centralized.py
 ┃ ┃ ┣ 📜datasplit.py
 ┃ ┃ ┣ 📜federated_client.py
 ┃ ┃ ┗ 📜independent.py
 ┃ ┣ 📂data
 ┃ ┃ ┣ 📂initial
 ┃ ┃ ┃ ┗ 📜combined_offset_misalignment.csv
 ┃ ┃ ┗ 📂transformed
 ┃ ┃ ┃ ┣ 📜1_0.001_1_4_0_DATASET_0.csv__client_independentX_test.pkl
 ┃ ┃ ┃ ┣ 📜1_0.001_1_4_0_DATASET_0.csv__client_independentX_train.pkl
 ┃ ┃ ┃ ┣ 📜1_0.001_1_4_0_DATASET_0.csv__client_independenty_test.pkl
 ┃ ┃ ┃ ┣ 📜1_0.001_1_4_0_DATASET_0.csv__client_independenty_train.pkl
 ┃ ┃ ┃ ┣ 📜DATASET_0.csv
 ┃ ┃ ┃ ┣ 📜DATASET_0.png
 ┃ ┃ ┃ ┗ 📜TOTAL_DATASET.png
 ┃ ┣ 📂server
 ┃ ┃ ┣ 📜CustomStrategy.py
 ┃ ┃ ┣ 📜NOTUSED.txt
 ┃ ┃ ┣ 📜aggregate.py
 ┃ ┃ ┣ 📜federated_server.py
 ┃ ┃ ┗ 📜strategy.py
 ┗ 📜README.md
</pre>


<h2>File Description:</h2>


<pre>
--->📂client<---
📜CustomNumpyClient.py: Inhered class with hadny function for federated client.

📜centralized.py: Run a deep learning model using a complete dataset.

📜datasplit.py: Builts in two operation modes for data generation 
(Dirichlet Distribution and manual mode [prefered mode]).

📜federated_client.py: Script for running the federated client.

📜independent.py: This script triggers a deep learning model on a small section
of the whole dataset.



--->📂data<---
📂initial: Folder that contains initial datasets.
📂transformed: Scripts generated files and data miscellanous data.


--->📂server<---
📜CustomStrategy.py: Custom FedAvg strategy implementation with built-in testing.
📜NOTUSED.txt: Not in use.
📜aggregate.py: Required file for federated server script.
📜federated_server.py: Script for running the federated server.
📜strategy.py: Abstract base class for server strategy.



</pre>





<h2>Execution Order:</h2>

Please, run the commands (script files) in this specific order for each specific ip:

1. datasplit.py (Clients)
2. centralized.py (Clients and server)
3. independent.py (Clients)
4. federated_server.py (Server)
5. federated_client.py (Clients)



<h2>Server access and pre-execution steps</h2>


Credentials

Vpn: development

<pre>
<h3>Server 96</h3>
User: beto
Password: ilab301
login: ssh beto@10.147.17.96
Please run: cd FL_AM_Defect-Detection && rm fl_testbed/version2/data/transformed/*

<h3>Client 104</h3>
User: ilab
Password: ilab301
login: ssh ilab@10.147.17.104
Please run: cd FL_AM_Defect-Detection && rm fl_testbed/version2/data/transformed/*

<h3>Client 111</h3>
User: ilab
Password: ilab301
login: ssh ilab@10.147.17.111
Please run: conda deactivate && conda activate tf && cd FL_AM_Defect-Detection && rm fl_testbed/version2/data/transformed/*

<h3>Client 234</h3>
User: ilabutk
Password: ilab301
login: ssh ilabutk@10.147.17.234
Please run: cd FL_AM_Defect-Detection && rm fl_testbed/version2/data/transformed/* 

<h3>Client 150</h3>
User: jose
Password: jatsOnTesla!
login: ssh jose@10.147.17.150
Please run: cd FL_AM_Defect-Detection && rm fl_testbed/version2/data/transformed/*



<h3>Important:<h3>

It is required for all clients to create 2 directories and place the follwoing files in it. After changing directory to "FL_AM_Defect-Detection" . Please, run:

"mkdir fl_testbed/version2/data/initial"
"mkdir fl_testbed/version2/data/transformed"

Paste both files under the initial folder:

combined_angular_misalignment_with_RUL.csv: https://drive.google.com/file/d/12Lvz0f56et1_-VXhgSEDkAU2xAUwCvIO/view?usp=sharing

combined_offset_misalignment.csv: https://drive.google.com/file/d/1-E5wqPmhtIlsde04fT2WDtzNXx-nufZa/view?usp=sharing


</pre>


<h2>Commands Explanation:</h2>

<pre>



-ml: Type of model executed.
-lr: Learning rate. :: It is important to fine tune this parameter as it may lead to overfitting
-e: Number of epochs. :: It is important to fine tune this parameter as it may lead to overfitting (EarlyStopping triggered).
-cm: Max number of clients.
-cn: Client number.
-dfn: Initial dataframe.
-ip: CLient/server ip.
--comparative_path_y_test: Initial dataset splitted train/test saved as pickle y_test
--comparative_path_X_test: Initial dataset splitted train/test saved as pickle X_test
--rounds: Number of federated rounds.
-l: Manual weights for data_split.py script.
-fq: fraction of sampled/residual dataset.


<!-- 
</pre>



<h2>Case 1: combined_offset_misalignment </h2>

Test case: "2 Clients and 1 Server"

<pre>


<h2>📜datasplit.py</h2>

<h3>Client 96</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/datasplit.py -ml 2 -cm 2 -dfn combined_offset_misalignment.csv -ip 10.147.17.104 -l  200 10 30 8 200 12 40 150 90 12  -fq 0.6  1

Then manually transfer to all involved clients

<h2>📜centralized.py</h2>
<h3>Client 104</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 0 -cm 2 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '10.147.17.104'

python3 fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 0 -cm 2 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '172.19.0.4'


<h3>Client 111</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 1 -cm 2 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '172.19.0.3'

python3 fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 1 -cm 2 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '172.19.0.3'


<h3>Server 96</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 2 -cm 2 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '10.147.17.96'

python3 fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 2 -cm 2 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '172.19.0.2'

<!-- python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/optimizer.py -->




<h2>📜independent.py</h2>
<h3>Client 104</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/independent.py  -ml 1 -lr 0.001 -e 10 -cm 2 -cn 0  -dfn DATASET_0.csv -ip 172.19.0.4 --comparative_path_y_test 10_0.001_1_2_0_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_2_0_combined_offset_misalignment.csv__client_centralizedX_test.pkl

python3 fl_testbed/version2/client/independent.py  -ml 1 -lr 0.001 -e 10 -cm 2 -cn 0  -dfn DATASET_0.csv -ip 172.19.0.4 --comparative_path_y_test 10_0.001_1_2_0_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_2_0_combined_offset_misalignment.csv__client_centralizedX_test.pkl


<h3>Client 111</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/independent.py  -ml 1 -lr 0.001 -e 10 -cm 2 -cn 1  -dfn DATASET_1.csv -ip 172.19.0.4 --comparative_path_y_test 10_0.001_1_2_1_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_2_1_combined_offset_misalignment.csv__client_centralizedX_test.pkl

python3 fl_testbed/version2/client/independent.py  -ml 1 -lr 0.001 -e 10 -cm 2 -cn 1  -dfn DATASET_1.csv -ip 172.19.0.3 --comparative_path_y_test 10_0.001_1_2_1_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_2_1_combined_offset_misalignment.csv__client_centralizedX_test.pkl




<h2>📜federated_server.py</h2>
<h3>Server 96</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/server/federated_server.py  -ml 1 -lr 0.001 -e 1 -cm 2  --rounds 100  -ip 10.147.17.96 --comparative_path_y_test 10_0.001_1_2_2_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_2_2_combined_offset_misalignment.csv__client_centralizedX_test.pkl


python3 fl_testbed/version2/server/federated_server.py  -ml 1 -lr 0.001 -e 1 -cm 2  --rounds 100  -ip 172.19.0.2 --comparative_path_y_test 10_0.001_1_2_2_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_2_2_combined_offset_misalignment.csv__client_centralizedX_test.pkl





<h2>📜federated_client.py</h2>
<h3>Client 104</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/federated_client.py  -ml 1 -lr 0.001 -e 1 -cm 2 -cn 0  -dfn DATASET_0.csv -ip 10.147.17.104

python3 fl_testbed/version2/client/federated_client.py  -ml 1 -lr 0.001 -e 1 -cm 2 -cn 0  -dfn DATASET_0.csv -ip 172.19.0.4




<h3>Client 111</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/federated_client.py  -ml 1 -lr 0.001 -e 1 -cm 2 -cn 1  -dfn DATASET_1.csv -ip 10.147.17.111

python3 fl_testbed/version2/client/federated_client.py  -ml 1 -lr 0.001 -e 1 -cm 2 -cn 1  -dfn DATASET_1.csv -ip 172.19.0.3

</pre>



#########CHECK HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
<h2>Case 2: combined_offset_misalignment </h2>


Test case: "4 Clients and 1 Server"


<pre>


<h2>📜datasplit.py</h2>
<h3>Client 96</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/datasplit.py -ml 2 -cm 2 -dfn combined_offset_misalignment.csv -ip 172.19.0.5 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1


python3 fl_testbed/version2/client/datasplit.py -ml 2 -cm 2 -dfn combined_offset_misalignment.csv -ip  172.19.0.5 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1


<!-- <h3>Client 111</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/datasplit.py -ml 1 -cm 4 -cn 1 -dfn combined_offset_misalignment.csv -ip 10.147.17.111 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1
<h3>Client 234</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/datasplit.py -ml 1 -cm 4 -cn 2 -dfn combined_offset_misalignment.csv -ip 10.147.17.234 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1
<h3>Client 150</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/datasplit.py -ml 1 -cm 4 -cn 3 -dfn combined_offset_misalignment.csv -ip 10.147.17.234 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1
 -->

#CONTINUE HERE!!!

<h2>📜centralized.py</h2>
<h3>Client 104</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 0 -cm 4 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '10.147.17.104'

python3 fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 0 -cm 4 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '172.19.0.9'

<h3>Client 111</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 1 -cm 4 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '10.147.17.111'

python3 fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 1 -cm 4 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '172.19.0.8'


<h3>Client 234</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 2 -cm 4 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '172.19.0.'

python3 fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 2 -cm 4 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '172.19.0.7'

<h3>Client 150</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 3 -cm 4 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '172.19.0.'

python3 fl_testbed/version2/client/centralized.py -ml 1 -lr 0.001 -cn 3 -cm 4 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '172.19.0.6'

<h3>Server 96</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py  -ml 1 -lr 0.001 -cn 4 -cm 4 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '172.19.0.'

python3 fl_testbed/version2/client/centralized.py  -ml 1 -lr 0.001 -cn 4 -cm 4 -e 10 -dfn 'combined_offset_misalignment.csv' -ip '172.19.0.5'



<h2>📜independent.py</h2>
<h3>Client 104</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/independent.py  -ml 1 -lr 0.001 -e 10 -cm 4 -cn 0  -dfn DATASET_0.csv -ip 10.147.17.104 --comparative_path_y_test 10_0.001_1_4_0_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_4_0_combined_offset_misalignment.csv__client_centralizedX_test.pkl


python3 fl_testbed/version2/client/independent.py  -ml 1 -lr 0.001 -e 10 -cm 4 -cn 0  -dfn DATASET_0.csv -ip 172.19.0.9 --comparative_path_y_test 10_0.001_1_4_0_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_4_0_combined_offset_misalignment.csv__client_centralizedX_test.pkl


<h3>Client 111</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/independent.py  -ml 1 -lr 0.001 -e 10 -cm 4 -cn 1  -dfn DATASET_1.csv -ip 172.19.0.8 --comparative_path_y_test 10_0.001_1_4_1_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_4_1_combined_offset_misalignment.csv__client_centralizedX_test.pkl

python3 fl_testbed/version2/client/independent.py  -ml 1 -lr 0.001 -e 10 -cm 4 -cn 1  -dfn DATASET_1.csv -ip 172.19.0.8 --comparative_path_y_test 10_0.001_1_4_1_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_4_1_combined_offset_misalignment.csv__client_centralizedX_test.pkl

<h3>Client 234</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/independent.py  -ml 1 -lr 0.001 -e 10 -cm 4 -cn 2  -dfn DATASET_2.csv -ip 10.147.17.234 --comparative_path_y_test 10_0.001_1_4_2_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_4_2_combined_offset_misalignment.csv__client_centralizedX_test.pkl

python3 fl_testbed/version2/client/independent.py  -ml 1 -lr 0.001 -e 10 -cm 4 -cn 2  -dfn DATASET_2.csv -ip 172.19.0.7 --comparative_path_y_test 10_0.001_1_4_2_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_4_2_combined_offset_misalignment.csv__client_centralizedX_test.pkl

<h3>Client 150</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/independent.py  -ml 1 -lr 0.001 -e 10 -cm 4 -cn 3  -dfn DATASET_3.csv -ip 10.147.17.150 --comparative_path_y_test 10_0.001_1_4_3_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_4_3_combined_offset_misalignment.csv__client_centralizedX_test.pkl

python3 fl_testbed/version2/client/independent.py  -ml 1 -lr 0.001 -e 10 -cm 4 -cn 3  -dfn DATASET_3.csv -ip 172.19.0.6 --comparative_path_y_test 10_0.001_1_4_3_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_4_3_combined_offset_misalignment.csv__client_centralizedX_test.pkl



<h2>📜federated_server.py</h2>
<h3>Server 96</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/server/federated_server.py  -ml 1 -lr 0.001 -e 10 -cm 4  --rounds 100  -ip 10.147.17.96 --comparative_path_y_test 10_0.001_1_4_4_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_4_4_combined_offset_misalignment.csv__client_centralizedX_test.pkl


python3 fl_testbed/version2/server/federated_server.py  -ml 1 -lr 0.001 -e 10 -cm 4  --rounds 100  -ip '172.19.0.5' --comparative_path_y_test 10_0.001_1_4_4_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_1_4_4_combined_offset_misalignment.csv__client_centralizedX_test.pkl > out_server_4.txt




<h2>📜federated_client.py</h2>
<h3>Client 104</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/federated_client.py  -ml 1 -lr 0.001 -e 1 -cm 4 -cn 0  -dfn DATASET_0.csv -ip 10.147.17.104

python3 fl_testbed/version2/client/federated_client.py  -ml 1 -lr 0.001 -e 1 -cm 4 -cn 0  -dfn DATASET_0.csv -ip '172.19.0.9' > out_server_4_9.txt

<h3>Client 111</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/federated_client.py  -ml 1 -lr 0.001 -e 1 -cm 4 -cn 1  -dfn DATASET_1.csv -ip 10.147.17.111

python3 fl_testbed/version2/client/federated_client.py  -ml 1 -lr 0.001 -e 1 -cm 4 -cn 1  -dfn DATASET_1.csv -ip '172.19.0.8' > out_server_4_8.txt

<h3>Client 234</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/federated_client.py  -ml 1 -lr 0.001 -e 1 -cm 4 -cn 2  -dfn DATASET_2.csv -ip 10.147.17.234

python3 fl_testbed/version2/client/federated_client.py  -ml 1 -lr 0.001 -e 1 -cm 4 -cn 2  -dfn DATASET_2.csv -ip '172.19.0.7' > out_server_4_7.txt

<h3>Client 234</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/federated_client.py  -ml 1 -lr 0.001 -e 1 -cm 4 -cn 3  -dfn DATASET_3.csv -ip 10.147.17.234

python3 fl_testbed/version2/client/federated_client.py  -ml 1 -lr 0.001 -e 1 -cm 4 -cn 3  -dfn DATASET_3.csv -ip '172.19.0.6' > out_server_4_6.txt


</pre>


 -->














<!-- 
<!--  -->








DOCKER NVIDIA

INSTALLATION DOCKER POWERED CONTAINERS
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

<!-- 







Docker supports a keyboard combination to gracefully detach from a container. Press Ctrl-P, followed by Ctrl-Q, to detach from your connection.

sudo docker attach 6ea25fed95eb  

<!--

https://4sysops.com/archives/macvlan-network-driver-assign-mac-address-to-docker-containers/


https://stackoverflow.com/questions/27273412/cannot-install-packages-inside-docker-ubuntu-image

sudo docker network create \
        --driver macvlan \
        --subnet 192.168.1.0/24 \
        --gateway 192.168.1.1 \
        --ip-range 192.168.1.80/28 \
        --aux-address 'host=192.168.1.80' \
        --opt parent=en1 \
        macvlan_net
  -->

sudo firewall-cmd --add-port=60000-61000/tcp 

sudo firewall-cmd --add-port=60000-61000/udp 

1.Dockerfile
docker build Dockerfile
docker build -t testbed:1.0 .
sudo chmod 775 FL_AM_Defect-Detection
 sudo docker stop * || sudo docker rm *



:8025:60001

-u $(id -u):$(id -g)

#SERVER
sudo docker run  --network skynet -d -it -v /home/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z   testbed:1.0 bin/bash 

#CLIENTS
sudo docker run  --network skynet -d -it -v /home/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash 
sudo docker run  --network skynet -d -it -v /home/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z   testbed:1.0 bin/bash



///IF WANT TO CREATE NEW EXPERIMENT!!
sudo docker run  --privileged --network skynet -d -it -v /home/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash  &&
sudo docker run --privileged  --network skynet -d -it -v /home/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&
sudo docker run --privileged --network skynet -d -it -v /home/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash  &&
sudo docker run  --privileged --network skynet -d -it -v /home/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash  &&

sudo docker run  --privileged --network skynet -d -it -v /home/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash && 
sudo docker run  --privileged --network skynet -d -it -v /home/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash 





e9f73e85f3bfcbe01ed9a14e69c4f68a459a5e8490c1195e86b2e95b9b1c1cf8 24
a9b625959270a681519b9baa3ebf6ebfa346a97b61b20240358d57450b3e9558 25
61fafa7d3fcf98f607118c8a0583190b76361ff4e733391516a0c8de0a2337c4 26
a86c3db8303c271119c10235ab6f55449e8ea27285cd291dab7820ca3171a0c3 27
b0f9b99938af297c9b52f8d584380f4b47f7f965912d447c27453393d02315f0 28
c3f02fc7609fb718c86709bce080cec10319ccc57862b529a5c618c591103df9 29





DOCKER SWARM




warm initialized: current node (ffzwxmzpya9sefjrqikp2gkjy) is now a manager.

To add a worker to this swarm, run the following command:

    docker swarm join --token SWMTKN-1-0oyrlf4qp9304ta7zc4ly5x4s0pp1sxohkynhbwp89o94q3ev1-5bzm7qa02is2131707wysdk6e 10.147.17.150:2377

To add a manager to this swarm, run 'docker swarm join-token manager' and follow the instructions.
 -->







 docker network create \
  --driver overlay \
  --subnet 10.0.9.0/24 \
  --gateway 10.0.9.99 \
  skynet2


<!-- --replicas-max-per-node=3 ORINGAIL -->

<!-- sudo docker run  --privileged --network skynet -d -it -v /home/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash  -->
docker service create --name federated --network skynet2 --replicas 1 testbed:1.0 


 --mount type=bind,src=/mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection,dst=/FL_AM_Defect-Detection:z    testbed:1.0 

bin/bash
 -d --mount type=bind,source=/mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection,target=/FL_AM_Defect-Detection:z  


https://stackoverflow.com/questions/68198392/docker-service-verify-detected-task-failure

docker service scale federated=5


/home/jose 10.147.17.0/24(rw,sync,no_root_squash)

umount /mnt/nfs/var/nfs_share_dir

docker service ps federated
docker service rm federated

g98vwpdng1kb2r8bs42o3ceq7



\NFS SHARING!


https://www.golinuxcloud.com/share-folder-with-nfs-ubuntu/

https://dev.to/prajwalmithun/setup-nfs-server-client-in-linux-and-unix-27id


FINAL RESULTS

10.147.17.150:/home/jose  837G  818G   19G  98% /mnt/nfs/var/nfs_share_dir



https://stackoverflow.com/questions/23935141/how-to-copy-docker-images-from-one-host-to-another-without-using-a-repository



<!-- 





###########NEW APPRAOACH#############

CREATE DOCKER IMAGE FROM DOCKER FILE NO GPU OFR MATI SERVER

docker network create \
skynet3

docker run  --privileged --name server0 --network skynet3 -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&
docker run  --privileged --name client1 --network skynet3 -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&
docker run  --privileged --name client2 --network skynet3 -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&
docker run  --privileged --name client3 --network skynet3 -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&
docker run  --privileged --name client4 --network skynet3 -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&
docker run  --privileged --name client5 --network skynet3 -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash 



sudo docker exec -it  <name>   bash


server  172.19.0.2
client1 172.19.0.3
client2 172.19.0.4
client3 172.19.0.5
client4 172.19.0.6 
client5 172.19.0.7



MATI
server0 172.18.0.2
client1 172.18.0.3
client2 172.18.0.4
client3 172.18.0.5
client4 172.18.0.6 
client5 172.18.0.7



server 0
python3 fl_testbed/version2/server/federated_server_RUL.py   -cm 5 -e 1 --rounds 4000 -ip  172.18.0.2  -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15
_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl' &> out_server_16_RUL_M3_fed_new.txt3




CLIENT 1     ok       
python3 fl_testbed/version2/client/federated_client_RUL.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.3 &> out_server_14_M3_4_0_ddf_LSTM_fed_new.txt2 




CLIENT 2     ok
python3 fl_testbed/version2/client/federated_client_RUL.py    -cn 2 -cm 5 -e 1 -dfn   'M3_4_2_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.4 &> out_server_14_M3_4_2_ddf_LSTM_fed_new.txt2 


CLIENT 3 ok
python3 fl_testbed/version2/client/federated_client_RUL.py   -cn 3 -cm 5 -e 1 -dfn   'M3_4_3_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.5 &> out_server_14_M3_4_3_ddf_LSTM_fed_new.txt2 


CLIENT 4 ok
python3 fl_testbed/version2/client/federated_client_RUL.py   -cn 4 -cm 5 -e 1 -dfn   'M3_4_4_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.6 &> out_server_14_M3_4_4_ddf_LSTM_fed_new.txt2 



CLIENT 74.  ok
python3 fl_testbed/version2/client/federated_client_RUL.py   -cn 1 -cm 15 -e 1 -dfn   'M3_4_1_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.7 &> out_server_14_M3_4_1_ddf_LSTM_fed_new.txt2 

 -->

####################
















!!!!!!!!NEW OF NEWEST VERSION!!!!!!!!


1.Requirement:
have nfs location up
have docker and image built
sudo su -
https://zerotier.atlassian.net/wiki/spaces/SD/pages/7536656/Running+ZeroTier+in+a+Docker+Container#:~:text=ZeroTier%20One%20makes%20ZeroTier%20virtual,dev%2Fnet%2Ftun%20device.



docker exec -it     bash


curl https://install.zerotier.com/ | bash 
/usr/sbin/zerotier-one -d 
/usr/sbin/zerotier-cli join c7c8172af153068f




Credentials:



tesla
ssh jose@tesla.ise.utk.edu
jatsOnTesla!

docker run --runtime=nvidia --gpus all --privileged  --name server --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash



docker run --runtime=nvidia --gpus all --privileged  --name client1 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&


docker run --runtime=nvidia --gpus all --privileged  --name client2 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&


docker run --runtime=nvidia --gpus all --privileged  --name client3 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&

docker run --runtime=nvidia --gpus all --privileged  --name client4 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&

docker run --runtime=nvidia --gpus all --privileged  --name client5 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash









docker run --runtime=nvidia --gpus all --privileged  --name server_2 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&



docker run --runtime=nvidia --gpus all --privileged  --name client1_2 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&


docker run --runtime=nvidia --gpus all --privileged  --name client2_2 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&


docker run --runtime=nvidia --gpus all --privileged  --name client3_2 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&

docker run --runtime=nvidia --gpus all --privileged  --name client4_2 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&

docker run --runtime=nvidia --gpus all --privileged  --name client5_2 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash



<!-- docker run --runtime=nvidia --gpus all --privileged  --name server_FedAvgM --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash

docker run  --runtime=nvidia --gpus all --privileged  --name server_FedOpt --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash


docker run --runtime=nvidia --gpus all  --privileged  --name server_QFedAvg --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash -->





mati
ssh josets@mati.ise.utk.edu
jts2023Pwd

docker run  --privileged --name client_1 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun  -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash

docker run  --privileged  --name client_1_FedAvgM --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&

docker run  --privileged  --name client_1_FedOpt --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&


docker run  --privileged  --name client_1_QFedAvg --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash



<!-- 
Using Ray:

docker run --shm-size=15gb --privileged  --name simu --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash


unsetopt nomatch #IF DIRECTLY UNDER TYHE TeRMINAL!
pip install -U flwr["simulation"]
pip install ray==1.11.1


export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.2

SomeCommand 2>&1 | tee simu_out.txt


windows
ssh ilab@smartshots.ise.utk.edu
ilab/ilab301


docker run  --privileged  --name client_2 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash  -->





Server4:
hostname: ssh jose@sadie.ise.utk.edu
user:   jose
password: jats2022Mushroom

docker run  --privileged  --name client_2 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /Volumes/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash

docker run  --privileged  --name client_2_FedAvgM --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&

docker run  --privileged  --name client_2_FedOpt --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&


docker run  --privileged  --name client_2_QFedAvg --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash






<!-- --mount type=bind,src=C:\web\dev,target=C:\web\dev my-docker

 -->
client:
hostname: ssh ilab@ilab2019wifi.ise.utk.edu
user: ilab
password: ilab301

docker run  --privileged  --name client_3 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /Volumes/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash


docker run  --privileged  --name client_3_FedAvgM --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&

docker run  --privileged  --name client_3_FedOpt --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash &&


docker run  --privileged  --name client_3_QFedAvg --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash





Server 2:
hostname: ssh ilabutk@ilabimac2020.ise.utk.edu
user: ilabutk
password: ilab301

docker run  --privileged  --name client_4 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /Volumes/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash

docker run  --privileged  --name client_4_FedAvgM --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash

docker run  --privileged  --name client_4_FedOpt --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash


docker run  --privileged  --name client_4_QFedAvg --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash




Server3:
hostname: ssh ilab@ilabimacpro.nomad.utk.edu /10.147.17.111
user: ilab
password: ilab301

docker run  --privileged  --name client_5 --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /Volumes/jose/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash

docker run  --privileged  --name client_5_FedAvgM --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash

docker run  --privileged  --name client_5_FedOpt --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash


docker run  --privileged  --name client_5_QFedAvg --cap-add=NET_ADMIN --cap-add=SYS_ADMIN --device=/dev/net/tun -d -it -v /mnt/nfs/var/nfs_share_dir/FL_AM_Defect-Detection:/FL_AM_Defect-Detection:z  testbed:1.0 bin/bash






running commands

server 0 ok

#FIRST RUN CENTRALIZED:

python3 fl_testbed/version2/client/centralized.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'combined_offset_misalignment_M3.csv'  -ip 172.19.0.5 2>&1 | tee out_server_14_OFFM3.txt2 


RUN IT AND RE RUN IT WITH 20,40 and 80 oFFSET NOW SEQUENCE LENGHT IS SET TO 80:

python3 fl_testbed/version2/client/centralized_new.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'combined_offset_misalignment_M3.csv'  -ip 172.19.0.5 --JUMPING_STEP 20 2>&1 | tee out_server_14_RULM3_SEQ20.txt2 

python3 fl_testbed/version2/client/centralized_new.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'combined_offset_misalignment_M3.csv'  -ip 172.19.0.5 --JUMPING_STEP 40 2>&1 | tee out_server_14_RULM3_SEQ40.txt2 


python3 fl_testbed/version2/client/centralized_new.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'combined_offset_misalignment_M3.csv'  -ip 172.19.0.5 --JUMPING_STEP 80 2>&1 | tee out_server_14_RULM3_SEQ80.txt2



#TO RUN DATA SPLIT

python3 fl_testbed/version2/client/datasplit.py -data_X_train 30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_train.pkl -data_X_vals 30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_vals.pkl -data_y_train 30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_train.pkl -data_y_vals 30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_vals.pkl -cm 4 -l 90 10 10 10 10     10 90 10 10 10    10 10 90 10 10      10 10 10 90 10    10 10 10 10 90 -fq  0.2 0.25 0.3333333 0.5 1  -motor 3 -type MLP 2>&1 | tee DATASPLIT_TYPE1_MLP_M3.txt 



python3 fl_testbed/version2/client/datasplit.py -data_X_train 30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_train.pkl -data_X_vals 30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_vals.pkl -data_y_train 30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_train.pkl -data_y_vals 30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_vals.pkl -cm 4 -l 50 50 50 50 50     50 50 50 50 50    50 50 50 50 50      50 50 50 50 50   50 50 50 50 50 -fq  0.2 0.25 0.3333333 0.5 1  -motor 3 -type MLP 2>&1 | tee DATASPLIT_TYPE2_MLP_M3.txt 



<!-- python3 fl_testbed/version2/client/datasplit.py -data_X_train 30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_train.pkl -data_X_vals 30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_vals.pkl -data_y_train 30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_train.pkl -data_y_vals 30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_vals.pkl -cm 4 -l 200 10 10 10 10     10 200 10 10 10    10 10 200 10 10      10 10 10 200 10    10 10 10 10 200 -fq  0.2 0.25 0.3333333 0.5 1  -motor 3 -type MLP 2>&1 | tee DATASPLIT_TYPE3_MLP_M3.txt 
 -->










OK
python3 fl_testbed/version2/client/datasplit.py -data_X_train 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_inputs.pkl -data_X_vals 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_inputs.pkl -data_y_train 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_out.pkl -data_y_vals 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_out.pkl -cm 4  -l 90 10 10 10 10     10 90 10 10 10    10 10 90 10 10      10 10 10 90 10    10 10 10 10 90 -fq  0.2 0.25 0.3333333 0.5 1  -motor 3 -type LSTM 2>&1 | tee DATASPLIT_TYPE1_LSTM_M3_SEQ80.txt

python3 fl_testbed/version2/client/datasplit.py -data_X_train 30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_inputs.pkl -data_X_vals 30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_inputs.pkl -data_y_train 30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_out.pkl -data_y_vals 30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_out.pkl -cm 4  -l 50 50 50 50 50     50 50 50 50 50    50 50 50 50 50      50 50 50 50 50   50 50 50 50 50 -fq  0.2 0.25 0.3333333 0.5 1  -motor 3 -type LSTM 2>&1 | tee DATASPLIT_TYPE2_LSTM_M3.txt


#INDEPENDENT




python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_0_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.19.0.5 2>&1 | tee out_server_M3_4_0_OFFSETM3_idp.txt4 &&

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_1_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.19.0.5 2>&1 | tee out_server_M3_4_1_OFFSETM3_idp.txt4 &&

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_2_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.19.0.5 2>&1 | tee out_server_M3_4_2_OFFSETM3_idp.txt4 &&

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_3_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.19.0.5 2>&1 | tee out_server_M3_4_3_OFFSETM3_idp.txt4

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_3_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.19.0.5 2>&1 | tee out_server_M3_4_4_OFFSETM3_idp.txt4






python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.19.0.5 2>&1 | tee out_server_14_M3_4_0_ddf_LSTM_idp_SEQ80.txt2  &&

python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_1_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.19.0.5 2>&1 | tee out_server_14_M3_4_1_ddf_LSTM_idp_SEQ80.txt2  &&

python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_2_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.19.0.5 2>&1 | tee out_server_14_M3_4_2_ddf_LSTM_idp_SEQ80.txt2 &&

python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_3_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.19.0.5 2>&1 | tee out_server_14_M3_4_3_ddf_LSTM_idp_SEQ80.txt2 &&	

python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_4_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.19.0.5 2>&1 | tee out_server_14_M3_4_4_ddf_LSTM_idp_SEQ80.txt2 








#FEDERATED ALL UNDER _FedAvg

python3 fl_testbed/version2/server/federated_server_RUL_FedAvg.py   -cm 5 -e 1 --rounds 4000 -ip  10.144.98.109  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15
_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_FedAvg_SEQ80.txt



python3 fl_testbed/version2/server/federated_server_OFFSET_FedAvg.py   -cm 5 -e 1 --rounds 4000 -ip  10.144.98.109  -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvg.txt




CLIENT 1     ok       
python3 fl_testbed/version2/client/federated_client_RUL_FedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.3 2>&1 | tee LSTM_MATTI_FedAvg_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_MATTI_FedAvg.txt




CLIENT 2     ok
python3 fl_testbed/version2/client/federated_client_RUL.py    -cn 2 -cm 5 -e 1 -dfn   'M3_4_2_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.4   2>&1 | tee LSTM_SADIE_FedAvg_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET.py   -cn 2 -cm 5 -e 1 -dfn   'M3_4_2_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_SADIE_FedAvg.txt





CLIENT 3 ok
python3 fl_testbed/version2/client/federated_client_RUL.py   -cn 3 -cm 5 -e 1 -dfn   'M3_4_3_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'  -ip 172.18.0.5  2>&1 | tee LSTM_2019_FedAvg_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET.py   -cn 3 -cm 5 -e 1 -dfn   'M3_4_3_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_2019_FedAvg.txt






CLIENT 4 ok
python3 fl_testbed/version2/client/federated_client_RUL.py   -cn 4 -cm 5 -e 1 -dfn   'M3_4_4_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.6 &> out_server_14_M3_4_4_ddf_LSTM_fed_new.txt2 2>&1 | tee LSTM_2020_FedAvg_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET.py   -cn 4 -cm 5 -e 1 -dfn   'M3_4_4_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_2020_FedAvg.txt






CLIENT 5  ok
python3 fl_testbed/version2/client/federated_client_RUL.py   -cn 1 -cm 15 -e 1 -dfn   'M3_4_1_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.7 &> out_server_14_M3_4_1_ddf_LSTM_fed_new.txt2 2>&1 | tee LSTM_PRO_FedAvg_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET.py   -cn 1 -cm 5 -e 1 -dfn   'M3_4_1_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_PRO_FedAvg.txt












#FEDERATED ALL UNDER _FedAvgM




python3 fl_testbed/version2/server/federated_server_RUL_FedAvgM.py   -cm 5 -e 1 --rounds 4000 -ip  10.144.98.109  -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15
_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_FedAvgM_SEQ80.txt



python3 fl_testbed/version2/server/federated_server_OFFSET_FedAvgM.py   -cm 5 -e 1 --rounds 4000 -ip  10.144.98.109  -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvgM.txt




CLIENT 1     ok       
python3 fl_testbed/version2/client/federated_client_RUL_FedAvgM.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.3 2>&1 | tee LSTM_MATTI_FedAvgM_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvgM.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_MATTI_FedAvgM.txt




CLIENT 2     ok
python3 fl_testbed/version2/client/federated_client_RUL_FedAvgM.py    -cn 2 -cm 5 -e 1 -dfn   'M3_4_2_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.4 &> out_server_14_M3_4_2_ddf_LSTM_fed_new.txt2  2>&1 | tee LSTM_SADIE_FedAvgM_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvgM.py   -cn 2 -cm 5 -e 1 -dfn   'M3_4_2_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_SADIE_FedAvgM.txt





CLIENT 3 ok
python3 fl_testbed/version2/client/federated_client_RUL_FedAvgM.py   -cn 3 -cm 5 -e 1 -dfn   'M3_4_3_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.5 &> out_server_14_M3_4_3_ddf_LSTM_fed_new.txt2 2>&1 | tee LSTM_2019_FedAvgM_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvgM.py   -cn 3 -cm 5 -e 1 -dfn   'M3_4_3_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_2019_FedAvgM.txt






CLIENT 4 ok
python3 fl_testbed/version2/client/federated_client_RUL_FedAvgM.py   -cn 4 -cm 5 -e 1 -dfn   'M3_4_4_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.6 &> out_server_14_M3_4_4_ddf_LSTM_fed_new.txt2 2>&1 | tee LSTM_2020_FedAvgM_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvgM.py   -cn 4 -cm 5 -e 1 -dfn   'M3_4_4_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_2020_FedAvgM.txt






CLIENT 5  ok
python3 fl_testbed/version2/client/federated_client_RUL_FedAvgM.py   -cn 1 -cm 15 -e 1 -dfn   'M3_4_1_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.7 &> out_server_14_M3_4_1_ddf_LSTM_fed_new.txt2 2>&1 | tee LSTM_PRO_FedAvgM_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvgM.py   -cn 1 -cm 5 -e 1 -dfn   'M3_4_1_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_PRO_FedAvgM.txt










#FEDERATED ALL UNDER _FedOpt




python3 fl_testbed/version2/server/federated_server_RUL_FedOpt.py   -cm 5 -e 1 --rounds 4000 -ip  10.144.98.109  -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15
_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_FedOpt_SEQ80.txt



python3 fl_testbed/version2/server/federated_server_OFFSET_FedOpt.py   -cm 5 -e 1 --rounds 4000 -ip  10.144.98.109  -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedOpt.txt




CLIENT 1     ok       
python3 fl_testbed/version2/client/federated_client_RUL_FedOpt.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.3 2>&1 | tee LSTM_MATTI_FedOpt_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_FedOpt.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_MATTI_FedOpt.txt




CLIENT 2     ok
python3 fl_testbed/version2/client/federated_client_RUL_FedOpt.py    -cn 2 -cm 5 -e 1 -dfn   'M3_4_2_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.4 &> out_server_14_M3_4_2_ddf_LSTM_fed_new.txt2  2>&1 | tee LSTM_SADIE_FedOpt_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_FedOpt.py   -cn 2 -cm 5 -e 1 -dfn   'M3_4_2_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_SADIE_FedOpt.txt





CLIENT 3 ok
python3 fl_testbed/version2/client/federated_client_RUL_FedOpt.py   -cn 3 -cm 5 -e 1 -dfn   'M3_4_3_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.5 &> out_server_14_M3_4_3_ddf_LSTM_fed_new.txt2 2>&1 | tee LSTM_2019_FedOpt_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_FedOpt.py   -cn 3 -cm 5 -e 1 -dfn   'M3_4_3_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_2019_FedOpt.txt






CLIENT 4 ok
python3 fl_testbed/version2/client/federated_client_RUL_FedOpt.py   -cn 4 -cm 5 -e 1 -dfn   'M3_4_4_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.6 &> out_server_14_M3_4_4_ddf_LSTM_fed_new.txt2 2>&1 | tee LSTM_2020_FedOpt_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_FedOpt.py   -cn 4 -cm 5 -e 1 -dfn   'M3_4_4_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_2020_FedOpt.txt






CLIENT 5  ok
python3 fl_testbed/version2/client/federated_client_RUL_FedOpt.py   -cn 1 -cm 15 -e 1 -dfn   'M3_4_1_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.7 &> out_server_14_M3_4_1_ddf_LSTM_fed_new.txt2 2>&1 | tee LSTM_PRO_FedOpt_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_FedOpt.py   -cn 1 -cm 5 -e 1 -dfn   'M3_4_1_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_PRO_FedOpt.txt





















#FEDERATED ALL UNDER _QFedAvg




python3 fl_testbed/version2/server/federated_server_RUL_QFedAvg.py   -cm 5 -e 1 --rounds 4000 -ip  10.144.98.109  -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15
_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_QFedAvg_SEQ80.txt



python3 fl_testbed/version2/server/federated_server_OFFSET_QFedAvg.py   -cm 5 -e 1 --rounds 4000 -ip  10.144.98.109  -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_QFedAvg.txt




CLIENT 1     ok       
python3 fl_testbed/version2/client/federated_client_RUL_QFedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.3 2>&1 | tee LSTM_MATTI_QFedAvg_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_QFedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_MATTI_QFedAvg.txt




CLIENT 2     ok
python3 fl_testbed/version2/client/federated_client_RUL__QFedAvg.py    -cn 2 -cm 5 -e 1 -dfn   'M3_4_2_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.4 &> out_server_14_M3_4_2_ddf_LSTM_fed_new.txt2  2>&1 | tee LSTM_SADIE_QFedAvg_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_QFedAvg.py   -cn 2 -cm 5 -e 1 -dfn   'M3_4_2_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_SADIE_QFedAvg.txt





CLIENT 3 ok
python3 fl_testbed/version2/client/federated_client_RUL_QFedAvg.py   -cn 3 -cm 5 -e 1 -dfn   'M3_4_3_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.5 &> out_server_14_M3_4_3_ddf_LSTM_fed_new.txt2 2>&1 | tee LSTM_2019_QFedAvg_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_QFedAvg.py   -cn 3 -cm 5 -e 1 -dfn   'M3_4_3_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_2019_QFedAvg.txt






CLIENT 4 ok
python3 fl_testbed/version2/client/federated_client_RUL_QFedAvg.py   -cn 4 -cm 5 -e 1 -dfn   'M3_4_4_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.6 &> out_server_14_M3_4_4_ddf_LSTM_fed_new.txt2 2>&1 | tee LSTM_2020_QFedAvg_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_QFedAvg.py   -cn 4 -cm 5 -e 1 -dfn   'M3_4_4_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_2020_QFedAvg.txt






CLIENT 5  ok
python3 fl_testbed/version2/client/federated_client_RUL_QFedAvg.py   -cn 1 -cm 15 -e 1 -dfn   'M3_4_1_ddf_LSTM.pkl' -dfn_test_x   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '30_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.7 &> out_server_14_M3_4_1_ddf_LSTM_fed_new.txt2 2>&1 | tee LSTM_PRO_QFedAvg_SEQ80.txt

python3 fl_testbed/version2/client/federated_client_OFFSET_QFedAvg.py   -cn 1 -cm 5 -e 1 -dfn   'M3_4_1_ddf_MLP.pkl' -dfn_test_x   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '30_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.3 2>&1 | tee MLP_PRO_QFedAvg.txt



<!-- 

SIMULATION 

#UNDER CENTOS 7 RUN: export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so
https://stackoverflow.com/questions/48831881/centos-7-libstdc-so-6-version-cxxabi-1-3-9-not-found

python3 fl_testbed/version2/server/federated_server_RUL_simulation.py 2>&1 | tee simupc1.txt -->


<!--




CONTAINER ID        IMAGE                         COMMAND                  CREATED             STATUS                   PORTS               NAMES
a8c7bf1c920f        testbed:0.2    172.19.0.9               "bin/bash"               29 seconds ago      Up 28 seconds                                clever_shirley
9293609c043a        testbed:0.2    172.19.0.8               "bin/bash"               29 seconds ago      Up 29 seconds                                festive_swirles
3010426e4ed6        testbed:0.2    172.19.0.7               "bin/bash"               31 seconds ago      Up 30 seconds                                ecstatic_elion
4c82cc89c90b        testbed:0.2    172.19.0.6               "bin/bash"               31 seconds ago      Up 31 seconds                                hopeful_lamarr
145094e499d7        testbed:0.2   SERVER  172.19.0.5           "bin/bash"               53 seconds ago      Up 52 seconds                                zealous_dubinsky

3cf5319aead2        testbed:0.1                   "bin/bash"               About an hour ago   Up About an hour                             zealous_agnesi
45f6e70d3a01        testbed:0.1                   "bin/bash"               About an hour ago   Up About an hour                             pedantic_noyce
6ea25fed95eb        testbed:0.1                   "bin/bash"               About an hour ago   Up 10 minutes                                elastic_saha
334a488fb852        hello-world                   "/hello"                 9 hours ago         Exited (0) 9 hours ago                       condescending_darwin
bbda03117150        jupyter/tensorflow-notebook   "tini -g -- start-..."   2 months ago        Created                                      romantic_varahamihira





     "Internal": false,
        "Attachable": false,
        "Containers": {
            "145094e499d7b85f1ef9ac624aee6afde705274c463e1de65cadfdb782060ffe": {
                "Name": "zealous_dubinsky", 145094e499d7
                "EndpointID": "77e85e36b9816018f966b5f8cd745ded5eb49bf41588e7e3a781122c3293e5cd",
                "MacAddress": "02:42:ac:13:00:05",
                "IPv4Address": "172.19.0.5/16",
                "IPv6Address": ""
            },
            "3010426e4ed65e23939221b9f2575f7295587ce7267a132077b98a83337a2926": {
                "Name": "ecstatic_elion",  3010426e4ed6
                "EndpointID": "60407d498ad90c34a8d2f19569cc0e692e8ff0b1a71b8fe5549f554f2549f32b",
                "MacAddress": "02:42:ac:13:00:07",
                "IPv4Address": "172.19.0.7/16",
                "IPv6Address": ""
            },
          
            "45f6e70d3a0158055b75ca34f05d36759793b0b57ede0c187ac119d642a5486c": {
                "Name": "pedantic_noyce", 45f6e70d3a01
                "EndpointID": "bf7c880ccae65c94a5aed94637e04d004272893ff5f8d03c7decb4c39e427e2e",
                "MacAddress": "02:42:ac:13:00:03",
                "IPv4Address": "172.19.0.3/16",
                "IPv6Address": ""
            },
            "4c82cc89c90bb552d4fb0f5ebfc4f97aa2d089f8ee9d9e6918cf2a34b8920c56": {
                "Name": "hopeful_lamarr", 4c82cc89c90b
                "EndpointID": "b1cc7443edf967e44e873df1497bfc62a2d8d083fc0cade47bf5954cdf28c32c",
                "MacAddress": "02:42:ac:13:00:06",
                "IPv4Address": "172.19.0.6/16",
                "IPv6Address": ""
            },
            "9293609c043aa77f8ba65c0a114d105f1c9d2d837083d373e73f57facd4d2102": {
                "Name": "festive_swirles", 9293609c043a
                "EndpointID": "3450098ffc7fda2d730be60da1901631c11d088de9cf7cb3c0c3c107436b2759",
                "MacAddress": "02:42:ac:13:00:08",
                "IPv4Address": "172.19.0.8/16",
                "IPv6Address": ""
            },


            "a8c7bf1c920fea4a8989d6ec23a853238bc7e9d9ebe2dd65d8555f8fbfaa104f": {
                "Name": "clever_shirley",
                "EndpointID": "506d8f1fd1c481d4519bc15dd86c52d540515bef37565bc2fe0225e2921b8c6c",
                "MacAddress": "02:42:ac:13:00:09",
                "IPv4Address": "172.19.0.9/16",
                "IPv6Address": ""





                    charming_jemison



ba677101c31b 51  testbed:1.0    "bin/bash"               4 seconds ago       Up 3 seconds                             focused_villani
6e6e7d991cba 52  testbed:1.0    "bin/bash"               5 seconds ago       Up 4 seconds                             festive_khayyam
29e172c989fc 53  testbed:1.0    "bin/bash"               6 seconds ago       Up 5 seconds                             mystifying_raman
9db5f5733830  54 testbed:1.0    "bin/bash"               8 seconds ago       Up 7 seconds                             vibrant_vaughan
06af3a09568a 55  testbed:1.0    "bin/bash"               9 seconds ago       Up 8 seconds                             nice_galileo
4f802378f551  56 testbed:1.0    "bin/bash"               12 seconds ago      Up 11 seconds                            dreamy_hopper



              "3cf5319aead2dd5937990b53072e964c152aec6a91577b2ad92b25bcd7e6880e": {
                "Name": "zealous_agnesi",  145094e499d7
                "EndpointID": "0db6cdb96e35c4c07ee11087c6a4c3abca2fafe29ac6a2290e1846b427a2cf11",
                "MacAddress": "02:42:ac:13:00:04",
                "IPv4Address": "172.19.0.4/16",
                "IPv6Address": ""
            },



                  "6ea25fed95eb426d32fb4560e47d461ece5edce4d13a004e2231ba672746171a": {
                "Name": "elastic_saha",
                "EndpointID": "b5eab565d6b94b63789172da0a9a54d955de20f355316f27f08312f5b6121905",
                "MacAddress": "02:42:ac:13:00:02",
                "IPv4Address": "172.19.0.2/16",
                "IPv6Address": ""




sudo docker ps -a 
    sudo docker exec -it     bash

docker service ps federated


5de986d7991beb01d78beb8d52dd5785aa28a19b870f2a377de7c5cd34d89019 68
26ee020742a9e419fcb6b25b31771a6f0028b21a28beeb5d2c01e334fe6e7092  69
a2f19dbd97b38747c8e5347943d61f9e600fa61ae94e691409414e828261ba8f 70
7c16b924540e1acc05c6cf2d08db39175f4f42c4389d182d44234ec5fa6700bb 71
47ea5678d41247a7941656250b0a2320c60556f02e88bb59f5b1b2fc230837f7 72
ebf95ab5a0813d37f2f7442ef5fadf898e073c1259db08ce7214dce7cd0fcf37 73



apt-get update && apt install -y iproute2 && apt-get install -y iputils-ping


docker commit 45f6e70d3a01 testbed:0.2


 -->
#create an updated image
docker network inspect skynet
































































<!-- 
<h2>Case 3: combined_angular_misalignment_with_RUL.csv </h2>

Test case: "4 Clients and 1 Server"

<pre>

<h2>📜datasplit.py</h2>
<h3>Client 104</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/datasplit.py -ml 2 -cm 4 -cn 0 -dfn combined_angular_misalignment_with_RUL.csv -ip 10.147.17.104 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1

#SERVER!!!!

python3 fl_testbed/version2/client/datasplit.py -ml 2 -cm 4 -cn 0 -dfn combined_angular_misalignment_with_RUL.csv -ip 10.147.17.104 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1

<!-- <h3>Client 111</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/datasplit.py -ml 2 -cm 4 -cn 1 -dfn combined_angular_misalignment_with_RUL.csv -ip 10.147.17.111 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1

python3 fl_testbed/version2/client/datasplit.py -ml 2 -cm 4 -cn 1 -dfn combined_angular_misalignment_with_RUL.csv -ip 10.147.17.111 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1

<h3>Client 234</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/datasplit.py -ml 2 -cm 4 -cn 2 -dfn combined_angular_misalignment_with_RUL.csv -ip 10.147.17.234 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1

python3 fl_testbed/version2/client/datasplit.py -ml 2 -cm 4 -cn 2 -dfn combined_angular_misalignment_with_RUL.csv -ip 10.147.17.234 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1


<h3>Client 150</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/datasplit.py -ml 2 -cm 4 -cn 3 -dfn combined_angular_misalignment_with_RUL.csv -ip 10.147.17.150 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1

python3 fl_testbed/version2/client/datasplit.py -ml 2 -cm 4 -cn 3 -dfn combined_angular_misalignment_with_RUL.csv -ip 10.147.17.150 -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 -fq 0.1 0.6 0.3 1 -->



<h2>📜centralized.py</h2>

<!-- <h3>Client 104</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py -ml 2 -lr 0.001 -cn 0 -cm 4 -e 400 -dfn 'combined_angular_misalignment_with_RUL.csv' -ip '10.147.17.104'

python3 fl_testbed/version2/client/centralized.py -ml 2 -lr 0.001 -cn 0 -cm 4 -e 400 -dfn 'combined_angular_misalignment_with_RUL.csv' -ip '10.147.17.104'

<h3>Client 111</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py -ml 2 -lr 0.001 -cn 1 -cm 4 -e 400 -dfn 'combined_angular_misalignment_with_RUL.csv' -ip '10.147.17.111'

python3 fl_testbed/version2/client/centralized.py -ml 2 -lr 0.001 -cn 1 -cm 4 -e 400 -dfn 'combined_angular_misalignment_with_RUL.csv' -ip '10.147.17.111'

<h3>Client 234</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py -ml 2 -lr 0.001 -cn 2 -cm 4 -e 400 -dfn 'combined_angular_misalignment_with_RUL.csv' -ip '10.147.17.234'

python3 fl_testbed/version2/client/centralized.py -ml 2 -lr 0.001 -cn 2 -cm 4 -e 400 -dfn 'combined_angular_misalignment_with_RUL.csv' -ip '10.147.17.234'


<h3>Client 150</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py -ml 2 -lr 0.001 -cn 3 -cm 4 -e 400 -dfn 'combined_angular_misalignment_with_RUL.csv' -ip '10.147.17.150'

python3 fl_testbed/version2/client/centralized.py -ml 2 -lr 0.001 -cn 3 -cm 4 -e 400 -dfn 'combined_angular_misalignment_with_RUL.csv' -ip '10.147.17.150'

<h3>Server 96</h3> python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/centralized.py -ml 2 -lr 0.001 -cn 4 -cm 4 -e 400 -dfn 'combined_angular_misalignment_with_RUL.csv' -ip '10.147.17.96'


python3 fl_testbed/version2/client/centralized.py -ml 2 -lr 0.001 -cn 4 -cm 4 -e 400 -dfn 'combined_angular_misalignment_with_RUL.csv' -ip '10.147.17.96'



<h2>📜independent.py</h2>

<h3>Client 104</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/independent.py  -ml 2 -lr 0.001 -e 400 -cm 4 -cn 0  -dfn DATASET_0.csv -ip 10.147.17.104 --comparative_path_y_test 400_0.001_2_4_0_combined_angular_misalignment_with_RUL.csv__client_centralizedy_test.pkl  --comparative_path_X_test 400_0.001_2_4_0_combined_angular_misalignment_with_RUL.csv__client_centralizedX_test.pkl

python3 fl_testbed/version2/client/independent.py  -ml 2 -lr 0.001 -e 400 -cm 4 -cn 0  -dfn DATASET_0.csv -ip 10.147.17.104 --comparative_path_y_test 400_0.001_2_4_0_combined_angular_misalignment_with_RUL.csv__client_centralizedy_test.pkl  --comparative_path_X_test 400_0.001_2_4_0_combined_angular_misalignment_with_RUL.csv__client_centralizedX_test.pkl


<h3>Client 111</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/independent.py  -ml 2 -lr 0.001 -e 400 -cm 4 -cn 1  -dfn DATASET_1.csv -ip 10.147.17.111 --comparative_path_y_test 400_0.001_2_4_1_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_2_4_1_combined_offset_misalignment.csv__client_centralizedX_test.pkl


python3 fl_testbed/version2/client/independent.py  -ml 2 -lr 0.001 -e 400 -cm 4 -cn 1  -dfn DATASET_1.csv -ip 10.147.17.111 --comparative_path_y_test 400_0.001_2_4_1_combined_offset_misalignment.csv__client_centralizedy_test.pkl  --comparative_path_X_test 10_0.001_2_4_1_combined_offset_misalignment.csv__client_centralizedX_test.pkl


<h3>Client 234</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/independent.py  -ml 2 -lr 0.001 -e 400 -cm 4 -cn 2  -dfn DATASET_2.csv -ip 10.147.17.234 --comparative_path_y_test 400_0.001_2_4_2_combined_angular_misalignment_with_RUL.csv__client_centralizedy_test.pkl  --comparative_path_X_test 400_0.001_2_4_2_combined_angular_misalignment_with_RUL.csv__client_centralizedX_test.pkl

python3 fl_testbed/version2/client/independent.py  -ml 2 -lr 0.001 -e 400 -cm 4 -cn 2  -dfn DATASET_2.csv -ip 10.147.17.234 --comparative_path_y_test 400_0.001_2_4_2_combined_angular_misalignment_with_RUL.csv__client_centralizedy_test.pkl  --comparative_path_X_test 400_0.001_2_4_2_combined_angular_misalignment_with_RUL.csv__client_centralizedX_test.pkl


<h3>Client 150</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/independent.py  -ml 2 -lr 0.001 -e 400 -cm 4 -cn 3  -dfn DATASET_3.csv -ip 10.147.17.150 --comparative_path_y_test 400_0.001_2_4_3_combined_angular_misalignment_with_RUL.csv__client_centralizedy_test.pkl  --comparative_path_X_test 400_0.001_2_4_3_combined_angular_misalignment_with_RUL.csv__client_centralizedX_test.pkl

python3 fl_testbed/version2/client/independent.py  -ml 2 -lr 0.001 -e 400 -cm 4 -cn 3  -dfn DATASET_3.csv -ip 10.147.17.150 --comparative_path_y_test 400_0.001_2_4_3_combined_angular_misalignment_with_RUL.csv__client_centralizedy_test.pkl  --comparative_path_X_test 400_0.001_2_4_3_combined_angular_misalignment_with_RUL.csv__client_centralizedX_test.pkl



<h2>📜federated_server.py</h2>

<h3>Server 96</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/server/federated_server.py  -ml 2 -lr 0.001 -e 10 -cm 4  --rounds 100  -ip 10.147.17.96 --comparative_path_y_test 400_0.001_2_4_4_combined_angular_misalignment_with_RUL.csv__client_centralizedy_test.pkl  --comparative_path_X_test 400_0.001_2_4_4_combined_angular_misalignment_with_RUL.csv__client_centralizedX_test.pkl

python3 fl_testbed/version2/server/federated_server.py  -ml 2 -lr 0.001 -e 10 -cm 4  --rounds 100  -ip 10.147.17.96 --comparative_path_y_test 400_0.001_2_4_4_combined_angular_misalignment_with_RUL.csv__client_centralizedy_test.pkl  --comparative_path_X_test 400_0.001_2_4_4_combined_angular_misalignment_with_RUL.csv__client_centralizedX_test.pkl




<h2>📜federated_client.py</h2>

<h3>Client 104</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/federated_client.py  -ml 2 -lr 0.001 -e 40 -cm 4 -cn 0  -dfn DATASET_0.csv -ip 10.147.17.104

python3 fl_testbed/version2/client/federated_client.py  -ml 2 -lr 0.001 -e 40 -cm 4 -cn 0  -dfn DATASET_0.csv -ip 10.147.17.104

<h3>Client 111</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/federated_client.py  -ml 2 -lr 0.001 -e 40 -cm 4 -cn 1  -dfn DATASET_1.csv -ip 10.147.17.111

python3 fl_testbed/version2/client/federated_client.py  -ml 2 -lr 0.001 -e 40 -cm 4 -cn 1  -dfn DATASET_1.csv -ip 10.147.17.111

<h3>Client 234</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/federated_client.py  -ml 2 -lr 0.001 -e 40 -cm 4 -cn 2  -dfn DATASET_2.csv -ip 10.147.17.234

python3 fl_testbed/version2/client/federated_client.py  -ml 2 -lr 0.001 -e 40 -cm 4 -cn 2  -dfn DATASET_2.csv -ip 10.147.17.234

<h3>Client 150</h3>python3 $HOME/FL_AM_Defect-Detection/fl_testbed/version2/client/federated_client.py  -ml 2 -lr 0.001 -e 40 -cm 4 -cn 3  -dfn DATASET_3.csv -ip 10.147.17.150

python3 fl_testbed/version2/client/federated_client.py  -ml 2 -lr 0.001 -e 40 -cm 4 -cn 3  -dfn DATASET_3.csv -ip 10.147.17.150



</pre>

 -->












<!-- <h2>Some highlights:</h2>
 -->

<!-- RUL MODEL Centralized with RUL data:

<pre>






<h2>96</h2>

List of possible mae: [11205.6640625, 10526.9052734375, 9904.2314453125, 8805.92578125, 11416.0478515625, 9454.7255859375, 13514.5625, 8636.7294921875, 8774.0927734375, 7873.4482421875]
Maximum mae That can be obtained from this model is: 13514.5625
Minimum mae: 7873.4482421875
Overall mae: 10011.23330078125

R^2: 0.9921982190658936
Mean Absolute Error (MAE): 27845.43008975196
Mean Squared Error (MSE): 3730215783.363027
Mean Absolute Percentage Error (MAPE): 0.221670562566138
Root Mean Squared Error (RMSE): 61075.49249382298
Explained Variance Score: 0.9923213341740174
Max Error: 2572092.0
Median Absolute Error: 14377.765625

<h2>104</h2>
List of possible mae: [11005.6044921875, 13002.6669921875, 10785.0380859375, 10293.9951171875, 8741.591796875, 7892.34716796875, 8058.63330078125, 8850.279296875, 6986.27587890625, 6953.142578125]
Maximum mae That can be obtained from this model is: 13002.6669921875
Minimum mae: 6953.142578125
Overall mae: 9256.957470703124

Name: rul, dtype: float64
(83583, 1)
R^2: 0.9832397201581122
Mean Absolute Error (MAE): 10781.154950057531
Mean Squared Error (MSE): 7983382309.35146
Mean Absolute Percentage Error (MAPE): 0.029516975619534584
Root Mean Squared Error (RMSE): 89349.77509401723
Explained Variance Score: 0.9832763186266061
Max Error: 24996002.0
Median Absolute Error: 6288.5
111


<h2>234</h2>
List of possible mae: [10250.4873046875, 6889.64892578125, 7406.3193359375, 7224.55078125, 6558.556640625, 5757.83837890625, 5933.31103515625, 5691.77294921875, 8334.9921875, 5472.89697265625]
Maximum mae That can be obtained from this model is: 10250.4873046875
Minimum mae: 5472.89697265625
Overall mae: 6952.037451171875

R^2: 0.8170189074402454
Mean Absolute Error (MAE): 23863.898258035668
Mean Squared Error (MSE): 87236457222.32507
Mean Absolute Percentage Error (MAPE): 9779058074935468.0
Root Mean Squared Error (RMSE): 295358.18462051306
Explained Variance Score: 0.8172640756907884
Max Error: 34324610.0
Median Absolute Error: 9597.75

<h2>150</h2>

R^2: 0.9896230423272036
Mean Absolute Error (MAE): 17265.76824377338
Mean Squared Error (MSE): 4963426278.73304
Mean Absolute Percentage Error (MAPE): 1.3553031241732274e+16
Root Mean Squared Error (RMSE): 70451.58819170111
Explained Variance Score: 0.9897316987405672
Max Error: 5830785.5
Median Absolute Error: 9835.65625

</pre> -->


#TO FOR

rsync -av --exclude='results' out_* results/tuesep26