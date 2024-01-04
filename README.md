## FL_AM_Defect-Detection
Paper for FL  

<h2>Publication Link</h2>
<pre>

[Overleaf URL](https://www.overleaf.com/project/641e11b0b280170c0afd8854)


📦Overleaf
 ┣ 📂IEEE (to_read)
 ┃ ┗ 📜main.tex (to_read)
 ┣ 📂ACM
 ┗ ┗📜main.tex


</pre>


<img align="center" src="https://federated.withgoogle.com/assets/comic/panel046.png">
</img>
<br>

<pre>
The Federated Learning code is organized within the 'fl_testbed' folder. The file structure adheres to the following schema.
</pre>
<!-- Federated Learning code is contained inside the fl_testbed folder. The file structure follows the shown schema. -->

<pre>
📦fl_testbed
 ┣ 📂version2
 ┃ ┣ 📂client
 ┃ ┣ 📂data
 ┃ ┃ ┣ 📂initial
 ┃ ┃ ┃ ┗ 📜combined_offset_misalignment.csv
 ┃ ┃ ┗ 📂transformed
 ┃ ┣ 📂server
 ┗ 📜README.md
</pre>


 <!-- ┃ ┃ ┃ ┣ 📜1_0.001_1_4_0_DATASET_0.csv__client_independentX_test.pkl
 ┃ ┃ ┃ ┣ 📜1_0.001_1_4_0_DATASET_0.csv__client_independentX_train.pkl
 ┃ ┃ ┃ ┣ 📜1_0.001_1_4_0_DATASET_0.csv__client_independenty_test.pkl
 ┃ ┃ ┃ ┣ 📜1_0.001_1_4_0_DATASET_0.csv__client_independenty_train.pkl
 ┃ ┃ ┃ ┣ 📜DATASET_0.csv
 ┃ ┃ ┃ ┣ 📜DATASET_0.png
 ┃ ┃ ┃ ┗ 📜TOTAL_DATASET.png -->

  <!-- ┃ ┃ ┣ 📜CustomStrategy.py
 ┃ ┃ ┣ 📜NOTUSED.txt
 ┃ ┃ ┣ 📜aggregate.py
 ┃ ┃ ┣ 📜federated_server.py
 ┃ ┃ ┗ 📜strategy.py -->

  <!-- ┃ ┃ ┣ 📜CustomNumpyClient.py
 ┃ ┃ ┣ 📜centralized.py
 ┃ ┃ ┣ 📜datasplit.py
 ┃ ┃ ┣ 📜federated_client.py
 ┃ ┃ ┗ 📜independent.py -->

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
<!-- 📜NOTUSED.txt: Not in use. -->
📜aggregate.py: Required file for federated server script.
📜federated_server.py: Script for running the federated server.
📜strategy.py: Abstract base class for server strategy.



</pre>





<h2>Execution Order:</h2>

Execution orchestrators labeled as (ex.'server_execution_LSTM_M1.sh') perform the following parameterized scripts execution while testing the FL hyperparameter parameter grid.

1. datasplit.py (Clients)
2. centralized.py (Clients and server)
3. independent.py (Clients)
4. federated_server.py (Server)
5. federated_client.py (Clients)



<h2>Server access and pre-execution steps</h2>


Credentials


<pre>
<h3>Server Tesla</h3>
User: jose
Password: ********
login: ssh jose@tesla.ise.utk.edu
Please run: cd FL_AM_Defect-Detection && rm fl_testbed/version2/data/transformed/*
</pre>
<!-- 
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
Please run: cd FL_AM_Defect-Detection && rm fl_testbed/version2/data/transformed/* -->



<h3>Important:<h3>
<pre>
It is required for all clients to create 2 directories and place the follwoing files in it. After changing directory to "FL_AM_Defect-Detection" . Please, run:

"mkdir fl_testbed/version2/data/initial"
"mkdir fl_testbed/version2/data/transformed"

Paste both files under the initial folder:

combined_angular_misalignment_with_RUL.csv: https://drive.google.com/file/d/12Lvz0f56et1_-VXhgSEDkAU2xAUwCvIO/view?usp=sharing

combined_offset_misalignment.csv: https://drive.google.com/file/d/1-E5wqPmhtIlsde04fT2WDtzNXx-nufZa/view?usp=sharing


</pre>


<h2>Parameters:</h2>

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


</pre> 
<!-- 