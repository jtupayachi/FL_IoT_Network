## FL_AM_Defect-Detection
<!-- File Structure -->

<h2>Abstract:</h2>

Internet of Things (IoT) sensors play a crucial role in collecting data and capturing patterns that enable health status evaluation of internal components in industrial machinery. These devices can actively collect signals to obtain multivariate temporal data. Machine learning fault detection methods rely on the availability of substantial quantities of high-quality information. In settings where collaborative data efforts face data sensitivity and computational workload challenges, federated learning (FL) can offer significant benefits. This new approach can greatly enhance fault detection capabilities in complex environments, promoting predictive maintenance plans to ensure the proper functioning of critical machinery. We present an innovative decentralized federated learning framework designed for fault detection in the context of condition-based monitoring. In-node local learning and a master-averaged weighted model effectively learn under isolated conditions. We implement and benchmark federated classification and regression models based on the remaining useful life (RUL) prediction and offset-type detection. A Dirichlet distribution generates non-identically and independently distributed data sets representing offset errors and use conditions. Computational results demonstrate a favorable performance in models based on q-Fair Federated Averaging (Q-FedAvg) and Federated Optimization (FedOpt) for multi-layer perceptron (MLP) based models, and Q-FedAvg and Federated Averaging with Momentum (FedAvgM) for long short-term memory (LSTM) models.

<!-- <h2>Publication Link</h2> -->
<!-- <pre>

📦Overleaf
 ┣ 📂IEEE (to_read)
 ┃ ┗ 📜main.tex (to_read)
 ┣ 📂ACM
 ┗ ┗📜main.tex


</pre> -->


<!-- <img align="center" src="https://federated.withgoogle.com/assets/comic/panel046.png">
</img> -->
<br>


<h2>Overview:</h2>

<pre>
The Federated Learning code is organized within the 'fl_testbed/version2' folder. The file structure adheres to the following schema.
</pre>
<!-- Federated Learning code is contained inside the fl_testbed folder. The file structure follows the shown schema. -->

<pre>
📦fl_testbed
 ┣ 📂version2
 ┃ ┣ 📂client
 ┃ ┣ 📂data
 ┃ ┃ ┣ 📂initial
 ┃ ┃ ┗ 📂transformed
 ┃ ┣ 📂server
 ┗ 📜README.md
</pre>



<h2>File Description:</h2>


<pre>
--->📂client<---
<!-- 📜CustomNumpyClient.py: Inhered class with hadny function for federated client. -->

📜centralized.py: Runs a deep learning model using a complete dataset.

📜datasplit.py: Builts in two operation modes for data generation (Dirichlet Distribution).

📜federated_client.py: Script for running the federated client.

📜independent.py: This script triggers a deep learning model on a small section
of the whole dataset.



--->📂data<---
📂initial: Folder that contains initial datasets.
📂transformed: Scripts generated files and data miscellanous data.


--->📂server<---
<!-- 📜CustomStrategy.py: Custom FedAvg strategy implementation with built-in testing.
📜aggregate.py: Required file for federated server script. -->
📜federated_server.py: Runs the federated server.
<!-- 📜strategy.py: Abstract base class for server strategy. -->



</pre>





<h2>Execution Order:</h2>

Execution orchestrators labeled as (ex.'server_execution_LSTM_M1.sh') perform the following parameterized scripts execution while testing the FL hyperparameter parameter grid.

1. datasplit.py (Clients)
2. centralized.py (Clients and Server)
3. independent.py (Clients)
4. federated_server.py (Server)
5. federated_client.py (Clients)



<h2>Server access and pre-execution steps</h2>


<!-- Credentials -->


<pre>
<h3>Running Server & Credentials</h3>
Server: Tesla
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



<h3>Important:</h3>

It is required for all clients to create 2 directories and place the follwoing files in it. After changing directory to "FL_AM_Defect-Detection" . Please, run:

"mkdir fl_testbed/version2/data/initial"
"mkdir fl_testbed/version2/data/transformed"

Paste both files under the initial folder:

<pre>


combined_angular_misalignment_with_RUL.csv: https://drive.google.com/file/d/12Lvz0f56et1_-VXhgSEDkAU2xAUwCvIO/view?usp=sharing

combined_offset_misalignment.csv: https://drive.google.com/file/d/1-E5wqPmhtIlsde04fT2WDtzNXx-nufZa/view?usp=sharing


</pre>


<h2>Main Parameters:</h2>

Included in the .sh files.
<pre>



-ml: Type of model executed.
-lr: Learning rate. 
-e: Number of epochs.
-cm: Max number of clients.
-cn: Client number.
-dfn: Initial dataframe.
-ip: CLient/server ip.
-fq: fraction of sampled/residual dataset.
--comparative_path_y_test: Initial dataset splitted train/test saved as pickle y_test
--comparative_path_X_test: Initial dataset splitted train/test saved as pickle X_test
--rounds: Number of federated rounds.
--JUMPING_STEP: Time window offset.




</pre> 



<h2>Results MLP:</h2>

<pre>

 
FL & Alpha & S-LR & FL Param & Accuracy & F1-Weighted & MCS & Loss & Runtime (s) \\
\hline
\midrule
   FedAvg &      0.001 &        - &           - &    0.280 &       0.160 & 0.246 &  4.031 &  1210.554 \\
  FedAvgM &      0.001 &    0.001 &         0.0 &    0.280 &       0.210 & 0.048 &  1.936 &  1279.013 \\
  FedAvgM &      0.001 &    0.001 &         0.7 &    0.370 &       0.270 & 0.152 &  1.751 &  1279.983 \\
  FedAvgM &      0.001 &    0.001 &         0.9 &    0.380 &       0.360 & 0.228 &  1.583 &  1275.875 \\
  \hline
   FedOpt &      0.001 &    0.001 &        1e-7 &    0.420 &       0.350 & 0.360 &  1.194 &  1380.077 \\
   FedOpt &      0.001 &    0.001 &        1e-8 &    0.170 &       0.060 & 0.100 &  2.581 &  1594.436 \\
   FedOpt &      0.001 &    0.001 &        1e-9 &    0.690 &       0.620 & 0.605 &  1.173 &  1287.955 \\
   \hline
  QFedAvg &      0.001 &    0.001 &         0.1 &    0.290 &       0.230 & 0.129 &  1.731 &  4306.860 \\
  QFedAvg &      0.001 &    0.001 &         0.2 &        - &           - &     - &      - &  1610.604 \\
  QFedAvg &      0.001 &    0.001 &         0.5 &        - &           - &     - &      - &  1611.923 \\
  \hline
  FedAvgM &      0.001 &     0.01 &         0.0 &    0.220 &       0.110 & 0.131 &  1.821 &  1281.111 \\
  FedAvgM &      0.001 &     0.01 &         0.7 &    0.560 &       0.560 & 0.469 &  1.442 &  1277.444 \\
  FedAvgM &      0.001 &     0.01 &         0.9 &    0.550 &       0.530 & 0.496 &  1.218 &  1275.843 \\
  \hline
   FedOpt &      0.001 &     0.01 &        1e-7 &    0.410 &       0.350 & 0.283 &  1.535 &  1295.318 \\
   FedOpt &      0.001 &     0.01 &        1e-8 &    0.480 &       0.400 & 0.381 &  1.012 &  1282.616 \\
   FedOpt &      0.001 &     0.01 &        1e-9 &    0.560 &       0.560 & 0.503 &  0.875 &  1281.948 \\
   \hline
  QFedAvg &      0.001 &     0.01 &         0.1 &    0.210 &       0.190 & 0.036 &  1.698 &  4305.479 \\
  QFedAvg &      0.001 &     0.01 &         0.2 &    0.240 &       0.210 & 0.044 &  1.824 &  1613.309 \\
  QFedAvg &      0.001 &     0.01 &         0.5 &    0.440 &       0.370 & 0.361 &  1.604 &  1607.695 \\
  \hline
  FedAvgM &      0.001 &        1 &         0.0 &    0.390 &       0.290 & 0.272 &  2.713 &  1272.881 \\
  FedAvgM &      0.001 &        1 &         0.7 &    0.340 &       0.220 & 0.109 &  1.842 &  1638.175 \\
  FedAvgM &      0.001 &        1 &         0.9 &    0.370 &       0.280 & 0.244 &  7.107 &  2998.370 \\
  \hline
   FedOpt &      0.001 &        1 &        1e-7 &    0.270 &       0.150 & 0.234 &  3.361 &  1292.318 \\
   FedOpt &      0.001 &        1 &        1e-8 &    0.680 &       0.690 & 0.620 &  0.990 &  1271.336 \\
   FedOpt &      0.001 &        1 &        1e-9 &    0.720 &       0.730 & 0.676 &  1.334 &  1284.942 \\
   \hline
  QFedAvg &      0.001 &        1 &         0.1 &    0.910 &       0.910 & 0.889 &  0.446 &  5940.035 \\
  QFedAvg &      0.001 &        1 &         0.2 &    0.760 &       0.720 & 0.696 &  0.715 &  3283.933 \\
  QFedAvg &      0.001 &        1 &         0.5 &    0.760 &       0.770 & 0.706 &  0.726 &  3160.853 \\
  \hline
   FedAvg &      0.005 &        - &           - &    0.420 &       0.350 & 0.376 &  4.318 &  2745.513 \\
  FedAvgM &      0.005 &    0.001 &         0.0 &    0.180 &       0.140 & 0.011 &  1.958 &  2245.466 \\
  FedAvgM &      0.005 &    0.001 &         0.7 &    0.190 &       0.150 & 0.082 &  1.867 &  2751.595 \\
  FedAvgM &      0.005 &    0.001 &         0.9 &    0.360 &       0.310 & 0.211 &  1.627 &  2233.973 \\
  \hline
   FedOpt &      0.005 &    0.001 &        1e-7 &    0.600 &       0.560 & 0.553 &  1.232 &  1154.881 \\
   FedOpt &      0.005 &    0.001 &        1e-8 &    0.450 &       0.350 & 0.404 &  1.181 &  2896.666 \\
   FedOpt &      0.005 &    0.001 &        1e-9 &    0.180 &       0.100 & 0.117 &  1.245 &  1281.921 \\
   \hline
  QFedAvg &      0.005 &    0.001 &         0.1 &        - &           - &     - &      - &  7034.284 \\
  QFedAvg &      0.005 &    0.001 &         0.2 &    0.310 &       0.300 & 0.129 &  1.736 &  3376.207 \\
  QFedAvg &      0.005 &    0.001 &         0.5 &    0.310 &       0.200 & 0.040 &  1.823 &  2245.047 \\
  \hline
  FedAvgM &      0.005 &     0.01 &         0.0 &    0.380 &       0.400 & 0.217 &  1.640 &  2741.302 \\
  FedAvgM &      0.005 &     0.01 &         0.7 &    0.600 &       0.550 & 0.493 &  1.539 &  1738.539 \\
  FedAvgM &      0.005 &     0.01 &         0.9 &    0.530 &       0.490 & 0.505 &  1.164 &  1408.076 \\
  \hline
   FedOpt &      0.005 &     0.01 &        1e-7 &    0.720 &       0.720 & 0.680 &  1.247 &  1298.812 \\
   FedOpt &      0.005 &     0.01 &        1e-8 &    0.320 &       0.210 & 0.281 &  2.468 &  2471.470 \\
   FedOpt &      0.005 &     0.01 &        1e-9 &    0.160 &       0.080 & 0.085 &  1.348 &  1285.359 \\
   \hline
  QFedAvg &      0.005 &     0.01 &         0.1 &    0.120 &       0.030 & 0.002 &  1.976 &  6111.755 \\
  QFedAvg &      0.005 &     0.01 &         0.2 &    0.440 &       0.360 & 0.255 &  1.654 &  3507.090 \\
  QFedAvg &      0.005 &     0.01 &         0.5 &        - &           - &     - &      - &  2765.469 \\
  \hline
  FedAvgM &      0.005 &        1 &         0.0 &    0.600 &       0.560 & 0.521 &  2.927 &  2770.307 \\
  FedAvgM &      0.005 &        1 &         0.7 &    0.410 &       0.360 & 0.421 &  6.472 &  2042.253 \\
  FedAvgM &      0.005 &        1 &         0.9 &    0.440 &       0.370 & 0.331 &  5.929 &  2807.720 \\
  \hline
   FedOpt &      0.005 &        1 &        1e-7 &    0.180 &       0.070 & 0.121 &  0.764 &  1280.215 \\
   FedOpt &      0.005 &        1 &        1e-8 &    0.450 &       0.360 & 0.383 &  1.588 &  2520.033 \\
   FedOpt &      0.005 &        1 &        1e-9 &    0.520 &       0.470 & 0.479 &  2.270 &  1284.866 \\
   \hline
  QFedAvg &      0.005 &        1 &         0.1 &    0.620 &       0.560 & 0.540 &  0.358 &  6192.477 \\
  QFedAvg &      0.005 &        1 &         0.2 &    0.790 &       0.760 & 0.736 &  0.707 &  2567.996 \\
  QFedAvg &      0.005 &        1 &         0.5 &    0.450 &       0.390 & 0.421 &  0.795 &  3349.176 \\
  \hline
   FedAvg &       0.01 &        - &           - &    0.550 &       0.460 & 0.497 &  1.479 &  1418.522 \\
  FedAvgM &       0.01 &    0.001 &         0.0 &    0.150 &       0.150 & 0.031 &  1.880 &  1304.965 \\
  FedAvgM &       0.01 &    0.001 &         0.7 &    0.220 &       0.120 & 0.207 &  2.011 &  1374.984 \\
  FedAvgM &       0.01 &    0.001 &         0.9 &    0.410 &       0.350 & 0.208 &  1.660 &  1873.002 \\
  \hline
   FedOpt &       0.01 &    0.001 &        1e-7 &    0.430 &       0.340 & 0.321 &  0.714 &  1148.910 \\
   FedOpt &       0.01 &    0.001 &        1e-8 &    0.470 &       0.400 & 0.417 &  1.391 &  1638.332 \\
   FedOpt &       0.01 &    0.001 &        1e-9 &    0.710 &       0.710 & 0.681 &  0.981 &  1281.673 \\
   \hline
  QFedAvg &       0.01 &    0.001 &         0.1 &        - &           - &     - &      - &  6529.223 \\
  QFedAvg &       0.01 &    0.001 &         0.2 &    0.260 &       0.280 & 0.160 &  1.759 &  3046.692 \\
  QFedAvg &       0.01 &    0.001 &         0.5 &    0.320 &       0.170 & 0.005 &  1.739 &  3051.956 \\
  \hline
  FedAvgM &       0.01 &     0.01 &         0.0 &    0.400 &       0.340 & 0.198 &  1.648 &  2892.551 \\
  FedAvgM &       0.01 &     0.01 &         0.7 &    0.690 &       0.600 & 0.613 &  1.494 &  2363.838 \\
  FedAvgM &       0.01 &     0.01 &         0.9 &    0.610 &       0.620 & 0.535 &  1.299 &  2380.934 \\
  \hline
   FedOpt &       0.01 &     0.01 &        1e-7 &    0.580 &       0.510 & 0.540 &  0.632 &  1288.414 \\
   FedOpt &       0.01 &     0.01 &        1e-8 &    0.390 &       0.320 & 0.352 &  0.632 &  1757.518 \\
   FedOpt &       0.01 &     0.01 &        1e-9 &    0.460 &       0.440 & 0.368 &  0.531 &  1780.786 \\
   \hline
  QFedAvg &       0.01 &     0.01 &         0.1 &    0.270 &       0.240 & 0.066 &  1.680 &  4651.629 \\
  QFedAvg &       0.01 &     0.01 &         0.2 &    0.280 &       0.170 & 0.022 &  1.785 &  1751.156 \\
  QFedAvg &       0.01 &     0.01 &         0.5 &        - &           - &     - &      - &  1746.899 \\
  \hline
  FedAvgM &       0.01 &        1 &         0.0 &    0.400 &       0.330 & 0.268 &  1.711 &  1638.610 \\
  FedAvgM &       0.01 &        1 &         0.7 &    0.750 &       0.670 & 0.675 &  1.509 &  2705.983 \\
  FedAvgM &       0.01 &        1 &         0.9 &    0.270 &       0.210 & 0.227 &  6.999 &  2250.001 \\
  \hline
   FedOpt &       0.01 &        1 &        1e-7 &    0.420 &       0.320 & 0.306 &  1.483 &  1284.505 \\
   FedOpt &       0.01 &        1 &        1e-8 &    0.530 &       0.520 & 0.478 &  1.448 &  2913.922 \\
   FedOpt &       0.01 &        1 &        1e-9 &    0.540 &       0.490 & 0.426 &  1.271 &  1298.697 \\
   \hline
  QFedAvg &       0.01 &        1 &         0.1 &    0.770 &       0.720 & 0.706 &  0.685 &  6433.538 \\
  QFedAvg &       0.01 &        1 &         0.2 &    0.650 &       0.600 & 0.552 &  0.810 &  3287.135 \\
  QFedAvg &       0.01 &        1 &         0.5 &    0.800 &       0.790 & 0.748 &  0.769 &  3351.137 \\
  \hline
   FedAvg &       0.02 &        - &           - &    0.410 &       0.310 & 0.309 &  1.162 &  2243.114 \\
  FedAvgM &       0.02 &    0.001 &         0.0 &    0.250 &       0.220 & 0.023 &  1.902 &  2643.627 \\
  FedAvgM &       0.02 &    0.001 &         0.7 &    0.200 &       0.100 & 0.101 &  2.069 &  2920.203 \\
  FedAvgM &       0.02 &    0.001 &         0.9 &    0.220 &       0.230 & 0.034 &  1.766 &  1852.229 \\
  \hline
   FedOpt &       0.02 &    0.001 &        1e-7 &    0.500 &       0.440 & 0.456 &  0.947 &  1642.449 \\
   FedOpt &       0.02 &    0.001 &        1e-8 &    0.430 &       0.350 & 0.327 &  0.616 &  3095.159 \\
   FedOpt &       0.02 &    0.001 &        1e-9 &    0.510 &       0.460 & 0.457 &  2.528 &  1280.040 \\
   \hline
  QFedAvg &       0.02 &    0.001 &         0.1 &        - &           - &     - &      - &  4928.401 \\
  QFedAvg &       0.02 &    0.001 &         0.2 &    0.220 &       0.190 & 0.055 &  1.864 &  1742.113 \\
  QFedAvg &       0.02 &    0.001 &         0.5 &    0.160 &       0.070 & 0.000 &  2.001 &  1756.518 \\
  \hline
  FedAvgM &       0.02 &     0.01 &         0.0 &    0.490 &       0.420 & 0.288 &  1.691 &  2726.681 \\
  FedAvgM &       0.02 &     0.01 &         0.7 &    0.760 &       0.750 & 0.696 &  1.265 &  2234.876 \\
  FedAvgM &       0.02 &     0.01 &         0.9 &    0.300 &       0.270 & 0.246 &  1.480 &  2774.526 \\
  \hline
   FedOpt &       0.02 &     0.01 &        1e-7 &    0.410 &       0.290 & 0.339 &  1.472 &  1289.025 \\
   FedOpt &       0.02 &     0.01 &        1e-8 &    0.310 &       0.190 & 0.287 &  0.621 &  2623.472 \\
   FedOpt &       0.02 &     0.01 &        1e-9 &    0.490 &       0.450 & 0.438 &  2.094 &  1287.257 \\
   \hline
  QFedAvg &       0.02 &     0.01 &         0.1 &    0.090 &       0.020 & 0.009 &  1.848 &  6051.387 \\
  QFedAvg &       0.02 &     0.01 &         0.2 &    0.160 &       0.170 & 0.031 &  1.664 &  3497.529 \\
  QFedAvg &       0.02 &     0.01 &         0.5 &    0.250 &       0.250 & 0.009 &  1.661 &  3789.658 \\
  \hline
  FedAvgM &       0.02 &        1 &         0.0 &    0.470 &       0.400 & 0.404 &  3.830 &  1649.486 \\
  FedAvgM &       0.02 &        1 &         0.7 &    0.170 &       0.100 & 0.115 &  4.632 &  2910.131 \\
  FedAvgM &       0.02 &        1 &         0.9 &    0.180 &       0.060 & 0.111 & 24.495 &  1434.800 \\
  \hline
   FedOpt &       0.02 &        1 &        1e-7 &    0.390 &       0.280 & 0.278 &  2.922 &  1287.952 \\
   FedOpt &       0.02 &        1 &        1e-8 &    0.470 &       0.360 & 0.423 &  2.036 &  2425.867 \\
   FedOpt &       0.02 &        1 &        1e-9 &    0.740 &       0.680 & 0.662 &  1.157 &  1282.801 \\
   \hline
  QFedAvg &       0.02 &        1 &         0.1 &    0.820 &       0.790 & 0.767 &  2.258 &  5826.659 \\
  QFedAvg &       0.02 &        1 &         0.2 &    0.940 &       0.930 & 0.915 &  0.567 &  3060.464 \\
  QFedAvg &       0.02 &        1 &         0.5 &    0.740 &       0.700 & 0.671 &  0.778 &  3124.191 \\
  \hline
   FedAvg &       0.05 &        - &           - &    0.710 &       0.630 & 0.630 &  0.988 &  1326.552 \\
  FedAvgM &       0.05 &    0.001 &         0.0 &    0.430 &       0.380 & 0.245 &  1.708 &  1370.562 \\
  FedAvgM &       0.05 &    0.001 &         0.7 &    0.130 &       0.090 & 0.023 &  1.716 &  1385.686 \\
  FedAvgM &       0.05 &    0.001 &         0.9 &    0.360 &       0.310 & 0.232 &  1.738 &  1996.326 \\
  \hline
   FedOpt &       0.05 &    0.001 &        1e-7 &    0.760 &       0.740 & 0.702 &  0.826 &  1639.983 \\
   FedOpt &       0.05 &    0.001 &        1e-8 &    0.900 &       0.890 & 0.868 &  1.766 &  3044.172 \\
   FedOpt &       0.05 &    0.001 &        1e-9 &    0.210 &       0.120 & 0.159 &  0.619 &  1290.653 \\
   \hline
  QFedAvg &       0.05 &    0.001 &         0.1 &        - &           - &     - &      - &  6455.773 \\
  QFedAvg &       0.05 &    0.001 &         0.2 &        - &           - &     - &      - &  3001.071 \\
  QFedAvg &       0.05 &    0.001 &         0.5 &    0.370 &       0.290 & 0.222 &  1.785 &  3146.799 \\
  \hline
  FedAvgM &       0.05 &     0.01 &         0.0 &    0.460 &       0.460 & 0.332 &  1.635 &  2940.077 \\
  FedAvgM &       0.05 &     0.01 &         0.7 &    0.620 &       0.620 & 0.523 &  1.276 &  2782.906 \\
  FedAvgM &       0.05 &     0.01 &         0.9 &    0.580 &       0.550 & 0.553 &  1.115 &  1996.281 \\
  \hline
   FedOpt &       0.05 &     0.01 &        1e-7 &    0.490 &       0.490 & 0.399 &  0.771 &  1286.672 \\
   FedOpt &       0.05 &     0.01 &        1e-8 &    0.540 &       0.490 & 0.450 &  1.101 &  2259.639 \\
   FedOpt &       0.05 &     0.01 &        1e-9 &    0.730 &       0.710 & 0.652 &  0.797 &  1282.435 \\
   \hline
  QFedAvg &       0.05 &     0.01 &         0.1 &    0.350 &       0.270 & 0.186 &  1.635 &  5419.123 \\
  QFedAvg &       0.05 &     0.01 &         0.2 &    0.480 &       0.380 & 0.336 &  1.632 &  1762.258 \\
  QFedAvg &       0.05 &     0.01 &         0.5 &        - &           - &     - &      - &  1881.516 \\
  \hline
  FedAvgM &       0.05 &        1 &         0.0 &    0.370 &       0.270 & 0.235 &  2.520 &  2707.163 \\
  FedAvgM &       0.05 &        1 &         0.7 &    0.550 &       0.490 & 0.488 &  3.241 &  2189.139 \\
  FedAvgM &       0.05 &        1 &         0.9 &    0.700 &       0.620 & 0.610 &  5.163 &  2720.669 \\
  \hline
   FedOpt &       0.05 &        1 &        1e-7 &    0.210 &       0.100 & 0.160 &  1.425 &  1291.017 \\
   FedOpt &       0.05 &        1 &        1e-8 &    0.470 &       0.390 & 0.363 &  1.345 &  2881.453 \\
   FedOpt &       0.05 &        1 &        1e-9 &    0.690 &       0.690 & 0.621 &  0.797 &  1288.712 \\
   \hline
  QFedAvg &       0.05 &        1 &         0.1 &    0.180 &       0.090 & 0.091 &  0.366 &  5669.815 \\
  QFedAvg &       0.05 &        1 &         0.2 &    0.490 &       0.480 & 0.412 &  0.516 &  3535.836 \\
  QFedAvg &       0.05 &        1 &         0.5 &    0.770 &       0.780 & 0.719 &  0.887 &  3823.680 \\
  \hline
   FedAvg &      0.075 &        - &           - &    0.320 &       0.310 & 0.263 &  2.975 &  2245.737 \\
  FedAvgM &      0.075 &    0.001 &         0.0 &    0.190 &       0.160 & 0.018 &  1.863 &  2690.760 \\
  FedAvgM &      0.075 &    0.001 &         0.7 &    0.320 &       0.280 & 0.126 &  1.718 &  2929.448 \\
  FedAvgM &      0.075 &    0.001 &         0.9 &    0.260 &       0.220 & 0.163 &  1.785 &  1837.243 \\
  \hline
   FedOpt &      0.075 &    0.001 &        1e-7 &    0.730 &       0.660 & 0.651 &  0.814 &  1149.189 \\
   FedOpt &      0.075 &    0.001 &        1e-8 &    0.720 &       0.730 & 0.668 &  0.707 &  2214.002 \\
   FedOpt &      0.075 &    0.001 &        1e-9 &    0.400 &       0.400 & 0.355 &  1.599 &  1288.940 \\
   \hline
  QFedAvg &      0.075 &    0.001 &         0.1 &        - &           - &     - &      - &  4943.256 \\
  QFedAvg &      0.075 &    0.001 &         0.2 &    0.490 &       0.440 & 0.329 &  1.716 &  1750.328 \\
  QFedAvg &      0.075 &    0.001 &         0.5 &    0.140 &       0.060 & 0.013 &  1.989 &  1773.635 \\
  \hline
  FedAvgM &      0.075 &     0.01 &         0.0 &    0.500 &       0.440 & 0.382 &  1.576 &  2738.733 \\
  FedAvgM &      0.075 &     0.01 &         0.7 &    0.540 &       0.540 & 0.483 &  1.387 &  2265.323 \\
  FedAvgM &      0.075 &     0.01 &         0.9 &    0.670 &       0.610 & 0.583 &  1.262 &  2727.130 \\
  \hline
   FedOpt &      0.075 &     0.01 &        1e-7 &    0.670 &       0.700 & 0.599 &  1.090 &  1285.646 \\
   FedOpt &      0.075 &     0.01 &        1e-8 &    0.720 &       0.700 & 0.666 &  1.705 &  2825.456 \\
   FedOpt &      0.075 &     0.01 &        1e-9 &    0.420 &       0.320 & 0.288 &  0.749 &  1284.441 \\
   \hline
  QFedAvg &      0.075 &     0.01 &         0.1 &    0.280 &       0.290 & 0.084 &  1.438 &  5743.474 \\
  QFedAvg &      0.075 &     0.01 &         0.2 &    0.390 &       0.310 & 0.225 &  1.690 &  3678.418 \\
  QFedAvg &      0.075 &     0.01 &         0.5 &    0.120 &       0.080 & 0.007 &  1.707 &  3882.005 \\
  \hline
  FedAvgM &      0.075 &        1 &         0.0 &    0.420 &       0.320 & 0.292 &  0.997 &  1620.301 \\
  FedAvgM &      0.075 &        1 &         0.7 &    0.340 &       0.300 & 0.285 &  2.314 &  2947.454 \\
  FedAvgM &      0.075 &        1 &         0.9 &    0.610 &       0.610 & 0.501 &  4.372 &  1408.044 \\
  \hline
   FedOpt &      0.075 &        1 &        1e-7 &    0.720 &       0.640 & 0.636 &  1.125 &  1286.284 \\
   FedOpt &      0.075 &        1 &        1e-8 &    0.730 &       0.700 & 0.669 &  0.610 &  2934.427 \\
   FedOpt &      0.075 &        1 &        1e-9 &    0.710 &       0.710 & 0.648 &  0.852 &  1290.130 \\
   \hline
  QFedAvg &      0.075 &        1 &         0.1 &    0.910 &       0.900 & 0.881 &  0.307 &  5275.194 \\
  QFedAvg &      0.075 &        1 &         0.2 &    0.770 &       0.710 & 0.715 &  0.523 &  3050.584 \\
  QFedAvg &      0.075 &        1 &         0.5 &    0.760 &       0.720 & 0.702 &  0.707 &  3575.840 \\
  \hline
   FedAvg &        0.1 &        - &           - &    0.890 &       0.870 & 0.859 &  1.607 &  2457.505 \\
  FedAvgM &        0.1 &    0.001 &         0.0 &    0.230 &       0.210 & 0.122 &  1.803 &  1534.532 \\
  FedAvgM &        0.1 &    0.001 &         0.7 &    0.250 &       0.150 & 0.177 &  2.073 &  2981.468 \\
  FedAvgM &        0.1 &    0.001 &         0.9 &    0.170 &       0.090 & 0.063 &  1.854 &  1419.741 \\
  \hline
   FedOpt &        0.1 &    0.001 &        1e-7 &    0.600 &       0.580 & 0.518 &  1.086 &  1145.380 \\
   FedOpt &        0.1 &    0.001 &        1e-8 &    0.880 &       0.860 & 0.848 &  0.960 &  3290.458 \\
   FedOpt &        0.1 &    0.001 &        1e-9 &    0.420 &       0.340 & 0.302 &  1.011 &  1290.873 \\
   \hline
  QFedAvg &        0.1 &    0.001 &         0.1 &        - &           - &     - &      - &  6137.342 \\
  QFedAvg &        0.1 &    0.001 &         0.2 &    0.090 &       0.030 & 0.003 &  2.016 &  3045.565 \\
  QFedAvg &        0.1 &    0.001 &         0.5 &    0.390 &       0.390 & 0.233 &  1.595 &  3093.148 \\
  \hline
  FedAvgM &        0.1 &     0.01 &         0.0 &    0.180 &       0.140 & 0.085 &  1.757 &  2710.664 \\
  FedAvgM &        0.1 &     0.01 &         0.7 &    0.790 &       0.780 & 0.724 &  1.134 &  2267.930 \\
  FedAvgM &        0.1 &     0.01 &         0.9 &    0.760 &       0.760 & 0.715 &  1.212 &  2930.887 \\
  \hline
   FedOpt &        0.1 &     0.01 &        1e-7 &    0.890 &       0.890 & 0.855 &  0.400 &  1292.593 \\
   FedOpt &        0.1 &     0.01 &        1e-8 &    0.660 &       0.630 & 0.635 &  0.898 &  2392.280 \\
   FedOpt &        0.1 &     0.01 &        1e-9 &    0.900 &       0.900 & 0.872 &  1.297 &  1282.900 \\
   \hline
  QFedAvg &        0.1 &     0.01 &         0.1 &    0.090 &       0.020 & 0.005 &  1.758 &  5914.353 \\
  QFedAvg &        0.1 &     0.01 &         0.2 &    0.170 &       0.180 & 0.042 &  1.597 &  3266.261 \\
  QFedAvg &        0.1 &     0.01 &         0.5 &    0.220 &       0.220 & 0.006 &  1.703 &  2661.993 \\
  \hline
  FedAvgM &        0.1 &        1 &         0.0 &    0.420 &       0.340 & 0.311 &  0.353 &  2321.703 \\
  FedAvgM &        0.1 &        1 &         0.7 &    0.480 &       0.430 & 0.387 &  0.325 &  1418.648 \\
  FedAvgM &        0.1 &        1 &         0.9 &    0.690 &       0.610 & 0.600 &  3.476 &  1401.674 \\
  \hline
   FedOpt &        0.1 &        1 &        1e-7 &    0.770 &       0.780 & 0.720 &  0.431 &  1281.467 \\
   FedOpt &        0.1 &        1 &        1e-8 &    0.610 &       0.570 & 0.485 &  0.883 &  2397.014 \\
   FedOpt &        0.1 &        1 &        1e-9 &    0.600 &       0.530 & 0.576 &  0.410 &  1288.406 \\
   \hline
  QFedAvg &        0.1 &        1 &         0.1 &    0.720 &       0.730 & 0.667 &  0.286 &  6255.950 \\
  QFedAvg &        0.1 &        1 &         0.2 &    0.910 &       0.910 & 0.879 &  0.330 &  3261.883 \\
  QFedAvg &        0.1 &        1 &         0.5 &    0.830 &       0.820 & 0.775 &  0.583 &  3499.362 \\
  \hline
   FedAvg &        0.2 &        - &           - &    0.810 &       0.760 & 0.762 &  0.222 &  2794.929 \\
  FedAvgM &        0.2 &    0.001 &         0.0 &    0.260 &       0.240 & 0.131 &  1.777 &  2122.969 \\
  FedAvgM &        0.2 &    0.001 &         0.7 &    0.380 &       0.380 & 0.154 &  1.627 &  2937.087 \\
  FedAvgM &        0.2 &    0.001 &         0.9 &    0.320 &       0.290 & 0.141 &  1.747 &  2376.400 \\
  \hline
   FedOpt &        0.2 &    0.001 &        1e-7 &    0.840 &       0.780 & 0.803 &  0.410 &  1151.226 \\
   FedOpt &        0.2 &    0.001 &        1e-8 &    0.810 &       0.770 & 0.766 &  0.201 &  2885.618 \\
   FedOpt &        0.2 &    0.001 &        1e-9 &    0.790 &       0.730 & 0.728 &  0.721 &  1282.361 \\
   \hline
  QFedAvg &        0.2 &    0.001 &         0.1 &        - &           - &     - &      - &  6550.722 \\
  QFedAvg &        0.2 &    0.001 &         0.2 &    0.270 &       0.220 & 0.024 &  1.713 &  2413.979 \\
  QFedAvg &        0.2 &    0.001 &         0.5 &        - &           - &     - &      - &  2937.973 \\
  \hline
  FedAvgM &        0.2 &     0.01 &         0.0 &    0.370 &       0.400 & 0.256 &  1.646 &  1422.083 \\
  FedAvgM &        0.2 &     0.01 &         0.7 &    0.770 &       0.750 & 0.707 &  1.137 &  1396.077 \\
  FedAvgM &        0.2 &     0.01 &         0.9 &    0.910 &       0.900 & 0.877 &  0.769 &  1435.639 \\
  \hline
   FedOpt &        0.2 &     0.01 &        1e-7 &    0.790 &       0.730 & 0.735 &  0.191 &  1290.602 \\
   FedOpt &        0.2 &     0.01 &        1e-8 &    0.910 &       0.900 & 0.889 &  0.430 &  2966.447 \\
   FedOpt &        0.2 &     0.01 &        1e-9 &    0.820 &       0.780 & 0.786 &  0.801 &  1782.568 \\
   \hline
  QFedAvg &        0.2 &     0.01 &         0.1 &    0.280 &       0.210 & 0.050 &  1.592 &  6269.745 \\
  QFedAvg &        0.2 &     0.01 &         0.2 &    0.240 &       0.230 & 0.067 &  1.465 &  3015.548 \\
  QFedAvg &        0.2 &     0.01 &         0.5 &    0.340 &       0.210 & 0.117 &  1.771 &  3505.382 \\
  \hline
  FedAvgM &        0.2 &        1 &         0.0 &    0.840 &       0.840 & 0.800 &  0.723 &  2885.397 \\
  FedAvgM &        0.2 &        1 &         0.7 &    0.800 &       0.760 & 0.757 &  1.960 &  2361.988 \\
  FedAvgM &        0.2 &        1 &         0.9 &    0.710 &       0.610 & 0.617 &  1.557 &  2827.480 \\
  \hline
   FedOpt &        0.2 &        1 &        1e-7 &    0.830 &       0.780 & 0.797 &  0.336 &  1288.246 \\
   FedOpt &        0.2 &        1 &        1e-8 &    0.520 &       0.470 & 0.396 &  1.755 &  2275.048 \\
   FedOpt &        0.2 &        1 &        1e-9 &    0.840 &       0.820 & 0.799 &  1.015 &  1281.978 \\
   \hline
  QFedAvg &        0.2 &        1 &         0.1 &    0.850 &       0.840 & 0.811 &  0.436 &  5495.455 \\
  QFedAvg &        0.2 &        1 &         0.2 &    0.930 &       0.920 & 0.905 &  0.421 &  1755.648 \\
  QFedAvg &        0.2 &        1 &         0.5 &    0.900 &       0.900 & 0.864 &  0.608 &  1770.797 \\
  \hline
   FedAvg &        0.5 &        - &           - &    0.780 &       0.780 & 0.742 &  1.386 &  1590.821 \\
  FedAvgM &        0.5 &    0.001 &         0.0 &    0.250 &       0.200 & 0.024 &  1.888 &  2934.613 \\
  FedAvgM &        0.5 &    0.001 &         0.7 &    0.220 &       0.160 & 0.067 &  1.967 &  1417.634 \\
  FedAvgM &        0.5 &    0.001 &         0.9 &    0.170 &       0.110 & 0.097 &  1.842 &  1360.160 \\
  \hline
   FedOpt &        0.5 &    0.001 &        1e-7 &    0.950 &       0.950 & 0.940 &  0.175 &  1154.104 \\
   FedOpt &        0.5 &    0.001 &        1e-8 &    0.830 &       0.790 & 0.802 &  0.858 &  3145.256 \\
   FedOpt &        0.5 &    0.001 &        1e-9 &    0.950 &       0.950 & 0.931 &  0.192 &  1290.717 \\
   \hline
  QFedAvg &        0.5 &    0.001 &         0.1 &        - &           - &     - &      - &  6060.063 \\
  QFedAvg &        0.5 &    0.001 &         0.2 &        - &           - &     - &      - &  3540.662 \\
  QFedAvg &        0.5 &    0.001 &         0.5 &    0.250 &       0.220 & 0.009 &  1.749 &  2739.744 \\
  \hline
  FedAvgM &        0.5 &     0.01 &         0.0 &    0.630 &       0.600 & 0.515 &  1.533 &  2756.052 \\
  FedAvgM &        0.5 &     0.01 &         0.7 &    0.390 &       0.360 & 0.294 &  1.455 &  2079.197 \\
  FedAvgM &        0.5 &     0.01 &         0.9 &    0.910 &       0.910 & 0.877 &  0.795 &  2615.035 \\
  \hline
   FedOpt &        0.5 &     0.01 &        1e-7 &    0.990 &       0.990 & 0.986 &  0.378 &  1292.924 \\
   FedOpt &        0.5 &     0.01 &        1e-8 &    0.900 &       0.900 & 0.875 &  0.264 &  2363.842 \\
   FedOpt &        0.5 &     0.01 &        1e-9 &    0.950 &       0.940 & 0.929 &  0.216 &  1788.138 \\
   \hline
  QFedAvg &        0.5 &     0.01 &         0.1 &    0.350 &       0.240 & 0.096 &  1.473 &  5930.990 \\
  QFedAvg &        0.5 &     0.01 &         0.2 &    0.140 &       0.100 & 0.023 &  1.864 &  2571.927 \\
  QFedAvg &        0.5 &     0.01 &         0.5 &    0.310 &       0.290 & 0.130 &  1.754 &  3338.041 \\
  \hline
  FedAvgM &        0.5 &        1 &         0.0 &    0.540 &       0.510 & 0.493 &  0.470 &  1517.458 \\
  FedAvgM &        0.5 &        1 &         0.7 &    0.830 &       0.770 & 0.791 &  0.714 &  1379.968 \\
  FedAvgM &        0.5 &        1 &         0.9 &    0.880 &       0.850 & 0.840 &  0.841 &  1362.437 \\
  \hline
   FedOpt &        0.5 &        1 &        1e-7 &    0.920 &       0.930 & 0.904 &  0.957 &  1287.756 \\
   FedOpt &        0.5 &        1 &        1e-8 &    0.590 &       0.570 & 0.546 &  0.986 &  2607.388 \\
   FedOpt &        0.5 &        1 &        1e-9 &    0.950 &       0.950 & 0.938 &  0.330 &  1282.426 \\
   \hline
  QFedAvg &        0.5 &        1 &         0.1 &    0.920 &       0.920 & 0.890 &  0.291 &  6244.481 \\
  QFedAvg &        0.5 &        1 &         0.2 &    0.860 &       0.840 & 0.814 &  0.335 &  3172.700 \\
  QFedAvg &        0.5 &        1 &         0.5 &    0.790 &       0.770 & 0.736 &  0.619 &  3473.126 \\
  \hline
   FedAvg &          1 &        - &           - &    0.820 &       0.780 & 0.790 &  0.522 &  1392.695 \\
  FedAvgM &          1 &    0.001 &         0.0 &    0.220 &       0.230 & 0.028 &  1.817 &  1294.054 \\
  FedAvgM &          1 &    0.001 &         0.7 &    0.170 &       0.070 & 0.039 &  1.774 &  1286.387 \\
  FedAvgM &          1 &    0.001 &         0.9 &    0.240 &       0.230 & 0.123 &  1.867 &  1292.964 \\
  \hline
   FedOpt &          1 &    0.001 &        1e-7 &    0.980 &       0.980 & 0.972 &  0.165 &  1648.930 \\
   FedOpt &          1 &    0.001 &        1e-8 &    0.910 &       0.910 & 0.884 &  0.424 &  2629.902 \\
   FedOpt &          1 &    0.001 &        1e-9 &    0.830 &       0.780 & 0.795 &  0.564 &  1286.318 \\
   \hline
  QFedAvg &          1 &    0.001 &         0.1 &    0.170 &       0.100 & 0.001 &  1.876 &  4322.234 \\
  QFedAvg &          1 &    0.001 &         0.2 &        - &           - &     - &      - &  1629.183 \\
  QFedAvg &          1 &    0.001 &         0.5 &    0.180 &       0.100 & 0.036 &  1.944 &  1622.808 \\
  \hline
  FedAvgM &          1 &     0.01 &         0.0 &    0.650 &       0.570 & 0.531 &  1.473 &  1288.067 \\
  FedAvgM &          1 &     0.01 &         0.7 &    0.810 &       0.820 & 0.760 &  1.008 &  1296.378 \\
  FedAvgM &          1 &     0.01 &         0.9 &    0.870 &       0.870 & 0.832 &  0.967 &  1288.944 \\
  \hline
   FedOpt &          1 &     0.01 &        1e-7 &    0.960 &       0.960 & 0.949 &  0.188 &  1284.544 \\
   FedOpt &          1 &     0.01 &        1e-8 &    0.960 &       0.970 & 0.953 &  0.695 &  2380.830 \\
   FedOpt &          1 &     0.01 &        1e-9 &    0.950 &       0.950 & 0.931 &  0.280 &  1285.729 \\
   \hline
  QFedAvg &          1 &     0.01 &         0.1 &    0.390 &       0.310 & 0.207 &  1.627 &  4613.593 \\
  QFedAvg &          1 &     0.01 &         0.2 &    0.330 &       0.240 & 0.053 &  1.656 &  1619.108 \\
  QFedAvg &          1 &     0.01 &         0.5 &    0.250 &       0.210 & 0.019 &  1.728 &  1617.979 \\
  \hline
  FedAvgM &          1 &        1 &         0.0 &    0.780 &       0.750 & 0.742 &  1.630 &  1285.239 \\
  FedAvgM &          1 &        1 &         0.7 &    0.840 &       0.790 & 0.807 &  2.000 &  1286.675 \\
  FedAvgM &          1 &        1 &         0.9 &    0.800 &       0.750 & 0.740 &  0.940 &  1291.021 \\
  \hline
   FedOpt &          1 &        1 &        1e-7 &    0.970 &       0.970 & 0.964 &  0.214 &  1279.161 \\
   FedOpt &          1 &        1 &        1e-8 &    0.960 &       0.960 & 0.941 &  0.201 &  2637.563 \\
   FedOpt &          1 &        1 &        1e-9 &    0.900 &       0.900 & 0.868 &  0.217 &  1288.664 \\
   \hline
  QFedAvg &          1 &        1 &         0.1 &    0.860 &       0.860 & 0.818 &  0.285 &  4925.205 \\
  QFedAvg &          1 &        1 &         0.2 &    0.910 &       0.900 & 0.883 &  0.341 &  1634.129 \\
  QFedAvg &          1 &        1 &         0.5 &    0.810 &       0.800 & 0.756 &  0.602 &  1622.623 \\
  \hline
   FedAvg &    1000000 &        - &           - &    0.930 &       0.930 & 0.913 &  0.735 &  1353.343 \\
  FedAvgM &    1000000 &    0.001 &         0.0 &    0.180 &       0.150 & 0.068 &  1.877 &  1288.517 \\
  FedAvgM &    1000000 &    0.001 &         0.7 &    0.300 &       0.290 & 0.133 &  1.768 &  1289.281 \\
  FedAvgM &    1000000 &    0.001 &         0.9 &    0.260 &       0.240 & 0.222 &  1.791 &  1291.179 \\
  \hline
   FedOpt &    1000000 &    0.001 &        1e-7 &    0.990 &       0.990 & 0.989 &  0.199 &  1153.495 \\
   FedOpt &    1000000 &    0.001 &        1e-8 &    0.990 &       0.990 & 0.988 &  0.151 &  2444.150 \\
   FedOpt &    1000000 &    0.001 &        1e-9 &    1.000 &       1.000 & 0.996 &  0.124 &  1280.262 \\
   \hline
  QFedAvg &    1000000 &    0.001 &         0.1 &    0.170 &       0.060 & 0.058 &  1.893 &  4304.251 \\
  QFedAvg &    1000000 &    0.001 &         0.2 &        - &           - &     - &      - &  1631.003 \\
  QFedAvg &    1000000 &    0.001 &         0.5 &        - &           - &     - &      - &  1623.545 \\
  \hline
  FedAvgM &    1000000 &     0.01 &         0.0 &    0.420 &       0.390 & 0.304 &  1.605 &  1288.961 \\
  FedAvgM &    1000000 &     0.01 &         0.7 &    0.630 &       0.650 & 0.531 &  1.309 &  1297.892 \\
  FedAvgM &    1000000 &     0.01 &         0.9 &    0.910 &       0.920 & 0.884 &  0.886 &  1291.020 \\
  \hline
   FedOpt &    1000000 &     0.01 &        1e-7 &    0.990 &       0.990 & 0.985 &  0.147 &  1294.049 \\
   FedOpt &    1000000 &     0.01 &        1e-8 &    0.850 &       0.810 & 0.826 &  0.545 &  2675.734 \\
   FedOpt &    1000000 &     0.01 &        1e-9 &    0.980 &       0.980 & 0.974 &  0.156 &  1780.395 \\
   \hline
  QFedAvg &    1000000 &     0.01 &         0.1 &    0.190 &       0.200 & 0.067 &  1.560 &  4615.418 \\
  QFedAvg &    1000000 &     0.01 &         0.2 &    0.330 &       0.260 & 0.074 &  1.502 &  1630.391 \\
  QFedAvg &    1000000 &     0.01 &         0.5 &    0.210 &       0.150 & 0.024 &  1.758 &  1634.046 \\
  \hline
  FedAvgM &    1000000 &        1 &         0.0 &    0.830 &       0.780 & 0.795 &  0.279 &  1293.985 \\
  FedAvgM &    1000000 &        1 &         0.7 &    0.960 &       0.960 & 0.948 &  0.220 &  1299.432 \\
  FedAvgM &    1000000 &        1 &         0.9 &    0.950 &       0.950 & 0.929 &  1.079 &  1301.024 \\
  \hline
   FedOpt &    1000000 &        1 &        1e-7 &    0.980 &       0.980 & 0.973 &  0.182 &  1290.299 \\
   FedOpt &    1000000 &        1 &        1e-8 &    1.000 &       1.000 & 0.994 &  0.154 &  2237.058 \\
   FedOpt &    1000000 &        1 &        1e-9 &    1.000 &       1.000 & 0.995 &  0.162 &  1289.716 \\
   \hline
  QFedAvg &    1000000 &        1 &         0.1 &    0.960 &       0.960 & 0.945 &  0.276 &  4311.827 \\
  QFedAvg &    1000000 &        1 &         0.2 &    0.960 &       0.960 & 0.948 &  0.315 &  1631.578 \\
  QFedAvg &    1000000 &        1 &         0.5 &    0.820 &       0.830 & 0.778 &  0.516 &  1630.251 \\
  \hline







</pre> 




<h2>Results LSTM:</h2>

<pre>





FL & Alpha & S-LR & FL Param  & $R^2$ & MSE & MAE & Loss & Runtime (s) \\
\hline
\midrule
   FedAvg &      0.001 &        - &           - & 0.711 & 0.029 & 0.128 & 0.014 &  4018.598 \\
  FedAvgM &      0.001 &    0.001 &         0.0 & 0.979 & 0.197 & 0.403 & 0.098 &  3765.202 \\
  FedAvgM &      0.001 &    0.001 &         0.7 & 0.939 & 0.193 & 0.401 & 0.096 &  3932.952 \\
  FedAvgM &      0.001 &    0.001 &         0.9 & 0.784 & 0.177 & 0.385 & 0.089 &  3960.095 \\
  \hline
   FedOpt &      0.001 &    0.001 &        1e-7 & 0.699 & 0.030 & 0.129 & 0.015 &  4048.795 \\
   FedOpt &      0.001 &    0.001 &        1e-8 & 0.669 & 0.033 & 0.135 & 0.017 &  3715.132 \\
   FedOpt &      0.001 &    0.001 &        1e-9 & 0.685 & 0.031 & 0.133 & 0.016 &  3705.501 \\
   \hline
  QFedAvg &      0.001 &    0.001 &         0.1 &     - &     - &     - &     - &   9279.19 \\
  QFedAvg &      0.001 &    0.001 &         0.2 &     - &     - &     - &     - &  4657.118 \\
  QFedAvg &      0.001 &    0.001 &         0.5 &     - &     - &     - &     - &   4792.91 \\
  \hline
  FedAvgM &      0.001 &     0.01 &         0.0 & 0.811 & 0.180 & 0.385 & 0.090 &  3973.814 \\
  FedAvgM &      0.001 &     0.01 &         0.7 & 0.451 & 0.144 & 0.346 & 0.072 &  4010.215 \\
  FedAvgM &      0.001 &     0.01 &         0.9 & 0.219 & 0.078 & 0.226 & 0.039 &  3826.738 \\
  \hline
   FedOpt &      0.001 &     0.01 &        1e-7 & 0.730 & 0.027 & 0.121 & 0.014 &   3702.29 \\
   FedOpt &      0.001 &     0.01 &        1e-8 & 0.719 & 0.028 & 0.123 & 0.014 &  3718.754 \\
   FedOpt &      0.001 &     0.01 &        1e-9 & 0.699 & 0.030 & 0.128 & 0.015 &  3691.368 \\
   \hline
  QFedAvg &      0.001 &     0.01 &         0.1 & 0.953 & 0.194 & 0.402 & 0.097 &    9414.5 \\
  QFedAvg &      0.001 &     0.01 &         0.2 & 0.979 & 0.197 & 0.405 & 0.098 &  4521.043 \\
  QFedAvg &      0.001 &     0.01 &         0.5 &     - &     - &     - &     - &  4712.592 \\
  \hline
  FedAvgM &      0.001 &        1 &         0.0 & 0.710 & 0.029 & 0.123 & 0.014 &   3916.03 \\
  FedAvgM &      0.001 &        1 &         0.7 & 0.691 & 0.031 & 0.127 & 0.015 &  4302.678 \\
  FedAvgM &      0.001 &        1 &         0.9 & 0.606 & 0.039 & 0.138 & 0.020 &  4894.611 \\
  \hline
   FedOpt &      0.001 &        1 &        1e-7 & 0.701 & 0.030 & 0.128 & 0.015 &  3717.609 \\
   FedOpt &      0.001 &        1 &        1e-8 & 0.692 & 0.031 & 0.130 & 0.015 &  3728.364 \\
   FedOpt &      0.001 &        1 &        1e-9 & 0.718 & 0.028 & 0.124 & 0.014 &  3696.752 \\
   \hline
  QFedAvg &      0.001 &        1 &         0.1 & 0.313 & 0.068 & 0.218 & 0.034 &   8484.12 \\
  QFedAvg &      0.001 &        1 &         0.2 & 0.011 & 0.098 & 0.275 & 0.049 &  4772.248 \\
  QFedAvg &      0.001 &        1 &         0.5 & 0.441 & 0.143 & 0.345 & 0.072 &  4955.259 \\
  \hline
   FedAvg &      0.005 &        - &           - & 0.704 & 0.030 & 0.125 & 0.015 &  9071.398 \\
  FedAvgM &      0.005 &    0.001 &         0.0 & 0.978 & 0.198 & 0.398 & 0.099 &  3915.442 \\
  FedAvgM &      0.005 &    0.001 &         0.7 & 0.925 & 0.193 & 0.399 & 0.096 &  3955.573 \\
  FedAvgM &      0.005 &    0.001 &         0.9 & 0.782 & 0.178 & 0.384 & 0.089 &  3950.564 \\
  \hline
   FedOpt &      0.005 &    0.001 &        1e-7 & 0.702 & 0.030 & 0.127 & 0.015 &  4142.367 \\
   FedOpt &      0.005 &    0.001 &        1e-8 & 0.724 & 0.028 & 0.122 & 0.014 &  3718.092 \\
   FedOpt &      0.005 &    0.001 &        1e-9 & 0.726 & 0.027 & 0.120 & 0.014 &  3742.055 \\
   \hline
  QFedAvg &      0.005 &    0.001 &         0.1 &     - &     - &     - &     - &  4728.792 \\
  QFedAvg &      0.005 &    0.001 &         0.2 &     - &     - &     - &     - &  4757.528 \\
  QFedAvg &      0.005 &    0.001 &         0.5 &     - &     - &     - &     - &  4740.476 \\
  \hline
  FedAvgM &      0.005 &     0.01 &         0.0 & 0.823 & 0.182 & 0.389 & 0.091 &  3979.118 \\
  FedAvgM &      0.005 &     0.01 &         0.7 & 0.358 & 0.136 & 0.328 & 0.068 &  3963.438 \\
  FedAvgM &      0.005 &     0.01 &         0.9 & 0.273 & 0.073 & 0.215 & 0.036 &  4001.067 \\
  \hline
   FedOpt &      0.005 &     0.01 &        1e-7 & 0.651 & 0.035 & 0.140 & 0.017 &  3704.574 \\
   FedOpt &      0.005 &     0.01 &        1e-8 & 0.709 & 0.029 & 0.126 & 0.015 &   3732.32 \\
   FedOpt &      0.005 &     0.01 &        1e-9 & 0.712 & 0.029 & 0.123 & 0.014 &  3721.665 \\
   \hline
  QFedAvg &      0.005 &     0.01 &         0.1 & 0.947 & 0.195 & 0.403 & 0.097 &  4734.855 \\
  QFedAvg &      0.005 &     0.01 &         0.2 & 0.976 & 0.198 & 0.408 & 0.099 &  4765.194 \\
  QFedAvg &      0.005 &     0.01 &         0.5 &     - &     - &     - &     - &  4460.221 \\
  \hline
  FedAvgM &      0.005 &        1 &         0.0 & 0.685 & 0.031 & 0.132 & 0.016 &  3690.438 \\
  FedAvgM &      0.005 &        1 &         0.7 & 0.680 & 0.032 & 0.123 & 0.016 &  3715.286 \\
  FedAvgM &      0.005 &        1 &         0.9 & 0.622 & 0.038 & 0.138 & 0.019 &  3701.955 \\
  \hline
   FedOpt &      0.005 &        1 &        1e-7 & 0.702 & 0.030 & 0.124 & 0.015 &  3695.399 \\
   FedOpt &      0.005 &        1 &        1e-8 & 0.699 & 0.030 & 0.128 & 0.015 &      None \\
   FedOpt &      0.005 &        1 &        1e-9 & 0.700 & 0.030 & 0.126 & 0.015 &  3736.308 \\
   \hline
  QFedAvg &      0.005 &        1 &         0.1 & 0.428 & 0.057 & 0.194 & 0.029 &  4401.996 \\
  QFedAvg &      0.005 &        1 &         0.2 & 0.062 & 0.094 & 0.270 & 0.047 &  4410.633 \\
  QFedAvg &      0.005 &        1 &         0.5 & 0.375 & 0.138 & 0.334 & 0.069 &  4606.923 \\
  \hline
   FedAvg &       0.01 &        - &           - & 0.738 & 0.026 & 0.122 & 0.013 &  4099.564 \\
  FedAvgM &       0.01 &    0.001 &         0.0 & 0.979 & 0.197 & 0.409 & 0.098 &  3821.408 \\
  FedAvgM &       0.01 &    0.001 &         0.7 & 0.931 & 0.192 & 0.402 & 0.096 &  3978.492 \\
  FedAvgM &       0.01 &    0.001 &         0.9 & 0.788 & 0.178 & 0.388 & 0.089 &  4944.231 \\
  \hline
   FedOpt &       0.01 &    0.001 &        1e-7 & 0.708 & 0.029 & 0.122 & 0.015 &   4119.19 \\
   FedOpt &       0.01 &    0.001 &        1e-8 & 0.676 & 0.032 & 0.134 & 0.016 &  3687.157 \\
   FedOpt &       0.01 &    0.001 &        1e-9 & 0.727 & 0.027 & 0.121 & 0.014 &  3687.722 \\
   \hline
  QFedAvg &       0.01 &    0.001 &         0.1 &     - &     - &     - &     - &  7983.825 \\
  QFedAvg &       0.01 &    0.001 &         0.2 &     - &     - &     - &     - &  4687.169 \\
  QFedAvg &       0.01 &    0.001 &         0.5 &     - &     - &     - &     - &  4732.988 \\
  \hline
  FedAvgM &       0.01 &     0.01 &         0.0 & 0.793 & 0.178 & 0.384 & 0.089 &  3971.801 \\
  FedAvgM &       0.01 &     0.01 &         0.7 & 0.393 & 0.138 & 0.330 & 0.069 &  3997.806 \\
  FedAvgM &       0.01 &     0.01 &         0.9 & 0.231 & 0.076 & 0.221 & 0.038 &  4777.871 \\
  \hline
   FedOpt &       0.01 &     0.01 &        1e-7 & 0.710 & 0.029 & 0.124 & 0.014 &  3708.456 \\
   FedOpt &       0.01 &     0.01 &        1e-8 & 0.710 & 0.029 & 0.125 & 0.014 &  3698.401 \\
   FedOpt &       0.01 &     0.01 &        1e-9 & 0.687 & 0.031 & 0.130 & 0.016 &  3695.516 \\
   \hline
  QFedAvg &       0.01 &     0.01 &         0.1 & 0.950 & 0.194 & 0.402 & 0.097 &  8477.221 \\
  QFedAvg &       0.01 &     0.01 &         0.2 & 0.973 & 0.196 & 0.403 & 0.098 &  4645.626 \\
  QFedAvg &       0.01 &     0.01 &         0.5 &     - &     - &     - &     - &  4599.221 \\
  \hline
  FedAvgM &       0.01 &        1 &         0.0 & 0.697 & 0.030 & 0.127 & 0.015 &  3976.748 \\
  FedAvgM &       0.01 &        1 &         0.7 & 0.764 & 0.023 & 0.101 & 0.012 &  3978.477 \\
  FedAvgM &       0.01 &        1 &         0.9 & 0.645 & 0.035 & 0.129 & 0.018 &   4944.88 \\
  \hline
   FedOpt &       0.01 &        1 &        1e-7 & 0.705 & 0.029 & 0.123 & 0.015 &  3684.658 \\
   FedOpt &       0.01 &        1 &        1e-8 & 0.720 & 0.028 & 0.124 & 0.014 &  3692.614 \\
   FedOpt &       0.01 &        1 &        1e-9 & 0.695 & 0.030 & 0.125 & 0.015 &   3700.76 \\
   \hline
  QFedAvg &       0.01 &        1 &         0.1 & 0.345 & 0.065 & 0.210 & 0.033 &  8119.071 \\
  QFedAvg &       0.01 &        1 &         0.2 & 0.069 & 0.093 & 0.265 & 0.046 &  4728.696 \\
  QFedAvg &       0.01 &        1 &         0.5 & 0.335 & 0.133 & 0.332 & 0.066 &  4921.196 \\
  \hline
   FedAvg &       0.02 &        - &           - & 0.732 & 0.027 & 0.125 & 0.013 &  4154.424 \\
  FedAvgM &       0.02 &    0.001 &         0.0 & 0.980 & 0.197 & 0.405 & 0.098 &  3757.283 \\
  FedAvgM &       0.02 &    0.001 &         0.7 & 0.937 & 0.193 & 0.400 & 0.096 &  3956.726 \\
  FedAvgM &       0.02 &    0.001 &         0.9 & 0.786 & 0.178 & 0.386 & 0.089 &  4857.583 \\
  \hline
   FedOpt &       0.02 &    0.001 &        1e-7 & 0.701 & 0.030 & 0.129 & 0.015 &  4154.633 \\
   FedOpt &       0.02 &    0.001 &        1e-8 & 0.707 & 0.029 & 0.125 & 0.015 &  3720.642 \\
   FedOpt &       0.02 &    0.001 &        1e-9 & 0.696 & 0.030 & 0.127 & 0.015 &  3731.696 \\
   \hline
  QFedAvg &       0.02 &    0.001 &         0.1 &     - &     - &     - &     - &  8339.512 \\
  QFedAvg &       0.02 &    0.001 &         0.2 &     - &     - &     - &     - &  4658.012 \\
  QFedAvg &       0.02 &    0.001 &         0.5 &     - &     - &     - &     - &  4711.791 \\
  \hline
  FedAvgM &       0.02 &     0.01 &         0.0 & 0.794 & 0.178 & 0.374 & 0.089 &   3942.93 \\
  FedAvgM &       0.02 &     0.01 &         0.7 & 0.426 & 0.142 & 0.342 & 0.071 &  3955.044 \\
  FedAvgM &       0.02 &     0.01 &         0.9 & 0.252 & 0.074 & 0.213 & 0.037 &  4703.602 \\
  \hline
   FedOpt &       0.02 &     0.01 &        1e-7 & 0.711 & 0.029 & 0.126 & 0.014 &  3710.572 \\
   FedOpt &       0.02 &     0.01 &        1e-8 & 0.703 & 0.030 & 0.127 & 0.015 &  3723.871 \\
   FedOpt &       0.02 &     0.01 &        1e-9 & 0.668 & 0.033 & 0.135 & 0.017 &  3755.415 \\
   \hline
  QFedAvg &       0.02 &     0.01 &         0.1 & 0.951 & 0.194 & 0.400 & 0.097 &  8468.831 \\
  QFedAvg &       0.02 &     0.01 &         0.2 & 0.976 & 0.196 & 0.403 & 0.098 &  4727.564 \\
  QFedAvg &       0.02 &     0.01 &         0.5 &     - &     - &     - &     - &  4595.178 \\
  \hline
  FedAvgM &       0.02 &        1 &         0.0 & 0.735 & 0.026 & 0.119 & 0.013 &  3990.567 \\
  FedAvgM &       0.02 &        1 &         0.7 & 0.730 & 0.027 & 0.113 & 0.013 &  4003.764 \\
  FedAvgM &       0.02 &        1 &         0.9 & 0.676 & 0.032 & 0.124 & 0.016 &  4904.264 \\
  \hline
   FedOpt &       0.02 &        1 &        1e-7 & 0.694 & 0.031 & 0.130 & 0.015 &  3715.024 \\
   FedOpt &       0.02 &        1 &        1e-8 & 0.674 & 0.033 & 0.132 & 0.016 &  3733.394 \\
   FedOpt &       0.02 &        1 &        1e-9 & 0.712 & 0.029 & 0.123 & 0.014 &  3725.746 \\
   \hline
  QFedAvg &       0.02 &        1 &         0.1 & 0.503 & 0.049 & 0.174 & 0.025 &  8404.705 \\
  QFedAvg &       0.02 &        1 &         0.2 & 0.193 & 0.080 & 0.242 & 0.040 &  4712.407 \\
  QFedAvg &       0.02 &        1 &         0.5 & 0.329 & 0.132 & 0.329 & 0.066 &  4914.408 \\
   FedAvg &       0.05 &        - &           - & 0.724 & 0.028 & 0.123 & 0.014 &  3642.169 \\
   \hline
  FedAvgM &       0.05 &    0.001 &         0.0 & 0.981 & 0.198 & 0.405 & 0.099 &  3681.653 \\
  FedAvgM &       0.05 &    0.001 &         0.7 & 0.928 & 0.193 & 0.399 & 0.096 &  3676.814 \\
  FedAvgM &       0.05 &    0.001 &         0.9 & 0.774 & 0.177 & 0.385 & 0.089 &  3686.945 \\
  \hline
   FedOpt &       0.05 &    0.001 &        1e-7 & 0.697 & 0.030 & 0.128 & 0.015 &  3670.005 \\
   FedOpt &       0.05 &    0.001 &        1e-8 & 0.711 & 0.029 & 0.126 & 0.014 &  3691.721 \\
   FedOpt &       0.05 &    0.001 &        1e-9 & 0.685 & 0.032 & 0.133 & 0.016 &  3712.324 \\
   \hline
  QFedAvg &       0.05 &    0.001 &         0.1 &     - &     - &     - &     - &  4392.969 \\
  QFedAvg &       0.05 &    0.001 &         0.2 &     - &     - &     - &     - &   4391.07 \\
  QFedAvg &       0.05 &    0.001 &         0.5 &     - &     - &     - &     - &  4379.599 \\
  \hline
  FedAvgM &       0.05 &     0.01 &         0.0 & 0.834 & 0.183 & 0.391 & 0.092 &  3693.714 \\
  FedAvgM &       0.05 &     0.01 &         0.7 & 0.413 & 0.141 & 0.342 & 0.071 &   3685.45 \\
  FedAvgM &       0.05 &     0.01 &         0.9 & 0.325 & 0.067 & 0.207 & 0.034 &   3697.96 \\
  \hline
   FedOpt &       0.05 &     0.01 &        1e-7 & 0.713 & 0.029 & 0.123 & 0.014 &  3681.093 \\
   FedOpt &       0.05 &     0.01 &        1e-8 & 0.683 & 0.032 & 0.128 & 0.016 &  3700.408 \\
   FedOpt &       0.05 &     0.01 &        1e-9 & 0.691 & 0.031 & 0.132 & 0.015 &  3720.563 \\
   \hline
  QFedAvg &       0.05 &     0.01 &         0.1 & 0.942 & 0.194 & 0.399 & 0.097 &  4429.756 \\
  QFedAvg &       0.05 &     0.01 &         0.2 & 0.978 & 0.198 & 0.405 & 0.099 &  4405.635 \\
  QFedAvg &       0.05 &     0.01 &         0.5 &     - &     - &     - &     - &  4362.115 \\
  \hline
  FedAvgM &       0.05 &        1 &         0.0 & 0.723 & 0.028 & 0.121 & 0.014 &  3716.743 \\
  FedAvgM &       0.05 &        1 &         0.7 & 0.742 & 0.026 & 0.110 & 0.013 &   3700.91 \\
  FedAvgM &       0.05 &        1 &         0.9 & 0.308 & 0.069 & 0.194 & 0.035 &  3707.327 \\
  \hline
   FedOpt &       0.05 &        1 &        1e-7 & 0.711 & 0.029 & 0.124 & 0.014 &  3700.004 \\
   FedOpt &       0.05 &        1 &        1e-8 & 0.709 & 0.029 & 0.127 & 0.015 &  3702.422 \\
   FedOpt &       0.05 &        1 &        1e-9 & 0.685 & 0.032 & 0.129 & 0.016 &  3700.565 \\
   \hline
  QFedAvg &       0.05 &        1 &         0.1 & 0.356 & 0.064 & 0.209 & 0.032 &  4408.695 \\
  QFedAvg &       0.05 &        1 &         0.2 & 0.066 & 0.093 & 0.266 & 0.047 &  4399.795 \\
  QFedAvg &       0.05 &        1 &         0.5 & 0.295 & 0.130 & 0.325 & 0.065 &  4620.235 \\
  \hline
   FedAvg &      0.075 &        - &           - & 0.704 & 0.030 & 0.125 & 0.015 &  3656.392 \\
  FedAvgM &      0.075 &    0.001 &         0.0 & 0.982 & 0.198 & 0.407 & 0.099 &  3711.438 \\
  FedAvgM &      0.075 &    0.001 &         0.7 & 0.933 & 0.193 & 0.400 & 0.097 &  3721.303 \\
  FedAvgM &      0.075 &    0.001 &         0.9 & 0.775 & 0.178 & 0.384 & 0.089 &  3707.502 \\
  \hline
   FedOpt &      0.075 &    0.001 &        1e-7 & 0.688 & 0.031 & 0.128 & 0.016 &  3740.531 \\
   FedOpt &      0.075 &    0.001 &        1e-8 & 0.701 & 0.030 & 0.128 & 0.015 &  3719.583 \\
   FedOpt &      0.075 &    0.001 &        1e-9 & 0.683 & 0.032 & 0.134 & 0.016 &  3744.439 \\
   \hline
  QFedAvg &      0.075 &    0.001 &         0.1 &     - &     - &     - &     - &  4406.367 \\
  QFedAvg &      0.075 &    0.001 &         0.2 &     - &     - &     - &     - &  4424.477 \\
  QFedAvg &      0.075 &    0.001 &         0.5 &     - &     - &     - &     - &  4408.012 \\
  \hline
  FedAvgM &      0.075 &     0.01 &         0.0 & 0.828 & 0.183 & 0.385 & 0.091 &  3718.837 \\
  FedAvgM &      0.075 &     0.01 &         0.7 & 0.413 & 0.141 & 0.339 & 0.071 &  3731.316 \\
  FedAvgM &      0.075 &     0.01 &         0.9 & 0.270 & 0.073 & 0.214 & 0.037 &  3723.418 \\
  \hline
   FedOpt &      0.075 &     0.01 &        1e-7 & 0.700 & 0.030 & 0.124 & 0.015 &  3721.138 \\
   FedOpt &      0.075 &     0.01 &        1e-8 & 0.697 & 0.030 & 0.129 & 0.015 &  3755.233 \\
   FedOpt &      0.075 &     0.01 &        1e-9 & 0.707 & 0.029 & 0.127 & 0.015 &  3710.466 \\
   \hline
  QFedAvg &      0.075 &     0.01 &         0.1 & 0.947 & 0.195 & 0.398 & 0.097 &  4419.944 \\
  QFedAvg &      0.075 &     0.01 &         0.2 & 0.974 & 0.198 & 0.403 & 0.099 &  4424.534 \\
  QFedAvg &      0.075 &     0.01 &         0.5 &     - &     - &     - &     - &  4411.812 \\
  \hline
  FedAvgM &      0.075 &        1 &         0.0 & 0.698 & 0.030 & 0.128 & 0.015 &  3714.743 \\
  FedAvgM &      0.075 &        1 &         0.7 & 0.717 & 0.028 & 0.119 & 0.014 &  3715.896 \\
  FedAvgM &      0.075 &        1 &         0.9 & 0.583 & 0.042 & 0.144 & 0.021 &  3727.534 \\
  \hline
   FedOpt &      0.075 &        1 &        1e-7 & 0.712 & 0.029 & 0.127 & 0.014 &  3716.352 \\
   FedOpt &      0.075 &        1 &        1e-8 & 0.686 & 0.031 & 0.130 & 0.016 &   3709.05 \\
   FedOpt &      0.075 &        1 &        1e-9 & 0.681 & 0.032 & 0.131 & 0.016 &  3733.965 \\
   \hline
  QFedAvg &      0.075 &        1 &         0.1 & 0.410 & 0.059 & 0.197 & 0.030 &  4473.667 \\
  QFedAvg &      0.075 &        1 &         0.2 & 0.040 & 0.096 & 0.274 & 0.048 &  4410.528 \\
  QFedAvg &      0.075 &        1 &         0.5 & 0.362 & 0.136 & 0.334 & 0.068 &  4644.231 \\
  \hline
   FedAvg &        0.1 &        - &           - & 0.667 & 0.033 & 0.134 & 0.017 &   4152.95 \\
  FedAvgM &        0.1 &    0.001 &         0.0 & 0.979 & 0.197 & 0.401 & 0.098 &  3845.039 \\
  FedAvgM &        0.1 &    0.001 &         0.7 & 0.942 & 0.193 & 0.397 & 0.097 &  3951.729 \\
  FedAvgM &        0.1 &    0.001 &         0.9 & 0.805 & 0.179 & 0.387 & 0.090 &   4907.03 \\
  \hline
   FedOpt &        0.1 &    0.001 &        1e-7 & 0.712 & 0.029 & 0.125 & 0.014 &  4134.122 \\
   FedOpt &        0.1 &    0.001 &        1e-8 & 0.693 & 0.031 & 0.128 & 0.015 &  3686.088 \\
   FedOpt &        0.1 &    0.001 &        1e-9 & 0.658 & 0.034 & 0.134 & 0.017 &  3724.021 \\
   \hline
  QFedAvg &        0.1 &    0.001 &         0.1 &     - &     - &     - &     - &  8244.406 \\
  QFedAvg &        0.1 &    0.001 &         0.2 &     - &     - &     - &     - &  4816.195 \\
  QFedAvg &        0.1 &    0.001 &         0.5 &     - &     - &     - &     - &  4729.274 \\
  \hline
  FedAvgM &        0.1 &     0.01 &         0.0 & 0.815 & 0.180 & 0.387 & 0.090 &  4013.364 \\
  FedAvgM &        0.1 &     0.01 &         0.7 & 0.419 & 0.141 & 0.340 & 0.071 &  4016.736 \\
  FedAvgM &        0.1 &     0.01 &         0.9 & 0.269 & 0.073 & 0.211 & 0.036 &  4813.541 \\
  \hline
   FedOpt &        0.1 &     0.01 &        1e-7 & 0.665 & 0.033 & 0.136 & 0.017 &   3721.84 \\
   FedOpt &        0.1 &     0.01 &        1e-8 & 0.670 & 0.033 & 0.134 & 0.016 &  3716.671 \\
   FedOpt &        0.1 &     0.01 &        1e-9 & 0.648 & 0.035 & 0.138 & 0.018 &  3706.826 \\
   \hline
  QFedAvg &        0.1 &     0.01 &         0.1 & 0.961 & 0.195 & 0.404 & 0.097 &  8516.412 \\
  QFedAvg &        0.1 &     0.01 &         0.2 & 0.976 & 0.196 & 0.404 & 0.098 &  4759.146 \\
  QFedAvg &        0.1 &     0.01 &         0.5 &     - &     - &     - &     - &  4522.861 \\
  \hline
  FedAvgM &        0.1 &        1 &         0.0 & 0.720 & 0.028 & 0.125 & 0.014 &  3975.267 \\
  FedAvgM &        0.1 &        1 &         0.7 & 0.699 & 0.030 & 0.114 & 0.015 &  3960.817 \\
  FedAvgM &        0.1 &        1 &         0.9 & 0.578 & 0.042 & 0.151 & 0.021 &  4959.182 \\
  \hline
   FedOpt &        0.1 &        1 &        1e-7 & 0.711 & 0.029 & 0.123 & 0.014 &  3710.098 \\
   FedOpt &        0.1 &        1 &        1e-8 & 0.680 & 0.032 & 0.130 & 0.016 &  3692.821 \\
   FedOpt &        0.1 &        1 &        1e-9 & 0.687 & 0.031 & 0.132 & 0.016 &  3685.647 \\
   \hline
  QFedAvg &        0.1 &        1 &         0.1 & 0.360 & 0.064 & 0.209 & 0.032 &  8405.274 \\
  QFedAvg &        0.1 &        1 &         0.2 & 0.125 & 0.087 & 0.255 & 0.043 &  4749.878 \\
  QFedAvg &        0.1 &        1 &         0.5 & 0.384 & 0.138 & 0.336 & 0.069 &  4896.551 \\
  \hline
   FedAvg &        0.2 &        - &           - & 0.719 & 0.028 & 0.121 & 0.014 &  4107.385 \\
  FedAvgM &        0.2 &    0.001 &         0.0 & 0.980 & 0.197 & 0.406 & 0.098 &  3742.821 \\
  FedAvgM &        0.2 &    0.001 &         0.7 & 0.938 & 0.193 & 0.400 & 0.096 &  3921.205 \\
  FedAvgM &        0.2 &    0.001 &         0.9 & 0.798 & 0.179 & 0.382 & 0.089 &  4869.805 \\
  \hline
   FedOpt &        0.2 &    0.001 &        1e-7 & 0.714 & 0.029 & 0.126 & 0.014 &  4137.837 \\
   FedOpt &        0.2 &    0.001 &        1e-8 & 0.678 & 0.032 & 0.133 & 0.016 &  3730.542 \\
   FedOpt &        0.2 &    0.001 &        1e-9 & 0.721 & 0.028 & 0.122 & 0.014 &  3747.735 \\
   \hline
  QFedAvg &        0.2 &    0.001 &         0.1 &     - &     - &     - &     - &  8249.368 \\
  QFedAvg &        0.2 &    0.001 &         0.2 &     - &     - &     - &     - &  4695.342 \\
  QFedAvg &        0.2 &    0.001 &         0.5 &     - &     - &     - &     - &  4785.048 \\
  \hline
  FedAvgM &        0.2 &     0.01 &         0.0 & 0.836 & 0.183 & 0.390 & 0.091 &  4015.138 \\
  FedAvgM &        0.2 &     0.01 &         0.7 & 0.447 & 0.144 & 0.342 & 0.072 &  4026.818 \\
  FedAvgM &        0.2 &     0.01 &         0.9 & 0.239 & 0.076 & 0.224 & 0.038 &  4766.698 \\
  \hline
   FedOpt &        0.2 &     0.01 &        1e-7 & 0.713 & 0.029 & 0.123 & 0.014 &  3713.327 \\
   FedOpt &        0.2 &     0.01 &        1e-8 & 0.712 & 0.029 & 0.123 & 0.014 &  3736.173 \\
   FedOpt &        0.2 &     0.01 &        1e-9 & 0.693 & 0.031 & 0.127 & 0.015 &  3723.323 \\
   \hline
  QFedAvg &        0.2 &     0.01 &         0.1 & 0.949 & 0.194 & 0.400 & 0.097 &  8482.245 \\
  QFedAvg &        0.2 &     0.01 &         0.2 & 0.980 & 0.197 & 0.405 & 0.098 &  4643.785 \\
  QFedAvg &        0.2 &     0.01 &         0.5 &     - &     - &     - &     - &  4598.547 \\
  \hline
  FedAvgM &        0.2 &        1 &         0.0 & 0.761 & 0.024 & 0.110 & 0.012 &  3938.139 \\
  FedAvgM &        0.2 &        1 &         0.7 & 0.652 & 0.035 & 0.130 & 0.017 &   3985.94 \\
  FedAvgM &        0.2 &        1 &         0.9 & 0.657 & 0.034 & 0.120 & 0.017 &  4930.265 \\
  \hline
   FedOpt &        0.2 &        1 &        1e-7 & 0.682 & 0.032 & 0.135 & 0.016 &  3712.524 \\
   FedOpt &        0.2 &        1 &        1e-8 & 0.681 & 0.032 & 0.132 & 0.016 &  3702.342 \\
   FedOpt &        0.2 &        1 &        1e-9 & 0.694 & 0.031 & 0.131 & 0.015 &  3750.784 \\
   \hline
  QFedAvg &        0.2 &        1 &         0.1 & 0.402 & 0.059 & 0.201 & 0.030 &   8485.82 \\
  QFedAvg &        0.2 &        1 &         0.2 & 0.004 & 0.100 & 0.281 & 0.050 &  4734.552 \\
  QFedAvg &        0.2 &        1 &         0.5 & 0.393 & 0.139 & 0.338 & 0.069 &      None \\
  \hline
   FedAvg &        0.5 &        - &           - & 0.699 & 0.030 & 0.125 & 0.015 &  3621.427 \\
  FedAvgM &        0.5 &    0.001 &         0.0 & 0.981 & 0.198 & 0.402 & 0.099 &  3658.501 \\
  FedAvgM &        0.5 &    0.001 &         0.7 & 0.941 & 0.194 & 0.403 & 0.097 &  3683.223 \\
  FedAvgM &        0.5 &    0.001 &         0.9 & 0.775 & 0.178 & 0.384 & 0.089 &  3679.711 \\
  \hline
   FedOpt &        0.5 &    0.001 &        1e-7 & 0.707 & 0.029 & 0.121 & 0.015 &  3711.238 \\
   FedOpt &        0.5 &    0.001 &        1e-8 & 0.705 & 0.030 & 0.125 & 0.015 &  3693.261 \\
   FedOpt &        0.5 &    0.001 &        1e-9 & 0.701 & 0.030 & 0.127 & 0.015 &  3702.508 \\
   \hline
  QFedAvg &        0.5 &    0.001 &         0.1 &     - &     - &     - &     - &  4378.176 \\
  QFedAvg &        0.5 &    0.001 &         0.2 &     - &     - &     - &     - &  4411.498 \\
  QFedAvg &        0.5 &    0.001 &         0.5 &     - &     - &     - &     - &  4378.575 \\
  \hline
  FedAvgM &        0.5 &     0.01 &         0.0 & 0.826 & 0.183 & 0.389 & 0.091 &   3658.62 \\
  FedAvgM &        0.5 &     0.01 &         0.7 & 0.418 & 0.142 & 0.340 & 0.071 &  3684.954 \\
  FedAvgM &        0.5 &     0.01 &         0.9 & 0.260 & 0.074 & 0.214 & 0.037 &  3680.444 \\
  \hline
   FedOpt &        0.5 &     0.01 &        1e-7 & 0.607 & 0.039 & 0.141 & 0.020 &  3677.584 \\
   FedOpt &        0.5 &     0.01 &        1e-8 & 0.675 & 0.032 & 0.127 & 0.016 &  3698.811 \\
   FedOpt &        0.5 &     0.01 &        1e-9 & 0.713 & 0.029 & 0.125 & 0.014 &  3689.004 \\
   \hline
  QFedAvg &        0.5 &     0.01 &         0.1 & 0.951 & 0.195 & 0.402 & 0.098 &  4371.691 \\
  QFedAvg &        0.5 &     0.01 &         0.2 & 0.974 & 0.198 & 0.408 & 0.099 &  4376.109 \\
  QFedAvg &        0.5 &     0.01 &         0.5 &     - &     - &     - &     - &  4389.336 \\
  \hline
  FedAvgM &        0.5 &        1 &         0.0 & 0.716 & 0.028 & 0.124 & 0.014 &  3699.566 \\
  FedAvgM &        0.5 &        1 &         0.7 & 0.647 & 0.035 & 0.133 & 0.018 &  3681.617 \\
  FedAvgM &        0.5 &        1 &         0.9 & 0.587 & 0.041 & 0.135 & 0.021 &   3686.34 \\
  \hline
   FedOpt &        0.5 &        1 &        1e-7 & 0.695 & 0.031 & 0.128 & 0.015 &  3754.853 \\
   FedOpt &        0.5 &        1 &        1e-8 & 0.706 & 0.029 & 0.126 & 0.015 &  3679.754 \\
   FedOpt &        0.5 &        1 &        1e-9 & 0.722 & 0.028 & 0.121 & 0.014 &  3690.291 \\
   \hline
  QFedAvg &        0.5 &        1 &         0.1 & 0.377 & 0.062 & 0.203 & 0.031 &  4402.415 \\
  QFedAvg &        0.5 &        1 &         0.2 & 0.106 & 0.089 & 0.260 & 0.045 &  4420.429 \\
  QFedAvg &        0.5 &        1 &         0.5 & 0.393 & 0.139 & 0.338 & 0.070 &  4601.539 \\
  \hline
   FedAvg &          1 &        - &           - & 0.680 & 0.032 & 0.132 & 0.016 &  3630.882 \\
  FedAvgM &          1 &    0.001 &         0.0 & 0.980 & 0.198 & 0.409 & 0.099 &  3739.519 \\
  FedAvgM &          1 &    0.001 &         0.7 & 0.929 & 0.193 & 0.394 & 0.097 &  3694.701 \\
  FedAvgM &          1 &    0.001 &         0.9 & 0.790 & 0.179 & 0.386 & 0.090 &  3700.117 \\
  \hline
   FedOpt &          1 &    0.001 &        1e-7 & 0.727 & 0.027 & 0.117 & 0.014 &  3711.373 \\
   FedOpt &          1 &    0.001 &        1e-8 & 0.745 & 0.026 & 0.117 & 0.013 &  3757.962 \\
   FedOpt &          1 &    0.001 &        1e-9 & 0.713 & 0.029 & 0.117 & 0.014 &  3764.011 \\
   \hline
  QFedAvg &          1 &    0.001 &         0.1 &     - &     - &     - &     - &  4428.732 \\
  QFedAvg &          1 &    0.001 &         0.2 &     - &     - &     - &     - &  4395.914 \\
  QFedAvg &          1 &    0.001 &         0.5 &     - &     - &     - &     - &   4415.55 \\
  \hline
  FedAvgM &          1 &     0.01 &         0.0 & 0.809 & 0.181 & 0.385 & 0.090 &  3714.443 \\
  FedAvgM &          1 &     0.01 &         0.7 & 0.464 & 0.146 & 0.345 & 0.073 &  3716.214 \\
  FedAvgM &          1 &     0.01 &         0.9 & 0.257 & 0.074 & 0.217 & 0.037 &  3739.978 \\
  \hline
   FedOpt &          1 &     0.01 &        1e-7 & 0.708 & 0.029 & 0.125 & 0.015 &  3723.973 \\
   FedOpt &          1 &     0.01 &        1e-8 & 0.730 & 0.027 & 0.119 & 0.013 &  3728.687 \\
   FedOpt &          1 &     0.01 &        1e-9 & 0.717 & 0.028 & 0.125 & 0.014 &  3703.243 \\
   \hline
  QFedAvg &          1 &     0.01 &         0.1 & 0.953 & 0.195 & 0.401 & 0.098 &  4412.321 \\
  QFedAvg &          1 &     0.01 &         0.2 & 0.977 & 0.198 & 0.407 & 0.099 &   4420.62 \\
  QFedAvg &          1 &     0.01 &         0.5 &     - &     - &     - &     - &  4479.057 \\
  \hline
  FedAvgM &          1 &        1 &         0.0 & 0.728 & 0.027 & 0.122 & 0.014 &  3736.921 \\
  FedAvgM &          1 &        1 &         0.7 & 0.698 & 0.030 & 0.112 & 0.015 &  3706.877 \\
  FedAvgM &          1 &        1 &         0.9 & 0.648 & 0.035 & 0.126 & 0.018 &  3723.511 \\
  \hline
   FedOpt &          1 &        1 &        1e-7 & 0.708 & 0.029 & 0.127 & 0.015 &  3710.632 \\
   FedOpt &          1 &        1 &        1e-8 & 0.713 & 0.029 & 0.123 & 0.014 &  3700.154 \\
   FedOpt &          1 &        1 &        1e-9 & 0.708 & 0.029 & 0.119 & 0.015 &  3720.961 \\
   \hline
  QFedAvg &          1 &        1 &         0.1 & 0.336 & 0.066 & 0.214 & 0.033 &  4404.949 \\
  QFedAvg &          1 &        1 &         0.2 & 0.111 & 0.089 & 0.260 & 0.044 &  4410.538 \\
  QFedAvg &          1 &        1 &         0.5 & 0.331 & 0.133 & 0.327 & 0.067 &  4592.872 \\
  \hline
   FedAvg &    1000000 &        - &           - & 0.707 & 0.029 & 0.123 & 0.015 &  3647.749 \\
  FedAvgM &    1000000 &    0.001 &         0.0 & 0.981 & 0.198 & 0.408 & 0.099 &  3698.387 \\
  FedAvgM &    1000000 &    0.001 &         0.7 & 0.941 & 0.194 & 0.401 & 0.097 &  3726.063 \\
  FedAvgM &    1000000 &    0.001 &         0.9 & 0.798 & 0.180 & 0.386 & 0.090 &  3698.957 \\
  \hline
   FedOpt &    1000000 &    0.001 &        1e-7 & 0.650 & 0.035 & 0.134 & 0.018 &  3723.296 \\
   FedOpt &    1000000 &    0.001 &        1e-8 & 0.661 & 0.034 & 0.137 & 0.017 &  3754.253 \\
   FedOpt &    1000000 &    0.001 &        1e-9 & 0.729 & 0.027 & 0.117 & 0.014 &  3749.374 \\
   \hline
  QFedAvg &    1000000 &    0.001 &         0.1 &     - &     - &     - &     - &  4417.479 \\
  QFedAvg &    1000000 &    0.001 &         0.2 &     - &     - &     - &     - &  4430.831 \\
  QFedAvg &    1000000 &    0.001 &         0.5 &     - &     - &     - &     - &  4784.405 \\
  \hline
  FedAvgM &    1000000 &     0.01 &         0.0 & 0.799 & 0.180 & 0.384 & 0.090 &  3716.385 \\
  FedAvgM &    1000000 &     0.01 &         0.7 & 0.431 & 0.143 & 0.343 & 0.072 &  3733.538 \\
  FedAvgM &    1000000 &     0.01 &         0.9 & 0.220 & 0.078 & 0.223 & 0.039 &  3747.649 \\
  \hline
   FedOpt &    1000000 &     0.01 &        1e-7 & 0.692 & 0.031 & 0.126 & 0.015 &    3712.2 \\
   FedOpt &    1000000 &     0.01 &        1e-8 & 0.728 & 0.027 & 0.118 & 0.014 &  3748.511 \\
   FedOpt &    1000000 &     0.01 &        1e-9 & 0.722 & 0.028 & 0.122 & 0.014 &  3738.497 \\
   \hline
  QFedAvg &    1000000 &     0.01 &         0.1 & 0.946 & 0.195 & 0.394 & 0.097 &  4445.261 \\
  QFedAvg &    1000000 &     0.01 &         0.2 & 0.977 & 0.198 & 0.397 & 0.099 &  4402.446 \\
  QFedAvg &    1000000 &     0.01 &         0.5 &     - &     - &     - &     - &  4391.106 \\
  \hline
  FedAvgM &    1000000 &        1 &         0.0 & 0.726 & 0.027 & 0.121 & 0.014 &  3709.114 \\
  FedAvgM &    1000000 &        1 &         0.7 & 0.707 & 0.029 & 0.110 & 0.015 &   3735.28 \\
  FedAvgM &    1000000 &        1 &         0.9 & 0.657 & 0.034 & 0.125 & 0.017 &  3715.087 \\
  \hline
   FedOpt &    1000000 &        1 &        1e-7 & 0.717 & 0.028 & 0.121 & 0.014 &  3755.469 \\
   FedOpt &    1000000 &        1 &        1e-8 & 0.725 & 0.028 & 0.123 & 0.014 &  3737.932 \\
   FedOpt &    1000000 &        1 &        1e-9 & 0.702 & 0.030 & 0.126 & 0.015 &  3725.321 \\
   \hline
  QFedAvg &    1000000 &        1 &         0.1 & 0.391 & 0.061 & 0.203 & 0.030 &   4416.36 \\
  QFedAvg &    1000000 &        1 &         0.2 & 0.016 & 0.098 & 0.276 & 0.049 &  4423.968 \\
  QFedAvg &    1000000 &        1 &         0.5 & 0.367 & 0.135 & 0.330 & 0.067 &  4410.475 \\
  \hline





</pre> 




# New Scale Experiments:


```bash

# Create a new conda environment called 'flwr-nasa'
conda create -n flwr-nasa python=3.11 -y

# Activate the environment
conda activate flwr-nasa





# Install packages via conda
conda install numpy pandas -y

# Install Flower via pip (usually more up-to-date)
pip install flwr


pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130


```