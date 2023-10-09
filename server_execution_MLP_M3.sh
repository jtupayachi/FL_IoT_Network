#!/bin/bash


#MLP
#TYPE1



#CENTRALIZED 
python3 fl_testbed/version2/client/centralized.py  -ml 1  -cn 15 -cm 15 -e 100 -dfn   'combined_offset_misalignment_M3.csv'  -ip 172.19.0.8 2>&1 | tee out_server_14_OFFM3_TYPE1.txt2 


#DATASPLIT
python3 fl_testbed/version2/client/datasplit.py -data_X_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_train.pkl -data_X_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_vals.pkl -data_y_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_train.pkl -data_y_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_vals.pkl -cm 4 -l 90 10 10 10 10     10 90 10 10 10    10 10 90 10 10      10 10 10 90 10    10 10 10 10 90 -fq  0.2 0.25 0.3333333 0.5 1  -motor 3 -type MLP 2>&1 | tee DATASPLIT_TYPE1_MLP_M3_TYPE1.txt

#INDEPENDENT
python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_0_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.8 2>&1 | tee out_server_M3_4_0_OFFSETM3_idp_TYPE1.txt4 

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_1_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.8 2>&1 | tee out_server_M3_4_1_OFFSETM3_idp_TYPE1.txt4 

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_2_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.8 2>&1 | tee out_server_M3_4_2_OFFSETM3_idp_TYPE1.txt4 

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_3_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.8 2>&1 | tee out_server_M3_4_3_OFFSETM3_idp_TYPE1.txt4

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_4_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.19.0.8 2>&1 | tee out_server_M3_4_4_OFFSETM3_idp_TYPE1.txt4


#FEDERATED
#_FedAvg
echo -n "_FedAvg"
python3 fl_testbed/version2/server/federated_server_OFFSET_FedAvg.py   -cm 5 -e 1 --rounds 4000 -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvg_TYPE1.txt
#_FedAvgM

echo -n "_FedAvgM"
python3 fl_testbed/version2/server/federated_server_OFFSET_FedAvgM.py   -cm 5 -e 1 --rounds 4000 -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvgM_TYPE1.txt
#_FedOpt

echo -n "_FedOpt"
python3 fl_testbed/version2/server/federated_server_OFFSET_FedOpt.py   -cm 5 -e 1 --rounds 4000 -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedOpt_TYPE1.txt
#_QFedAvg
  
echo -n "_QFedAvg"
python3 fl_testbed/version2/server/federated_server_OFFSET_QFedAvg.py   -cm 5 -e 1 --rounds 4000 -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_QFedAvg_TYPE1.txt








# #MLP
# #TYPE2



#CENTRALIZED 
python3 fl_testbed/version2/client/centralized.py  -ml 1  -cn 15 -cm 15 -e 100 -dfn   'combined_offset_misalignment_M3.csv'  -ip 172.17.0.8 2>&1 | tee out_server_14_OFFM3_TYPE2.txt2 


#DATASPLIT
python3 fl_testbed/version2/client/datasplit.py -data_X_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_train.pkl -data_X_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_vals.pkl -data_y_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_train.pkl -data_y_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_vals.pkl -cm 4 -l 50 50 50 50 50     50 50 50 50 50    50 50 50 50 50      50 50 50 50 50   50 50 50 50 50 -fq  0.2 0.25 0.3333333 0.5 1  -motor 3 -type MLP 2>&1 | tee DATASPLIT_TYPE2_MLP_M3.txt 


#INDEPENDENT
python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_0_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.8 2>&1 | tee out_server_M3_4_0_OFFSETM3_idp_TYPE2.txt4 

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_1_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.8 2>&1 | tee out_server_M3_4_1_OFFSETM3_idp_TYPE2.txt4 

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_2_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.8 2>&1 | tee out_server_M3_4_2_OFFSETM3_idp_TYPE2.txt4 

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_3_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.8 2>&1 | tee out_server_M3_4_3_OFFSETM3_idp_TYPE2.txt4

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_4_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.82>&1 | tee out_server_M3_4_4_OFFSETM3_idp_TYPE2.txt4


#FEDERATED
#_FedAvg
echo -n "_FedAvg"
python3 fl_testbed/version2/server/federated_server_OFFSET_FedAvg.py   -cm 5 -e 1 --rounds 4000 -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvg_TYPE2.txt
#_FedAvgM

echo -n "_FedAvgM"
python3 fl_testbed/version2/server/federated_server_OFFSET_FedAvgM.py   -cm 5 -e 1 --rounds 4000 -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvgM_TYPE2.txt
#_FedOpt

echo -n "_FedOpt"
python3 fl_testbed/version2/server/federated_server_OFFSET_FedOpt.py   -cm 5 -e 1 --rounds 4000 -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedOpt_TYPE2.txt
#_QFedAvg
  
echo -n "_QFedAvg"
python3 fl_testbed/version2/server/federated_server_OFFSET_QFedAvg.py   -cm 5 -e 1 --rounds 4000 -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_QFedAvg_TYPE2.txt
























