#!/bin/bash




alphas="0.001 0.01 0.1 0.02 0.2 0.005 0.05 0.5 0.075 1.0 1000000.0"

for var in $alphas; do
  echo $var
  #HERE GOES THE WHOLE SEQUENCE:






#MLP
#${var}



#CENTRALIZED 
python3 fl_testbed/version2/client/centralized.py  -ml 1  -cn 15 -cm 15 -e 100 -dfn   'combined_offset_misalignment_M3.csv'  -ip 172.19.0.8 2>&1 | tee out_server_14_OFFM3_${var}.txt2 

#DATASPLIT
python3 fl_testbed/version2/client/dirichelet_split.py -data_X_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_train.pkl -data_X_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_vals.pkl -data_y_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_train.pkl -data_y_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_vals.pkl -cm 5 -alpha $var -beta 0.2 -motor 3 -type MLP 2>&1 | tee DATASPLIT_${var}_MLP_M3_${var}.txt


#INDEPENDENT
python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_0_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.8 2>&1 | tee out_server_M3_4_0_OFFSETM3_idp_${var}.txt4 

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_1_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.8 2>&1 | tee out_server_M3_4_1_OFFSETM3_idp_${var}.txt4 

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_2_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.8 2>&1 | tee out_server_M3_4_2_OFFSETM3_idp_${var}.txt4 

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_3_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.17.0.8 2>&1 | tee out_server_M3_4_3_OFFSETM3_idp_${var}.txt4

python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_4_4_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.19.0.8 2>&1 | tee out_server_M3_4_4_OFFSETM3_idp_${var}.txt4


#FEDERATED
#_FedAvg
echo -n "_FedAvg"
python3 fl_testbed/version2/server/federated_server_OFFSET_FedAvg.py   -cm 5 -e 1 --rounds 100 -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvg_${var}.txt
#_FedAvgM

echo -n "_FedAvgM"
python3 fl_testbed/version2/server/federated_server_OFFSET_FedAvgM.py   -cm 5 -e 1 --rounds 100 -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvgM_${var}.txt
#_FedOpt

echo -n "_FedOpt"
python3 fl_testbed/version2/server/federated_server_OFFSET_FedOpt.py   -cm 5 -e 1 --rounds 100 -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedOpt_${var}.txt
#_QFedAvg
  
echo -n "_QFedAvg"
python3 fl_testbed/version2/server/federated_server_OFFSET_QFedAvg.py   -cm 5 -e 1 --rounds 100 -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_4_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_QFedAvg_${var}.txt














  echo "done"
  sleep 200
done





















