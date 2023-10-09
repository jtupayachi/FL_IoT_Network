#!/bin/bash
#MLP
#TYPE1


sleep 300
#_FedAvg
echo -n "_FedAvg"
python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvg.py -cn 3  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedy_test.pkl' -dfn 'M2_4_2_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvg_TYPE1.txt
#_FedAvgM

echo -n "_FedAvgM"
python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvgM.py -cn 3  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedy_test.pkl' -dfn 'M2_4_2_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvgM_TYPE1.txt
#_FedOpt

echo -n "_FedOpt"
python3 fl_testbed/version2/client/federated_client_OFFSET_FedOpt.py -cn 3  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedy_test.pkl' -dfn 'M2_4_2_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedOpt_TYPE1.txt
#_QFedAvg
  
echo -n "_QFedAvg"
python3 fl_testbed/version2/client/federated_client_OFFSET_QFedAvg.py -cn 3  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedy_test.pkl' -dfn 'M2_4_2_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_QFedAvg_TYPE1.txt




#TYPE2

sleep 300
#_FedAvg
echo -n "_FedAvg"
python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvg.py -cn 3  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedy_test.pkl' -dfn 'M2_4_2_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvg_TYPE2.txt
#_FedAvgM

echo -n "_FedAvgM"
python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvgM.py -cn 3  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedy_test.pkl' -dfn 'M2_4_2_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvgM_TYPE2.txt
#_FedOpt

echo -n "_FedOpt"
python3 fl_testbed/version2/client/federated_client_OFFSET_FedOpt.py -cn 3 -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedy_test.pkl' -dfn 'M2_4_2_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedOpt_TYPE2.txt
#_QFedAvg
  
echo -n "_QFedAvg"
python3 fl_testbed/version2/client/federated_client_OFFSET_QFedAvg.py -cn 3  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M2.csv__client_centralizedy_test.pkl' -dfn 'M2_4_2_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_QFedAvg_TYPE2.txt
