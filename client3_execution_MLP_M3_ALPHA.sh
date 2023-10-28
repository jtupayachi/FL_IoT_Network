#!/bin/bash
#MLP
#${var}

alphas="0.001 0.01 0.1 0.02 0.2 0.005 0.05 0.5 0.075 1.0 1000000.0"

for var in $alphas; do
  echo $var
  #HERE GOES THE WHOLE SEQUENCE:



sleep 500
#_FedAvg
echo -n "_FedAvg"
python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvg.py -cn 3  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_2_ddf_MLP.pkl' 2>&1 | tee MLP_CLIENT3_FedAvg_${var}.txt
#_FedAvgM
sleep 300
echo -n "_FedAvgM"
python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvgM.py -cn 3  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_2_ddf_MLP.pkl' 2>&1 | tee MLP_CLIENT3_FedAvgM_${var}.txt
#_FedOpt
sleep 300
echo -n "_FedOpt"
python3 fl_testbed/version2/client/federated_client_OFFSET_FedOpt.py -cn 3  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_2_ddf_MLP.pkl' 2>&1 | tee MLP_CLIENT3_FedOpt_${var}.txt
#_QFedAvg
  sleep 300
echo -n "_QFedAvg"
python3 fl_testbed/version2/client/federated_client_OFFSET_QFedAvg.py -cn 3  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_2_ddf_MLP.pkl' 2>&1 | tee MLP_CLIENT3_QFedAvg_${var}.txt





  echo "done"
done



