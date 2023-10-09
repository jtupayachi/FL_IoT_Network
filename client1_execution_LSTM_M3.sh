#!/bin/bash
#LSTM
#TYPE1

#FOR SEQ80
#_FedAvg
echo 'ONLY RUN AFTER THE CENTRALIZED AND INDEPENDENT APPRAOCHES ARE DONE SLEEP 200'
sleep 200
python3 fl_testbed/version2/client/federated_client_RUL_FedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedAvg_SEQ80_TYPE1.txt
#_FedAvgM
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_FedAvgM.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedAvgM_SEQ80_TYPE1.txt
#_FedOpt
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_FedOpt.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedOpt_SEQ80_TYPE1.txt
#_QFedAvg
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_QFedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_QFedAvg_SEQ80_TYPE1.txt



#FOR SEQ40
#_FedAvg
sleep 200
python3 fl_testbed/version2/client/federated_client_RUL_FedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedAvg_SEQ40_TYPE1.txt
#_FedAvgM
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_FedAvgM.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedAvgM_SEQ40_TYPE1.txt
#_FedOpt
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_FedOpt.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedOpt_SEQ40_TYPE1.txt
#_QFedAvg
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_QFedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_QFedAvg_SEQ40_TYPE1.txt




#FOR SEQ20
#_FedAvg
sleep 200
python3 fl_testbed/version2/client/federated_client_RUL_FedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedAvg_SEQ20_TYPE1.txt
#_FedAvgM
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_FedAvgM.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedAvgM_SEQ20_TYPE1.txt
#_FedOpt
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_FedOpt.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedOpt_SEQ20_TYPE1.txt
#_QFedAvg
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_QFedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_QFedAvg_SEQ20_TYPE1.txt





# TYPE2

# FOR SEQ80
_FedAvg
echo 'ONLY RUN AFTER THE CENTRALIZED AND INDEPENDENT APPRAOCHES ARE DONE SLEEP 200'
sleep 200
python3 fl_testbed/version2/client/federated_client_RUL_FedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedAvg_SEQ80_TYPE2.txt
#_FedAvgM
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_FedAvgM.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedAvgM_SEQ80_TYPE2.txt
_FedOpt
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_FedOpt.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedOpt_SEQ80_TYPE2.txt
#_QFedAvg
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_QFedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_QFedAvg_SEQ80_TYPE2.txt



# FOR SEQ40
_FedAvg
sleep 200
python3 fl_testbed/version2/client/federated_client_RUL_FedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedAvg_SEQ40_TYPE2.txt
#_FedAvgM
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_FedAvgM.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedAvgM_SEQ40_TYPE2.txt
_FedOpt
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_FedOpt.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedOpt_SEQ40_TYPE2.txt
#_QFedAvg
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_QFedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_QFedAvg_SEQ40_TYPE2.txt




#FOR SEQ20
# _FedAvg
sleep 200
python3 fl_testbed/version2/client/federated_client_RUL_FedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedAvg_SEQ20_TYPE2.txt
#_FedAvgM
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_FedAvgM.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedAvgM_SEQ20_TYPE2.txt
_FedOpt
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_FedOpt.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_FedOpt_SEQ20_TYPE2.txt
#_QFedAvg
sleep 50
python3 fl_testbed/version2/client/federated_client_RUL_QFedAvg.py   -cn 0 -cm 5 -e 1 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.3 2>&1 | tee LSTM_MATTI_QFedAvg_SEQ20_TYPE2.txt



