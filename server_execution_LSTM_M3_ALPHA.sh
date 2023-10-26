# !/bin/bash


#LSTM
#TYPE1

#FOR SEQ80

#CENTRALIZED SEQ80
python3 fl_testbed/version2/client/centralized_new.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'combined_offset_misalignment_M3.csv' --JUMPING_STEP 3  -ip 172.17.0.2 2>&1 | tee out_server_14_RULM3_SEQ80_TYPE1.txt2

#DATASPLIT
python3 fl_testbed/version2/client/datasplit.py -data_X_train 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_inputs.pkl -data_X_vals 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_inputs.pkl -data_y_train 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_out.pkl -data_y_vals 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_out.pkl -cm 4  -l 90 10 10 10 10     10 90 10 10 10    10 10 90 10 10      10 10 10 90 10    10 10 10 10 90 -fq  0.2 0.25 0.3333333 0.5 1  -motor 3 -type LSTM 2>&1 | tee DATASPLIT_TYPE1_LSTM_M3_SEQ80.txt

#INDEPENDENT
python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.2 2>&1 | tee out_server_14_M3_4_0_ddf_LSTM_idp_SEQ80_TYPE1.txt2  

python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_1_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.2 2>&1 | tee out_server_14_M3_4_1_ddf_LSTM_idp_SEQ80_TYPE1.txt2 

python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_2_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.2 2>&1 | tee out_server_14_M3_4_2_ddf_LSTM_idp_SEQ80_TYPE1.txt2 

python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_3_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.19.0.2 2>&1 | tee out_server_14_M3_4_3_ddf_LSTM_idp_SEQ80_TYPE1.txt2 	

python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_4_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.19.0.2 2>&1 | tee out_server_14_M3_4_4_ddf_LSTM_idp_SEQ80_TYPE1.txt2 

#FEDERATED
#_FedAvg
echo -n "_FedAvg"

python3 fl_testbed/version2/server/federated_server_RUL_FedAvg.py   -cm 5 -e 1 --rounds 1000 -ip  172.17.0.2  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_FedAvg_SEQ80_TYPE1.txt
#_FedAvgM

echo -n "_FedAvgM"
python3 fl_testbed/version2/server/federated_server_RUL_FedAvgM.py   -cm 5 -e 1 --rounds 1000 -ip  172.17.0.2  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_FedAvgM_SEQ80_TYPE1.txt
#_FedOpt

echo -n "_FedOpt"
python3 fl_testbed/version2/server/federated_server_RUL_FedOpt.py   -cm 5 -e 1 --rounds 1000 -ip  172.17.0.2  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_FedOpt_SEQ80_TYPE1.txt
#_QFedAvg
  
echo -n "_QFedAvg"
python3 fl_testbed/version2/server/federated_server_RUL_QFedAvg.py   -cm 5 -e 1 --rounds 1000 -ip  172.17.0.2  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_QFedAvg_SEQ80_TYPE1.txt












# TYPE2


# FOR SEQ80

# CENTRALIZED SEQ80
python3 fl_testbed/version2/client/centralized_new.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'combined_offset_misalignment_M3.csv' --JUMPING_STEP 80  -ip 172.17.0.2 2>&1 | tee out_server_14_RULM3_SEQ80_TYPE2.txt2

#DATASPLIT
python3 fl_testbed/version2/client/datasplit.py -data_X_train 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_inputs.pkl -data_X_vals 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_inputs.pkl -data_y_train 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_out.pkl -data_y_vals 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_out.pkl -cm 4  -l 50 50 50 50 50     50 50 50 50 50    50 50 50 50 50      50 50 50 50 50   50 50 50 50 50 -fq  0.2 0.25 0.3333333 0.5 1  -motor 3 -type LSTM 2>&1 | tee DATASPLIT_TYPE2_LSTM_M3_SEQ80.txt


#INDEPENDENT
python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.2 2>&1 | tee out_server_14_M3_4_0_ddf_LSTM_idp_SEQ80_TYPE2.txt2  

python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_1_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.2 2>&1 | tee out_server_14_M3_4_1_ddf_LSTM_idp_SEQ80_TYPE2.txt2 

python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_2_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.17.0.2 2>&1 | tee out_server_14_M3_4_2_ddf_LSTM_idp_SEQ80_TYPE2.txt2 

python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_3_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.19.0.2 2>&1 | tee out_server_14_M3_4_3_ddf_LSTM_idp_SEQ80_TYPE2.txt2 	

python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_4_4_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.19.0.2 2>&1 | tee out_server_14_M3_4_4_ddf_LSTM_idp_SEQ80_TYPE2.txt2 

#FEDERATED
#_FedAvg
echo -n "_FedAvg"

python3 fl_testbed/version2/server/federated_server_RUL_FedAvg.py   -cm 5 -e 1 --rounds 1000 -ip  172.17.0.2  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_FedAvg_SEQ80_TYPE2.txt
#_FedAvgM

echo -n "_FedAvgM"
python3 fl_testbed/version2/server/federated_server_RUL_FedAvgM.py   -cm 5 -e 1 --rounds 1000 -ip  172.17.0.2  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_FedAvgM_SEQ80_TYPE2.txt
#_FedOpt

echo -n "_FedOpt"
python3 fl_testbed/version2/server/federated_server_RUL_FedOpt.py   -cm 5 -e 1 --rounds 1000 -ip  172.17.0.2  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_FedOpt_SEQ80_TYPE2.txt
#_QFedAvg
  
echo -n "_QFedAvg"
python3 fl_testbed/version2/server/federated_server_RUL_QFedAvg.py   -cm 5 -e 1 --rounds 1000 -ip  172.17.0.2  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_4_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_QFedAvg_SEQ80_TYPE2.txt





