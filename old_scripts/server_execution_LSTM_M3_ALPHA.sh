# !/bin/bash



#CENTRALIZED SEQ80
python3 fl_testbed/version2/client/centralized_new.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'combined_offset_misalignment_M3.csv' --JUMPING_STEP 20  -ip 172.18.0.2 2>&1 | tee out_server_14_RULM3_SEQ80.txt2

# alphas="0.001 0.01 0.1 0.02 0.2 0.005 0.05 0.5 0.075 1.0 1000000.0"
alphas="0.001 0.01 0.1 0.02 0.2 0.005 0.05 0.5 0.075 1.0 1000000.0" #0.005 0.05 0.5 0.075 1.0 
slr="0.001 0.01 1"

FedAvgM_momentum="0.0 0.7 0.9"
FedOpt_tau="1e-7 1e-8 1e-9"
QFedAvg_q="0.5" #0.1 0.2 

for var in $alphas; do
    echo $var
    
    
    python3 fl_testbed/version2/client/dirichelet_split.py -data_X_train 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_inputs.pkl -data_X_vals 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_inputs.pkl -data_y_train 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_out.pkl -data_y_vals 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_out.pkl -cm 5 -alpha ${var} -beta 0.2 -motor 3 -type LSTM 2>&1 | tee DATASPLIT_${var}_LSTM_M3_${var}.txt
    
    #INDEPENDENT
    python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_5_0_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.2 2>&1 | tee out_server_14_M3_4_0_ddf_LSTM_idp_${var}.txt2
    python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_5_1_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.2 2>&1 | tee out_server_14_M3_4_1_ddf_LSTM_idp_${var}.txt2
    
    python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_5_2_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.18.0.2 2>&1 | tee out_server_14_M3_4_2_ddf_LSTM_idp_${var}.txt2
    
    python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_5_3_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.19.0.2 2>&1 | tee out_server_14_M3_4_3_ddf_LSTM_idp_${var}.txt2
    
    python3 fl_testbed/version2/client/independent.py  -ml 2  -cn 15 -cm 15 -e 100 -dfn   'M3_5_4_ddf_LSTM.pkl' -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl'   -ip 172.19.0.2 2>&1 | tee out_server_14_M3_4_4_ddf_LSTM_idp_${var}.txt2
    
    #FEDERATED
    #_FedAvg
    # echo -n "_FedAvg"
    # python3 fl_testbed/version2/server/federated_server_RUL_FedAvg.py   -cm 5 -e 1 --rounds 1000 -ip  172.18.0.2  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_5_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_FedAvg_${var}.txt
    #_FedAvgM
    for var2 in $slr; do
        echo $var2
        
        # for var3 in $FedAvgM_momentum; do
        #     echo $var3
        #     echo -n "_FedAvgM"
        #     python3 fl_testbed/version2/server/federated_server_RUL_FedAvgM.py -momentum ${var3} -slr ${var2} -cm 5 -e 1 --rounds 1000 -ip  172.18.0.2  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_5_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_FedAvgM_${var}_slr_${var2}_${var3}.txt
        #     echo "done"
        # done
        #_FedOpt
        for var4 in $FedOpt_tau; do
            echo $var4
            echo -n "_FedOpt"
            python3 fl_testbed/version2/server/federated_server_RUL_FedOpt.py -tau ${var4}  -slr ${var2}  -cm 5 -e 1 --rounds 1000 -ip  172.18.0.2  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_5_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_FedOpt_${var}_slr_${var2}_${var4}.txt
            echo "done"
        done
        #_QFedAvg
        # for var5 in $QFedAvg_q; do
        #     echo $var5
        #     echo -n "_QFedAvg"
        #     python3 fl_testbed/version2/server/federated_server_RUL_QFedAvg.py -q ${var5} -slr ${var2}  -cm 5 -e 1 --rounds 1000 -ip  172.18.0.2  -dfn_test_x   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' -dfn_test_y   '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' -dfn 'M3_5_0_ddf_LSTM.pkl'  2>&1 | tee LSTM_TESLA_QFedAvg_${var}_slr_${var2}_${var5}.txt
            
        #     echo "done"
        # done
        
        echo "done"
    done
    
    echo "done"
done
