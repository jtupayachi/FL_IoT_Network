#!/bin/bash



#CENTRALIZED
python3 fl_testbed/version2/client/centralized.py  -ml 1  -cn 15 -cm 15 -e 100 -dfn   'combined_offset_misalignment_M3.csv'  -ip 172.19.0.8 2>&1 | tee out_server_14_OFFM3.txt2


alphas="0.001 0.01 0.1 0.02 0.2 0.005 0.05 0.5 0.075 1.0 1000000.0"
slr="0.001 0.01 1"

FedAvgM_momentum="0.0 0.7 0.9"
FedOpt_tau="1e-8"
# FedOpt_tau="1e-7 1e-8 1e-9"
QFedAvg_q="0.1 0.2 0.5"

for var in $alphas; do
    echo $var
    #HERE GOES THE WHOLE SEQUENCE:

    #DATASPLIT
    python3 fl_testbed/version2/client/dirichelet_split.py -data_X_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_train.pkl -data_X_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_vals.pkl -data_y_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_train.pkl -data_y_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_vals.pkl -cm 5 -alpha ${var} -beta 0.2 -motor 3 -type MLP 2>&1 | tee DATASPLIT_${var}_MLP_M3_${var}.txt
    
    
    #INDEPENDENT
    python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_5_0_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.8 2>&1 | tee out_server_M3_5_0_OFFSETM3_idp_${var}.txt4
    
    python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_5_1_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.8 2>&1 | tee out_server_M3_5_1_OFFSETM3_idp_${var}.txt4
    
    python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_5_2_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.8 2>&1 | tee out_server_M3_5_2_OFFSETM3_idp_${var}.txt4
    
    python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_5_3_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.18.0.8 2>&1 | tee out_server_M3_5_3_OFFSETM3_idp_${var}.txt4
    
    python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn   'M3_5_4_ddf_MLP.pkl' -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl'   -ip 172.19.0.8 2>&1 | tee out_server_M3_5_4_OFFSETM3_idp_${var}.txt4
    
    
    #FEDERATED

    # #_FedAvg
    # echo -n "_FedAvg"
    # python3 fl_testbed/version2/server/federated_server_OFFSET_FedAvg.py   -cm 5 -e 1 --rounds 100 -ip  172.18.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvg_${var}.txt
    
    
    
    for var2 in $slr; do
        echo $var2
        
        
        #_FedAvgM
        # for var3 in $FedAvgM_momentum; do
        #     echo $var3
        #     echo -n "_FedAvgM"
        #     python3 fl_testbed/version2/server/federated_server_OFFSET_FedAvgM.py -momentum ${var3} -slr ${var2}  -cm 5 -e 1 --rounds 100 -ip  172.18.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedAvgM_${var}_slr_${var2}_${var3}.txt
        #     echo "done"
        # done

        #_FedOpt
        for var4 in $FedOpt_tau; do
            echo $var4
            echo -n "_FedOpt"
            python3 fl_testbed/version2/server/federated_server_OFFSET_FedOpt.py -tau ${var4}  -slr ${var2}  -cm 5 -e 1 --rounds 100 -ip  172.18.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedOpt_${var}_slr_${var2}_${var4}.txt
            echo "done"
        done
        
        # #_QFedAvg
        # for var5 in $QFedAvg_q; do
        #     echo $var5
        #     echo -n "_QFedAvg"
        #     python3 fl_testbed/version2/server/federated_server_OFFSET_QFedAvg.py  -q ${var5} -slr ${var2}  -cm 5 -e 1 --rounds 100 -ip  172.18.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_QFedAvg_${var}_slr_${var2}_${var5}.txt
        #     echo "done"
        # done
        
        echo "done"
    done
    
    echo "done"
done
