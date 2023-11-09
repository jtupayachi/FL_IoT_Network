#!/bin/bash
#MLP
#${var}

alphas="0.001 0.01 0.1 0.02 0.2 0.005 0.05 0.5 0.075 1.0 1000000.0"
slr="0.001 0.01 1"

FedAvgM_momentum="0.0 0.7 0.9"
# FedOpt_tau="1e-7 1e-8 1e-9"
FedOpt_tau="1e-8"
QFedAvg_q="0.1 0.2 0.5"

for var in $alphas; do
    echo $var
    sleep 500


    # #_FedAvg
    # echo -n "_FedAvg"
    # python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvg.py -cn 4  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_4_ddf_MLP.pkl' 2>&1 | tee MLP_CLIENT5_FedAvg_${var}.txt
    
    
    for var2 in $slr; do
        echo $var2
        
        
        #_FedAvgM
        # for var3 in $FedAvgM_momentum; do
        #     echo $var3
        #     sleep 300
        #     echo -n "_FedAvgM"
        #     python3 fl_testbed/version2/client/federated_client_OFFSET_FedAvgM.py -cn 4  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_4_ddf_MLP.pkl' 2>&1 | tee MLP_CLIENT5_FedAvgM_${var}_slr_${var2}_${var3}.txt
        #     echo "done"
        # done

        #_FedOpt
        for var4 in $FedOpt_tau; do
            echo $var4
            sleep 300
            echo -n "_FedOpt"
            python3 fl_testbed/version2/client/federated_client_OFFSET_FedOpt.py -cn 4  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_4_ddf_MLP.pkl' 2>&1 | tee MLP_CLIENT5_FedOpt_${var}_slr_${var2}_${var4}.txt
            echo "done"
        done

        # #_QFedAvg
        # for var5 in $QFedAvg_q; do
        #     echo $var5
        #     sleep 300
        #     echo -n "_QFedAvg"
        #     python3 fl_testbed/version2/client/federated_client_OFFSET_QFedAvg.py -cn 4  -cm 5 -e 1  -ip  172.17.0.8  -dfn_test_x   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y   '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_4_ddf_MLP.pkl' 2>&1 | tee MLP_CLIENT5_QFedAvg_${var}_slr_${var2}_${var5}.txt
        #     echo "done"
        # done
        
        echo "done"
    done
    
    echo "done"
done
