#!/bin/bash
# New FL Methods: MOON, FedALA, StatAvg, DASHA for MLP (OFFSET)
# Complete parameter coverage for all methods

# Core parameters
alphas="0.001 0.01 0.1 1.0"
slr="0.001 0.01 0.1"

# MOON parameters: mu (contrastive loss weight), temperature
MOON_mu="0.5 1.0 5.0"
MOON_temperature="0.1 0.5 1.0"

# FedALA parameters: eta (adaptive learning rate), threshold
FedALA_eta="0.01 0.1 1.0"
FedALA_threshold="0.1 0.5 1.0"

# StatAvg parameters: stat_weight, use_variance
StatAvg_stat_weight="0.01 0.1 0.5"
StatAvg_use_variance="true false"

# DASHA parameters: alpha (step size), gamma (compression), momentum
DASHA_alpha="0.01 0.1 0.5"
DASHA_gamma="0.3 0.5 0.7"
DASHA_momentum="0.7 0.9 0.95"

for var in $alphas; do
    echo "Alpha: $var"
    
    # Data split
    python3 fl_testbed/version2/client/dirichelet_split.py -data_X_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_train.pkl -data_X_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_vals.pkl -data_y_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_train.pkl -data_y_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_vals.pkl -cm 5 -alpha ${var} -beta 0.2 -motor 3 -type MLP 2>&1 | tee DATASPLIT_${var}_MLP_M3_${var}.txt
    
    # INDEPENDENT training for all clients
    for client in 0 1 2 3 4; do
        python3 fl_testbed/version2/client/independent.py  -ml 1  -cn 15 -cm 15 -e 30 -dfn "M3_5_${client}_ddf_MLP.pkl" -dfn_test_x '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -ip 172.18.0.8 2>&1 | tee out_server_M3_5_${client}_OFFSETM3_idp_${var}.txt4
    done
    
    # MOON Method
    for var2 in $slr; do
        for mu in $MOON_mu; do
            for temp in $MOON_temperature; do
                echo "MOON - mu: $mu, temperature: $temp, slr: $var2"
                python3 fl_testbed/version2/server/federated_server_OFFSET_MOON.py -mu ${mu} -temperature ${temp} -slr ${var2} -cm 5 -e 1 --rounds 100 -ip 172.18.0.8 -dfn_test_x '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_MOON_${var}_slr_${var2}_mu_${mu}_temp_${temp}.txt
                echo "MOON done"
            done
        done
    done
    
    # FedALA Method
    for var2 in $slr; do
        for eta in $FedALA_eta; do
            for threshold in $FedALA_threshold; do
                echo "FedALA - eta: $eta, threshold: $threshold, slr: $var2"
                python3 fl_testbed/version2/server/federated_server_OFFSET_FedALA.py -eta ${eta} -threshold ${threshold} -slr ${var2} -cm 5 -e 1 --rounds 100 -ip 172.18.0.8 -dfn_test_x '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_FedALA_${var}_slr_${var2}_eta_${eta}_threshold_${threshold}.txt
                echo "FedALA done"
            done
        done
    done
    
    # StatAvg Method
    for var2 in $slr; do
        for stat_w in $StatAvg_stat_weight; do
            for use_var in $StatAvg_use_variance; do
                echo "StatAvg - stat_weight: $stat_w, use_variance: $use_var, slr: $var2"
                python3 fl_testbed/version2/server/federated_server_OFFSET_StatAvg.py -stat_weight ${stat_w} -use_variance ${use_var} -slr ${var2} -cm 5 -e 1 --rounds 100 -ip 172.18.0.8 -dfn_test_x '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_StatAvg_${var}_slr_${var2}_statw_${stat_w}_usevar_${use_var}.txt
                echo "StatAvg done"
            done
        done
    done
    
    # DASHA Method
    for var2 in $slr; do
        for alpha_val in $DASHA_alpha; do
            for gamma_val in $DASHA_gamma; do
                for mom in $DASHA_momentum; do
                    echo "DASHA - alpha: $alpha_val, gamma: $gamma_val, momentum: $mom, slr: $var2"
                    python3 fl_testbed/version2/server/federated_server_OFFSET_DASHA.py -alpha ${alpha_val} -gamma ${gamma_val} -momentum ${mom} -slr ${var2} -cm 5 -e 1 --rounds 100 -ip 172.18.0.8 -dfn_test_x '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_0_ddf_MLP.pkl' 2>&1 | tee MLP_TESLA_DASHA_${var}_slr_${var2}_alpha_${alpha_val}_gamma_${gamma_val}_mom_${mom}.txt
                    echo "DASHA done"
                done
            done
        done
    done
    
    echo "Alpha $var completed"
done
