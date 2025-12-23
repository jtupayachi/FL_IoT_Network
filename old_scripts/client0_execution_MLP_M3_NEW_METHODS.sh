#!/bin/bash
# Client 0 execution for MOON, FedALA, StatAvg, and DASHA methods - MLP

alphas="0.001 0.01 0.1 1.0"
slr="0.001 0.01 0.1"

# MOON parameters
MOON_mu="0.5 1.0 5.0"
MOON_temperature="0.1 0.5 1.0"

# FedALA parameters
FedALA_eta="0.01 0.1 1.0"
FedALA_threshold="0.1 0.5 1.0"

# StatAvg parameters
StatAvg_stat_weight="0.01 0.1 0.5"
StatAvg_use_variance="true false"

# DASHA parameters
DASHA_alpha="0.01 0.1 0.5"
DASHA_gamma="0.3 0.5 0.7"
DASHA_momentum="0.7 0.9 0.95"

# Wait for server to be ready
echo "Waiting for server to be ready..."
while ! nc -z 172.18.0.8 5000 2>/dev/null; do
    sleep 2
done
echo "Server is ready!"

for var in $alphas; do
    echo "Alpha: $var"
    
    # MOON Method
    for var2 in $slr; do
        for mu in $MOON_mu; do
            for temp in $MOON_temperature; do
                echo "MOON - Client 0 - mu: $mu, temperature: $temp"
                python3 fl_testbed/version2/client/federated_client_OFFSET_MOON.py -cn 0 -cm 5 -e 1 -ip 172.18.0.8 -dfn_test_x '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_0_ddf_MLP.pkl' 2>&1 | tee MLP_CLIENT0_MOON_${var}_slr_${var2}_mu_${mu}_temp_${temp}.txt
                echo "done"
            done
        done
    done
    
    # FedALA Method
    for var2 in $slr; do
        for eta in $FedALA_eta; do
            for threshold in $FedALA_threshold; do
                echo "FedALA - Client 0 - eta: $eta, threshold: $threshold"
                python3 fl_testbed/version2/client/federated_client_OFFSET_FedALA.py -cn 0 -cm 5 -e 1 -ip 172.18.0.8 -dfn_test_x '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_0_ddf_MLP.pkl' 2>&1 | tee MLP_CLIENT0_FedALA_${var}_slr_${var2}_eta_${eta}_threshold_${threshold}.txt
                echo "done"
            done
        done
    done
    
    # StatAvg Method
    for var2 in $slr; do
        for stat_w in $StatAvg_stat_weight; do
            for use_var in $StatAvg_use_variance; do
                echo "StatAvg - Client 0 - stat_weight: $stat_w, use_variance: $use_var"
                python3 fl_testbed/version2/client/federated_client_OFFSET_StatAvg.py -cn 0 -cm 5 -e 1 -ip 172.18.0.8 -dfn_test_x '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_0_ddf_MLP.pkl' 2>&1 | tee MLP_CLIENT0_StatAvg_${var}_slr_${var2}_statw_${stat_w}_usevar_${use_var}.txt
                echo "done"
            done
        done
    done
    
    # DASHA Method
    for var2 in $slr; do
        for alpha_val in $DASHA_alpha; do
            for gamma_val in $DASHA_gamma; do
                for mom in $DASHA_momentum; do
                    echo "DASHA - Client 0 - alpha: $alpha_val, gamma: $gamma_val, momentum: $mom"
                    python3 fl_testbed/version2/client/federated_client_OFFSET_DASHA.py -cn 0 -cm 5 -e 1 -ip 172.18.0.8 -dfn_test_x '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_test.pkl' -dfn_test_y '100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_test.pkl' -dfn 'M3_5_0_ddf_MLP.pkl' 2>&1 | tee MLP_CLIENT0_DASHA_${var}_slr_${var2}_alpha_${alpha_val}_gamma_${gamma_val}_mom_${mom}.txt
                    echo "done"
                done
            done
        done
    done
    
    echo "Alpha $var completed"
done
