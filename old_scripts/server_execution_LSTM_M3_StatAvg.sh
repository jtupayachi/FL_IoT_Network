#!/bin/bash
# Auto-generated server script for StatAvg

alphas="0.001 0.01 0.1 1.0"
slr="0.001 0.01 0.1"

StatAvg_stat_weight="0.01 0.1 0.5"
StatAvg_use_variance="true false"

for alpha in $alphas; do
    echo "Alpha: $alpha"
    for lr in $slr; do
        for stat_w in $StatAvg_stat_weight; do
            for use_var in $StatAvg_use_variance; do
                echo "StatAvg - stat_weight: $stat_w, use_variance: $use_var, slr: $lr"
                python3 fl_testbed/version2/server/federated_server_RUL_StatAvg.py \
                    -stat_weight ${stat_w} -use_variance ${use_var} -slr ${lr} -cm 5 -e 1 --rounds 100 \
                    -ip 172.18.3.10 \
                    -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
                    -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
                    -dfn 'M3_5_0_ddf_LSTM.pkl'
            done
        done
    done
done
