#!/bin/bash
cd /workspace
alphas="0.001 0.01 0.1 1.0"
StatAvg_stat_weight="0.01 0.1 0.5"  # ONLY vary this

for alpha in $alphas; do
    for stat_w in $StatAvg_stat_weight; do
        echo "StatAvg - alpha: $alpha, stat_weight: $stat_w (use_variance=true, slr=0.01)"
        python3 fl_testbed/version2/server/federated_server_RUL_StatAvg.py \
            -stat_weight ${stat_w} -use_variance true -slr 0.01 -cm 5 -e 1 --rounds 100 \
            -ip 172.18.3.10 \
            -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
            -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
            -dfn 'M3_5_0_ddf_LSTM.pkl' \
            2>&1 | tee LSTM_StatAvg_alpha_${alpha}_statw_${stat_w}.txt
        echo "StatAvg alpha=$alpha stat_weight=$stat_w done"
    done
done
echo "=== StatAvg ALL DONE ==="
