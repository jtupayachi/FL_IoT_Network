#!/bin/bash
cd /workspace
alphas="0.001 0.01 0.1 1.0"
slr_values="0.001 0.01 0.1"

for alpha in $alphas; do
    for slr_val in $slr_values; do
        echo "DASHA - alpha: $alpha, slr: $slr_val"
        python3 fl_testbed/version2/server/federated_server_RUL_DASHA.py \
            -tau 1.0 -slr ${slr_val} -cm 5 -e 1 --rounds 100 \
            -ip 172.18.4.10 \
            -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
            -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
            -dfn 'M3_5_0_ddf_LSTM.pkl' \
            2>&1 | tee LSTM_DASHA_alpha_${alpha}_slr_${slr_val}.txt
        echo "DASHA alpha=$alpha slr=$slr_val done"
    done
done
echo "=== DASHA ALL DONE ==="
