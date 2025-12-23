#!/bin/bash
cd /workspace
alphas="0.001 0.01 0.1 1.0"
tau_values="0.01 0.1 1.0"  # Vary tau since method params not implemented

for alpha in $alphas; do
    for tau_val in $tau_values; do
        echo "MOON - alpha: $alpha, tau: $tau_val"
        python3 fl_testbed/version2/server/federated_server_RUL_MOON.py \
            -mu 1.0 -temperature 0.5 -tau ${tau_val} -slr 0.01 -cm 5 -e 1 --rounds 100 \
            -ip 172.18.1.10 \
            -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
            -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
            -dfn 'M3_5_0_ddf_LSTM.pkl' \
            2>&1 | tee LSTM_MOON_alpha_${alpha}_tau_${tau_val}.txt
        echo "MOON alpha=$alpha tau=$tau_val done"
    done
done
echo "=== MOON ALL DONE ==="
