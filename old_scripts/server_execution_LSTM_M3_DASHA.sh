#!/bin/bash
# Auto-generated server script for DASHA

alphas="0.001 0.01 0.1 1.0"
slr="0.001 0.01 0.1"

DASHA_alpha="0.01 0.1 0.5"
DASHA_gamma="0.3 0.5 0.7"
DASHA_momentum="0.7 0.9 0.95"

for alpha in $alphas; do
    echo "Alpha: $alpha"
    for lr in $slr; do
        for alpha_val in $DASHA_alpha; do
            for gamma_val in $DASHA_gamma; do
                for mom in $DASHA_momentum; do
                    echo "DASHA - alpha: $alpha_val, gamma: $gamma_val, momentum: $mom, slr: $lr"
                    python3 fl_testbed/version2/server/federated_server_RUL_DASHA.py \
                        -alpha ${alpha_val} -gamma ${gamma_val} -momentum ${mom} -slr ${lr} -cm 5 -e 1 --rounds 100 \
                        -ip 172.18.4.10 \
                        -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
                        -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
                        -dfn 'M3_5_0_ddf_LSTM.pkl'
                done
            done
        done
    done
done
