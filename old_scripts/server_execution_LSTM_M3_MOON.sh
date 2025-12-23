#!/bin/bash
# Auto-generated server script for MOON

alphas="0.001 0.01 0.1 1.0"
slr="0.001 0.01 0.1"

MOON_mu="0.5 1.0 5.0"
MOON_temperature="0.1 0.5 1.0"

for alpha in $alphas; do
    echo "Alpha: $alpha"
    for lr in $slr; do
        for mu in $MOON_mu; do
            for temp in $MOON_temperature; do
                echo "MOON - mu: $mu, temp: $temp, slr: $lr"
                python3 fl_testbed/version2/server/federated_server_RUL_MOON.py \
                    -mu ${mu} -temperature ${temp} -slr ${lr} -cm 5 -e 1 --rounds 100 \
                    -ip 172.18.1.10 \
                    -dfn_test_x '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_inputs.pkl' \
                    -dfn_test_y '100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtest_out.pkl' \
                    -dfn 'M3_5_0_ddf_LSTM.pkl'
            done
        done
    done
done
