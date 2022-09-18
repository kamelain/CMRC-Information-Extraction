#!/bin/bash
pypath=$(dirname $(dirname $(readlink -f $0)))
CUDA_VISIBLE_DEVICES=1 python $pypath/my_preprocess/3_my_run_app.py \
--ex_index=1 \
--device_id=0 \
--mode=app \
--corpus_type=QuAC_test \
--ensure_corres \
--ensure_rel \
--corres_threshold=0.5 \
--rel_threshold=0.1