my_cls_type=bg
my_cls_numb=10
my_batch_size=4
# my_output_dir=bert_out/${my_cls_type}_${my_cls_numb}_test
my_output_dir=bert_out/check_cls${my_cls_numb}_for30000
my_quac_path=cls/quac_${my_cls_type}/QuAC${my_cls_numb}
my_learning_rate=3e-5      #[5e-5, 3e-5, 2e-5]

# kl03: 10_std_3e-5 (ham)
# kl28: 10_5e-5
# esp2080: 10_2e-5

# if [ ! -d ${my_cache_dir} ]
# then
#     mkdir ${my_cache_dir}
#     mkdir ${my_cache_dir}/cache/
# fi

if [ ! -d ${my_output_dir} ]
then
    -p mkdir ${my_output_dir}/quac/
    -p mkdir ${my_output_dir}/cache/
fi

CUDA_VISIBLE_DEVICES=0 python hae.py \
    --history=6 \
    --num_train_epochs=3.0 \
    --train_steps=24000 \
    --max_considered_history_turns=11 \
    --learning_rate=${my_learning_rate} \
    --warmup_proportion=0.1 \
    --evaluation_steps=1000 \
    --evaluate_after=22000 \
    --load_small_portion=False \
    --train_batch_size=${my_batch_size} \
    --max_answer_length=40 \
    --output_dir=${my_output_dir} \
    --cache_dir=${my_output_dir}/ \
    --quac_train_file=${my_quac_path}/train_v0.2.json \
    --quac_predict_file=${my_quac_path}/val_v0.2.json>${my_cls_type}_${my_cls_numb}_test.log
    
unset "${!my_@}"

    # https://github.com/prdwb/bert_hae

    
