my_batch_size=2
my_type_name=kbert_dyn_pos_small
my_output_dir=bert_out/${my_type_name}
my_quac_path=wordnet/dataset_wn/quac10_v4/small
my_learning_rate=3e-5      #[5e-5, 3e-5, 2e-5]

if [ ! -d ${my_output_dir} ]
then
    mkdir -p ${my_output_dir}/quac/
    mkdir -p ${my_output_dir}/cache/
fi

CUDA_VISIBLE_DEVICES=1 python hae.py \
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
    --cache_dir=${my_output_dir}/cache/ \
    --quac_train_file=${my_quac_path}/train_v0.2.json \
    --quac_predict_file=${my_quac_path}/val_v0.2.json>run_wn_${my_type_name}.log
    
unset "${!my_@}"

    # https://github.com/prdwb/bert_hae
