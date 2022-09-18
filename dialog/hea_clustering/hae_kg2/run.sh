my_batch_size=1
my_kg_type=prgc
my_type_name=${my_kg_type}_kpos3
# my_type_name=cls_2
my_output_dir=bert_out/${my_type_name}/
# my_quac_path=wordnet/dataset_wn/quac10_v5
# my_quac_path=plsa/dataset_plsa/quac10_v2/
my_quac_path=prgc/
# my_quac_path=dataset_plsa/quac10_v1/
my_learning_rate=3e-5      #[5e-5, 3e-5, 2e-5]

###
# --cache_dir=${my_output_dir}/cache/ \
###

if [ ! -d ${my_output_dir} ]
then
    mkdir -p ${my_output_dir}/quac/
    mkdir -p ${my_output_dir}/cache/
fi

CUDA_VISIBLE_DEVICES=2 python hae.py \
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
    --quac_predict_file=${my_quac_path}/val_v0.2.json>run_${my_kg_type}_${my_type_name}.log
    
unset "${!my_@}"

    # https://github.com/prdwb/bert_hae

    # --evaluation_steps=1000 \ 2
    # --evaluate_after=22000 \  10
    # --train_steps=24000 \     16
