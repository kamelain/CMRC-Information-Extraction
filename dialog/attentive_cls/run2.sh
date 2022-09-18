my_cls_type=bg
my_cls_numb=10
my_batch_size=8
my_output_dir=bert_out/${my_cls_type}_${my_cls_numb}_std
my_quac_path=cls/quac_${my_cls_type}/QuAC${my_cls_numb}
my_learning_rate=3e-5   

if [ ! -d ${my_output_dir} ]
then
    mkdir ${my_output_dir}
    mkdir ${my_output_dir}/quac/
fi

CUDA_VISIBLE_DEVICES=2 python cqa_run_his_atten.py \
	--max_considered_history_turns=4 \
	--num_train_epochs=20.0 \
	--train_steps=58000 \
	--learning_rate=${my_learning_rate} \
	--n_best_size=20 \
	--better_hae=True \
	--MTL=True \
	--MTL_lambda=0.1 \
	--MTL_mu=0.8 \
	--train_batch_size=${my_batch_size} \
	--predict_batch_size=${my_batch_size} \
	--evaluate_after=50000 \
	--evaluation_steps=1000 \
	--fine_grained_attention=True \
	--bert_hidden=768 \
	--max_answer_length=50 \
	--load_small_portion=False \
	--cache_dir=cache_large/  \
	--mtl_input=reduce_mean \
	--quac_train_file=${my_quac_path}/train_v0.2.json \
	--quac_predict_file=${my_quac_path}/val_v0.2.json \
	--warmup_proportion=0.1 > ${my_cls_type}_${my_cls_numb}_${my_learning_rate}.log

unset "${!my_@}"