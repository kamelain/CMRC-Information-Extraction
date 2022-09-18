my_batch_size=4
my_cache_dir=cache/3/					# --cache_dir=cache_large/test		cache_large/test/	
my_output_dir=bert_out/

if [ ! -d ${my_output_dir} ]
then
    mkdir ${my_output_dir}
    mkdir ${my_output_dir}/quac/
	mkdir ${my_output_dir}/coqa/
fi

if [ ! -d ${my_cache_dir} ]
then
    mkdir ${my_cache_dir}
	mkdir ${my_cache_dir}/quac/
	mkdir ${my_cache_dir}/coqa/
fi

CUDA_VISIBLE_DEVICES=3 python cqa_run_his_atten.py \
	--max_considered_history_turns=4 \
	--num_train_epochs=20.0 \
	--train_steps=58000 \
	--learning_rate=3e-5 \
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
	--cache_dir=${my_cache_dir} \
	--mtl_input=reduce_mea \
	--warmup_proportion=0.1>run_log_sabert3.log

unset "${!my_@}"