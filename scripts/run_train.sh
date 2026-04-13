TRAINED_MODEL_PATH='grpo-qwen2.5_7b_ins'
LOG_PATH='logs/train_log.log'

python src/train.py \
    --model_folder ${TRAINED_MODEL_PATH} \
    --default_dir hf_model \
    --input_delta_name delta_svd_matrix,last_delta_svd_matrix \
    --output_delta_name last_delta_svd_matrix \
    --output_path ${TRAINED_MODEL_PATH}/NExt \
    --svd_topk 1 \
    --max_steps 100 \
    --learning_rate 2e-5 \
    --epoch 1000 \
    --save_epoch 200 \
    --loss_func l1 \
&> ${LOG_PATH}