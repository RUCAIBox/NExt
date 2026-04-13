MAX_TRAIN_STEP=150
TRAINED_MODEL_PATH='grpo-qwen2.5_7b_ins'
LOG_PATH='logs/extrapolate_log.log'

python src/extrapolate.py \
    --backbone_model_path ${TRAINED_MODEL_PATH}/global_step_${MAX_TRAIN_STEP}/hf_model \
    --model_folder ${TRAINED_MODEL_PATH} \
    --default_dir hf_model \
    --delta_name last_delta_svd_matrix \
    --ckpt_path ${TRAINED_MODEL_PATH}/NExt \
    --svd_topk 1 \
    --max_steps ${MAX_TRAIN_STEP} \
    --learning_rate 2e-5 \
    --epoch 1000 \
    --pred_step 405 \
    --inter_step 10 \
    --min_coef 0.5 \
    --max_coef 4.0 \
    --inter_coef 0.5 \
&> ${LOG_PATH}