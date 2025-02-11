GPU=0
model_base=./ckpt
model_name_or_path=$model_base/PersuGPT

CUDA_VISIBLE_DEVICES=$GPU python src/PersuGPT_cli_demo.py \
    --model_name_or_path $model_name_or_path \
    --template default \
    --temperature 0.75 \
    --top_p 0.7 \
    --repetition_penalty 1.01 \
    --finetuning_type freeze