CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --dataset_name cnn_dailymail \
    --model_name_or_path t5-base \
    --dataset_config 3.0.0 \
    --output_dir output \
    --max_source_length 128 \
    --per_device_train_batch_size 16 \
