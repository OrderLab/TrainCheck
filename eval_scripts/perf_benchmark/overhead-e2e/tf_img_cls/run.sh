CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset_name beans \
    --output_dir ./beans_outputs/ \
    --label_column_name labels \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --seed 1337