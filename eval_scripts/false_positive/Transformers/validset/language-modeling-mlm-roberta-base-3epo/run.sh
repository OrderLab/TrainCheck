python run_mlm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path FacebookAI/roberta-base \
    --output_dir /tmp/test-mlm \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2