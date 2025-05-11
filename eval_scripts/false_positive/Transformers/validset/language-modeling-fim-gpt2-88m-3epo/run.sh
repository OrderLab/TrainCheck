python run_fim_no_trainer.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path distilbert/distilgpt2 \
    --fim_rate 0.5 \
    --fim_spm_rate 0.2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --output_dir /tmp/test-clm