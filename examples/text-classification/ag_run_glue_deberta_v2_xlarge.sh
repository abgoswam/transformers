python run_glue.py \
    --model_name_or_path microsoft/deberta-v2-xlarge \
    --task_name mnli \
    --do_train \
    --do_eval \
    --max_steps 1000 \
    --max_seq_length 128 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --logging_first_step \
    --output_dir tmp/mnli_output/
