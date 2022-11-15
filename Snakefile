submit_description = "val_loss_min"
model_name = "baseline_v3"

submit_name = model_name + "_" + submit_description
experiment = "baseline"


configfile: f"configs/{experiment}.yml"


rule all:
    input:
        f"data/submission/{submit_name}.csv",


rule test_model:
    input:
        "data/processed/test",
        "data/raw/g2net-detecting-continuous-gravitational-waves/sample_submission.csv",
        f"data/trained_models/{model_name}.pt",
    output:
        f"data/submission/{submit_name}.csv",
    shell:
        "python3 -m src.inference --data_path {input[0]} "
        f"    --model_base_type {config['model_base_type']} "
        f"    --experiment {experiment} "
        f"    --use_nfolds {config['use_nfolds']} "
        "     --data_csv_path {input[1]} "
        "     --models_path {input[2]} "
        "     --submission_path {output} "
        f"    --batch_size {config['test_batch_size']} "
        f"    --use_best_model {config['use_best_models']} "


rule processing_test_data:
    input:
        "data/raw/g2net-detecting-continuous-gravitational-waves/test",
        "data/raw/g2net-detecting-continuous-gravitational-waves/sample_submission.csv",
    output:
        directory(f"data/processed/test_{experiment}"),
        f"data/processed/test_labels_{experiment}.csv",
    shell:
        "python -m src.processing --data {input[0]} "
        "    --data_csv {input[1]} "
        "    --output {output[0]} "
        "    --output_csv {output[1]} "
        "    --n_workers 1 "
        "    --mode test "
        f"   --processing_type {experiment}"


rule train_model:
    input:
        f"data/processed/",
        f"data/processed/all_data_labels_{experiment}.csv",
    output:
        f"data/trained_models/{model_name}.pt",
    shell:
        "python -m src.train --data_path {input[0]}"
        "     --data_csv_path {input[1]} "
        "     --model_save_path {output} "
        f"     --experiment {experiment} "
        f"    --batch_size {config['batch_size']} "
        f"    --epochs {config['epochs']} "
        f"    --model_base_type {config['model_base_type']} "
        f"    --nfold {config['nfold']} "
        f"    --n_workers {config['n_workers']} "
        f"    --weight_decay {config['weight_decay']} "
        f"    --max_grad_norm {config['max_grad_norm']} "
        f"    --lr_max {config['lr_max']} "
        f"    --epochs_warmup {config['epochs_warmup']} "
        f"    --flip_rate {config['flip_rate']} "
        f"    --freq_shift_rate {config['freq_shift_rate']} "
        f"    --time_mask_num {config['time_mask_num']} "
        f"    --freq_mask_num {config['freq_mask_num']} "
        f"    --augmentaion_train {config['augmentaion_train']} "
        f"    --pretrained {config['pretrained']}"
        f"    --random_state {config['random_state']} "


rule merge_train_data:
    input:
        f"data/processed/train_labels_{experiment}.csv",
        f"data/processed/generated_noise_{experiment}.csv",
        f"data/processed/generated_signal_{experiment}.csv",
    output:
        f"data/processed/all_data_labels_{experiment}.csv",
    shell:
        "python -m src.merge_data --data {input[0]} {input[1]} {input[2]} "
        "    --output {output[0]}  "


rule processing_train_data:
    input:
        "data/raw/g2net-detecting-continuous-gravitational-waves/train",
        "data/raw/g2net-detecting-continuous-gravitational-waves/train_labels.csv",
    output:
        directory(f"data/processed/train_{experiment}"),
        f"data/processed/train_labels_{experiment}.csv",
    shell:
        "python -m src.processing --data {input[0]} "
        "    --data_csv {input[1]} "
        "    --output {output[0]} "
        "    --output_csv {output[1]} "
        "    --n_workers 1 "
        "    --mode train "
        f"   --processing_type {experiment} "


rule generate_processed_signal_data:
    output:
        directory(f"data/processed/generated_signal_{experiment}"),
        f"data/processed/generated_signal_{experiment}.csv",
    shell:
        "python -m src.data_generation --output {output[0]} "
        "    --output_csv {output[1]} "
        f"   --num_signals {config['num_signals']} "
        f"   --processing {experiment} "
        f"   --random_state {config['random_state']} "
        f"   --data_type generated_signal > /dev/null 2>&1  "


rule generate_processed_noise_data:
    output:
        directory(f"data/processed/generated_noise_{experiment}"),
        f"data/processed/generated_noise_{experiment}.csv",
    shell:
        "python -m src.data_generation --output {output[0]} "
        "    --output_csv {output[1]} "
        f"   --num_signals {config['num_noise']} "
        f"   --processing {experiment} "
        f"   --random_state {config['random_state']} "
        f"   --data_type generated_noise > /dev/null 2>&1 "
