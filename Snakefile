config_path = "configs/spectrogram-all-signal.yml"


configfile: config_path


# Trained model name
model_description = "spectrogramm+large-kernel-proc"
model_name = config["model_name"] + "_" + model_description

# Submit name
model_featured = ""
submit_description = model_description + model_featured
submit_name = config["model_name"] + "_" + submit_description

processing = config["processing"]


rule all:
    input:
        f"data/submission/{submit_name}.csv",


rule test_model:
    input:
        f"data/processed/test_{processing}",
        f"data/raw/g2net-detecting-continuous-gravitational-waves/sample_submission.csv",
        f"data/trained_models/{model_name}.pt",
    output:
        f"data/submission/{submit_name}.csv",
    shell:
        "python3 -m src.inference --data_path {input[0]} "
        "     --data_csv_path {input[1]} "
        "     --models_path {input[2]} "
        "     --submission_path {output} "
        f"    --config_path {config_path} "


rule processing_test_data:
    input:
        "data/raw/g2net-detecting-continuous-gravitational-waves/test",
        "data/raw/g2net-detecting-continuous-gravitational-waves/sample_submission.csv",
    output:
        directory(f"data/processed/test_{processing}"),
        f"data/processed/test_labels_{processing}.csv",
    shell:
        "python -m src.processing --data {input[0]} "
        "    --data_csv {input[1]} "
        "    --output {output[0]} "
        "    --output_csv {output[1]} "
        f"   --mode test_{processing} "
        f"   --config_path {config_path} "


rule train_model:
    input:
        f"data/processed/",
        f"data/processed/all_data_labels_{processing}.csv",
    output:
        f"data/trained_models/{model_name}.pt",
    shell:
        "python -m src.train --data_path {input[0]}"
        "     --data_csv_path {input[1]} "
        "     --model_save_path {output} "
        f"    --config_path {config_path} "


rule merge_train_data:
    input:
        f"data/processed/train_labels_{processing}.csv",
        f"data/processed/generated_noise_{processing}.csv",
        f"data/processed/generated_signal_{processing}.csv",
    output:
        f"data/processed/all_data_labels_{processing}.csv",
    shell:
        "python -m src.merge_data --data {input[0]} {input[1]} {input[2]} "
        "    --output {output[0]}  "


rule processing_train_data:
    input:
        "data/raw/g2net-detecting-continuous-gravitational-waves/train",
        "data/raw/g2net-detecting-continuous-gravitational-waves/train_labels.csv",
    output:
        directory(f"data/processed/train_{processing}"),
        f"data/processed/train_labels_{processing}.csv",
    shell:
        "python -m src.processing --data {input[0]} "
        "    --data_csv {input[1]} "
        "    --output {output[0]} "
        "    --output_csv {output[1]} "
        f"   --mode train_{processing} "
        f"   --config_path {config_path} "


rule generate_processed_signal_data:
    output:
        directory(f"data/processed/generated_signal_{processing}"),
        f"data/processed/generated_signal_{processing}.csv",
    shell:
        "python -m src.data_generation --output {output[0]} "
        "    --output_csv {output[1]} "
        f"   --config_path {config_path} "
        f"   --data_type generated_signal > /dev/null 2>&1 "
        #" > /dev/null 2>&1  "


rule generate_processed_noise_data:
    output:
        directory(f"data/processed/generated_noise_{processing}"),
        f"data/processed/generated_noise_{processing}.csv",
    shell:
        "python -m src.data_generation --output {output[0]} "
        "    --output_csv {output[1]} "
        f"   --config_path {config_path}          "
        f"   --data_type generated_noise > /dev/null 2>&1 "
        # > /dev/null 2>&1 "
