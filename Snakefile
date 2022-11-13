sumbit_description = "baseline_with_noise_10000_noisedsignal_10000_random_F0_fixed_random_state_augmentations_best_model_efficientb7"


rule all:
    input:
        f"data/submission/{sumbit_description}.csv",


rule test_baseline_model:
    input:
        "data/processed/test",
        "data/raw/g2net-detecting-continuous-gravitational-waves/sample_submission.csv",
        f"data/trained_models/{sumbit_description}.pt",
    output:
        f"data/submission/{sumbit_description}.csv",
    shell:
        """
        python3 -m src.inference --data_path {input[0]} \
            --model_name tf_efficientnet_b7_ns \
            --data_csv_path {input[1]} \
            --models_path {input[2]} \
            --submission_path {output} \
            --batch_size 64 
        """


rule processing_test_data:
    input:
        "data/raw/g2net-detecting-continuous-gravitational-waves/test",
        "data/raw/g2net-detecting-continuous-gravitational-waves/sample_submission.csv",
    output:
        directory("data/processed/test"),
        "data/processed/test_labels.csv",
    shell:
        """
        python -m src.processing --data {input[0]} \
            --data_csv {input[1]} \
            --output {output[0]} \
            --output_csv {output[1]} \
            --n_workers 1 \
            --mode test
        """


rule train_baseline_model:
    input:
        "data/processed/",
        "data/processed/all_data_labels.csv",
    output:
        f"data/trained_models/{sumbit_description}.pt",
    shell:
        """
        python -m src.train --data_path {input[0]}\
            --data_csv_path {input[1]} \
            --model_save_path {output} \
            --batch_size 14 \
            --epochs 8 \
            --augmentaion_train \
            --save_best_model \
            --pretrained \
            --model_type tf_efficientnet_b7_ns
        """


rule merge_train_data:
    input:
        "data/processed/train_labels.csv",
        "data/processed/generated_noise.csv",
        "data/processed/generated_signal.csv",
    output:
        "data/processed/all_data_labels.csv",
    shell:
        """
        python -m src.merge_data --data {input[0]} {input[1]} \
            --output {output[0]} \
        """


rule processing_train_data:
    input:
        "data/raw/g2net-detecting-continuous-gravitational-waves/train",
        "data/raw/g2net-detecting-continuous-gravitational-waves/train_labels.csv",
    output:
        directory("data/processed/train"),
        "data/processed/train_labels.csv",
    shell:
        """
        python -m src.processing --data {input[0]} \
            --data_csv {input[1]} \
            --output {output[0]} \
            --output_csv {output[1]} \
            --n_workers 1 \
            --mode train
        """


rule generate_processed_signal_data:
    output:
        directory("data/processed/generated_signal"),
        "data/processed/generated_signal.csv",
    shell:
        """
        python3 -m src.data_generation --output {output[0]} \
        --output_csv {output[1]} \
        --num_signals 10000 \
        --processing baseline \
        --data_type generated_signal  > /dev/null 2>&1 
        """
        # > /dev/null 2>&1 


rule generate_processed_noise_data:
    output:
        directory("data/processed/generated_noise"),
        "data/processed/generated_noise.csv",
    shell:
        """
        python3 -m src.data_generation --output {output[0]} \
        --output_csv {output[1]} \
        --num_signals 10000 \
        --processing baseline \
        --data_type generated_noise > /dev/null 2>&1 
        """
