rule all:
    input:
        "data/submission/baseline_with_noise_gen_submit_effi6.csv",


rule test_baseline_model:
    input:
        "data/processed/test",
        "data/raw/g2net-detecting-continuous-gravitational-waves/sample_submission.csv",
        "data/trained_models/baseline_noise_effi6.pt",
    output:
        "data/submission/baseline_with_noise_gen_submit_effi6.csv",
    shell:
        """
        python3 -m src.inference --data_path {input[0]} \
            --model_name tf_efficientnet_b6_ns \
            --data_csv_path {input[1]} \
            --models_path {input[2]} \
            --submission_path {output} \
            --batch_size 64 
        """


rule train_baseline_model:
    input:
        "data/processed/",
        "data/processed/all_data_labels.csv",
    output:
        "data/trained_models/baseline_noise_effi6.pt",
    shell:
        """
        python -m src.train --data_path {input[0]}\
            --data_csv_path {input[1]} \
            --model_save_path {output} \
            --batch_size 18 \
            --epochs 8 \
            --model_type tf_efficientnet_b6_ns
        """


rule merge_train_data:
    input:
        "data/processed/train_labels.csv",
        "data/processed/generated_noise.csv",
    output:
        "data/processed/all_data_labels.csv",
    shell:
        """
        python -m src.merge_data --data {input[0]} {input[1]} \
            --output {output[0]} \
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


rule processing_generated_noise_data:
    input:
        "data/generated/noise",
        "data/generated/noise.csv",
    output:
        directory("data/processed/generated_noise"),
        "data/processed/generated_noise.csv",
    shell:
        """
        python -m src.processing --data {input[0]} \
            --data_csv {input[1]} \
            --output {output[0]} \
            --output_csv {output[1]} \
            --n_workers 1 \
            --mode generated_noise
        """


rule generate_noise_data:
    output:
        directory("data/generated/noise"),
        "data/generated/noise.csv",
    shell:
        """
        python3 -m src.data_generation --output {output[0]} \
        --output_csv {output[1]} \
        --num_signals 500 \
        --data_type noise  > /dev/null 2>&1
        """
