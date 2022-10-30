rule all:
    input:
        "data/submission/submission.csv",


rule test_model:
    input:
        "data/processed/test",
        "data/processed/test_labels.csv",
        "data/trained_models",
    output:
        "data/submission/submission.csv",
    shell:
        """
        python3 -m src.inference --data_path {input[0]} \
            --models_path {input[1]} \
            --submission_path {output} \
            --batch_size=24
        """


rule train_model:
    input:
        "data/processed/train",
        "data/processed/train_labels.csv",
    output:
        "data/trained_models",
    shell:
        """
        python -m src.train --data_path {input[0]} \
            --data_csv_path {input[1]} \
            --model_save_path {output} \
            --batch_size 24
        """


rule processing_test_data:
    input:
        "data/raw/g2net-detecting-continuous-gravitational-waves",
        "data/raw/g2net-detecting-continuous-gravitational-waves/sample_submission.csv",
    output:
        "data/processed/test",
        "data/processed/test_labels.csv",
    shell:
        """
        python -m src.processing --data {input[0]} \
            --data_csv {input[1]} \
            --output {output[0]} \
            --output_csv {output[1]} \
            --mode test \
            --n_workers 3
        """


rule processing_train_data:
    input:
        "data/raw/g2net-detecting-continuous-gravitational-waves",
        "data/raw/g2net-detecting-continuous-gravitational-waves/train_labels.csv",
    output:
        "data/processed/train",
        "data/processed/train_labels.csv",
    shell:
        """
        python -m src.processing --data {input[0]} \
            --data_csv {input[1]} \
            --output {output[0]} \
            --output_csv {output[1]} \
            --mode train \
            --n_workers 3
        """