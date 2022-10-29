import torch
import pandas as pd

from dataset import Dataset
from train import evaluate


def predict(model_name, model_path, data_path, batch_size, n_workers, device, logger):
    # Load model (if necessary)
    model = Model(model_name, pretrained=False)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict
    submit = pd.read_csv(os.path.join(data_path, "/sample_submission.csv"))
    dataset_test = Dataset("test", data_path.submit)
    loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, num_workers=n_workers, pin_memory=True
    )

    test = evaluate(model, loader_test, device, compute_score=False, pbar=len(submit))

    # Write prediction
    submit["target"] = test["y_pred"]
    submit.to_csv("submission.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="tf_efficientnet_b5_ns")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_workers", type=int, default=6)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--INFERENCE MODEL--")
    predict(
        args.model_name,
        args.model_path,
        args.data_path,
        args.batch_size,
        args.n_workers,
        args.device,
        logger,
    )


if __name__ == "__main__":
    main()
