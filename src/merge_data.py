import logging
import argparse
import pandas as pd


def main():
    """Runs data processing scripts"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+")
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.info("--MERGE DATAFRAME--")
    logger.info(f"config arguments: {args}")

    dfs = []
    for df_path in args.data:
        df = pd.read_csv(df_path)
        dfs.append(df)

    dfs = pd.concat(dfs)

    logger.info(f"Data size: {len(dfs)}")
    logger.info(f"is noise (label 0) has {len(dfs[dfs.target == 0])} signals")
    logger.info(f"is CW (label 1) has {len(dfs[dfs.target == 1])} signals")
    logger.info(f"Target distribution - target 0: {len(dfs[dfs.target == 0]) / len(dfs)}, target 1: {len(dfs[dfs.target == 1]) / len(dfs)}")

    dfs.to_csv(args.output, index=False)
    logger.info(f"Data saved to {args.output}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
