from pipeline import *
import argparse, os, random, pandas as pd
from pipeline.common import HEADER
import uuid, logging, sys

if __name__ == "__main__":
    logging.basicConfig(filename="training.log", level=logging.INFO, format="%(asctime)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, required=True, choices=["inference", "update", "summary"])
    parser.add_argument("-p", "--path_to_data", type=str, required=False)

    os.makedirs("data/batches/", exist_ok=True)

    args = parser.parse_args()

    match(args.mode):
        case "inference":
            if not os.path.exists("models/best_model.pkl"):
                print("ERROR: run program in update mode first")
                exit(0)
            X = pd.read_csv(args.path_to_data, names=HEADER)
            pred = ModelManager.inference(DataPreprocessor.transform_for_inference(X))
            X["predict"] = pred
            fn = f"inference/{uuid.uuid4()}.csv"
            X.to_csv(fn)
            print(f"saved at {fn}")

        case "update":
            batch_filename = DataCollector.get_batch()
            batch = pd.read_csv(batch_filename)
            DataCollector.get_metafeatures(batch_filename)
            DataCollector.get_metafeatures()
            DataAnalyser.calc_dq_metrics(len(os.listdir("data/batches/")) - 1)
            DataAnalyser.calc_dq_metrics(0, all=True)
            batch = DataAnalyser.clean_data(batch)
            X, y = DataPreprocessor.transform(batch)
            ModelTrainer.fit_models(X, y)


        case "summary":
            if not os.path.exists("models/best_model.pkl"):
                print("ERROR: run program in update mode first")
                exit(0)
            ModelManager.summarize()