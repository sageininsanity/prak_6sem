import json
import pandas as pd
import logging
import os
from .common import HEADER, IMPROVED_HEADER, NUM_FEATURES, CAT_FEATURES

class DataCollector:
    logger = logging.getLogger(__name__)
    @staticmethod
    def get_batch():
        """
        Extracts a batch from the raw storage and saves it elsewhere (stream emulation).
        """
        with open("config.json", "r") as f:    
            conf_dict = json.load(f)
        batch_numerical = pd.read_csv(f"data/preprocessed_raw/num.csv", names=NUM_FEATURES, header=None, nrows=conf_dict["batch_size"], 
                         skiprows=conf_dict["batch_offset"])
        batch_categorical = pd.read_csv(f"data/preprocessed_raw/cat.csv", names=IMPROVED_HEADER, header=None, nrows=conf_dict["batch_size"], 
                         skiprows=conf_dict["batch_offset"])
        batch = pd.concat([batch_numerical, batch_categorical], axis=1)
        batch_filename = f"data/batches/batch_{conf_dict['batch_num']}.csv"
        batch.to_csv(batch_filename, index=False)
        conf_dict["batch_num"] += 1
        conf_dict["batch_offset"] += conf_dict["batch_size"]
        with open("config.json", "w") as f:
            json.dump(conf_dict, f)
        DataCollector.logger.info(f"Batch received, saved at {batch_filename}.")
        return batch_filename
    
    @staticmethod
    def get_whole_df():
        """
        Collates all the available data in a single pandas DataFrame.
        """
        i = 1
        df = pd.read_csv("data/batches/batch_0.csv")
        while (path := f"data/batches/batch_{i}.csv") and os.path.exists(path):
            batch = pd.read_csv(path)
            df = pd.concat([df, batch], axis=0)
            i += 1
        return df

    @staticmethod
    def get_metafeatures(batch_filename=None):
        """
        Computes some meta features (i.e, rate of missing values).
        ### Args:
        batch_filename: the name of the batch to extract metaparameters from. If None (default), metaparameters are gathered across ALL the data.
        """
        df = DataCollector.get_whole_df() if batch_filename is None else pd.read_csv(batch_filename)
        key = "ALL" if batch_filename is None else f"{batch_filename.split('_')[-1].split('.')[0]}"
        with open("meta.json", "r") as f:
            meta = json.load(f)
        meta[key] = {}
        meta[key]["n_rows"] = len(df)
        for col in CAT_FEATURES:
            cols = [name for name in IMPROVED_HEADER if (col in name and "_nan" not in name and df[name].any())]
            meta[key][f"{col}_distr"] = list(df[cols].sum(axis=1) / df[cols].sum())
        meta[key]["missing_rate"] = float((df[NUM_FEATURES].isnull().sum().sum() + sum([df[name].sum() for name in IMPROVED_HEADER if "_nan" in name])) / (len(df) * len(CAT_FEATURES)))
        with open("meta.json", "w") as f:
            json.dump(meta, f)
        DataCollector.logger.info("Metafeatures updated and saved at meta.json.")
