import json
import pandas as pd
import logging
import os
from pipeline.common import HEADER

class DataCollector:
    logger = logging.getLogger(__name__)
    @staticmethod
    def get_batch(file_num: int):
        """
        Extracts a batch from the raw storage and saves it elsewhere (stream emulation).
        
        ### Args:
        file_num (int): randomly selected 0 or 1 (for multiple source simulation)
        """
        file_key = "file0" if file_num == 0 else "file1"
        with open("config.json", "r") as f:    
            conf_dict = json.load(f)
        batch = pd.read_csv(f"data/raw/{file_num}.csv", names=HEADER, header=None, nrows=conf_dict[file_key]["batch_size"], 
                         skiprows=conf_dict[file_key]["batch_offset"])
        batch_filename = f"data/batches/batch_{conf_dict['batch_num']}.csv"
        batch.to_csv(batch_filename, index=False)
        conf_dict["batch_num"] += 1
        conf_dict[file_key]["batch_offset"] += conf_dict[file_key]["batch_size"]
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
        for col in df.columns:
            meta[key][f"{col}_distr"] = dict(df[col].value_counts(normalize=True))
        meta[key]["missing_rate"] = df.isnull().mean().mean()
        with open("meta.json", "w") as f:
            json.dump(meta, f)
        DataCollector.logger.info("Metafeatures updated and saved at meta.json.")
