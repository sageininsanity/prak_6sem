import json, pandas as pd, numpy as np
from pipeline.common import CAT_FEATURES

class DataAnalyser:
    @staticmethod
    def calc_dq_metrics(batch_num, all=False):
        """
        Calculates the Data Quality metrics for a batch, or, if all is True, for all of the available data.
        ### Args:
        batch_num: the index of the batch
        all: a flag for partial/whole analysis.
        """
        with open("dq.json") as f:
            dq_stat = json.load(f)

        with open("meta.json") as f:
            meta = json.load(f)
        
        if not all:
            key = f"{batch_num}"
        else:
            key = "ALL"
        
        dq_stat[key] = {}
        dq_stat[key]["Completeness"] = meta[key]["missing_rate"]
        entropies = []
        for col in CAT_FEATURES:
            p = np.array(list(meta[key][f"{col}_distr"].values()))
            entropy = np.sum(-(p * np.log(p)))
            entropies.append(float(entropy))
        dq_stat[key]["min_entropy"] = min(entropies)
        dq_stat[key]["max_entropy"] = max(entropies)
        with open("dq.json", "w") as f:
            json.dump(dq_stat, f)

    @staticmethod
    def clean_data(batch: pd.DataFrame):
        """
        Perform cleaning duties, right now it's the deletion of the outliers by some features.
        ### Args:
        batch: batch to perform cleaning on.
        """
        batch.drop_duplicates()
        OUTLIER_FT = ["PROD_YEAR", "SEATS_NUM", "CARRYING_CAPACITY"]
        quantiles = batch[OUTLIER_FT].quantile([0.25, 0.75])
        iqr = quantiles.diff().values[1]
        q1 = quantiles.values[0]
        q3 = quantiles.values[1]
        lower = pd.Series(q1 - 1.5 * iqr, batch[OUTLIER_FT].keys())
        upper = pd.Series(q3 + 1.5 * iqr, batch[OUTLIER_FT].keys())
        return batch.loc[((batch[OUTLIER_FT] >= lower) & (batch[OUTLIER_FT] <= upper)).all(axis="columns")]