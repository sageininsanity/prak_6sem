import pandas as pd
from pipeline.common import NUM_FEATURES, CAT_FEATURES
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import os, pickle, numpy as np

class DataPreprocessor:
    @staticmethod
    def transform(batch: pd.DataFrame):
        """
        Transforms a batch for it to be fed to the model.
        ### Args
        batch: batch to be transformed.
        """
        batch["HAS_CLAIM"] = ~batch["CLAIM_PAID"].isna()
        batch["DURATION"] = (pd.to_datetime(batch["INSR_END"], format=r'%d-%b-%y') - pd.to_datetime(batch["INSR_BEGIN"], format=r'%d-%b-%y')).dt.days
        batch["CLAIM_PAID"] = batch["CLAIM_PAID"].fillna(0) # NaN means no claim
        if os.path.exists("encoder.pkl"):
            with open("encoder.pkl", "rb") as f:
                encoder = pickle.load(f)
        else:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoder.fit(batch[CAT_FEATURES])
        imputer_cat = SimpleImputer(missing_values=pd.NA, strategy="most_frequent")
        imputer_num = SimpleImputer(missing_values=pd.NA, strategy="median")
        batch[CAT_FEATURES] = imputer_cat.fit_transform(batch[CAT_FEATURES].astype("category"))
        batch[NUM_FEATURES] = imputer_num.fit_transform(batch[NUM_FEATURES])
        cat = encoder.transform(batch[CAT_FEATURES])
        with open("encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
        ft = NUM_FEATURES + ["DURATION", "HAS_CLAIM"]
        num = np.array(batch[ft], dtype=np.float64)
        return np.concatenate([num, cat], axis=1), np.array(batch["CLAIM_PAID"])