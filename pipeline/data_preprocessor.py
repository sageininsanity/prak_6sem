import pandas as pd
from .common import NUM_FEATURES, CAT_FEATURES, IMPROVED_HEADER
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
        nf = [ft for ft in NUM_FEATURES if ft not in ["INSR_BEGIN", "INSR_END"]]
        batch["OLD"] = batch["PROD_YEAR"] <= 2000
        batch["IN_USE"] = pd.to_datetime(batch["INSR_BEGIN"], format=r'%d-%b-%y').dt.year - batch["PROD_YEAR"]
        batch["DURATION"] = (pd.to_datetime(batch["INSR_END"], format=r'%d-%b-%y') - pd.to_datetime(batch["INSR_BEGIN"], format=r'%d-%b-%y')).dt.days
        batch["CLAIM_PAID"] = batch["CLAIM_PAID"].fillna(0) # NaN means no claim
        imputer_num = SimpleImputer(missing_values=pd.NA, strategy="median")
        batch[nf] = imputer_num.fit_transform(batch[nf])
        ft = nf + ["DURATION", "OLD", "IN_USE"]
        num = np.array(batch[ft], dtype=np.float64)
        cat = np.array(batch[IMPROVED_HEADER].drop([name for name in IMPROVED_HEADER if "_nan" in name], axis=1))
        return np.concatenate([num, cat], axis=1), np.array(batch["CLAIM_PAID"])
    
    def transform_for_inference(batch: pd.DataFrame):
        """
        Transforms a batch for it to be fed to the model.
        ### Args
        batch: batch to be transformed.
        """
        nf = [ft for ft in NUM_FEATURES if ft not in ["INSR_BEGIN", "INSR_END"]]
        batch["OLD"] = batch["PROD_YEAR"] <= 2000
        batch["IN_USE"] = pd.to_datetime(batch["INSR_BEGIN"], format=r'%d-%b-%y').dt.year - batch["PROD_YEAR"]
        batch["DURATION"] = (pd.to_datetime(batch["INSR_END"], format=r'%d-%b-%y') - pd.to_datetime(batch["INSR_BEGIN"], format=r'%d-%b-%y')).dt.days
        imputer_num = SimpleImputer(missing_values=pd.NA, strategy="median")
        batch[nf] = imputer_num.fit_transform(batch[nf])
        ft = nf + ["DURATION", "OLD", "IN_USE"]
        num = np.array(batch[ft], dtype=np.float64)
        cat = np.array(batch[IMPROVED_HEADER].drop([name for name in IMPROVED_HEADER if "_nan" in name], axis=1))
        return np.concatenate([num, cat], axis=1)