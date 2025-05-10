import pandas as pd
from .common import HEADER, NUM_FEATURES, CAT_FEATURES

class RawPreprocessor:
    @staticmethod
    def process_raw_data():
        df1 = pd.read_csv("data/raw/0.csv", names=HEADER)
        df2 = pd.read_csv("data/raw/1.csv", names=HEADER)
        df = pd.concat([df1, df2], axis=0)
        num, cat = df[NUM_FEATURES], df[CAT_FEATURES]
        new_cat = pd.get_dummies(cat, dummy_na=True, columns=CAT_FEATURES)
        num.to_csv("data/preprocessed_raw/num.csv", header=False, index=False)
        new_cat.to_csv("data/preprocessed_raw/cat.csv", index=False)