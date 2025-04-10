from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_squared_error
import json, pickle, numpy as np

models = ["MLPRegressor", "PARegressor", "SGDRegressor"]

class ModelTrainer:
    @staticmethod
    def fit_models(X, y):
        """
        Trains all three models on new data and saves the best of them.
        ### Args:
        X: training objects
        y: training responses
        """
        val_len = int(0.1 * len(X))
        X_train, y_train = X[val_len:, :], y[val_len:]
        X_val, y_val = X[:val_len, :], y[:val_len]

        with open("config.json") as f:
            conf_dict = json.load(f)
        idx = conf_dict["model_version"]
        if idx == 0:
            m1 = MLPRegressor()
            m2 = SGDRegressor()
            m3 = PassiveAggressiveRegressor()
        else:
            with open(f"models/MLPRegressor/model_{idx - 1}", "rb") as f:
                m1 = pickle.load(f)
            with open(f"models/SGDRegressor/model_{idx - 1}", "rb") as f:
                m2 = pickle.load(f)
            with open(f"models/PARegressor/model_{idx - 1}", "rb") as f:
                m3 = pickle.load(f)
        
        m1.partial_fit(X_train, y_train)
        m2.partial_fit(X_train, y_train)
        m3.partial_fit(X_train, y_train)

        metr1, metr2, metr3 = mean_squared_error(y_val, m1.predict(X_val)), mean_squared_error(y_val, m2.predict(X_val)), mean_squared_error(y_val, m3.predict(X_val))
        best_model = [m1, m2, m3][np.argmin(np.array([metr1, metr2, metr3]))]

        with open("models/best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)

        conf_dict["model_version"] += 1
        with open("config.json", "w") as f:
            json.dump(conf_dict, f)

        with open(f"models/MLPRegressor/model_{idx}", "wb") as f:
            pickle.dump(m1, f)
        with open(f"models/SGDRegressor/model_{idx}", "wb") as f:
            pickle.dump(m2, f)
        with open(f"models/PARegressor/model_{idx}", "wb") as f:
            pickle.dump(m3, f)

        with open("summary/MLPRegressor.txt", "a") as f:
            print(metr1, file=f)
        with open("summary/SGDRegressor.txt", "a") as f:
            print(metr2, file=f)
        with open("summary/PARegressor.txt", "a") as f:
            print(metr3, file=f)