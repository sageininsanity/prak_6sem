import pickle, json
import matplotlib.pyplot as plt

class ModelManager:
    @staticmethod
    def inference(X):
        with open("models/best_model.pkl", "rb") as f:
            model = pickle.load(f)

        X["predict"] = model.predict(X)

        return X
    
    @staticmethod
    def summarize():
        with open("summary/MLPRegressor.txt", "r") as f:
            mlp_stat = [float(l) for l in f.readlines()]

        with open("summary/PARegressor.txt") as f:
            pa_stat = [float(l) for l in f.readlines()]

        with open("summary/SGDRegressor.txt") as f:
            sgd_stat = [float(l) for l in f.readlines()]

        plt.plot(range(len(mlp_stat)), mlp_stat)
        plt.plot(range(len(pa_stat)), pa_stat)
        plt.plot(range(len(sgd_stat)), sgd_stat)
        plt.legend(("MLP", "PassiveAgressive", "SGD"))
        plt.title("MSE over model versions.")
        plt.savefig("summary/report.png")