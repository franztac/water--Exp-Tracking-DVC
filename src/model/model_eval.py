from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import pickle
import json
import logging
from dvclive import Live
import yaml


logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s - %(filename)s] %(message)s"
)


def load_data(filepath: str) -> pd.DataFrame:
    try:
        logging.info(f"loading data from {filepath}")
        return pd.read_csv(filepath)
    except Exception as e:
        logging.error(f"error loading data from {filepath}: {e}")
        raise


# test_data = pd.read_csv("./data/processed/test_processed.csv")
# X_test = test_data.iloc[:, :-1].values
# y_test = test_data.iloc[:, -1].values


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    try:
        logging.info("Preparing data: separing features from target!")
        X = data.drop(columns=["Potability"])
        y = data.Potability
        return X, y
    except Exception as e:
        logging.error(f"error preparing data: {e}")
        raise


def load_model(filepath: str):
    try:
        logging.info(f"loading model from {filepath}...")
        with open(filepath, "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        logging.error(f"error loading model from {filepath}: {e}")
        raise


# with open("model.pkl", "rb") as file:
#     model = pickle.load(file)


def evaluation_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    try:
        params = yaml.safe_load(open("params.yaml", "r"))
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]

        logging.info(f"preparing metrics dict for {model}")
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)

        with Live(save_dvc_exp=True) as live:
            logging.info("preparing report for dvclive/metrics.json...")

            live.log_metric("accuracy", acc)
            live.log_metric("precision", pre)
            live.log_metric("recall", recall)
            live.log_metric("f1-score", f1score)

            live.log_param("test_size", test_size)
            live.log_param("n_estimators", n_estimators)

        metrics_dict = {
            "accuracy": acc,
            "precision": pre,
            "recall": recall,
            "f1_score": f1score,
        }
        return metrics_dict

    except Exception as e:
        logging.error(f"error evaluating model: {e}")
        raise


def save_metrics(metrics_dict: dict, metrics_path: str) -> None:
    try:
        logging.info(f"saving metrics to {metrics_path}")
        with open(metrics_path, "w") as file:
            json.dump(metrics_dict, file, indent=4)
    except Exception as e:
        logging.error(f"error saving metrics to {metrics_path}: {e}")
        raise


def main():
    try:
        test_datapath = "./data/processed/test_processed_median.csv"
        model_path = "models/model.pkl"
        metrics_path = "reports/metrics.json"

        test_data = load_data(test_datapath)
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluation_model(model, X_test, y_test)
        save_metrics(metrics, metrics_path)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
