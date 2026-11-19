
from pathlib import Path
import pickle

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
TRACKING = f"sqlite:///{ROOT / 'mlflow.db'}"


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    # Load dataset or create it if missing
    feature_file = DATA / "features.parquet"

    if feature_file.exists():
        df = pd.read_parquet(feature_file)
    else:
        digits = load_digits()
        df = pd.DataFrame(
            digits.data,
            columns=[f"pixel_{i}" for i in range(64)]
        )
        df["label"] = digits.target
        df.to_parquet(feature_file, index=False)

    X = df[[f"pixel_{i}" for i in range(64)]]
    y = df["label"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Two classifiers and their parameter grids
    candidates = [
        (
            "logistic_regression",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000))
            ]),
            {"model__C": [0.1, 1.0, 10.0]}
        ),
        (
            "random_forest",
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 20]
            }
        )
    ]

    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment("handwritten-digits-classification")

    for name, estimator, parameters in candidates:
        print("\nTraining:", name)

        search = GridSearchCV(
            estimator,
            parameters,
            cv=3,
            scoring="accuracy",
            n_jobs=1
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        predictions = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="weighted")

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("model_name", name)
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_accuracy", float(search.best_score_))
            mlflow.log_metric("test_accuracy", float(accuracy))
            mlflow.log_metric("test_f1_score", float(f1))

            # Save model using pickle and log as an artifact
            model_file = DATA / f"{name}.pkl"
            with open(model_file, "wb") as file:
                pickle.dump(best_model, file)

            mlflow.log_artifact(str(model_file), artifact_path="model")

            print("Best parameters:", search.best_params_)
            print("Accuracy:", round(accuracy, 4))
            print("F1 Score:", round(f1, 4))
            print("Run ID:", run.info.run_id)

    # Save fixed test data for Member 6
    test_df = X_test.copy()
    test_df["label"] = y_test.values
    test_df.to_csv(DATA / "test_data.csv", index=False)

    print("\nTest dataset saved successfully!")
    print("All models trained and logged successfully!")


if __name__ == "__main__":
    main()
