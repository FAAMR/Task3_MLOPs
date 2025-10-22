import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import mlflow
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Load preprocessed Titanic dataset
df = pd.read_csv('data/preprocessed_titanic.csv')
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set main experiment
mlflow.set_experiment("Titanic RandomForest Tuning")

# Hyperparameter options
n_estimators_options = [50, 100]
max_depth_options = [3, 5]

best_auc = 0
best_metrics = {}
best_params = {}

with mlflow.start_run(run_name="RandomForest_Hyperparameter_Tuning") as parent_run:
    mlflow.set_tag("model_type", "RandomForestClassifier")

    for n in n_estimators_options:
        for depth in max_depth_options:
            with mlflow.start_run(run_name=f"n{n}_d{depth}", nested=True):
                params = {"n_estimators": n, "max_depth": depth, "random_state": 42}
                mlflow.log_params(params)

                model = RandomForestClassifier(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]

                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("auc", auc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                print(
                    f"Run n_estimators={n}, max_depth={depth} | "
                    f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
                )

                # Update best metrics
                if auc > best_auc:
                    best_auc = auc
                    best_metrics = {
                        "accuracy": accuracy,
                        "auc": auc,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    }
                    best_params = params

    # Log best run metrics to parent run (so it shows in UI)
    mlflow.log_params(best_params)
    mlflow.log_metrics(best_metrics)

print("RandomForest hyperparameter tuning complete.")
