import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import lightgbm as lgb
import mlflow
from dotenv import load_dotenv
load_dotenv()

mlflow.lightgbm.autolog()  # Enable autologging for LightGBM
mlflow.set_experiment("Titanic Survival Prediction")

# Load preprocessed Titanic data
df = pd.read_csv('data/preprocessed_titanic.csv')
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="LightGBM Autolog"):
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Manually log extra metrics for fair comparison
    y_pred = model.predict(X_test)
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

print("LightGBM training and MLflow logging complete.")
