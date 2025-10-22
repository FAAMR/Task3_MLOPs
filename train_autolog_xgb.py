import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
import mlflow
from dotenv import load_dotenv
load_dotenv()

mlflow.xgboost.autolog()  # Enable autologging for XGBoost
mlflow.set_experiment("Titanic Survival Prediction")

# Load preprocessed Titanic data
df = pd.read_csv('data/preprocessed_titanic.csv')
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="XGBoost Autolog"):
    model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    
    # Autologging captures training metrics by providing an eval_set
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Manually log extra metrics that autolog might not capture
    y_pred = model.predict(X_test)
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

print("XGBoost training and MLflow logging complete.")
