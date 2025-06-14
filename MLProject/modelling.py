import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline

import mlflow
import mlflow.sklearn

def load_data():
    df = pd.read_csv('dataset_anime_clean.csv')
    df = df.drop(['anime_id', 'name', 'rank'], axis=1, errors='ignore')
    df = df.dropna()
    X = df.drop('rank_encoded', axis=1)
    y = df['rank_encoded']
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y), X.columns

def create_pipeline(params):
    return imbpipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=8,
            class_weight='balanced',
            random_state=42
        ))
    ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()

    # Params
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth
    }

    # Load data & train model
    (X_train, X_test, y_train, y_test), feature_names = load_data()
    pipeline = create_pipeline(params)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Logging
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_param("n_estimators", params["n_estimators"])
    mlflow.log_param("max_depth", params["max_depth"])
    mlflow.log_metric("accuracy", acc)

    report = classification_report(y_test, y_pred, output_dict=True)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{label}_{metric_name}", value)

    mlflow.sklearn.log_model(pipeline, "model")
