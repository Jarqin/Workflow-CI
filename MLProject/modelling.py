import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
import argparse
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    args = parser.parse_args()
    
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth
    }
    
    (X_train, X_test, y_train, y_test), feature_names = load_data()
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Create and train pipeline
        pipeline = create_pipeline(params)
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        for label in report:
            if label in ['macro avg', 'weighted avg']:
                mlflow.log_metric(f"{label}_precision", report[label]['precision'])
                mlflow.log_metric(f"{label}_recall", report[label]['recall'])
                mlflow.log_metric(f"{label}_f1-score", report[label]['f1-score'])
        
        # Log model
        mlflow.sklearn.log_model(pipeline, "model")
        
        print(f"Model trained with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()