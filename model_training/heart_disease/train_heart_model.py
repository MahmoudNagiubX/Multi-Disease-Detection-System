import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main() -> None: # Train RandomForest model ->  then save it as a pickle file with the feature names
    # Resolve project paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]  # go up 2 levels to project root
    
    data_path = project_root / "app" / "data" / "datasets" / "cardio_train.csv"
    model_dir = project_root / "app" / "data" / "saved_models"
    model_dir.mkdir(parents = True, exist_ok = True)
    model_path = model_dir / "heart_model.pkl"
    
    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Loading dataset from: {data_path}")
    
    # Load dataset
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {data_path}. "
            f"Make sure your CSV is placed there and named 'Heart Disease UCI.csv'."
        )

    df = pd.read_csv(data_path, sep = ";")
    print(f"[INFO] Dataset shape: {df.shape}")
    print("[INFO] Columns:", list(df.columns))
    
    # 2. DATA CLEANING (Crucial for Accuracy!)
    # Filter out impossible values (outliers)
    print(f"[INFO] Original rows: {len(df)}")
    
    # Keep Systolic BP between 50 and 250
    df = df[(df["ap_hi"] >= 50) & (df["ap_hi"] <= 250)]
    
    # Keep Diastolic BP between 30 and 150
    df = df[(df["ap_lo"] >= 30) & (df["ap_lo"] <= 150)]
    
    # Keep Height > 100cm (to remove errors)
    df = df[df["height"] >= 100]
    
    # 3. FEATURE ENGINEERING: Add BMI
    # BMI = weight (kg) / height (m)^2
    df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
    
    # Drop the 'id' column as it's not a feature
    if "id" in df.columns:
        df.drop(columns = ["id"], inplace = True)
    
    # Define features (X) and target (y)
    if "cardio" not in df.columns:
        raise KeyError(
            "Column 'cardio' not found. Check CSV format."
        )

    X = df.drop(columns = ["cardio"])
    y = df["cardio"]    # target (0 = no disease, 1 = disease)
    feature_names = list(X.columns)
    print("[INFO] Using features:", feature_names)
    
    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 42,
        stratify = y,
    )
    print(f"[INFO] Train size: {X_train.shape[0]}")
    print(f"[INFO] Test size:  {X_test.shape[0]}")
    
    # Create and train RandomForest model
    rf = RandomForestClassifier(
        n_estimators = 300,
        max_depth = None,
        random_state = 42,
        class_weight = "balanced",
        n_jobs = -1,
    )

    print("[INFO] Training RandomForest model...")
    rf.fit(X_train, y_train)
    
    # Evaluate model
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))

    print(f"[RESULT] Train Accuracy: {train_acc:.4f}")
    print(f"[RESULT] Test Accuracy:  {test_acc:.4f}")

    # Save model + feature names
    model_bundle = {
        "model": rf,
        "feature_names": feature_names,
    }
    joblib.dump(model_bundle, model_path)
    print(f"[INFO] Model saved to: {model_path}")

if __name__ == "__main__":
    main()