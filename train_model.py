import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def train_and_save():
    print("🔄 Loading dataset...")
    # ตรวจสอบ path
    csv_path = "data/dataset.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found at {csv_path}")
        print("Please ensure 'dataset.csv' is in the 'data/' folder.")
        return

    df = pd.read_csv(csv_path)
    
    print("🧹 Preprocessing data...")
    # Handle missing values
    df = df.dropna()
    
    # Define features and target
    X = df.drop('Hydration Level', axis=1)
    y = df['Hydration Level'].map({'Good': 0, 'Poor': 1}) # 1 = Dehydration Risk
    
    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Preprocessing Pipeline
    numeric_features = ['Age', 'Weight (kg)', 'Daily Water Intake (liters)']
    categorical_features = ['Gender', 'Physical Activity Level', 'Weather']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    print("🤖 Training SVM Model...")
    # SVM with probability=True
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'))
    ])
    
    model.fit(X_train, y_train)
    
    print("📊 Evaluating Model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("\n" + "="*40)
    print("MODEL EVALUATION RESULTS")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Good (0)', 'Poor (1)']))
    print("="*40)
    
    print("💾 Saving models...")
    os.makedirs("model", exist_ok=True)
    
    # Save full pipeline as model.pkl (best practice to avoid preprocessing mismatch)
    joblib.dump(model, 'model/model.pkl')
    
    # Save individual components for dashboard/inspection if needed
    joblib.dump(model.named_steps['preprocessor'], 'model/preprocessor.pkl')
    
    print("✅ Training complete! Models saved to 'model/' directory.")

if __name__ == "__main__":
    train_and_save()