import os
import pickle
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# ✅ Paths
MODEL_DIR = "models"
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.h5")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")

# ✅ API for live reefer data
LIVE_DATA_API = "http://127.0.0.1:5000/live_data"

# ✅ Features required for model input
FEATURES = ['temperature', 'target_temperature', 'ambient_temperature', 'humidity_level',
            'oxygen_level', 'nitrogen_level', 'carbon_dioxide_level']


# ✅ Fetch live reefer data from API
def fetch_live_data():
    try:
        response = requests.get(LIVE_DATA_API)
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                raise ValueError(f"Live data error: {data['error']}")
            return pd.DataFrame([data])  # Convert JSON response to Pandas DataFrame
        else:
            raise ConnectionError(f"API request failed with status {response.status_code}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to fetch live data: {e}")


# ✅ Load trained models
def load_models():
    lstm_model = load_model(LSTM_MODEL_PATH)
    with open(RF_MODEL_PATH, 'rb') as f:
        rf_model = pickle.load(f)
    return lstm_model, rf_model


# ✅ Preprocess live data for evaluation
def preprocess_data(df):
    X = df[FEATURES]

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM input
    X_lstm = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

    return X_lstm


# ✅ Evaluate the model using live data
def evaluate_live_data():
    try:
        df = fetch_live_data()  # ✅ Get live data from API
        X_lstm = preprocess_data(df)

        # ✅ Load trained models
        lstm_model, rf_model = load_models()

        # ✅ Extract features from LSTM
        lstm_features = lstm_model.predict(X_lstm).astype(np.float32)

        # ✅ Predict using Random Forest
        y_pred_prob = rf_model.predict_proba(lstm_features)[:, 1]  # Probability of failure
        y_pred = (y_pred_prob > 0.7).astype(int)  # Thresholding

        # ✅ Get true label from live data
        y_true = np.array([df.get('maintenance_required', 0)]).astype(int)

        # ✅ Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # ✅ Compute Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)

        # ✅ Compute ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # ✅ Print Model Performance
        print("\n📊 **Live Model Performance Metrics**")
        print(f"🔹 Accuracy:  {accuracy:.4f}")
        print(f"🔹 Precision: {precision:.4f}")
        print(f"🔹 Recall:    {recall:.4f}")
        print(f"🔹 F1 Score:  {f1:.4f}")

        # ✅ Plot Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Live Confusion Matrix")
        plt.show()

        # ✅ Plot ROC Curve
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Live ROC Curve")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"\n❌ Error during live evaluation: {e}")


# ✅ Run Live Evaluation
if __name__ == "__main__":
    print("\n🔄 Evaluating Model with Live Data...")
    evaluate_live_data()
