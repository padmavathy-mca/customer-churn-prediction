"""
Model Module for Telco Customer Churn Prediction.
Modular helper functions for building, compiling, training, evaluating,
and predicting with the Artificial Neural Network (ANN).
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# TensorFlow / Keras imports with fallback safety
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


def build_ann(
    input_dim: int = 4,
    hidden_units: Tuple[int, ...] = (16, 8),
    activation: str = "relu",
    output_activation: str = "sigmoid"
) -> Any:
    """
    Build the Sequential Artificial Neural Network architecture.
    Default architecture matching the notebook:
    - Input: shape=(4,)
    - Dense(16, activation='relu')
    - Dense(8, activation='relu')
    - Dense(1, activation='sigmoid')

    Parameters:
        input_dim (int): Number of input features (default: 4).
        hidden_units (Tuple[int, ...]): Number of nodes in hidden layers.
        activation (str): Activation function for hidden layers.
        output_activation (str): Activation function for output layer.

    Returns:
        Sequential: Compiled or uncompiled Keras Sequential model.
    """
    if not HAS_TENSORFLOW:
        raise ImportError("TensorFlow is required to build the ANN model.")

    layers = [Input(shape=(input_dim,))]
    for units in hidden_units:
        layers.append(Dense(units, activation=activation))
    layers.append(Dense(1, activation=output_activation))

    model = Sequential(layers)
    return model


def compile_model(
    model: Any,
    optimizer: str = "adam",
    loss: str = "binary_crossentropy",
    metrics: Optional[List[str]] = None
) -> Any:
    """
    Compile the Keras model with optimizer, loss, and evaluation metrics.

    Parameters:
        model: Keras Sequential model.
        optimizer (str): Optimizer name (default 'adam').
        loss (str): Loss function (default 'binary_crossentropy').
        metrics (List[str], optional): Metrics to track (default ['accuracy']).

    Returns:
        Keras model: Compiled model.
    """
    if metrics is None:
        metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def train_ann(
    model: Any,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    val_split: float = 0.2,
    patience: int = 10,
    restore_best_weights: bool = True,
    verbose: int = 1
) -> Tuple[Any, Any]:
    """
    Train the ANN model using EarlyStopping to prevent overfitting.

    Parameters:
        model: Compiled Keras model.
        X_train_scaled (np.ndarray): Scaled training features.
        y_train (np.ndarray): Target vector.
        epochs (int): Maximum training epochs (default 100).
        batch_size (int): Training batch size (default 32).
        val_split (float): Validation split ratio (default 0.2).
        patience (int): Number of epochs without val_loss improvement before stopping.
        restore_best_weights (bool): Restore model weights from epoch with lowest val_loss.
        verbose (int): Verbosity level (0, 1, or 2).

    Returns:
        Tuple[model, history]: Trained model and Keras History object.
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=restore_best_weights
    )

    history = model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=[early_stop],
        verbose=verbose
    )

    return model, history


def evaluate_ann(
    model: Any,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate the ANN model on the test dataset.

    Computes:
    - Test Loss
    - Test Accuracy
    - Confusion Matrix
    - Classification Report (Precision, Recall, F1-Score)
    - Predictions & Probabilities comparison DataFrame

    Parameters:
        model: Trained Keras model.
        X_test_scaled (np.ndarray): Scaled test features.
        y_test (np.ndarray): True target labels.
        threshold (float): Decision threshold for classification (default 0.5).

    Returns:
        Dict[str, Any]: Comprehensive evaluation metrics dictionary.
    """
    # Evaluate loss and accuracy
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

    # Predict probabilities
    y_probability = model.predict(X_test_scaled, verbose=0).flatten()

    # Apply threshold for binary classification
    y_pred = (y_probability >= threshold).astype(int)

    # True labels array
    y_true = np.array(y_test)

    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    cr_dict = classification_report(y_true, y_pred, output_dict=True, target_names=["Retained (No)", "Churned (Yes)"])
    cr_text = classification_report(y_true, y_pred, target_names=["Retained (No)", "Churned (Yes)"])

    results_df = pd.DataFrame({
        "Actual_Churn": y_true,
        "Predicted_Churn": y_pred,
        "Churn_Probability": y_probability
    })

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "accuracy_pct": float(accuracy * 100),
        "confusion_matrix": cm,
        "classification_report_dict": cr_dict,
        "classification_report_text": cr_text,
        "probabilities": y_probability,
        "predictions": y_pred,
        "results_dataframe": results_df
    }


def predict_customer(
    model: Any,
    scaler: Any,
    tenure: int,
    monthly_charges: float,
    total_charges: float,
    senior_citizen: int,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Predict churn for an individual customer profile.

    Parameters:
        model: Trained ANN model.
        scaler: Fitted StandardScaler.
        tenure (int): Months as customer.
        monthly_charges (float): Monthly bill.
        total_charges (float): Total cumulative bill.
        senior_citizen (int): 1 if senior, 0 otherwise.
        threshold (float): Classification decision threshold (default 0.5).

    Returns:
        Dict[str, Any]: Result with churn probability, binary prediction, risk category, and drivers.
    """
    input_df = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "SeniorCitizen": senior_citizen
    }])

    scaled_input = scaler.transform(input_df)
    prob = float(model.predict(scaled_input, verbose=0)[0][0])
    is_churn = int(prob >= threshold)

    # Determine risk tier
    if prob < 0.30:
        risk_tier = "Low Risk"
        risk_color = "green"
    elif prob < 0.60:
        risk_tier = "Moderate Risk"
        risk_color = "orange"
    else:
        risk_tier = "High Risk"
        risk_color = "red"

    # Identify primary churn drivers
    drivers = []
    if tenure <= 6:
        drivers.append("Short customer tenure (new subscriber under 6 months)")
    if monthly_charges > 70.0:
        drivers.append("Elevated monthly charges (> $70.00/mo)")
    if senior_citizen == 1:
        drivers.append("Senior citizen demographic profile")
    if total_charges < 200.0 and tenure < 4:
        drivers.append("Low total commitment / initial contract period")

    if not drivers:
        drivers.append("Stable tenure and standard pricing profile")

    return {
        "probability": prob,
        "probability_pct": round(prob * 100, 2),
        "prediction": is_churn,
        "prediction_label": "Likely to Churn" if is_churn == 1 else "Likely to Stay",
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "drivers": drivers
    }


def batch_predict(
    model: Any,
    scaler: Any,
    df: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Score a batch DataFrame and append churn probability and prediction columns.

    Parameters:
        model: Trained ANN model.
        scaler: Fitted StandardScaler.
        df (pd.DataFrame): Dataframe with required feature columns.
        threshold (float): Classification decision threshold.

    Returns:
        pd.DataFrame: Augmented dataframe with 'Churn_Probability' and 'Predicted_Churn'.
    """
    feature_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    X = df[feature_cols].copy()
    X_scaled = scaler.transform(X)

    probabilities = model.predict(X_scaled, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(int)

    out_df = df.copy()
    out_df["Churn_Probability"] = np.round(probabilities, 4)
    out_df["Predicted_Churn"] = predictions
    out_df["Risk_Level"] = pd.cut(
        probabilities,
        bins=[-0.01, 0.30, 0.60, 1.0],
        labels=["Low", "Medium", "High"]
    )

    return out_df


def save_pipeline(
    model: Any,
    scaler: Any,
    model_path: str = "ann_churn_model.keras",
    scaler_path: str = "scaler.joblib"
) -> None:
    """
    Save the trained Keras model and fitted StandardScaler to disk.
    """
    model.save(model_path)
    joblib.dump(scaler, scaler_path)


def load_pipeline(
    model_path: str = "ann_churn_model.keras",
    scaler_path: str = "scaler.joblib"
) -> Tuple[Any, Any]:
    """
    Load saved Keras model and fitted StandardScaler from disk.
    """
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
