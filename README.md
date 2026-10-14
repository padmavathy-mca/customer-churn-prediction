# Telco Customer Churn Prediction Application

An Artificial Neural Network (ANN) binary classification application for predicting whether a telecom customer is likely to churn, built with **Streamlit** (frontend) and **modular Python backend helper functions** derived from the Jupyter Notebook.

---

## 📋 Problem Overview

- **Algorithm**: Artificial Neural Network (ANN)
- **Task**: Binary Classification (0 = Retained, 1 = Churned)
- **Framework**: TensorFlow / Keras, Scikit-Learn, Pandas, Streamlit
- **Key Features Selected**:
  1. `tenure`: Number of months the customer has stayed with the company
  2. `MonthlyCharges`: Current monthly recurring bill amount
  3. `TotalCharges`: Cumulative charges over customer lifetime
  4. `SeniorCitizen`: Binary demographic indicator (0 = No, 1 = Yes)

---

## 🏗️ Modular Architecture

The original monolithic Jupyter notebook logic has been decoupled into modular, single-responsibility Python files:

```
├── app.py                     # Streamlit frontend (only py UI file)
├── data_preprocessing.py      # Modular data ingestion, cleaning, splitting & scaling
├── model.py                   # ANN definition, compilation, training, evaluation & inference
├── visualization.py           # Reusable plotting helpers (loss/acc curves, confusion matrix)
├── Telco-Customer-Churn.csv   # Telco sample dataset
├── requirements.txt           # Python package dependencies
└── README.md                  # Documentation and execution instructions
```

---

## 🛠️ Modular Helper Functions Breakdown

### 1. `data_preprocessing.py`
- `load_data(filepath_or_buffer)`: Ingests CSV dataset from disk or memory buffer.
- `clean_dataset(df, drop_id=True)`: Coerces `TotalCharges` to float, handles NaNs, removes duplicates, and drops `customerID`.
- `extract_features_and_target(df, feature_cols, target_col)`: Isolates the 4 input features and maps `Churn` (`Yes` -> `1`, `No` -> `0`).
- `encode_categorical_features(X)`: Runs `pd.get_dummies(drop_first=True)` on features.
- `split_dataset(X, y, test_size=0.2, random_state=42, stratify=True)`: Creates stratified 80/20 train/test split.
- `scale_features(X_train, X_test, scaler=None)`: Fits and transforms features using `StandardScaler`.
- `preprocess_single_customer(...)`: Scales an individual customer profile for real-time inference.
- `generate_eda_statistics(df)`: Computes statistical summary, shape, missing value audit, and churn rates.

### 2. `model.py`
- `build_ann(input_dim=4, hidden_units=(16, 8), activation='relu', output_activation='sigmoid')`: Constructs Sequential model matching notebook:
  - Input: shape (4,)
  - Dense Layer 1: 16 nodes, ReLU
  - Dense Layer 2: 8 nodes, ReLU
  - Output Layer: 1 node, Sigmoid
- `compile_model(model, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])`: Compiles model.
- `train_ann(model, X_train_scaled, y_train, epochs=100, batch_size=32, val_split=0.2, patience=10)`: Trains with `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`.
- `evaluate_ann(model, X_test_scaled, y_test, threshold=0.5)`: Generates test loss, test accuracy, confusion matrix, and classification report.
- `predict_customer(...)`: Scores single customer profile, evaluates churn probability and risk tier (Low, Moderate, High).
- `batch_predict(model, scaler, df, threshold=0.5)`: Generates predictions across a dataframe batch.
- `save_pipeline(model, scaler, ...)` & `load_pipeline(...)`: Saves and restores trained model and scaler artifacts.

### 3. `visualization.py`
- `plot_training_curves(history)`: Plots Training vs. Validation Accuracy and Training vs. Validation Loss side-by-side.
- `plot_confusion_matrix_heatmap(cm)`: Styled 2x2 confusion matrix with counts and percentages.
- `plot_feature_distributions(df)`: Histograms of `tenure`, `MonthlyCharges`, and `TotalCharges` grouped by churn outcome.
- `plot_churn_donut(df)`: Donut chart displaying overall churn vs. retention ratio.

---

## ⚡ Quickstart Guide

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Setup Virtual Environment
```bash
# Clone the repository and navigate to the project directory
cd customer-churn-app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Notebook Benchmark Results

- **Input Features**: 4 (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`)
- **Total Model Parameters**: 225 trainable parameters
- **Final Test Accuracy**: ~77.54%
- **Final Test Loss**: ~0.4435
- **Confusion Matrix**:
  - True Negatives (Retained correctly): 921
  - False Positives: 112
  - False Negatives: 204
  - True Positives (Churned correctly): 170
- **Classification Report**:
  - Class 0 (Retained): Precision 0.82 | Recall 0.89 | F1-Score 0.85
  - Class 1 (Churned): Precision 0.60 | Recall 0.45 | F1-Score 0.52
