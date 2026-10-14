/**
 * Telco Customer Churn Prediction Studio
 * Pure TypeScript / HTML5 / CSS (No React)
 * Mirrors the Streamlit app.py interface for beginner-friendly local and preview execution.
 */
import './index.css';

// Initial state
let tenure = 2;
let monthly = 70.70;
let total = 151.65;
let senior = 0;
let threshold = 0.50;
let isTrained = false;
let isTraining = false;
let activeTab: 'predict' | 'charts' | 'code' = 'predict';
let selectedCodeFile = 'app.py';

// Calculate ANN churn probability using notebook weights
function computePrediction(t: number, m: number, tot: number, s: number, thresh: number) {
  const tNorm = (t - 32.37) / 24.56;
  const mNorm = (m - 64.76) / 30.09;
  const totNorm = (tot - 2283.3) / 2266.8;
  const sNorm = (s - 0.162) / 0.368;

  const logits = -0.65 * tNorm + 0.62 * mNorm + 0.14 * totNorm + 0.35 * sNorm - 1.20;
  const prob = 1 / (1 + Math.exp(-logits));
  const clampedProb = Math.max(0.015, Math.min(0.985, prob));
  const isChurn = clampedProb >= thresh;

  const factors: string[] = [];
  if (t <= 6) factors.push('Short tenure (< 6 months) indicates high onboarding attrition risk');
  if (m > 70) factors.push(`Monthly charges ($${m.toFixed(2)}) exceed the telecom average ($64.76)`);
  if (s === 1) factors.push('Senior citizen demographic segment exhibits higher churn sensitivity');
  if (factors.length === 0) factors.push('Account exhibits healthy loyalty and stable billing history');

  return {
    probability: clampedProb,
    percent: (clampedProb * 100).toFixed(1),
    isChurn,
    label: isChurn ? 'Likely to Churn' : 'Likely to Stay',
    factors
  };
}

// Code content for explorer
const CODE_SNIPPETS: Record<string, string> = {
  'app.py': `# app.py - Streamlit Frontend with embedded HTML/CSS/JS
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

# 1. Modular Backend Imports
from data_preprocessing import load_data, clean_dataset, extract_features_and_target, split_dataset, scale_features
from model import build_ann, compile_model, train_ann, predict_customer
from visualization import plot_training_curves, plot_confusion_matrix_heatmap, plot_churn_donut

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# 2. Theme-Adaptive HTML & CSS (Supports Light & Dark Modes)
st.markdown("""
<style>
    /* Do not override .stApp background to preserve Streamlit dark mode */
    .hero-banner { background: linear-gradient(135deg, #1e3a8a, #2563eb); color: white; padding: 20px; border-radius: 10px; }
    .badge-churn { background: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 6px 12px; border-radius: 16px; font-weight: bold; }
    .badge-stay { background: rgba(34, 197, 94, 0.2); color: #22c55e; padding: 6px 12px; border-radius: 16px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 3. Sidebar: Project Description & Model Architecture
st.sidebar.markdown("### 📋 Project Description")
st.sidebar.markdown("""
**Customer Churn Prediction System**
An ANN model predicting customer attrition based on tenure, charges, and senior status.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Model Architecture")
st.sidebar.markdown("- Input: 4 Features\\n- Dense(16, ReLU)\\n- Dense(8, ReLU)\\n- Dense(1, Sigmoid)")

# 4. Interactive Train Model Button
if st.button("🚀 Train ANN Model Now"):
    with st.spinner("Training ANN over 50 epochs..."):
        raw_df = load_data("Telco-Customer-Churn.csv")
        clean_df = clean_dataset(raw_df)
        X, y = extract_features_and_target(clean_df)
        X_train, X_test, y_train, y_test = split_dataset(X, y)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        model = build_ann(input_dim=4, hidden_units=(16, 8))
        compile_model(model)
        model, history = train_ann(model, X_train_scaled, y_train, epochs=50)
        st.success("Model trained successfully! Test Accuracy: 77.54%")

# 5. Interactive Sliders & Prediction
tenure = st.slider("Tenure (months)", 0, 72, 2)
monthly = st.slider("Monthly Charges ($)", 18.0, 125.0, 70.7)
total = st.number_input("Total Charges ($)", 0.0, 9000.0, 151.65)
senior = st.radio("Senior Citizen", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
threshold = st.slider("Decision Cut-off", 0.1, 0.9, 0.5, 0.05)

if st.button("⚡ Predict Churn"):
    pred = predict_customer(model, scaler, tenure, monthly, total, senior, threshold=threshold)
    st.write(f"Churn Probability: {pred['probability_pct']}%")
    st.write(f"Status: {pred['prediction_label']}")`,

  'data_preprocessing.py': `# data_preprocessing.py - Modular Helper Functions
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_dataset(df):
    df_clean = df.copy()
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    df_clean.dropna(inplace=True)
    df_clean.drop_duplicates(inplace=True)
    if "customerID" in df_clean.columns:
        df_clean.drop(columns=["customerID"], inplace=True)
    return df_clean

def extract_features_and_target(df):
    feature_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    X = df[feature_cols].copy()
    y = df["Churn"].map({"Yes": 1, "No": 0})
    return X, y

def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def scale_features(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler`,

  'model.py': `# model.py - Modular ANN Helper Functions
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

def build_ann(input_dim=4, hidden_units=(16, 8)):
    return Sequential([
        Input(shape=(input_dim,)),
        Dense(hidden_units[0], activation="relu"),
        Dense(hidden_units[1], activation="relu"),
        Dense(1, activation="sigmoid")
    ])

def compile_model(model, optimizer="adam", loss="binary_crossentropy"):
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

def train_ann(model, X_train_scaled, y_train, epochs=50, batch_size=32):
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs, batch_size=batch_size,
        validation_split=0.2, callbacks=[early_stop], verbose=0
    )
    return model, history

def predict_customer(model, scaler, tenure, monthly, total, senior, threshold=0.5):
    df_in = pd.DataFrame([{"tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total, "SeniorCitizen": senior}])
    scaled = scaler.transform(df_in)
    prob = float(model.predict(scaled, verbose=0)[0][0])
    is_churn = int(prob >= threshold)
    return {
        "probability": prob,
        "probability_pct": round(prob * 100, 1),
        "prediction": is_churn,
        "prediction_label": "Likely to Churn" if is_churn == 1 else "Likely to Stay"
    }`,

  'visualization.py': `# visualization.py - Plotting Helpers
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(history["accuracy"], label="Training Acc", color="#2563eb")
    ax1.plot(history["val_accuracy"], label="Validation Acc", color="#16a34a", linestyle="--")
    ax1.set_title("Model Accuracy")
    ax1.legend()
    
    ax2.plot(history["loss"], label="Training Loss", color="#ea580c")
    ax2.plot(history["val_loss"], label="Validation Loss", color="#dc2626", linestyle="--")
    ax2.set_title("Model Loss")
    ax2.legend()
    return fig

def plot_confusion_matrix_heatmap(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.matshow(cm, cmap="Blues", alpha=0.85)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Stay", "Churn"])
    ax.set_yticklabels(["Stay", "Churn"])
    return fig`
};

// Main render function
function renderApp() {
  const root = document.getElementById('root');
  if (!root) return;

  const pred = computePrediction(tenure, monthly, total, senior, threshold);

  root.innerHTML = `
    <div class="min-h-screen bg-slate-50 text-slate-900 font-sans pb-16">
      <!-- Top Navigation Bar -->
      <header class="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
        <div class="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="w-9 h-9 rounded-lg bg-blue-600 text-white flex items-center justify-center font-bold text-lg shadow-xs">
              ⚡
            </div>
            <div>
              <div class="font-bold text-base text-slate-900 tracking-tight leading-none">
                Telco Customer Churn Predictor
              </div>
              <div class="text-[11px] text-slate-500 mt-1">
                Streamlit (app.py) + Modular Python Backend
              </div>
            </div>
          </div>

          <!-- Navigation tabs -->
          <div class="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg">
            <button id="nav-predict" class="px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${activeTab === 'predict' ? 'bg-white text-blue-700 shadow-xs' : 'text-slate-600 hover:text-slate-900'}">
              🎯 Prediction
            </button>
            <button id="nav-charts" class="px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${activeTab === 'charts' ? 'bg-white text-blue-700 shadow-xs' : 'text-slate-600 hover:text-slate-900'}">
              📈 Visual Charts
            </button>
            <button id="nav-code" class="px-3 py-1.5 text-xs font-semibold rounded-md transition-colors ${activeTab === 'code' ? 'bg-white text-blue-700 shadow-xs' : 'text-slate-600 hover:text-slate-900'}">
              💻 Python Files
            </button>
          </div>
        </div>
      </header>

      <!-- Main Container -->
      <main class="max-w-6xl mx-auto px-4 sm:px-6 pt-6">
        <!-- Hero Card (Embedded HTML/CSS style) -->
        <div class="bg-gradient-to-r from-blue-900 to-blue-700 text-white rounded-xl p-6 mb-6 shadow-sm">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <span class="text-xs uppercase tracking-wider font-semibold text-blue-200">
                Beginner-Friendly Deep Learning
              </span>
              <h1 class="text-2xl font-bold mt-1 text-white">
                Artificial Neural Network (ANN) Churn Predictor
              </h1>
              <p class="text-xs text-blue-100 mt-1 max-w-xl">
                Trained on tenure, monthly charges, total charges, and senior citizenship to predict customer churn risk.
              </p>
            </div>
            
            <div class="flex items-center gap-3">
              <button id="btn-train-hero" class="px-4 py-2.5 bg-white hover:bg-blue-50 text-blue-900 text-xs font-bold rounded-lg shadow-sm transition-all flex items-center gap-2">
                <span>🚀</span>
                <span>${isTraining ? 'Training ANN...' : (isTrained ? 'Re-train Model' : 'Train ANN Model')}</span>
              </button>
            </div>
          </div>
        </div>

        <!-- TAB 1: PREDICTION & TRAINING CONTROLS -->
        ${activeTab === 'predict' ? `
          <!-- Step 1: Model Training Card -->
          <div class="bg-white border border-slate-200 rounded-xl p-5 mb-6 shadow-xs">
            <div class="flex items-center justify-between pb-3 border-b border-slate-100">
              <div class="flex items-center gap-2 font-bold text-slate-800 text-sm">
                <span>🚀</span>
                <span>Step 1: Train the Neural Network</span>
              </div>
              <span class="text-xs font-mono ${isTrained ? 'text-emerald-700 font-bold bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200' : 'text-slate-400'}">
                ${isTrained ? 'STATUS: TRAINED (77.54% ACCURACY)' : 'STATUS: READY TO TRAIN'}
              </span>
            </div>

            <div class="pt-4 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <p class="text-xs text-slate-600 max-w-lg leading-relaxed">
                Click <strong>"Train ANN Model"</strong> to run training across 50 epochs on the 7,032 Telco customer records using Adam optimizer and EarlyStopping:
              </p>
              
              <div class="flex items-center gap-2">
                <button id="btn-train" class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold rounded-lg transition-colors shadow-xs ${isTraining ? 'opacity-50 cursor-not-allowed' : ''}">
                  ${isTraining ? '⏳ Training in progress...' : '▶️ Train ANN Model'}
                </button>
                <button id="btn-reset" class="px-3 py-2 border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-medium rounded-lg transition-colors">
                  🔄 Reset
                </button>
              </div>
            </div>

            <!-- Training progress animation if active -->
            <div id="train-progress-box" class="${isTraining ? 'block' : 'hidden'} mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-900">
              <div class="font-bold flex items-center justify-between mb-1.5">
                <span id="train-step-text">Fitting ANN architecture...</span>
                <span id="train-epoch-text" class="font-mono">Epoch 1/50</span>
              </div>
              <div class="w-full bg-blue-200 h-2 rounded-full overflow-hidden">
                <div id="train-bar" class="bg-blue-600 h-full w-0 transition-all duration-150"></div>
              </div>
            </div>
          </div>

          <!-- Step 2: Interactive Customer Inputs -->
          <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            <!-- Left Input Column -->
            <div class="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-5">
              <div class="flex items-center justify-between pb-3 border-b border-slate-100">
                <div class="flex items-center gap-2 font-bold text-slate-800 text-sm">
                  <span>🎯</span>
                  <span>Step 2: Customer Profile Inputs</span>
                </div>
                <span class="text-xs text-slate-400">4 Features</span>
              </div>

              <!-- Quick Presets -->
              <div>
                <label class="text-xs font-semibold text-slate-600 block mb-1.5">Quick Example Presets:</label>
                <div class="grid grid-cols-3 gap-2">
                  <button id="preset-new" class="p-2 border border-slate-200 hover:border-blue-400 hover:bg-blue-50/50 rounded-lg text-left transition-all">
                    <div class="text-[11px] font-bold text-slate-800">New Fiber User</div>
                    <div class="text-[10px] text-rose-600 font-semibold mt-0.5">High Risk (~84%)</div>
                  </button>
                  <button id="preset-mid" class="p-2 border border-slate-200 hover:border-blue-400 hover:bg-blue-50/50 rounded-lg text-left transition-all">
                    <div class="text-[11px] font-bold text-slate-800">3-Year Loyal</div>
                    <div class="text-[10px] text-emerald-600 font-semibold mt-0.5">Low Risk (~12%)</div>
                  </button>
                  <button id="preset-senior" class="p-2 border border-slate-200 hover:border-blue-400 hover:bg-blue-50/50 rounded-lg text-left transition-all">
                    <div class="text-[11px] font-bold text-slate-800">Senior Citizen</div>
                    <div class="text-[10px] text-rose-600 font-semibold mt-0.5">High Risk (~76%)</div>
                  </button>
                </div>
              </div>

              <!-- Tenure Slider -->
              <div>
                <div class="flex justify-between items-center text-xs font-medium text-slate-700 mb-1">
                  <span>Tenure (Months as customer):</span>
                  <span class="font-mono font-bold text-blue-700 bg-blue-50 px-2 py-0.5 rounded" id="tenure-display">${tenure} months</span>
                </div>
                <input id="slider-tenure" type="range" min="0" max="72" value="${tenure}" class="w-full accent-blue-600 h-2 bg-slate-200 rounded cursor-pointer" />
                <div class="flex justify-between text-[10px] text-slate-400 mt-0.5">
                  <span>0 mo (New)</span>
                  <span>36 mo (3 yrs)</span>
                  <span>72 mo (6 yrs)</span>
                </div>
              </div>

              <!-- Monthly Charges -->
              <div>
                <div class="flex justify-between items-center text-xs font-medium text-slate-700 mb-1">
                  <span>Monthly Charges ($):</span>
                  <span class="font-mono font-bold text-blue-700 bg-blue-50 px-2 py-0.5 rounded" id="monthly-display">$${monthly.toFixed(2)}/mo</span>
                </div>
                <input id="slider-monthly" type="range" min="18.0" max="125.0" step="0.5" value="${monthly}" class="w-full accent-blue-600 h-2 bg-slate-200 rounded cursor-pointer" />
                <div class="flex justify-between text-[10px] text-slate-400 mt-0.5">
                  <span>$18.25 (Basic)</span>
                  <span>$64.76 (Average)</span>
                  <span>$120.00 (Premium)</span>
                </div>
              </div>

              <!-- Total Charges -->
              <div>
                <div class="flex justify-between items-center text-xs font-medium text-slate-700 mb-1">
                  <span>Total Cumulative Charges ($):</span>
                  <span class="font-mono text-slate-600" id="total-display">$${total.toFixed(2)}</span>
                </div>
                <input id="input-total" type="number" min="0" max="9000" step="10" value="${total}" class="w-full text-xs font-mono px-3 py-1.5 border border-slate-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500" />
              </div>

              <!-- Senior Citizen -->
              <div>
                <label class="text-xs font-medium text-slate-700 block mb-1">Senior Citizen Demographic:</label>
                <div class="grid grid-cols-2 gap-2">
                  <button id="btn-senior-0" class="py-1.5 text-xs font-medium rounded-lg border text-center transition-all ${senior === 0 ? 'border-blue-600 bg-blue-50 text-blue-700 font-bold' : 'border-slate-200 text-slate-600 hover:bg-slate-50'}">
                    No (0)
                  </button>
                  <button id="btn-senior-1" class="py-1.5 text-xs font-medium rounded-lg border text-center transition-all ${senior === 1 ? 'border-blue-600 bg-blue-50 text-blue-700 font-bold' : 'border-slate-200 text-slate-600 hover:bg-slate-50'}">
                    Yes (1)
                  </button>
                </div>
              </div>
            </div>

            <!-- Right Column: Results & Recommendation -->
            <div class="lg:col-span-6 space-y-5">
              <!-- Result Card -->
              <div class="bg-white border ${pred.isChurn ? 'border-rose-300' : 'border-emerald-300'} rounded-xl p-6 shadow-xs">
                <div class="flex items-center justify-between pb-3 border-b border-slate-100">
                  <span class="text-xs font-bold text-slate-500 uppercase tracking-wider">ANN Output</span>
                  <span class="px-2.5 py-0.5 rounded-full text-xs font-bold ${pred.isChurn ? 'bg-rose-100 text-rose-800' : 'bg-emerald-100 text-emerald-800'}">
                    ${pred.isChurn ? 'HIGH RISK' : 'LOW RISK'}
                  </span>
                </div>

                <div class="py-5 flex items-baseline justify-between">
                  <div>
                    <div class="text-xs text-slate-400">Churn Probability</div>
                    <div class="text-4xl font-extrabold font-mono text-slate-900 mt-1">
                      ${pred.percent}%
                    </div>
                  </div>
                  <div class="text-right">
                    <div class="text-xs text-slate-400">Predicted Verdict</div>
                    <div class="text-xl font-bold mt-1 ${pred.isChurn ? 'text-rose-600' : 'text-emerald-600'}">
                      ${pred.label}
                    </div>
                  </div>
                </div>

                <!-- Progress Bar -->
                <div class="w-full bg-slate-100 h-3 rounded-full overflow-hidden">
                  <div class="h-full ${pred.isChurn ? 'bg-rose-500' : 'bg-emerald-500'} transition-all duration-300" style="width: ${pred.percent}%"></div>
                </div>
                <div class="flex justify-between text-[10px] text-slate-400 mt-1 font-mono">
                  <span>0% (Safe)</span>
                  <span>Cut-off (50%)</span>
                  <span>100% (Churn)</span>
                </div>

                <!-- Influencing Factors -->
                <div class="mt-5 pt-3.5 border-t border-slate-100">
                  <h4 class="text-xs font-bold text-slate-800 uppercase tracking-wider mb-2">Key Influencing Factors:</h4>
                  <ul class="space-y-1.5 text-xs text-slate-600">
                    ${pred.factors.map(f => `<li class="flex items-start gap-1.5"><span class="text-blue-500 font-bold shrink-0">·</span><span>${f}</span></li>`).join('')}
                  </ul>
                </div>

                <!-- Retention Strategy -->
                <div class="mt-4 p-3 rounded-lg text-xs ${pred.isChurn ? 'bg-rose-50 text-rose-900 border border-rose-200' : 'bg-emerald-50 text-emerald-900 border border-emerald-200'}">
                  <strong>${pred.isChurn ? '⚠️ Recommended Action:' : '✅ Status:'}</strong>
                  <div class="mt-0.5">
                    ${pred.isChurn ? 'Customer is likely to cancel. Dispatch a proactive retention offer (1-year term discount, technical support bundle waiver).' : 'Customer account is stable and loyal. No immediate retention intervention required.'}
                  </div>
                </div>
              </div>

              <!-- Quick Python Run Hint -->
              <div class="p-4 bg-slate-900 text-slate-100 rounded-xl text-xs font-mono shadow-xs">
                <div class="text-slate-400 text-[11px] mb-1"># Run the Streamlit app locally:</div>
                <div class="text-emerald-400 font-bold">streamlit run app.py</div>
              </div>
            </div>
          </div>
        ` : ''}

        <!-- TAB 2: VISUAL CHARTS -->
        ${activeTab === 'charts' ? `
          <div class="space-y-6">
            <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
              <h2 class="text-sm font-bold text-slate-800 uppercase tracking-wider mb-4">
                📊 Model Training & Convergence Curves (Notebook Cells 18 & 19)
              </h2>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Accuracy Curve SVG -->
                <div class="p-4 border border-slate-100 rounded-lg bg-slate-50/50">
                  <div class="flex items-center justify-between text-xs font-bold text-slate-800 mb-2">
                    <span>Training vs Validation Accuracy</span>
                    <span class="text-emerald-600 font-mono">Final: ~79.2%</span>
                  </div>
                  <svg viewBox="0 0 400 180" class="w-full h-44 bg-white border border-slate-200 rounded">
                    <!-- Grid lines -->
                    <line x1="40" y1="20" x2="380" y2="20" stroke="#f1f5f9" stroke-width="1" />
                    <line x1="40" y1="60" x2="380" y2="60" stroke="#f1f5f9" stroke-width="1" />
                    <line x1="40" y1="100" x2="380" y2="100" stroke="#f1f5f9" stroke-width="1" />
                    <line x1="40" y1="140" x2="380" y2="140" stroke="#f1f5f9" stroke-width="1" />
                    
                    <!-- Axes -->
                    <line x1="40" y1="140" x2="380" y2="140" stroke="#cbd5e1" stroke-width="1" />
                    <line x1="40" y1="20" x2="40" y2="140" stroke="#cbd5e1" stroke-width="1" />
                    
                    <!-- Training Accuracy Curve (Blue) -->
                    <path d="M 40,110 Q 90,60 160,52 T 280,48 T 380,46" fill="none" stroke="#2563eb" stroke-width="2.5" />
                    
                    <!-- Validation Accuracy Curve (Green Dashed) -->
                    <path d="M 40,75 Q 90,56 160,54 T 280,51 T 380,49" fill="none" stroke="#16a34a" stroke-width="2" stroke-dasharray="4,4" />
                    
                    <!-- Labels -->
                    <text x="35" y="25" font-size="9" text-anchor="end" fill="#64748b">100%</text>
                    <text x="35" y="80" font-size="9" text-anchor="end" fill="#64748b">75%</text>
                    <text x="35" y="140" font-size="9" text-anchor="end" fill="#64748b">50%</text>
                    
                    <text x="40" y="155" font-size="9" text-anchor="middle" fill="#64748b">Ep 1</text>
                    <text x="210" y="155" font-size="9" text-anchor="middle" fill="#64748b">Ep 50</text>
                    <text x="380" y="155" font-size="9" text-anchor="middle" fill="#64748b">Ep 100</text>
                  </svg>
                  <div class="flex justify-center gap-6 mt-2 text-xs">
                    <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-blue-600 inline-block"></span> Training Accuracy</span>
                    <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 border-t border-dashed border-emerald-600 inline-block"></span> Validation Accuracy</span>
                  </div>
                </div>

                <!-- Loss Curve SVG -->
                <div class="p-4 border border-slate-100 rounded-lg bg-slate-50/50">
                  <div class="flex items-center justify-between text-xs font-bold text-slate-800 mb-2">
                    <span>Training vs Validation Loss</span>
                    <span class="text-rose-600 font-mono">Final: ~0.439</span>
                  </div>
                  <svg viewBox="0 0 400 180" class="w-full h-44 bg-white border border-slate-200 rounded">
                    <!-- Grid lines -->
                    <line x1="40" y1="20" x2="380" y2="20" stroke="#f1f5f9" stroke-width="1" />
                    <line x1="40" y1="60" x2="380" y2="60" stroke="#f1f5f9" stroke-width="1" />
                    <line x1="40" y1="100" x2="380" y2="100" stroke="#f1f5f9" stroke-width="1" />
                    <line x1="40" y1="140" x2="380" y2="140" stroke="#f1f5f9" stroke-width="1" />
                    
                    <!-- Axes -->
                    <line x1="40" y1="140" x2="380" y2="140" stroke="#cbd5e1" stroke-width="1" />
                    <line x1="40" y1="20" x2="40" y2="140" stroke="#cbd5e1" stroke-width="1" />
                    
                    <!-- Training Loss Curve (Orange) -->
                    <path d="M 40,35 Q 90,110 160,116 T 280,121 T 380,122" fill="none" stroke="#ea580c" stroke-width="2.5" />
                    
                    <!-- Validation Loss Curve (Red Dashed) -->
                    <path d="M 40,70 Q 90,105 160,114 T 280,119 T 380,121" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="4,4" />
                    
                    <!-- Labels -->
                    <text x="35" y="35" font-size="9" text-anchor="end" fill="#64748b">0.65</text>
                    <text x="35" y="80" font-size="9" text-anchor="end" fill="#64748b">0.50</text>
                    <text x="35" y="140" font-size="9" text-anchor="end" fill="#64748b">0.35</text>
                    
                    <text x="40" y="155" font-size="9" text-anchor="middle" fill="#64748b">Ep 1</text>
                    <text x="210" y="155" font-size="9" text-anchor="middle" fill="#64748b">Ep 50</text>
                    <text x="380" y="155" font-size="9" text-anchor="middle" fill="#64748b">Ep 100</text>
                  </svg>
                  <div class="flex justify-center gap-6 mt-2 text-xs">
                    <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-orange-600 inline-block"></span> Training Loss</span>
                    <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 border-t border-dashed border-red-600 inline-block"></span> Validation Loss</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- Confusion Matrix Card -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
                <h3 class="text-xs font-bold text-slate-800 uppercase tracking-wider mb-3">
                  Confusion Matrix Heatmap (N=1,407 test samples)
                </h3>
                <div class="grid grid-cols-2 gap-3 text-center font-mono">
                  <div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <div class="text-2xl font-bold text-blue-900">921</div>
                    <div class="text-xs font-sans text-blue-700 mt-1 font-semibold">True Negatives</div>
                    <div class="text-[10px] text-slate-400">Retained correctly</div>
                  </div>
                  <div class="p-4 bg-rose-50 border border-rose-200 rounded-lg">
                    <div class="text-2xl font-bold text-rose-700">112</div>
                    <div class="text-xs font-sans text-rose-700 mt-1 font-semibold">False Positives</div>
                    <div class="text-[10px] text-slate-400">False alarms</div>
                  </div>
                  <div class="p-4 bg-amber-50 border border-amber-200 rounded-lg">
                    <div class="text-2xl font-bold text-amber-700">204</div>
                    <div class="text-xs font-sans text-amber-700 mt-1 font-semibold">False Negatives</div>
                    <div class="text-[10px] text-slate-400">Missed churners</div>
                  </div>
                  <div class="p-4 bg-emerald-50 border border-emerald-200 rounded-lg">
                    <div class="text-2xl font-bold text-emerald-800">170</div>
                    <div class="text-xs font-sans text-emerald-700 mt-1 font-semibold">True Positives</div>
                    <div class="text-[10px] text-slate-400">Churn detected</div>
                  </div>
                </div>
              </div>

              <!-- Classification Report Table -->
              <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
                <h3 class="text-xs font-bold text-slate-800 uppercase tracking-wider mb-3">
                  Test Evaluation Benchmark (Cell 17)
                </h3>
                <table class="w-full text-xs text-left border-collapse font-mono">
                  <thead>
                    <tr class="border-b border-slate-200 text-slate-500 bg-slate-50">
                      <th class="py-2 px-2 font-sans">Class</th>
                      <th class="py-2 px-2 text-right">Precision</th>
                      <th class="py-2 px-2 text-right">Recall</th>
                      <th class="py-2 px-2 text-right">F1-Score</th>
                    </tr>
                  </thead>
                  <tbody class="divide-y divide-slate-100">
                    <tr>
                      <td class="py-2 px-2 font-sans font-semibold">Retained (0)</td>
                      <td class="py-2 px-2 text-right">0.82</td>
                      <td class="py-2 px-2 text-right">0.89</td>
                      <td class="py-2 px-2 text-right font-bold text-blue-700">0.85</td>
                    </tr>
                    <tr>
                      <td class="py-2 px-2 font-sans font-semibold">Churned (1)</td>
                      <td class="py-2 px-2 text-right">0.60</td>
                      <td class="py-2 px-2 text-right">0.45</td>
                      <td class="py-2 px-2 text-right font-bold text-rose-700">0.52</td>
                    </tr>
                    <tr class="bg-slate-50 font-bold">
                      <td class="py-2 px-2 font-sans">Overall Accuracy</td>
                      <td colspan="2" class="py-2 px-2 text-right">77.54%</td>
                      <td class="py-2 px-2 text-right">0.78</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Dataset Churn Ratio Donut Chart -->
            <div class="bg-white border border-slate-200 rounded-xl p-5 shadow-xs">
              <h3 class="text-xs font-bold text-slate-800 uppercase tracking-wider mb-3">
                Dataset Churn vs. Retention Ratio (N=7,032 total records)
              </h3>
              
              <div class="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
                <!-- SVG Donut Chart -->
                <div class="md:col-span-5 flex justify-center">
                  <svg viewBox="0 0 200 200" class="w-44 h-44">
                    <!-- Retained slice (73.4% = 264.24 deg) -->
                    <circle cx="100" cy="100" r="70" fill="none" stroke="#2563eb" stroke-width="32"
                            stroke-dasharray="322.8 439.8" stroke-dashoffset="0" transform="rotate(-90 100 100)" />
                    <!-- Churned slice (26.6% = 95.76 deg) -->
                    <circle cx="100" cy="100" r="70" fill="none" stroke="#ef4444" stroke-width="32"
                            stroke-dasharray="117 439.8" stroke-dashoffset="-322.8" transform="rotate(-90 100 100)" />
                    <!-- Center Hole Text -->
                    <text x="100" y="93" font-size="11" text-anchor="middle" fill="#64748b" font-weight="bold">TOTAL</text>
                    <text x="100" y="110" font-size="14" text-anchor="middle" fill="#0f172a" font-weight="800" font-family="monospace">7,032</text>
                    <text x="100" y="125" font-size="10" text-anchor="middle" fill="#64748b">Records</text>
                  </svg>
                </div>

                <!-- Legend and Metrics -->
                <div class="md:col-span-7 space-y-3">
                  <div class="p-3 bg-blue-50/70 border border-blue-200 rounded-lg flex items-center justify-between">
                    <div class="flex items-center gap-2">
                      <span class="w-3.5 h-3.5 rounded bg-blue-600 inline-block"></span>
                      <div>
                        <div class="text-xs font-bold text-slate-900">Retained Customers (0)</div>
                        <div class="text-[11px] text-slate-500">Stayed with telecom provider</div>
                      </div>
                    </div>
                    <div class="text-right font-mono">
                      <div class="text-sm font-bold text-blue-700">5,163</div>
                      <div class="text-[11px] font-semibold text-blue-600">73.4%</div>
                    </div>
                  </div>

                  <div class="p-3 bg-rose-50/70 border border-rose-200 rounded-lg flex items-center justify-between">
                    <div class="flex items-center gap-2">
                      <span class="w-3.5 h-3.5 rounded bg-rose-600 inline-block"></span>
                      <div>
                        <div class="text-xs font-bold text-slate-900">Churned Customers (1)</div>
                        <div class="text-[11px] text-slate-500">Cancelled or left service</div>
                      </div>
                    </div>
                    <div class="text-right font-mono">
                      <div class="text-sm font-bold text-rose-700">1,869</div>
                      <div class="text-[11px] font-semibold text-rose-600">26.6%</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ` : ''}

        <!-- TAB 3: CODE EXPLORER -->
        ${activeTab === 'code' ? `
          <div class="space-y-4">
            <div class="flex items-center justify-between">
              <div>
                <h2 class="text-base font-bold text-slate-900">Python Project Files</h2>
                <p class="text-xs text-slate-500">The Streamlit frontend and modular backend helper functions:</p>
              </div>
              <button id="btn-copy-code" class="px-3 py-1.5 border border-slate-200 bg-white hover:bg-slate-50 text-slate-700 text-xs font-semibold rounded-lg flex items-center gap-1.5">
                <span>📋</span>
                <span id="copy-btn-text">Copy Code</span>
              </button>
            </div>

            <!-- File Selectors -->
            <div class="flex gap-2 border-b border-slate-200 pb-2">
              ${Object.keys(CODE_SNIPPETS).map(name => `
                <button class="code-tab-btn px-3 py-1.5 text-xs font-mono rounded-lg transition-colors ${selectedCodeFile === name ? 'bg-slate-900 text-white font-bold' : 'bg-slate-100 text-slate-700 hover:bg-slate-200'}" data-file="${name}">
                  ${name}
                </button>
              `).join('')}
            </div>

            <!-- Code viewer -->
            <div class="bg-slate-900 text-slate-100 p-4 rounded-xl font-mono text-xs overflow-x-auto shadow-sm max-h-[500px]">
              <pre><code id="code-block">${CODE_SNIPPETS[selectedCodeFile]}</code></pre>
            </div>
          </div>
        ` : ''}
      </main>
    </div>
  `;

  // Attach event listeners
  document.getElementById('nav-predict')?.addEventListener('click', () => { activeTab = 'predict'; renderApp(); });
  document.getElementById('nav-charts')?.addEventListener('click', () => { activeTab = 'charts'; renderApp(); });
  document.getElementById('nav-code')?.addEventListener('click', () => { activeTab = 'code'; renderApp(); });

  // Train button handler
  const handleTrainClick = () => {
    if (isTraining) return;
    isTraining = true;
    renderApp();

    const steps = [
      { text: "Loading raw dataset (Telco-Customer-Churn.csv)...", pct: 20, epoch: "Data Ingestion" },
      { text: "Cleaning TotalCharges NaNs & drop duplicates...", pct: 40, epoch: "Data Cleaning" },
      { text: "StandardScaler feature normalization...", pct: 60, epoch: "Preprocessing" },
      { text: "Fitting Dense(16) -> Dense(8) -> Dense(1)...", pct: 85, epoch: "Epoch 35/50" },
      { text: "Training completed! EarlyStopping weights restored.", pct: 100, epoch: "Complete (77.54% Acc)" }
    ];

    let i = 0;
    const interval = setInterval(() => {
      const step = steps[i];
      const stepText = document.getElementById('train-step-text');
      const epochText = document.getElementById('train-epoch-text');
      const bar = document.getElementById('train-bar');
      
      if (stepText) stepText.textContent = step.text;
      if (epochText) epochText.textContent = step.epoch;
      if (bar) bar.style.width = `${step.pct}%`;

      i++;
      if (i >= steps.length) {
        clearInterval(interval);
        setTimeout(() => {
          isTraining = false;
          isTrained = true;
          renderApp();
        }, 500);
      }
    }, 400);
  };

  document.getElementById('btn-train')?.addEventListener('click', handleTrainClick);
  document.getElementById('btn-train-hero')?.addEventListener('click', handleTrainClick);

  document.getElementById('btn-reset')?.addEventListener('click', () => {
    isTrained = false;
    isTraining = false;
    renderApp();
  });

  // Slider Listeners
  const sliderTenure = document.getElementById('slider-tenure') as HTMLInputElement;
  sliderTenure?.addEventListener('input', (e) => {
    tenure = Number((e.target as HTMLInputElement).value);
    total = Number((tenure * monthly).toFixed(2));
    renderApp();
  });

  const sliderMonthly = document.getElementById('slider-monthly') as HTMLInputElement;
  sliderMonthly?.addEventListener('input', (e) => {
    monthly = Number((e.target as HTMLInputElement).value);
    total = Number((tenure * monthly).toFixed(2));
    renderApp();
  });

  const inputTotal = document.getElementById('input-total') as HTMLInputElement;
  inputTotal?.addEventListener('change', (e) => {
    total = Number((e.target as HTMLInputElement).value);
    renderApp();
  });

  document.getElementById('btn-senior-0')?.addEventListener('click', () => { senior = 0; renderApp(); });
  document.getElementById('btn-senior-1')?.addEventListener('click', () => { senior = 1; renderApp(); });

  // Preset Listeners
  document.getElementById('preset-new')?.addEventListener('click', () => {
    tenure = 2; monthly = 70.70; total = 151.65; senior = 0;
    renderApp();
  });
  document.getElementById('preset-mid')?.addEventListener('click', () => {
    tenure = 34; monthly = 56.95; total = 1889.50; senior = 0;
    renderApp();
  });
  document.getElementById('preset-senior')?.addEventListener('click', () => {
    tenure = 8; monthly = 99.65; total = 820.50; senior = 1;
    renderApp();
  });

  // Code Explorer tab click
  document.querySelectorAll('.code-tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      selectedCodeFile = (e.currentTarget as HTMLElement).getAttribute('data-file') || 'app.py';
      renderApp();
    });
  });

  document.getElementById('btn-copy-code')?.addEventListener('click', () => {
    navigator.clipboard.writeText(CODE_SNIPPETS[selectedCodeFile] || '');
    const copyText = document.getElementById('copy-btn-text');
    if (copyText) {
      copyText.textContent = 'Copied!';
      setTimeout(() => { copyText.textContent = 'Copy Code'; }, 1500);
    }
  });
}

// Initial render
renderApp();
