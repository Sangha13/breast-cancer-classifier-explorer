import streamlit as st
import pandas as pd

from components.utils import load_csv_data, preprocess_data, split_and_scale
from components.model_trainer import get_model, train_and_predict, evaluate_model
from components.plots import plot_confusion_matrix, plot_roc_curve, plot_decision_boundary

# Streamlit UI
st.title("🔬 Breast Cancer Classifier Explorer")
st.write("Upload your dataset or use the built-in Breast Cancer dataset.")

# Load Dataset
df = load_csv_data()
st.subheader("📊 Data Preview")
st.dataframe(df.head())

# Feature Selection
all_features = list(df.columns[:-1])  # exclude target
features = st.multiselect("Select exactly 2 features for decision boundary visualization:", all_features, default=["mean radius", "mean texture"])

if len(features) != 2:
    st.warning("Please select exactly 2 features to continue.")
    st.stop()

# Model Selection
st.sidebar.title("🧠 Model & Parameters")
model_type = st.sidebar.selectbox("Choose model", ["KNN", "SVM", "Decision Tree"])

params = {}

if model_type == "KNN":
    k = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
    params["k"] = k
elif model_type == "SVM":
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
    params["C"] = C
    params["kernel"] = kernel
elif model_type == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 4)
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
    params["max_depth"] = max_depth
    params["criterion"] = criterion

# Preprocessing
X, y = preprocess_data(df, features)
X_train, X_test, y_train, y_test = split_and_scale(X, y)

# Train & Predict
model = get_model(model_type, params)
y_pred, y_prob = train_and_predict(model, X_train, X_test, y_train)
acc, f1, cm = evaluate_model(y_test, y_pred)

# Results
st.subheader("✅ Model Evaluation")
st.write(f"**Accuracy:** {acc:.2f}")
st.write(f"**F1 Score:** {f1:.2f}")

st.subheader("🔲 Confusion Matrix")
fig_cm = plot_confusion_matrix(cm)
st.pyplot(fig_cm)

if y_prob is not None:
    st.subheader("📈 ROC Curve")
    fig_roc = plot_roc_curve(y_test, y_prob, label=model_type)
    st.pyplot(fig_roc)

st.subheader("🧠 Decision Boundary (Training Data)")
fig_db = plot_decision_boundary(model, X_train, y_train, features)
st.pyplot(fig_db)
