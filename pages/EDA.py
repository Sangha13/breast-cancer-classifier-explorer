import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import os

# ✅ Fix ImportError: add parent dir to sys.path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from components.utils import load_csv_data
st.title("Exploratory Data Analysis (EDA)")

# Load dataset
df = load_csv_data()

st.subheader("Dataset Overview")
st.write("Shape of dataset:", df.shape)
st.dataframe(df.head())

st.subheader("Class Distribution")
class_counts = df['target'].value_counts()
st.bar_chart(class_counts)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.subheader("Feature Summary")
st.dataframe(df.describe())

st.subheader("Boxplot of Selected Feature")
selected_feature = st.selectbox("Choose a feature to plot:", df.columns[:-1])
fig2, ax2 = plt.subplots()
sns.boxplot(data=df, x="target", y=selected_feature, ax=ax2)
st.pyplot(fig2)
