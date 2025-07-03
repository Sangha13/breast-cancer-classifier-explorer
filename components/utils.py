import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

#Load Breast Cancer CSV from data folder
def load_csv_data():
    data_path = os.path.join("data", "breast_cancer.csv")
    df = pd.read_csv(data_path)
    return df

#Extract features and target
def preprocess_data(df, selected_features, target_column='target'):
    X = df[selected_features].values
    y = df[target_column].values
    return X, y

#Split data and apply Standard Scaling
def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
