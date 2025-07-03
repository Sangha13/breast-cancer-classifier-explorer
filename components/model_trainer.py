from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

#Initialize the model based on user input
def get_model(model_type, params):
    if model_type == "KNN":
        return KNeighborsClassifier(n_neighbors=params.get("k", 5))
    elif model_type == "SVM":
        return SVC(C=params.get("C", 1.0), kernel=params.get("kernel", "rbf"), probability=True)
    elif model_type == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=params.get("max_depth", 4),
            criterion=params.get("criterion", "gini")
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

#Train the model and return predictions + probabilities
def train_and_predict(model, X_train, X_test, y_train):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, y_prob

#Evaluate model performance
def evaluate_model(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, f1, cm
