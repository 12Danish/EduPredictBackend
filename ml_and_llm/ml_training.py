import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Load and clean dataset
df = pd.read_csv("data.csv", sep=";")

# Remove quotes and extra spaces from column names and string columns
df.columns = (df.columns
              .str.replace('"', '', regex=False)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip())

for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.replace('"', '', regex=False).str.replace(r"\s+", " ", regex=True).str.strip()

# Convert 3-class target to 2-class: 'Graduate' and 'Enrolled' -> 'Success'
df["Target"] = df["Target"].replace({"Graduate": "Success", "Enrolled": "Success"})
y = (df["Target"] == "Dropout").astype(int)
X = df.drop(columns=["Target"])

# Split data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Encode categorical features and scale numerical features
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

# One-hot encode categorical variables
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(X_train[cat_cols])
ohe_features = ohe.get_feature_names_out(cat_cols)

# ColumnTransformer for scaling numeric and encoding categorical features
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", ohe, cat_cols)
])
preprocessor.fit(X_train)

# Transform training and test sets
X_train_trans = preprocessor.transform(X_train)
X_test_trans = preprocessor.transform(X_test)
all_features = num_cols + ohe_features.tolist()

# Identify top 10 important features using Random Forest

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_trans, y_train)

importances = pd.Series(rf.feature_importances_, index=all_features)
top_features = importances.sort_values(ascending=False).head(8).index.tolist()

print("\nTop 10 features selected based on importance:")
for i, f in enumerate(top_features, 1):
    print(f"{i}. {f}")

# Keep only top features for training and test sets
X_train_top = pd.DataFrame(X_train_trans, columns=all_features)[top_features]
X_test_top = pd.DataFrame(X_test_trans, columns=all_features)[top_features]

# MLP + SMOTE
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True,
)

pipe = ImbPipeline([
    ("smote", SMOTE(random_state=42)),  # Handle class imbalance
    ("clf", mlp_clf)
])

# Hyperparameter grid for GridSearch
param_grid = {
    "clf__hidden_layer_sizes": [(64, 32), (128, 64), (32, 16)],
    "clf__learning_rate_init": [0.001, 0.01],
    "clf__alpha": [0.0001, 0.001],
}

# hyper-parameter tuning
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=1,
    refit=True
)

# Fit GridSearch on training data
grid.fit(X_train_top, y_train)
best_model = grid.best_estimator_
print("\nBest hyperparameters found:")
print(grid.best_params_)


# evaluation function
def eval_model(model, X_te, y_te, threshold=0.5, label="Default 0.5"):
    proba = model.predict_proba(X_te)[:, 1]
    y_hat = (proba >= threshold).astype(int)
    acc = accuracy_score(y_te, y_hat)
    prec = precision_score(y_te, y_hat, zero_division=0)
    rec = recall_score(y_te, y_hat, zero_division=0)
    f1 = f1_score(y_te, y_hat, zero_division=0)
    print(f"\n[{label}]")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_te, y_hat))
    return acc, prec, rec, f1


# test set check on default threshold
eval_model(best_model, X_test_top, y_test, threshold=0.5, label="Default threshold 0.5")

# threshold optimization
X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(
    X_train_top, y_train, test_size=0.2, stratify=y_train, random_state=42
)

best_model_sub = grid.best_estimator_
best_model_sub.fit(X_tr_sub, y_tr_sub)
proba_val = best_model_sub.predict_proba(X_val_sub)[:, 1]


def find_best_threshold(y_true, p, metric="f1"):
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t, best_score = 0.5, -1
    for t in thresholds:
        y_hat = (p >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_hat, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_hat, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_hat, zero_division=0)
        else:
            raise ValueError("metric must be one of: f1, precision, recall")
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score


best_t, best_t_f1 = find_best_threshold(y_val_sub, proba_val, metric="f1")
print(f"\nBest threshold on validation (based on F1): {best_t:.2f}  (F1={best_t_f1:.4f})")

# training metrics on orignal data  (non-SMOTE)
best_model.fit(X_train_top, y_train)  # Fit final model

train_metrics = eval_model(best_model, X_train_top, y_train, threshold=best_t, label="Training set (real)")
test_metrics = eval_model(best_model, X_test_top, y_test, threshold=best_t, label="Test set")
# train vs test
metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
print("\n=== Training vs Test Metrics ===")
for i, name in enumerate(metrics_names):
    print(f"{name}: Train = {train_metrics[i]:.4f}, Test = {test_metrics[i]:.4f}")

print(f"\nSaving model trained on {len(top_features)} top features...")

model_artifacts = {
    'model': best_model,  # Pipeline already trained on top 10 features
    'top_features': top_features,  # The 10 selected feature names
    'optimal_threshold': best_t,  # Optimized decision threshold
    'best_params': grid.best_params_,  # Best hyperparameters from GridSearch
    'class_mapping': {0: 'Success', 1: 'Dropout'},
    'model_performance': {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'validation_f1_at_optimal_threshold': best_t_f1,
        'n_features_used': len(top_features)
    }
}
# Save the complete model package
joblib.dump(model_artifacts, 'student_dropout_model_top10.pkl')
