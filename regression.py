import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import Parallel, delayed
import os

# === Load Dataset Files Separately ===
dataset_files = {
    'Train1': r'F:\Python\Research Day 2025\Datasets\Training\signals_labels_binary_Tr0.csv',
    'Train2': r'F:\Python\Research Day 2025\Datasets\Training\signals_labels_binary_Tr1.csv',
    'Validation': r'F:\Python\Research Day 2025\Datasets\Validation\signals_labels_binary_Va0.csv',
    'Test1': r'F:\Python\Research Day 2025\Datasets\Testing\signals_labels_binary_Te0.csv',
    'Test2': r'F:\Python\Research Day 2025\Datasets\Testing\signals_labels_binary_Te1.csv'
}

datasets = {}
for name, path in dataset_files.items():
    if os.path.exists(path):
        try:
            datasets[name] = pd.read_csv(path, low_memory=False)
            print(f"OK Loaded {name} - Shape: {datasets[name].shape}")
        except Exception as e:
            print(f"ERROR loading {name}: {e}")
    else:
        print(f"ERROR: {name} file not found at {path}")

# === Optimized Feature Extraction ===
def extract_features(signal_str):
    """Extracts statistical features from a signal string"""
    try:
        signal = np.fromstring(signal_str, sep=',', dtype=np.float32)
        if signal.size == 0:
            return [np.nan] * 6
        return [
            np.mean(signal),
            np.std(signal),
            np.max(signal),
            np.min(signal),
            skew(signal),
            kurtosis(signal)
        ]
    except Exception as e:
        print(f"ERROR processing signal: {e}")
        return [np.nan] * 6

# === Process Datasets Using Parallel Processing ===
def process_dataset(df):
    if 'Signal Values' not in df.columns or 'Label' not in df.columns:
        print("ERROR: Missing required columns")
        return None
    df = df.dropna(subset=['Signal Values'])
    
    features = Parallel(n_jobs=-1)(delayed(extract_features)(sig) for sig in df['Signal Values'])
    feature_columns = ['mean', 'std', 'max', 'min', 'skew', 'kurtosis']
    
    features_df = pd.DataFrame(features, columns=feature_columns)
    features_df['Label'] = df['Label'].astype(int)
    return features_df.dropna()

processed_datasets = {name: process_dataset(df) for name, df in datasets.items() if df is not None}

# === Merge Training Data ===
if 'Train1' in processed_datasets and 'Train2' in processed_datasets:
    train_df = pd.concat([processed_datasets['Train1'], processed_datasets['Train2']], ignore_index=True)
    X_train, y_train = train_df.drop(columns=['Label']), train_df['Label']
else:
    print("ERROR: Missing training datasets.")
    exit()

# === Validation & Test Data ===
def get_features_labels(name):
    df = processed_datasets.get(name)
    if df is not None:
        return df.drop(columns=['Label']), df['Label']
    return None, None

X_val, y_val = get_features_labels('Validation')
X_test1, y_test1 = get_features_labels('Test1')
X_test2, y_test2 = get_features_labels('Test2')

if X_val is None:
    print("ERROR: Missing validation dataset.")
    exit()

# === Train Optimized Random Forest Model ===
rf_model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)

# === Evaluate Model on Validation Set ===
y_val_pred = rf_model.predict(X_val)
print("\nValidation Results:")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(classification_report(y_val, y_val_pred))

# === Evaluate Model on Test Sets ===
def evaluate_model(X_test, y_test, test_name):
    if X_test is not None and y_test is not None:
        y_pred = rf_model.predict(X_test)
        print(f"\n{test_name} Results:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
        return confusion_matrix(y_test, y_pred)
    return None

cm_test1 = evaluate_model(X_test1, y_test1, "Test1")
cm_test2 = evaluate_model(X_test2, y_test2, "Test2")

# === Confusion Matrices ===
def plot_confusion_matrix(cm, title, ax, cmap):
    if cm is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_confusion_matrix(cm_test1, "Confusion Matrix - Test1", axes[0], "Blues")
plot_confusion_matrix(cm_test2, "Confusion Matrix - Test2", axes[1], "Reds")
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# === Polynomial Regression on Most Important Feature ===
# Step 1: Get feature importance
feature_names = X_train.columns
importances = rf_model.feature_importances_
most_important_idx = np.argmax(importances)
most_important_feature = feature_names[most_important_idx]
print(f"\nMost important feature: {most_important_feature} (Importance: {importances[most_important_idx]:.4f})")

# Step 2: Prepare data for polynomial regression
X_feature = X_val[most_important_feature].values.reshape(-1, 1)
y_prob = rf_model.predict_proba(X_val)[:, 1]  # Predicted probabilities for PD

# Step 3: Fix any NaN or infinite values
mask = np.isfinite(X_feature.flatten()) & np.isfinite(y_prob)
X_feature = X_feature[mask].reshape(-1, 1)
y_prob = y_prob[mask]

# Step 4: Fit polynomial regression (degree 3)
degree = 3
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(X_feature, y_prob)

# Generate points for the regression curve
X_range = np.linspace(X_feature.min(), X_feature.max(), 100).reshape(-1, 1)
y_pred = polyreg.predict(X_range)

# Step 5: Calculate R², MSE, and MAE
y_pred_val = polyreg.predict(X_feature)  # Predictions on validation set
r2 = r2_score(y_prob, y_pred_val)
mse = mean_squared_error(y_prob, y_pred_val)
mae = mean_absolute_error(y_prob, y_pred_val)
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Step 6: Extract polynomial coefficients
poly_features = polyreg.named_steps['polynomialfeatures']
lin_reg = polyreg.named_steps['linearregression']
coefs = lin_reg.coef_
intercept = lin_reg.intercept_

# Construct the polynomial equation
terms = [f"{intercept:.4f}"]
for i, coef in enumerate(coefs[1:], 1):
    if coef != 0:
        terms.append(f"{coef:+.4f}x^{i}")
equation = "y = " + " ".join(terms)
print(f"Polynomial equation: {equation}")

# Step 7: Plot the regression
plt.figure(figsize=(10, 6))
plt.scatter(X_feature, y_prob, color='blue', alpha=0.5, label='Data points')
plt.plot(X_range, y_pred, color='red', linewidth=2, label=f'Polynomial regression (degree {degree})')
plt.xlabel(most_important_feature)
plt.ylabel('Predicted Probability of PD')
plt.title(f'Polynomial Regression: {most_important_feature} vs. PD Probability\nR²={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}')
plt.legend()
plt.grid(True)
plt.savefig('polynomial_regression_pd.png')
plt.show()  # Display the plot interactively
