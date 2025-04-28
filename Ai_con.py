# === 📦 Import Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import Parallel, delayed
import os

# === 📂 Load Dataset Files ===
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
            print(f"✅ Loaded {name} - Shape: {datasets[name].shape}")
        except Exception as e:
            print(f"❌ Error loading {name}: {e}")
    else:
        print(f"❌ Warning: {name} file not found at {path}")

# === 🛠 Feature Extraction Function ===
def extract_features(signal_str):
    """Extracts statistical features from a signal string."""
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
        print(f"❌ Error processing signal: {e}")
        return [np.nan] * 6

# === 🚀 Process Datasets ===
def process_dataset(df):
    if 'Signal Values' not in df.columns or 'Label' not in df.columns:
        print("❌ Error: Missing required columns")
        return None
    df = df.dropna(subset=['Signal Values'])
    
    features = Parallel(n_jobs=-1)(delayed(extract_features)(sig) for sig in df['Signal Values'])
    feature_columns = ['mean', 'std', 'max', 'min', 'skew', 'kurtosis']
    
    features_df = pd.DataFrame(features, columns=feature_columns)
    features_df['Label'] = df['Label'].astype(int)
    return features_df.dropna()

processed_datasets = {name: process_dataset(df) for name, df in datasets.items() if df is not None}

# === 🔹 Merge Training Data ===
if 'Train1' in processed_datasets and 'Train2' in processed_datasets:
    train_df = pd.concat([processed_datasets['Train1'], processed_datasets['Train2']], ignore_index=True)
    X_train, y_train = train_df.drop(columns=['Label']), train_df['Label']
else:
    print("❌ Error: Missing training datasets.")
    exit()

# === 🔹 Validation & Test Data Preparation ===
def get_features_labels(name):
    df = processed_datasets.get(name)
    if df is not None:
        return df.drop(columns=['Label']), df['Label']
    return None, None

X_val, y_val = get_features_labels('Validation')
X_test1, y_test1 = get_features_labels('Test1')
X_test2, y_test2 = get_features_labels('Test2')

if X_val is None:
    print("❌ Error: Missing validation dataset.")
    exit()

# === 🤖 Train Random Forest Model ===
rf_model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)

# === 🔍 Evaluate Model ===
def evaluate_model(X_test, y_test, test_name):
    if X_test is not None and y_test is not None:
        y_pred = rf_model.predict(X_test)
        print(f"\n🔹 {test_name} Results:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))
        return confusion_matrix(y_test, y_pred)
    return None

print("\n🔹 Validation Set Evaluation:")
cm_val = evaluate_model(X_val, y_val, "Validation Set")

cm_test1 = evaluate_model(X_test1, y_test1, "Test1 Set")
cm_test2 = evaluate_model(X_test2, y_test2, "Test2 Set")

# === 📊 Plot Confusion Matrices ===
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
plt.show()

# === 📈 Measure Model Confidence ===
def measure_confidence(model, X, y, dataset_name):
    if X is None or y is None:
        print(f"❌ {dataset_name} is missing, skipping confidence analysis.")
        return
    
    y_probs = model.predict_proba(X)
    y_pred = np.argmax(y_probs, axis=1)
    y_confidences = y_probs[np.arange(len(y_probs)), y_pred]

    avg_confidence = np.mean(y_confidences)
    std_confidence = np.std(y_confidences)
    high_confidence_rate = np.mean(y_confidences > 0.9) * 100

    print(f"\n🔹 Confidence Analysis on {dataset_name}:")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Standard Deviation: {std_confidence:.4f}")
    print(f"High Confidence Predictions (>90%): {high_confidence_rate:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.hist(y_confidences, bins=20, color='lightgreen', edgecolor='black')
    plt.title(f'Confidence Distribution - {dataset_name}')
    plt.xlabel('Confidence Level')
    plt.ylabel('Number of Samples')
    plt.grid(True)
    plt.show()

# 🚀 Run Confidence Measurement
measure_confidence(rf_model, X_val, y_val, "Validation Set")
measure_confidence(rf_model, X_test1, y_test1, "Test1 Set")
measure_confidence(rf_model, X_test2, y_test2, "Test2 Set")
