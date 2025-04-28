import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from statsmodels.stats.weightstats import ztest
from joblib import Parallel, delayed
import os

# ====================== 🚀 MAIN ANALYSIS PIPELINE ======================

def main():
    # === 📂 1. Load Dataset Files ===
    print("\n" + "="*50)
    print("📂 LOADING DATASETS")
    print("="*50)
    dataset_files = {
        'Train1': r'F:\\Python\\Research Day 2025\\Datasets\\Training\\signals_labels_binary_Tr0.csv',
        'Train2': r'F:\\Python\\Research Day 2025\\Datasets\\Training\\signals_labels_binary_Tr1.csv',
        'Validation': r'F:\\Python\\Research Day 2025\\Datasets\\Validation\\signals_labels_binary_Va0.csv',
        'Test1': r'F:\\Python\\Research Day 2025\\Datasets\\Testing\\signals_labels_binary_Te0.csv',
        'Test2': r'F:\\Python\\Research Day 2025\\Datasets\\Testing\\signals_labels_binary_Te1.csv'
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

    # === 🛠 2. Feature Extraction ===
    print("\n" + "="*50)
    print("🛠 FEATURE EXTRACTION")
    print("="*50)

    def extract_features(signal_str):
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

    def process_dataset(df):
        if 'Signal Values' not in df.columns or 'Label' not in df.columns:
            print("❌ Error: Missing required columns")
            return None
        df = df.dropna(subset=['Signal Values'])

        features = Parallel(n_jobs=-1)(
            delayed(extract_features)(sig) for sig in df['Signal Values']
        )
        feature_columns = ['mean', 'std', 'max', 'min', 'skew', 'kurtosis']

        features_df = pd.DataFrame(features, columns=feature_columns)
        features_df['Label'] = df['Label'].astype(int)
        return features_df.dropna()

    processed_datasets = {name: process_dataset(df) for name, df in datasets.items() if df is not None}

    # === 🔬 3. Hypothesis Testing ===
    print("\n" + "="*50)
    print("🔬 HYPOTHESIS TESTING: PD vs NON-PD FEATURES")
    print("="*50)

    def perform_hypothesis_tests(features_df):
        pd_signals = features_df[features_df['Label'] == 1]
        non_pd_signals = features_df[features_df['Label'] == 0]

        results = []
        for feature in ['mean', 'std', 'max', 'min', 'skew', 'kurtosis']:
            group1 = pd_signals[feature]
            group0 = non_pd_signals[feature]

            try:
                z_stat, p_val = ztest(group1, group0)
            except Exception as e:
                print(f"❌ Z-test failed for {feature}: {e}")
                z_stat, p_val = np.nan, np.nan

            pooled_std = np.sqrt((group1.std() ** 2 + group0.std() ** 2) / 2)
            cohen_d = (group1.mean() - group0.mean()) / pooled_std

            results.append({
                'Feature': feature,
                'Test': "Z-test",
                'Z-statistic': z_stat,
                'p-value': p_val,
                'Cohen_d': cohen_d,
                'Significant': p_val < 0.05 if not np.isnan(p_val) else False
            })

        return pd.DataFrame(results)

    if 'Train1' in processed_datasets:
        test_results = perform_hypothesis_tests(processed_datasets['Train1'])
        print("\n📊 Hypothesis Test Results:")
        print(test_results.sort_values('p-value'))

        plt.figure(figsize=(14, 8))
        for i, feature in enumerate(['mean', 'std', 'max', 'min', 'skew', 'kurtosis']):
            plt.subplot(2, 3, i+1)
            sns.boxplot(x='Label', y=feature, data=processed_datasets['Train1'])
            plt.title(f"{feature}\n(p={test_results[test_results['Feature']==feature]['p-value'].values[0]:.2e})")
        plt.tight_layout()
        plt.savefig('feature_distributions.pdf')
        plt.show()

    # === 🤖 4. Machine Learning Model ===
    print("\n" + "="*50)
    print("🤖 MACHINE LEARNING MODEL")
    print("="*50)

    if 'Train1' in processed_datasets and 'Train2' in processed_datasets:
        train_df = pd.concat([processed_datasets['Train1'], processed_datasets['Train2']], ignore_index=True)
        X_train, y_train = train_df.drop(columns=['Label']), train_df['Label']
    else:
        print("❌ Error: Missing training datasets.")
        return

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
        return

    rf_model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)

    y_val_pred = rf_model.predict(X_val)
    print("\n🔹 Validation Results:")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(classification_report(y_val, y_val_pred))

    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\n🔝 Feature Importance:")
    print(feature_importance)

    if 'Train1' in processed_datasets:
        combined_results = test_results.merge(feature_importance, on='Feature')
        print("\n📊 Combined Statistical and Model Results:")
        print(combined_results.sort_values('p-value'))

    # === 📊 5. Evaluation on Test Sets ===
    print("\n" + "="*50)
    print("📊 MODEL EVALUATION")
    print("="*50)

    def evaluate_model(X_test, y_test, test_name):
        if X_test is not None and y_test is not None:
            y_pred = rf_model.predict(X_test)
            print(f"\n🔹 {test_name} Results:")
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(classification_report(y_test, y_pred))
            return confusion_matrix(y_test, y_pred)
        return None

    cm_test1 = evaluate_model(X_test1, y_test1, "Test1")
    cm_test2 = evaluate_model(X_test2, y_test2, "Test2")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if cm_test1 is not None:
        sns.heatmap(cm_test1, annot=True, fmt='d', cmap="Blues", ax=axes[0])
        axes[0].set_title("Test1 Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

    if cm_test2 is not None:
        sns.heatmap(cm_test2, annot=True, fmt='d', cmap="Reds", ax=axes[1])
        axes[1].set_title("Test2 Confusion Matrix")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()

if _name_ == "_main_":
    main()
