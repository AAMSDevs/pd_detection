import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# === Load Data ===
file_path = r'D:\Training\signals_labels_binary_Tr0.csv'
te0_df = pd.read_csv(file_path)

# === Convert Signal Strings to Arrays ===
te0_df['Signal Array'] = te0_df['Signal Values'].apply(
    lambda x: np.array([float(i) for i in x.split(',')])
)

# === Extract Statistical Features ===
def extract_features(signal):
    return {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'max': np.max(signal),
        'min': np.min(signal),
        'skew': skew(signal),
        'kurtosis': kurtosis(signal)
    }

features_df = te0_df['Signal Array'].apply(extract_features).apply(pd.Series)

# === Handle Labels ===
te0_df['Label'] = te0_df['Label'].astype(int)               # Integer for filtering
features_df['Label'] = te0_df['Label'].astype(str)         # String for plotting

# === Plot One PD and One Non-PD Signal ===
pd_signal = te0_df[te0_df['Label'] == 1]['Signal Array'].iloc[0]
non_pd_signal = te0_df[te0_df['Label'] == 0]['Signal Array'].iloc[0]

plt.figure()
plt.plot(pd_signal, color='red', label='PD Signal')
plt.title('Partial Discharge (PD) Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(non_pd_signal, color='blue', label='Non-PD Signal')
plt.title('Non-PD (Noise) Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

# === Histograms of Signal Amplitudes ===
plt.figure()
plt.hist(pd_signal, bins=50, color='red', edgecolor='black', alpha=0.7)
plt.title('Histogram of PD Signal')
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

plt.figure()
plt.hist(non_pd_signal, bins=50, color='blue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Non-PD Signal')
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# === Correlation Matrices for Features ===
pd_features = features_df[features_df['Label'] == '1'].drop('Label', axis=1)
non_pd_features = features_df[features_df['Label'] == '0'].drop('Label', axis=1)

plt.figure(figsize=(8, 6))
sns.heatmap(pd_features.corr(), annot=True, cmap='Reds', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - PD Signals')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(non_pd_features.corr(), annot=True, cmap='Blues', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix - Non-PD Signals')
plt.tight_layout()
plt.show()

# === Boxplots for Features in 2x3 Grid with Wider Y-Axis ===
feature_names = ['mean', 'std', 'max', 'min', 'skew', 'kurtosis']
palette_colors = {'0': 'blue', '1': 'red'}

# Wider y-axis limits for clarity
y_axis_limits = {
    'mean': (-0.1, 0.1),
    'std': (0, 10),
    'max': (0, 10),

    'min': (-10, 0),
    'skew': (-1, 1),
    'kurtosis': (-5, 30)
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Boxplots of Extracted Features (PD vs Non-PD)', fontsize=16)

for i, feature in enumerate(feature_names):
    row, col = divmod(i, 3)
    ax = axes[row, col]

    sns.boxplot(x='Label', y=feature, data=features_df, hue='Label',
                palette=palette_colors, ax=ax, dodge=False, width=0.5, linewidth=1.2, legend=False)

    ax.set_title(f'{feature.capitalize()} Distribution', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel(feature)
    ax.set_ylim(y_axis_limits[feature])
    ax.grid(True, linestyle='--', alpha=0.5)

    if i == 0:
        ax.legend(title='Signal Type', labels=['Non-PD (0)', 'PD (1)'])
    else:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
