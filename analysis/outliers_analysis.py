import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# File Paths
ROOT = Path('.') / "analysis"

DATA_DIR = ROOT / 'data'
PLOT_DIR = ROOT / 'plots'
PLOT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
np.random.seed(42)

train_data = pd.read_csv(DATA_DIR / 'pirate_pain_train.csv')
labels = pd.read_csv(DATA_DIR / 'pirate_pain_train_labels.csv')
train_data = train_data.merge(labels, on='sample_index', how='left')

pain_survey_cols = ['pain_survey_1', 'pain_survey_2', 'pain_survey_3', 'pain_survey_4']
joint_cols = [f'joint_{i:02d}' for i in range(31)]
all_time_features = pain_survey_cols + joint_cols

def savefig(fig, name):
    fig.savefig(PLOT_DIR / name, bbox_inches='tight', dpi=100)
    plt.close(fig)

print("="*80)
print("OUTLIERS ANALYSIS")
print("="*80)
print(f"Dataset: {len(train_data)} timesteps from {train_data['sample_index'].nunique()} samples")
print(f"Features analyzed: {len(all_time_features)} temporal features")
print("="*80)

# =============================================================================
# OUTLIER DETECTION METHODS
# =============================================================================

def detect_zscore_outliers(series, threshold=3):
    """Z-score method: Good for normally distributed data"""
    z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
    return z_scores > threshold

def detect_iqr_outliers(series, k=1.5):
    """IQR method: Robust to non-normal distributions"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return (series < lower_bound) | (series > upper_bound)

def detect_mad_outliers(series, threshold=3.5):
    """Median Absolute Deviation: Very robust method"""
    median = series.median()
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return pd.Series([False] * len(series), index=series.index)
    modified_z_scores = 0.6745 * (series - median) / mad
    return np.abs(modified_z_scores) > threshold

def detect_rolling_outliers(series, window=10, threshold=3):
    """Rolling Z-score: Time-aware outlier detection"""
    rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window=window, center=True, min_periods=1).std()
    rolling_std = rolling_std.replace(0, 1e-10)  # Avoid division by zero
    z_scores = np.abs((series - rolling_mean) / rolling_std)
    return z_scores > threshold

def detect_temporal_outliers_per_sample(df, feature_cols, method='rolling', **kwargs):
    """Detect outliers within each time series sample"""
    df = df.sort_values(['sample_index', 'time']).copy()
    outlier_results = {}
    
    for col in feature_cols:
        outliers_list = []
        for sample_idx in df['sample_index'].unique():
            sample_data = df[df['sample_index'] == sample_idx][col].copy()
            
            if method == 'rolling':
                outliers = detect_rolling_outliers(sample_data, **kwargs)
            elif method == 'zscore':
                outliers = detect_zscore_outliers(sample_data, **kwargs)
            elif method == 'iqr':
                outliers = detect_iqr_outliers(sample_data, **kwargs)
            elif method == 'mad':
                outliers = detect_mad_outliers(sample_data, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            outliers_list.extend(outliers.values)
        
        outlier_results[col] = outliers_list
    
    return pd.DataFrame(outlier_results, index=df.index)

# =============================================================================
# ANALYZE OUTLIERS ACROSS ALL FEATURES
# =============================================================================

print("\n" + "="*80)
print("GLOBAL OUTLIER DETECTION (across all timesteps)")
print("="*80)

outlier_summary = []

for col in all_time_features:
    data = train_data[col].dropna()
    
    # Apply different methods
    zscore_out = detect_zscore_outliers(data, threshold=3)
    iqr_out = detect_iqr_outliers(data, k=1.5)
    mad_out = detect_mad_outliers(data, threshold=3.5)
    
    outlier_summary.append({
        'feature': col,
        'zscore_outliers': zscore_out.sum(),
        'zscore_pct': (zscore_out.sum() / len(data)) * 100,
        'iqr_outliers': iqr_out.sum(),
        'iqr_pct': (iqr_out.sum() / len(data)) * 100,
        'mad_outliers': mad_out.sum(),
        'mad_pct': (mad_out.sum() / len(data)) * 100,
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'q1': data.quantile(0.25),
        'median': data.median(),
        'q3': data.quantile(0.75)
    })

outlier_df = pd.DataFrame(outlier_summary)

print("\nTop 10 features with most outliers (Z-score method):")
print(outlier_df.nlargest(10, 'zscore_pct')[['feature', 'zscore_outliers', 'zscore_pct', 'mean', 'std']])

print("\nTop 10 features with most outliers (IQR method):")
print(outlier_df.nlargest(10, 'iqr_pct')[['feature', 'iqr_outliers', 'iqr_pct', 'median', 'q1', 'q3']])

# =============================================================================
# TIME-AWARE OUTLIER DETECTION
# =============================================================================

print("\n" + "="*80)
print("TIME-AWARE OUTLIER DETECTION (within each time series)")
print("="*80)

# Analyze a subset of features for temporal outliers
sample_features = pain_survey_cols + joint_cols[:5]  # Pain surveys + first 5 joints

temporal_outliers_rolling = detect_temporal_outliers_per_sample(
    train_data, sample_features, method='rolling', window=10, threshold=3
)

print("\nTemporal outliers detected (Rolling Z-score, window=10):")
for col in sample_features:
    n_outliers = temporal_outliers_rolling[col].sum()
    pct = (n_outliers / len(train_data)) * 100
    print(f"  {col}: {n_outliers} outliers ({pct:.2f}%)")

# =============================================================================
# 4. OUTLIER IMPACT ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("OUTLIER IMPACT ANALYSIS")
print("="*80)

# Analyze how outliers affect statistics
impact_analysis = []

for col in all_time_features:
    data = train_data[col].dropna()
    outliers = detect_iqr_outliers(data, k=1.5)
    
    mean_with = data.mean()
    mean_without = data[~outliers].mean()
    std_with = data.std()
    std_without = data[~outliers].std()
    
    impact_analysis.append({
        'feature': col,
        'outlier_count': outliers.sum(),
        'outlier_pct': (outliers.sum() / len(data)) * 100,
        'mean_change': abs(mean_with - mean_without),
        'mean_change_pct': abs((mean_with - mean_without) / mean_with) * 100 if mean_with != 0 else 0,
        'std_change': abs(std_with - std_without),
        'std_change_pct': abs((std_with - std_without) / std_with) * 100 if std_with != 0 else 0
    })

impact_df = pd.DataFrame(impact_analysis)

print("\nFeatures most affected by outliers (mean change):")
print(impact_df.nlargest(10, 'mean_change_pct')[['feature', 'outlier_pct', 'mean_change', 'mean_change_pct']])

print("\nFeatures most affected by outliers (std change):")
print(impact_df.nlargest(10, 'std_change_pct')[['feature', 'outlier_pct', 'std_change', 'std_change_pct']])

# =============================================================================
# 5. VISUALIZATION: OUTLIER DISTRIBUTION
# =============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS...")
print("="*80)

# Plot 1: Comprehensive outlier overview by IQR method
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top left: Top features with most outliers (IQR method)
top_10 = outlier_df.nlargest(15, 'iqr_pct')
axes[0, 0].barh(range(len(top_10)), top_10['iqr_pct'], color='coral')
axes[0, 0].set_yticks(range(len(top_10)))
axes[0, 0].set_yticklabels(top_10['feature'], fontsize=9)
axes[0, 0].set_xlabel('Outlier Percentage (%)', fontsize=10)
axes[0, 0].set_title('Top 15 Features by Outlier Count (IQR Method)', fontsize=11, fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].axvline(5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Top right: Distribution comparison for worst feature
feature_example = outlier_df.nlargest(1, 'iqr_pct')['feature'].values[0]
data_full = train_data[feature_example].dropna()
outliers_mask = detect_iqr_outliers(data_full, k=1.5)
data_clean = data_full[~outliers_mask]

axes[0, 1].hist(data_full, bins=50, alpha=0.5, color='blue', edgecolor='black', label='With outliers')
axes[0, 1].hist(data_clean, bins=50, alpha=0.7, color='green', edgecolor='black', label='Without outliers')
axes[0, 1].set_xlabel('Value', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].set_title(f'{feature_example} Distribution\n({outliers_mask.sum()} outliers, {(outliers_mask.sum()/len(data_full)*100):.1f}%)', 
                     fontsize=11, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Bottom left: Outlier impact on mean
top_impact = impact_df.nlargest(10, 'mean_change_pct')
axes[1, 0].barh(range(len(top_impact)), top_impact['mean_change_pct'], color='steelblue')
axes[1, 0].set_yticks(range(len(top_impact)))
axes[1, 0].set_yticklabels(top_impact['feature'], fontsize=9)
axes[1, 0].set_xlabel('Mean Change (%)', fontsize=10)
axes[1, 0].set_title('Top 10 Features - Mean Impact from Outliers', fontsize=11, fontweight='bold')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Bottom right: Outlier severity categories
severity_counts = {
    'High (>5%)': len(outlier_df[outlier_df['iqr_pct'] > 5]),
    'Medium (2-5%)': len(outlier_df[(outlier_df['iqr_pct'] >= 2) & (outlier_df['iqr_pct'] <= 5)]),
    'Low (<2%)': len(outlier_df[outlier_df['iqr_pct'] < 2])
}
colors_pie = ['#ff6b6b', '#ffd93d', '#6bcf7f']
axes[1, 1].pie(severity_counts.values(), labels=severity_counts.keys(), autopct='%1.1f%%',
               colors=colors_pie, startangle=90, textprops={'fontsize': 10})
axes[1, 1].set_title(f'Features by Outlier Severity\n(Total: {len(outlier_df)} features)', 
                     fontsize=11, fontweight='bold')

plt.tight_layout()
savefig(fig, 'outliers_overview.png')
print("Saved: outliers_overview.png")

# Plot 2: Time series examples with outliers highlighted
sample_ids = train_data['sample_index'].unique()[:4]
features_to_plot = ['joint_06', 'joint_07', 'pain_survey_1', 'pain_survey_2']  # High outlier features

fig, axes = plt.subplots(4, 4, figsize=(18, 14))

for row_idx, feature in enumerate(features_to_plot):
    for col_idx, sample_id in enumerate(sample_ids):
        ax = axes[row_idx, col_idx]
        sample_data = train_data[train_data['sample_index'] == sample_id].copy()
        sample_data = sample_data.sort_values('time')
        
        # Detect outliers using IQR method on this sample
        outliers = detect_iqr_outliers(sample_data[feature], k=1.5)
        
        # Plot time series
        ax.plot(sample_data['time'], sample_data[feature], 
                'b-', linewidth=1.5, alpha=0.7)
        
        # Highlight outliers
        if outliers.sum() > 0:
            outlier_times = sample_data.loc[outliers, 'time']
            outlier_values = sample_data.loc[outliers, feature]
            ax.scatter(outlier_times, outlier_values, 
                      color='red', s=30, zorder=5, alpha=0.8)
        
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        
        if col_idx == 0:
            ax.set_ylabel(f'{feature}\nValue', fontsize=9, fontweight='bold')
        
        if row_idx == 0:
            ax.set_title(f'Sample {sample_id}', fontsize=9, fontweight='bold')
        else:
            ax.set_title(f'{outliers.sum()} outliers', fontsize=8, color='red' if outliers.sum() > 0 else 'green')
        
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

plt.tight_layout()
savefig(fig, 'outliers_timeseries_detailed.png')
print("Saved: outliers_timeseries_detailed.png")

# Plot 3: Box plots for high-outlier features
high_outlier_features = outlier_df.nlargest(9, 'iqr_pct')['feature'].values

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, col in enumerate(high_outlier_features):
    data = train_data[col].dropna()
    bp = axes[idx].boxplot(data, vert=True, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2),
                           flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.3))
    
    outlier_pct = outlier_df[outlier_df["feature"]==col]["iqr_pct"].values[0]
    axes[idx].set_title(f'{col}\n{outlier_pct:.1f}% outliers', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Value', fontsize=9)
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].tick_params(labelsize=8)

plt.suptitle('Box Plots - Top 9 Features with Most Outliers', fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
savefig(fig, 'outliers_boxplots.png')
print("Saved: outliers_boxplots.png")