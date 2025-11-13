import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

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

def savefig(fig, name):
    fig.savefig(PLOT_DIR / name, bbox_inches='tight', dpi=100)
    plt.close(fig)

print("="*80)
print("TEMPORAL FEATURE ANALYSIS")
print("="*80)

def create_temporal_features(df, feature_cols, window=5):
    df = df.sort_values(['sample_index', 'time']).copy()
    temporal_map = {}
    
    for col in feature_cols:
        df[f'{col}_diff'] = df.groupby('sample_index')[col].diff().fillna(0)
        temporal_map[f'{col}_diff'] = 'diff'
        
        df[f'{col}_rollmean'] = df.groupby('sample_index')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        temporal_map[f'{col}_rollmean'] = 'rolling_mean'
        
        df[f'{col}_rollstd'] = df.groupby('sample_index')[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        ).fillna(0)
        temporal_map[f'{col}_rollstd'] = 'rolling_std'
        
        df[f'{col}_ewm'] = df.groupby('sample_index')[col].transform(
            lambda x: x.ewm(span=window, adjust=False).mean()
        )
        temporal_map[f'{col}_ewm'] = 'ewm_mean'
    
    return df, temporal_map

analysis_features = pain_survey_cols + joint_cols[:5]
train_with_temporal, temporal_map = create_temporal_features(train_data, analysis_features)

def compute_anova_scores(df, feature_cols, label_col='label'):
    results = []
    for col in feature_cols:
        groups = [df[df[label_col] == label][col].dropna() for label in df[label_col].unique()]
        if any(len(g) == 0 for g in groups):
            continue
        f_stat, p_value = stats.f_oneway(*groups)
        results.append({
            'feature': col,
            'f_statistic': f_stat,
            'p_value': p_value,
            'feature_type': temporal_map.get(col, 'base')
        })
    return pd.DataFrame(results).sort_values('f_statistic', ascending=False)

base_importance = compute_anova_scores(train_with_temporal, analysis_features)
base_importance['feature_type'] = 'base'
temporal_importance = compute_anova_scores(train_with_temporal, list(temporal_map.keys()))
all_importance = pd.concat([base_importance, temporal_importance], ignore_index=True)
all_importance = all_importance.sort_values('f_statistic', ascending=False)

print("\nTop 10 Features:")
print(all_importance.head(10)[['feature', 'feature_type', 'f_statistic']])

print("\nMean F-statistic by Type:")
print(all_importance.groupby('feature_type')['f_statistic'].mean().sort_values(ascending=False))

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS...")
print("="*80)

# Plot 1: Comprehensive temporal feature overview
fig = plt.figure(figsize=(16, 10), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)

# Top left: Top 25 features with color coding
ax1 = fig.add_subplot(gs[0, :])
top_25 = all_importance.head(25).copy()
colors = top_25['feature_type'].map({
    'base': '#7f8c8d', 'diff': '#e74c3c', 'rolling_mean': '#3498db',
    'rolling_std': '#2ecc71', 'ewm_mean': '#9b59b6'
})
bars = ax1.barh(range(len(top_25)), top_25['f_statistic'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(top_25)))
ax1.set_yticklabels([f[:35] for f in top_25['feature']], fontsize=9)
ax1.set_xlabel('ANOVA F-Statistic (Discriminative Power)', fontsize=10, fontweight='bold')
ax1.set_title('Top 25 Most Discriminative Features', fontsize=12, fontweight='bold', pad=10)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3, linestyle='--')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#7f8c8d', alpha=0.8, label='Base Features', edgecolor='black'),
    Patch(facecolor='#e74c3c', alpha=0.8, label='Diff (Rate of Change)', edgecolor='black'),
    Patch(facecolor='#3498db', alpha=0.8, label='Rolling Mean', edgecolor='black'),
    Patch(facecolor='#2ecc71', alpha=0.8, label='Rolling Std', edgecolor='black'),
    Patch(facecolor='#9b59b6', alpha=0.8, label='EWM (Exp. Weighted)', edgecolor='black')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.95)

# Middle left: Box plot by feature type
ax2 = fig.add_subplot(gs[1, 0])
feature_type_order = ['base', 'diff', 'rolling_mean', 'rolling_std', 'ewm_mean']
bp = sns.boxplot(data=all_importance, x='feature_type', y='f_statistic', 
                 order=feature_type_order, ax=ax2, hue='feature_type', palette='Set2', legend=False)
ax2.set_xlabel('Feature Type', fontsize=10, fontweight='bold')
ax2.set_ylabel('F-Statistic Distribution', fontsize=10)
ax2.set_title('Discriminative Power Distribution', fontsize=11, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Middle right: Average discriminative power
ax3 = fig.add_subplot(gs[1, 1])
mean_f = all_importance.groupby('feature_type')['f_statistic'].mean().sort_values(ascending=False)
bars = mean_f.plot(kind='bar', ax=ax3, color=['#3498db', '#9b59b6', '#7f8c8d', '#2ecc71', '#e74c3c'], 
                   alpha=0.8, edgecolor='black', linewidth=1)
ax3.set_xlabel('Feature Type', fontsize=10, fontweight='bold')
ax3.set_ylabel('Mean F-Statistic', fontsize=10)
ax3.set_title('Average Discriminative Power', fontsize=11, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (idx, val) in enumerate(mean_f.items()):
    ax3.text(i, val + max(mean_f) * 0.02, f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Bottom: Feature type count and percentage
ax4 = fig.add_subplot(gs[2, :])
type_counts = all_importance['feature_type'].value_counts()
base_mean = all_importance[all_importance['feature_type'] == 'base']['f_statistic'].mean()
temporal_stats = all_importance[all_importance['feature_type'] != 'base'].groupby('feature_type')['f_statistic'].mean()

improvement_data = []
for feat_type in ['diff', 'rolling_mean', 'rolling_std', 'ewm_mean']:
    if feat_type in temporal_stats.index:
        mean_val = temporal_stats[feat_type]
        improvement = ((mean_val - base_mean) / base_mean * 100)
        improvement_data.append({'type': feat_type, 'improvement': improvement, 'mean': mean_val})

improvement_df = pd.DataFrame(improvement_data).sort_values('improvement', ascending=False)
colors_imp = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvement_df['improvement']]
bars = ax4.barh(range(len(improvement_df)), improvement_df['improvement'], color=colors_imp, alpha=0.8, edgecolor='black', linewidth=0.5)
ax4.set_yticks(range(len(improvement_df)))
ax4.set_yticklabels(improvement_df['type'], fontsize=10)
ax4.set_xlabel('Improvement over Base Features (%)', fontsize=10, fontweight='bold')
ax4.set_title('Temporal Feature Engineering Impact', fontsize=11, fontweight='bold')
ax4.axvline(0, color='black', linewidth=2, linestyle='-', alpha=0.5)
ax4.grid(axis='x', alpha=0.3, linestyle='--')
ax4.invert_yaxis()

# Add value labels
for i, (idx, row) in enumerate(improvement_df.iterrows()):
    ax4.text(row['improvement'] + (5 if row['improvement'] > 0 else -5), i, 
             f"{row['improvement']:+.1f}% (F={row['mean']:.1f})", 
             ha='left' if row['improvement'] > 0 else 'right', va='center', fontsize=9, fontweight='bold')

savefig(fig, 'temporal_features_comprehensive.png')
print("Saved: temporal_features_comprehensive.png")

# Plot 2: Temporal transformation examples - comparing different features
sample_per_class = {}
for label in sorted(train_with_temporal['label'].unique()):
    sample_idx = train_with_temporal[train_with_temporal['label'] == label]['sample_index'].iloc[0]
    sample_per_class[label] = sample_idx

features_to_show = ['pain_survey_1', 'joint_00']
fig, axes = plt.subplots(len(features_to_show) * 3, 1, figsize=(14, 4 * len(features_to_show) * 3), sharex=False)

label_names = {'no_pain': 'No Pain', 'low_pain': 'Low Pain', 'high_pain': 'High Pain'}

for feat_idx, feature in enumerate(features_to_show):
    for class_idx, (label, sample_idx) in enumerate(sample_per_class.items()):
        ax_idx = feat_idx * 3 + class_idx
        ax = axes[ax_idx]
        
        sample = train_with_temporal[train_with_temporal['sample_index'] == sample_idx].sort_values('time')
        
        # Plot original and temporal features
        ax.plot(sample['time'], sample[feature], 'o-', label='Original', linewidth=2.5, markersize=4, color='#34495e', alpha=0.8)
        ax.plot(sample['time'], sample[f'{feature}_diff'], 's-', label='Diff', linewidth=1.8, markersize=3, color='#e74c3c', alpha=0.7)
        ax.plot(sample['time'], sample[f'{feature}_rollmean'], '^-', label='Rolling Mean (W=5)', linewidth=1.8, markersize=3, color='#3498db', alpha=0.7)
        ax.plot(sample['time'], sample[f'{feature}_ewm'], 'D-', label='EWM', linewidth=1.8, markersize=3, color='#9b59b6', alpha=0.7)
        
        ax.set_ylabel('Value', fontsize=9, fontweight='bold')
        ax.set_title(f'{feature} | {label_names.get(label, label)} (Sample {sample_idx})', fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=8, framealpha=0.95, ncol=4)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        if ax_idx == len(axes) - 1:
            ax.set_xlabel('Time', fontsize=10, fontweight='bold')

plt.suptitle('Temporal Feature Transformations by Pain Class', fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout()
savefig(fig, 'temporal_transformations_detailed.png')
print("Saved: temporal_transformations_detailed.png")

print("\n" + "="*80)
print("RECOMMENDATIONS FOR TEMPORAL FEATURE ENGINEERING")
print("="*80)

temporal_stats = all_importance[all_importance['feature_type'] != 'base'].groupby('feature_type')['f_statistic'].mean()
base_mean = all_importance[all_importance['feature_type'] == 'base']['f_statistic'].mean()

print(f"\nBase features average F-statistic: {base_mean:.2f}")
print("\nTemporal feature types ranked by discriminative power:\n")

for feat_type, mean_val in temporal_stats.sort_values(ascending=False).items():
    improvement = ((mean_val - base_mean) / base_mean * 100)
    status = "ENABLE" if improvement > 0 else "SKIP"
    symbol = "↑" if improvement > 0 else "↓"
    print(f"{status:10s} {feat_type:15s}: F={mean_val:7.2f}  {symbol} {improvement:+6.1f}% vs base")