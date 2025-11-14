# ===============
# Setup
# ===============

import os

import random
import numpy as np
import pandas as pd
from pathlib import Path

# ===============
# Configuration
# ===============

# Dataset: 661 samples, ~160 timesteps/sample, 35 raw features

# Random Seed
SEED = 42

# File Paths
ROOT = Path('.')
TRAIN_CSV = ROOT / 'data' / 'pirate_pain_train.csv'
LABEL_CSV = ROOT / 'data' / 'pirate_pain_train_labels.csv'
TEST_CSV = ROOT / 'data' / 'pirate_pain_test.csv'
OUT_DIR = ROOT / 'out'

# Column Definitions
pain_survey_cols = ['pain_survey_1','pain_survey_2','pain_survey_3','pain_survey_4']
static_cat_cols = ['n_legs','n_hands','n_eyes']
joint_cols = [f'joint_{i:02d}' for i in range(31)]
time_feature_cols = pain_survey_cols + joint_cols
id_cols = ['sample_index','time']
label_column_name = 'label'

# Label Mapping
LABEL_MAP = {'no_pain':0, 'low_pain':1, 'high_pain':2}

# Temporal feature configuration

TEMPORAL_FEATURES_CONFIG = {
    'diff': False,                # Rate of change (velocity) - captures dynamics
    'rolling_mean': False,        # Smoothed trend - reduces noise
    'rolling_std': False,         # Local volatility - captures variation patterns
    'ewm_mean': True,           # Exponentially weighted mean - emphasizes recent values
}

# Temporal Feature Parameters
ROLLING_WINDOW = 5              # Window size for rolling statistics
ROLLING_MIN_PERIODS = 1         # Minimum periods for rolling calculations
EWM_SPAN = 5                    # Span for exponential weighted mean

# Outlier Handling
ENABLE_OUTLIER_HANDLING = True  # Enable/disable outlier preprocessing
OUTLIER_CLIP_LOWER_PCT = 1.0    # Lower percentile for clipping (per sample)
OUTLIER_CLIP_UPPER_PCT = 99.0   # Upper percentile for clipping (per sample)
OUTLIER_LOG_TRANSFORM = True    # Apply log transform to near-zero skewed features

# Normalization
SCALING_METHOD = 'minmax'       # Options: 'robust' (median+IQR) or 'minmax'

# Duplicate Column Detection
CORRELATION_THRESHOLD = 0.9999999999999999999  # Threshold for detecting duplicate columns
DUPLICATE_TOLERANCE_ABS = 1e-40  # Absolute tolerance for np.allclose
DUPLICATE_TOLERANCE_REL = 1e-40  # Relative tolerance for np.allclose

# Initialize
random.seed(SEED)
np.random.seed(SEED)
OUT_DIR.mkdir(exist_ok=True)


# ===============
# Data Loading
# ===============

# Verify data directory and expected CSV files exist and give helpful messages
DATA_DIR = ROOT / 'data'
if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Data directory not found: {DATA_DIR}\n"
        "Place your CSV files in this folder or update the paths at the top of the script.\n"
        "Expected files: pirate_pain_train.csv, pirate_pain_train_labels.csv, pirate_pain_test.csv"
    )

missing_files = [p for p in (TRAIN_CSV, LABEL_CSV, TEST_CSV) if not p.exists()]
if missing_files:
    missing_names = ', '.join(p.name for p in missing_files)
    raise FileNotFoundError(
        f"Missing data files in {DATA_DIR}: {missing_names}.\n"
        "Make sure these files are present or adjust the CSV path variables at the top of the script."
    )

# Load CSVs with clearer errors
try:
    X_train_df = pd.read_csv(TRAIN_CSV)
except Exception as e:
    raise RuntimeError(f"Failed to read training CSV '{TRAIN_CSV}': {e}")

try:
    X_test_df = pd.read_csv(TEST_CSV)
except Exception as e:
    raise RuntimeError(f"Failed to read test CSV '{TEST_CSV}': {e}")

try:
    y_df = pd.read_csv(LABEL_CSV)
except Exception as e:
    raise RuntimeError(f"Failed to read labels CSV '{LABEL_CSV}': {e}")

# numeric label mapping if labels are strings
if y_df[label_column_name].dtype == object:
    y_df[label_column_name] = y_df[label_column_name].map(LABEL_MAP)
else:
    y_df[label_column_name] = y_df[label_column_name].astype(int)

print('Train shape:', X_train_df.shape)
print('Test shape:', X_test_df.shape)
print('Labels shape:', y_df.shape)



# ===============
# Preprocessing
# ===============

# --- Static Categorical Encoding ---

def encode_static_binary(df, static_cols):
    df = df.copy()
    encoding_info = {}
    for col in static_cols:
        unique_vals = sorted(df[col].unique().tolist())
        if len(unique_vals) != 2:
            mapping = {v:i for i,v in enumerate(unique_vals)}
            print(f"Warning: {col} has {len(unique_vals)} unique values, mapping -> {mapping}")
        else:
            mapping = {unique_vals[0]:0, unique_vals[1]:1}
        df[col+'_enc'] = df[col].map(mapping).astype(int)
        encoding_info[col] = mapping
    return df, encoding_info

print("\n--- Encoding Static Categorical Features ---")

# apply encoding on TRAIN
sample_static = X_train_df.groupby('sample_index')[static_cat_cols].first().reset_index()
sample_static_encoded, encoding_info = encode_static_binary(sample_static, static_cat_cols)

# Merge encoded static features back into the time-series dataframe
X_train_df = X_train_df.merge(sample_static_encoded[['sample_index'] + [c+'_enc' for c in static_cat_cols]],
            on='sample_index', how='left')

# Drop original static columns
X_train_df = X_train_df.drop(columns=static_cat_cols)

# Apply same encoding to TEST using learned mappings
sample_static_test = X_test_df.groupby('sample_index')[static_cat_cols].first().reset_index()
for col in static_cat_cols:
    sample_static_test[col+'_enc'] = sample_static_test[col].map(encoding_info[col]).astype(int)

X_test_df = X_test_df.merge(sample_static_test[['sample_index'] + [c+'_enc' for c in static_cat_cols]],
            on='sample_index', how='left')
X_test_df = X_test_df.drop(columns=static_cat_cols)

static_encoded_cols = [c+'_enc' for c in static_cat_cols]
time_feature_cols += static_encoded_cols

print('Encoded static columns:', static_encoded_cols)

# --- Temporal Feature Engineering ---

def create_temporal_features(df, feature_cols, sample_col='sample_index', config=None):
    """
    Create temporal features for time-series classification.
    All features computed within each sample to prevent data leakage.
    """
    if config is None:
        config = TEMPORAL_FEATURES_CONFIG
    
    df = df.sort_values([sample_col, 'time']).reset_index(drop=True)
    new_features = []
    
    print(f"Creating temporal features for {len(feature_cols)} base features...")
    
    # Rate of change (captures dynamics and trends)
    if config.get('diff', False):
        print("Computing rate of change (diff)")
        for col in feature_cols:
            diff_col = f'{col}_diff'
            df[diff_col] = df.groupby(sample_col)[col].diff().fillna(0)
            new_features.append(diff_col)
    
    # Rolling mean (smooths noise, captures local trends)
    if config.get('rolling_mean', False):
        print(f"Computing rolling mean (window={ROLLING_WINDOW})")
        for col in feature_cols:
            roll_col = f'{col}_rollmean{ROLLING_WINDOW}'
            df[roll_col] = df.groupby(sample_col)[col].transform(
                lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).mean()
            )
            new_features.append(roll_col)
    
    # Rolling standard deviation (captures local volatility/variation)
    if config.get('rolling_std', False):
        print(f"Computing rolling std (window={ROLLING_WINDOW})")
        for col in feature_cols:
            roll_col = f'{col}_rollstd{ROLLING_WINDOW}'
            df[roll_col] = df.groupby(sample_col)[col].transform(
                lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).std()
            ).fillna(0)
            new_features.append(roll_col)
    
    # Exponentially weighted mean (emphasizes recent observations)
    if config.get('ewm_mean', False):
        print(f"Computing exponentially weighted mean (span={EWM_SPAN})")
        for col in feature_cols:
            ewm_col = f'{col}_ewm{EWM_SPAN}'
            df[ewm_col] = df.groupby(sample_col)[col].transform(
                lambda x: x.ewm(span=EWM_SPAN, adjust=False).mean()
            )
            new_features.append(ewm_col)
    
    enabled_count = sum(1 for v in config.values() if v)
    print(f"Created {len(new_features)} temporal features ({enabled_count} feature types enabled)\n")
    
    return df, new_features

print("\n--- Creating Temporal Features ---")

# Only apply to non-static features
temporal_base_cols = pain_survey_cols + joint_cols
X_train_df, train_temporal_features = create_temporal_features(
    X_train_df, temporal_base_cols, config=TEMPORAL_FEATURES_CONFIG
)
X_test_df, test_temporal_features = create_temporal_features(
    X_test_df, temporal_base_cols, config=TEMPORAL_FEATURES_CONFIG
)

# Add temporal features to the feature list
time_feature_cols += train_temporal_features

# --- Dropping Useless Features ---

print("\n--- Dropping Useless Features ---")

# Drop constant (zero-variance) columns across entire training dataframe
const_cols = []
for c in time_feature_cols:
    if X_train_df[c].nunique(dropna=False) <= 1:
        const_cols.append(c)
        print("Found constant column to drop:", c)

if const_cols:
    X_train_df = X_train_df.drop(columns=const_cols)
    X_test_df = X_test_df.drop(columns=const_cols)
    time_feature_cols = [c for c in time_feature_cols if c not in const_cols]

# Detect exact-duplicate columns (value-wise equality) and drop duplicates
print("Checking for duplicate columns...")
cols_to_drop = []
cols_to_check = time_feature_cols.copy()

# Build correlation matrix once
corr_matrix = X_train_df[time_feature_cols].corr().abs()

# Find perfectly correlated pairs (correlation = 1.0)
checked = set()
for i, col_i in enumerate(time_feature_cols):
    if col_i in cols_to_drop or col_i in checked:
        continue
    for j, col_j in enumerate(time_feature_cols[i+1:], start=i+1):
        if col_j in cols_to_drop or col_j in checked:
            continue
        # Check if perfectly correlated
        if corr_matrix.loc[col_i, col_j] > CORRELATION_THRESHOLD:
            # Verify with exact comparison
            if np.allclose(X_train_df[col_i].values, X_train_df[col_j].values, 
                          atol=DUPLICATE_TOLERANCE_ABS, rtol=DUPLICATE_TOLERANCE_REL):
                cols_to_drop.append(col_j)
                print(f"Found duplicate column to drop: {col_j} (duplicate of {col_i})")
    checked.add(col_i)

if cols_to_drop:
    X_train_df = X_train_df.drop(columns=cols_to_drop)
    X_test_df = X_test_df.drop(columns=cols_to_drop)
    time_feature_cols = [c for c in time_feature_cols if c not in cols_to_drop]

# --- Outlier Handling ---

if ENABLE_OUTLIER_HANDLING:
    print("\n--- Handling Outliers ---")
    print("It can take a while...")
    
    def handle_outliers_per_sample(df, feature_cols, static_cols, discrete_cols, 
                                    lower_pct=1.0, upper_pct=99.0, 
                                    apply_log_transform=True, sample_col='sample_index'):
        """
        Handle outliers in time series data with sample-aware processing.
        
        Strategy:
        - Static/discrete features: Skip (no modification)
        - Near-zero features (median < 0.01): Apply log1p transform for skewed distributions
        - Regular features: Clip to percentiles within each sample (prevents data leakage)
        
        Args:
            df: DataFrame to process
            feature_cols: All feature columns
            static_cols: Static encoded columns to skip
            discrete_cols: Discrete columns (pain surveys) to skip
            lower_pct: Lower percentile for clipping
            upper_pct: Upper percentile for clipping
            apply_log_transform: Whether to log-transform near-zero features
            sample_col: Column name for sample identifier
        
        Returns:
            df: Processed DataFrame
            transform_info: Dictionary with transformation details
        """
        df = df.copy()
        transform_info = {'clipped': [], 'log_transformed': [], 'skipped': []}
        
        # Identify features to skip (static + discrete)
        skip_features = set(static_cols + discrete_cols)
        
        for col in feature_cols:
            if col in skip_features:
                transform_info['skipped'].append(col)
                continue
            
            # Check if feature is near-zero (skewed distribution)
            median_val = df[col].median()
            
            if apply_log_transform and median_val < 0.01:
                # Log transform for near-zero skewed features
                df[col] = np.log1p(df[col])  # log(1+x) handles zeros
                transform_info['log_transformed'].append(col)
            else:
                # Clip outliers within each sample (time-series aware)
                # This prevents data leakage across samples
                df[col] = df.groupby(sample_col)[col].transform(
                    lambda x: x.clip(
                        lower=x.quantile(lower_pct/100), 
                        upper=x.quantile(upper_pct/100)
                    )
                )
                transform_info['clipped'].append(col)
        
        return df, transform_info
    
    # Apply to training data
    X_train_df, train_transform_info = handle_outliers_per_sample(
        X_train_df, 
        time_feature_cols,
        static_encoded_cols,
        pain_survey_cols,
        lower_pct=OUTLIER_CLIP_LOWER_PCT,
        upper_pct=OUTLIER_CLIP_UPPER_PCT,
        apply_log_transform=OUTLIER_LOG_TRANSFORM
    )
    
    # Apply same transformations to test data
    X_test_df_processed = X_test_df.copy()
    
    # Log transform same features as train
    for col in train_transform_info['log_transformed']:
        X_test_df_processed[col] = np.log1p(X_test_df_processed[col])
    
    # Clip same features as train (per sample)
    for col in train_transform_info['clipped']:
        X_test_df_processed[col] = X_test_df_processed.groupby('sample_index')[col].transform(
            lambda x: x.clip(
                lower=x.quantile(OUTLIER_CLIP_LOWER_PCT/100),
                upper=x.quantile(OUTLIER_CLIP_UPPER_PCT/100)
            )
        )
    
    X_test_df = X_test_df_processed
    
    print(f"Outlier handling complete:")
    print(f"  - Skipped (static/discrete): {len(train_transform_info['skipped'])} features")
    print(f"  - Log-transformed (near-zero): {len(train_transform_info['log_transformed'])} features")
    print(f"  - Clipped (per-sample, {OUTLIER_CLIP_LOWER_PCT}-{OUTLIER_CLIP_UPPER_PCT}th percentile): {len(train_transform_info['clipped'])} features")
    
    if train_transform_info['log_transformed']:
        print(f"\n  Log-transformed features: {', '.join(train_transform_info['log_transformed'][:5])}{'...' if len(train_transform_info['log_transformed']) > 5 else ''}")

else:
    print("\n--- Outlier Handling: DISABLED ---")


# --- Normalization ---

print(f"\n--- Normalizing Time-Series Features ({SCALING_METHOD.upper()} Scaling) ---")

# Exclude static encoded columns from normalization (keep them as binary 0/1)
temporal_only_cols = [c for c in time_feature_cols if c not in static_encoded_cols]
print(f"Normalizing {len(temporal_only_cols)} temporal features (excluding {len(static_encoded_cols)} static binary features)")

if SCALING_METHOD == 'robust':
    # Robust Scaling: Uses Median + IQR (better for outliers)
    print("Using Robust Scaler (Median + IQR) - better for data with outliers")
    
    # Compute median and IQR from TRAIN data (after capping)
    medians = X_train_df[temporal_only_cols].median(axis=0)
    Q1 = X_train_df[temporal_only_cols].quantile(0.25, axis=0)
    Q3 = X_train_df[temporal_only_cols].quantile(0.75, axis=0)
    IQR = Q3 - Q1
    
    # Avoid division by zero
    IQR[IQR == 0] = 1
    
    # Transform Training set: (X - median) / IQR
    X_train_df[temporal_only_cols] = (X_train_df[temporal_only_cols] - medians) / IQR
    
    # Transform Test set using TRAIN statistics
    X_test_df[temporal_only_cols] = (X_test_df[temporal_only_cols] - medians) / IQR
    
    print('Robust scaling complete. Temporal features centered at 0 with IQR-based scaling.')
    print(f'Static features ({", ".join(static_encoded_cols)}) kept as binary (0/1) values.')

elif SCALING_METHOD == 'minmax':
    # Min-Max Scaling: Scales to [0, 1] range
    print("Using Min-Max Scaler - scales features to [0, 1] range")
    
    # Compute min and max from TRAIN data
    min_vals = X_train_df[temporal_only_cols].min(axis=0)
    data_range = X_train_df[temporal_only_cols].max(axis=0) - min_vals
    
    # Avoid division by zero
    data_range[data_range == 0] = 1
    
    # Transform Training set
    X_train_df[temporal_only_cols] = (X_train_df[temporal_only_cols] - min_vals) / data_range
    
    # Transform Test set using TRAIN statistics
    X_test_df[temporal_only_cols] = (X_test_df[temporal_only_cols] - min_vals) / data_range
    
    print('Min-Max scaling complete. Temporal features scaled to [0, 1] range.')
    print(f'Static features ({", ".join(static_encoded_cols)}) kept as binary (0/1) values.')

else:
    raise ValueError(f"Unknown SCALING_METHOD: {SCALING_METHOD}. Use 'robust' or 'minmax'.")

print('Scaling complete. Temporal features normalized, static features preserved as binary.')


# --- Final Processed Data ---

print("\n\n=== PROCESSED TRAINING DATA ===")

# Shape and Feature Count
print("\n" + "-"*35)
print(f"- Total Records (Rows): {X_train_df.shape[0]}")
print(f"- Total Features (Columns): {X_train_df.shape[1]}")

# Detailed Structure and Missing Values
print("\n" + "-"*35)
print("- Data Structure:")
X_train_df.info(verbose=False, memory_usage='deep')

# Descriptive Statistics for Numerical Features
print("\n" + "-"*35)
print("Descriptive Statistics:")
print(X_train_df.describe().T)

# A Peek at the Data
print("\n" + "-"*35)
print("First 5 Records (Head):")
print(X_train_df.head())

print("\n=== PROCESSED TEST DATA ===")

# Shape and Feature Count
print("\n" + "-"*35)
print(f"- Total Records (Rows): {X_test_df.shape[0]}")
print(f"- Total Features (Columns): {X_test_df.shape[1]}")

# Detailed Structure and Missing Values
print("\n" + "-"*35)
print("- Data Structure:")
X_test_df.info(verbose=False, memory_usage='deep')

# Descriptive Statistics for Numerical Features
print("\n" + "-"*35)
print("Descriptive Statistics:")
print(X_test_df.describe().T)

# A Peek at the Data
print("\n" + "-"*35)
print("First 5 Records (Head):")
print(X_test_df.head())


# ===============
# Save Data
# ===============

print("\n\n=== SAVING PROCESSED DATA ===")

# Define output file paths
PROCESSED_TRAIN_CSV = OUT_DIR / 'preprocessed_train.csv'
PROCESSED_TEST_CSV = OUT_DIR / 'preprocessed_test.csv'
PROCESSED_LABELS_CSV = OUT_DIR / 'preprocessed_labels.csv'

# Save training data
X_train_df.to_csv(PROCESSED_TRAIN_CSV, index=False)
print(f"Saved training data: {PROCESSED_TRAIN_CSV}")
print(f"   Shape: {X_train_df.shape}")

# Save test data
X_test_df.to_csv(PROCESSED_TEST_CSV, index=False)
print(f"Saved test data: {PROCESSED_TEST_CSV}")
print(f"   Shape: {X_test_df.shape}")

# Save labels
y_df.to_csv(PROCESSED_LABELS_CSV, index=False)
print(f"Saved labels: {PROCESSED_LABELS_CSV}")
print(f"   Shape: {y_df.shape}")

print(f"\nAll preprocessed files saved to: {OUT_DIR.absolute()}")
print("\n" + "="*50)
print("PREPROCESSING COMPLETE!")
print("="*50)