import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime

# Set working directory to project root
project_root = r"C:\Users\risha\steel_production_analysis"
os.chdir(project_root)

# Setup directories
os.makedirs('results', exist_ok=True)

# -------------------------
# 1. Remove duplicates
# -------------------------
def remove_duplicates(df):
    """Remove duplicate rows from dataset"""
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after = df.shape[0]
    duplicates_removed = before - after
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows")
    return df, duplicates_removed

# -------------------------
# 2. Handle missing values
# -------------------------
def handle_missing_values(df, strategy='median'):
    """
    Impute numeric columns using median (default) or mean.
    Leaves non-numeric untouched except optional simple filling.
    """
    missing_before = df.isnull().sum().sum()
    
    # Numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().sum() > 0:
            if strategy == 'median':
                val = df[col].median()
            else:
                val = df[col].mean()
            df[col] = df[col].fillna(val)
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    missing_after = df.isnull().sum().sum()
    if missing_before > 0:
        print(f"Filled {missing_before - missing_after} missing values")
    return df, missing_before

# -------------------------
# 3. Detect outliers (IQR method)
# -------------------------
def detect_outliers_iqr(df, columns=None, multiplier=1.5):
    """
    Mark outliers (True) using IQR method for given columns.
    Returns a boolean DataFrame same shape as df[columns]
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    is_outlier = pd.DataFrame(False, index=df.index, columns=columns)
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        is_outlier[col] = (df[col] < lower) | (df[col] > upper)
    
    return is_outlier

# -------------------------
# 4. Treat outliers (IQR method)
# -------------------------
def treat_outliers_iqr(df, columns=None, method='clip', multiplier=1.5):
    """
    Treat outliers using IQR method
    method: 'clip' - clip to bounds
            'median' - replace with median
            'drop' - remove outlier rows
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    before = df.shape[0]
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        
        if method == 'clip':
            df[col] = df[col].clip(lower, upper)
        elif method == 'median':
            median = df[col].median()
            df.loc[(df[col] < lower) | (df[col] > upper), col] = median
        elif method == 'drop':
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    after = df.shape[0]
    outliers_removed = before - after
    if outliers_removed > 0:
        print(f"Removed {outliers_removed} outlier rows")
    return df, outliers_removed

# -------------------------
# 5. Encode categorical variables
# -------------------------
def encode_categorical_variables(df, method='label'):
    """
    Encode categorical columns using LabelEncoder or OneHotEncoder
    method: 'label' or 'onehot'
    """
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(cat_cols) == 0:
        return df, {}
    
    le_dict = {}
    
    if method == 'label':
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
    elif method == 'onehot':
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df, le_dict

# -------------------------
# 6. Full preprocessing pipeline
# -------------------------
def preprocess_pipeline(df, name='Dataset', remove_dup=True, handle_missing=True, 
                       treat_outliers=True, encode_cat=True, outlier_method='clip'):
    """
    Complete preprocessing pipeline
    """
    print(f"Processing {name}...")
    
    stats = {
        'initial_shape': df.shape,
        'duplicates_removed': 0,
        'missing_values': 0,
        'outliers_handled': 0
    }
    
    if remove_dup:
        df, stats['duplicates_removed'] = remove_duplicates(df)
    
    if handle_missing:
        df, stats['missing_values'] = handle_missing_values(df, strategy='median')
    
    if treat_outliers:
        df, stats['outliers_handled'] = treat_outliers_iqr(df, method=outlier_method)
    
    if encode_cat:
        df, _ = encode_categorical_variables(df, method='label')
    
    stats['final_shape'] = df.shape
    
    return df, stats

# -------------------------
# 7. Data splitting and normalization
# -------------------------
def split_and_normalize_data(df, target_column, val_size=0.1, test_size=0.1, random_state=42):
    """
    Splits into train/val/test and standardizes features.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    print("\n📐 Splitting and normalizing data...")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # First split: test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: validation
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, random_state=random_state
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), columns=X_val.columns, index=X_val.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

# -------------------------
# 8. Main execution
# -------------------------
if __name__ == "__main__":
    try:
        print("Starting data preprocessing...\n")
        
        # Load datasets
        print("Loading datasets...")
        df_train = pd.read_csv("data/raw/normalized_train_data.csv", engine="python")
        df_test = pd.read_csv("data/raw/normalized_test_data.csv", engine="python")
        print(f"Train: {df_train.shape}")
        print(f"Test: {df_test.shape}\n")
        
        # Preprocess training data
        df_train_clean, train_stats = preprocess_pipeline(
            df_train, 
            name='Training Data',
            remove_dup=True, 
            handle_missing=True,
            treat_outliers=True,
            encode_cat=True,
            outlier_method='clip'
        )
        
        # Preprocess test data
        df_test_clean, test_stats = preprocess_pipeline(
            df_test,
            name='Test Data',
            remove_dup=True,
            handle_missing=True,
            treat_outliers=True,
            encode_cat=True,
            outlier_method='clip'
        )
        
        # Save cleaned datasets
        print("Saving cleaned datasets...")
        df_train_clean.to_csv("results/clean_train.csv", index=False)
        df_test_clean.to_csv("results/clean_test.csv", index=False)
        print(f"Saved clean_train.csv ({df_train_clean.shape})")
        print(f"Saved clean_test.csv ({df_test_clean.shape})\n")
        
        # Save preprocessing statistics
        preprocessing_stats = {
            'timestamp': datetime.now().isoformat(),
            'training_data': {
                'initial_shape': list(train_stats['initial_shape']),
                'final_shape': list(train_stats['final_shape']),
                'duplicates_removed': int(train_stats['duplicates_removed']),
                'missing_values': int(train_stats['missing_values']),
                'outliers_handled': int(train_stats['outliers_handled'])
            },
            'test_data': {
                'initial_shape': list(test_stats['initial_shape']),
                'final_shape': list(test_stats['final_shape']),
                'duplicates_removed': int(test_stats['duplicates_removed']),
                'missing_values': int(test_stats['missing_values']),
                'outliers_handled': int(test_stats['outliers_handled'])
            },
            'outlier_method': 'clip',
            'missing_value_strategy': 'median'
        }
        
        with open('results/preprocessing_statistics.json', 'w') as f:
            json.dump(preprocessing_stats, f, indent=4)
        print(f"Saved preprocessing_statistics.json")
        
        print("\nPreprocessing complete!")
        print(f"\nTraining: {train_stats['initial_shape']} -> {train_stats['final_shape']}")
        print(f"  Duplicates removed: {train_stats['duplicates_removed']}")
        print(f"  Missing values filled: {train_stats['missing_values']}")
        print(f"  Outliers handled: {train_stats['outliers_handled']}")
        print(f"\nTest: {test_stats['initial_shape']} -> {test_stats['final_shape']}")
        print(f"  Duplicates removed: {test_stats['duplicates_removed']}")
        print(f"  Missing values filled: {test_stats['missing_values']}")
        print(f"  Outliers handled: {test_stats['outliers_handled']}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the following files exist:")
        print("  - data/raw/normalized_train_data.csv")
        print("  - data/raw/normalized_test_data.csv")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()