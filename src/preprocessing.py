import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def initial_preprocessing(data: pd.DataFrame, categorical_cols: list):
    
    # Create a copy of the dataset
    cleaned_df = data.copy()
    
    # Remove unecessary columns (this is to prevent dangerous bias and anonymisation)
    cleaned_df = cleaned_df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Apply one-hot encoding to categorical features
    cleaned_df = pd.get_dummies(cleaned_df, columns=categorical_cols, prefix=categorical_cols)
        
    return cleaned_df

def model_preprocessing(
    cleaned_data: pd.DataFrame,
    use_smote: bool = True,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """
    Create train/val/test splits without leakage. No scaling or resampling occurs here.
    """
    clean_data = cleaned_data.copy()
    
    # Drop complaints column to prevent data leakage
    if 'Complain' in clean_data.columns:
        clean_data = clean_data.drop('Complain', axis=1)
    
    # Split the dataset into X and y 
    y = clean_data['Exited']
    X = clean_data.drop('Exited', axis=1)
    feature_names = list(X.columns)
    
    # First split: train vs temp (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state, stratify=y
    )
    
    # Compute relative size of test within temp
    temp_test_ratio = test_size / (val_size + test_size)
    
    # Second split: val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=temp_test_ratio, random_state=(random_state + 1), stratify=y_temp
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test,


def build_training_pipeline(estimator, use_scaler: bool = True, use_smote: bool = True):
    """
    Build an imblearn Pipeline to ensure all preprocessing occurs inside fit only on training data.
    - StandardScaler is included by default (safe for most models; can be disabled for trees)
    - SMOTE is included by default for class imbalance; during inference it's ignored
    """
    steps = []
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    if use_smote:
        steps.append(('smote', SMOTE(random_state=42)))
    steps.append(('clf', estimator))
    return ImbPipeline(steps)
    
    
    
    
    
    
        
    
    
    
    
    