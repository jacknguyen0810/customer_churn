import pandas as pd
import numpy as np
import time
import warnings
import os
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from joblib import dump, load
from src.preprocessing import build_training_pipeline

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def save_model(model, model_name, threshold, metrics):
    """Save trained model using joblib"""
    # Create models directory if it doesn't exist
    models_dir = 'data/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}_model.joblib')
    dump(model, model_path)
    
    # Save metadata (threshold and metrics)
    metadata_path = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}_metadata.joblib')
    metadata = {
        'threshold': threshold,
        'metrics': metrics,
        'model_name': model_name,
        'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    dump(metadata, metadata_path)
    
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    return model_path, metadata_path

def load_model(model_name):
    """Load saved model and metadata"""
    models_dir = 'data/models'
    model_path = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}_model.joblib')
    metadata_path = os.path.join(models_dir, f'{model_name.lower().replace(" ", "_")}_metadata.joblib')
    
    if os.path.exists(model_path) and os.path.exists(metadata_path):
        model = load(model_path)
        metadata = load(metadata_path)
        print(f"Model loaded: {model_path}")
        return model, metadata
    else:
        print(f"Model not found: {model_path}")
        return None, None

def find_best_threshold(model, X_val, y_val):
    """Find optimal threshold for maximizing recall"""
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
        
        # Find threshold that maximizes recall while maintaining reasonable precision
        valid_indices = recall[:-1] >= 0.7
        if np.any(valid_indices):
            valid_precision = precision[:-1][valid_indices]
            best_idx = np.argmax(valid_precision)
            best_threshold = thresholds[valid_indices][best_idx]
        else:
            recall_values = recall[:-1]
            best_idx = np.argmax(recall_values)
            best_threshold = thresholds[best_idx] if len(thresholds) > best_idx else 0.5
    else:
        best_threshold = 0.5
        
    return best_threshold

def plot_confusion_matrix(y_true, y_pred, model_name, threshold=None):
    """Plot confusion matrix for a single model"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['No Churn', 'Churn'], 
               yticklabels=['No Churn', 'Churn'])
    
    title = f'{model_name} Confusion Matrix'
    if threshold:
        title += f' (Threshold: {threshold:.3f})'
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_feature_importance(model, model_name, feature_names, top_n=10):
    """Plot feature importance for models that support it"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Ensure we don't exceed the number of available features
        actual_top_n = min(top_n, len(importances))
        indices = np.argsort(importances)[::-1][:actual_top_n]
        
        # Ensure indices are within bounds of feature_names
        valid_indices = [idx for idx in indices if idx < len(feature_names)]
        
        if not valid_indices:
            print(f"❌ {model_name}: No valid feature indices found")
            return
            
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(valid_indices)), importances[valid_indices])
        plt.yticks(range(len(valid_indices)), [feature_names[i] for i in valid_indices])
        plt.title(f'{model_name} - Top {len(valid_indices)} Most Important Features')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        # Print feature importance
        print(f"\n{model_name} - Top {len(valid_indices)} Most Important Features:")
        print("-" * 50)
        for i, idx in enumerate(valid_indices):
            print(f"{i+1:2d}. {feature_names[idx]:25s} {importances[idx]:.4f}")
    else:
        print(f"\n{model_name} does not support feature importance analysis.")

def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test, class_weights=None, feature_names=None, n_trials=20):
    """Train and optimize Random Forest with Optuna CV and pipeline."""
    print("🌲 Training Random Forest (Optuna CV)...")
    print("=" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1,
        }

        estimator = RandomForestClassifier(**params)
        pipeline = build_training_pipeline(estimator, use_scaler=True, use_smote=True)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    start_time = time.time()
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    train_time = time.time() - start_time

    print(f"\n Best Parameters Found:")
    print("-" * 40)
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    print(f"Best CV ROC AUC: {study.best_value:.4f}")

    best_estimator = RandomForestClassifier(**{**study.best_params})
    best_pipeline = build_training_pipeline(best_estimator, use_scaler=True, use_smote=True)
    best_pipeline.fit(X_train, y_train)

    threshold = find_best_threshold(best_pipeline, X_val, y_val)

    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 Random Forest Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Train Time (opt+fit): {train_time:.2f}s")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    plot_confusion_matrix(y_test, y_pred, 'Random Forest', threshold)

    if feature_names:
        try:
            plot_feature_importance(best_pipeline.named_steps['clf'], 'Random Forest', feature_names)
        except Exception:
            pass

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'train_time': train_time,
        'cv_best_mean_roc_auc': study.best_value,
        'cv_best_params': study.best_params,
        'cv_n_splits': 5,
        'cv_scoring': 'roc_auc',
    }

    save_model(best_pipeline, 'Random Forest', threshold, metrics)

    return best_pipeline, threshold, metrics

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test, class_weights=None, feature_names=None, n_trials=20):
    """Train and optimize XGBoost with Optuna CV and pipeline."""
    print("🚀 Training XGBoost (Optuna CV)...")
    print("=" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': -1,
            'tree_method': 'hist',
        }

        estimator = xgb.XGBClassifier(**params)
        pipeline = build_training_pipeline(estimator, use_scaler=True, use_smote=True)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    start_time = time.time()
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    train_time = time.time() - start_time

    print(f"\n🎯 Best Parameters Found:")
    print("-" * 40)
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    print(f"Best CV ROC AUC: {study.best_value:.4f}")

    best_estimator = xgb.XGBClassifier(**{**study.best_params})
    best_pipeline = build_training_pipeline(best_estimator, use_scaler=True, use_smote=True)
    best_pipeline.fit(X_train, y_train)

    threshold = find_best_threshold(best_pipeline, X_val, y_val)

    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 XGBoost Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Train Time (opt+fit): {train_time:.2f}s")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    plot_confusion_matrix(y_test, y_pred, 'XGBoost', threshold)

    if feature_names:
        try:
            plot_feature_importance(best_pipeline.named_steps['clf'], 'XGBoost', feature_names)
        except Exception:
            pass

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'train_time': train_time,
        'cv_best_mean_roc_auc': study.best_value,
        'cv_best_params': study.best_params,
        'cv_n_splits': 5,
        'cv_scoring': 'roc_auc',
    }

    save_model(best_pipeline, 'XGBoost', threshold, metrics)

    return best_pipeline, threshold, metrics

def train_adaboost(X_train, y_train, X_val, y_val, X_test, y_test, class_weights=None, feature_names=None, n_trials=20):
    """Train and optimize AdaBoost"""
    print("🚀 Training AdaBoost...")
    print("=" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0, log=True),
            'random_state': 42,
        }

        estimator = AdaBoostClassifier(**params)
        pipeline = build_training_pipeline(estimator, use_scaler=True, use_smote=True)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    start_time = time.time()
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    train_time = time.time() - start_time

    print(f"\nBest Parameters Found:")
    print("-" * 40)
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    print(f"Best CV ROC AUC: {study.best_value:.4f}")

    best_estimator = AdaBoostClassifier(**{**study.best_params})
    best_pipeline = build_training_pipeline(best_estimator, use_scaler=True, use_smote=True)
    best_pipeline.fit(X_train, y_train)

    threshold = find_best_threshold(best_pipeline, X_val, y_val)

    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 AdaBoost Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Train Time (opt+fit): {train_time:.2f}s")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    plot_confusion_matrix(y_test, y_pred, 'AdaBoost', threshold)

    if feature_names:
        try:
            plot_feature_importance(best_pipeline.named_steps['clf'], 'AdaBoost', feature_names)
        except Exception:
            pass

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'train_time': train_time,
        'cv_best_mean_roc_auc': study.best_value,
        'cv_best_params': study.best_params,
        'cv_n_splits': 5,
        'cv_scoring': 'roc_auc',
    }

    save_model(best_pipeline, 'AdaBoost', threshold, metrics)

    return best_pipeline, threshold, metrics

def train_decision_tree(X_train, y_train, X_val, y_val, X_test, y_test, class_weights=None, feature_names=None, n_trials=20):
    """Train and optimize Decision Tree"""
    print("🌳 Training Decision Tree...")
    print("=" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
        }

        estimator = DecisionTreeClassifier(**params)
        pipeline = build_training_pipeline(estimator, use_scaler=False, use_smote=True)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    start_time = time.time()
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    train_time = time.time() - start_time

    print(f"\n Best Parameters Found:")
    print("-" * 40)
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    print(f"Best CV ROC AUC: {study.best_value:.4f}")

    best_estimator = DecisionTreeClassifier(**{**study.best_params})
    best_pipeline = build_training_pipeline(best_estimator, use_scaler=False, use_smote=True)
    best_pipeline.fit(X_train, y_train)

    threshold = find_best_threshold(best_pipeline, X_val, y_val)

    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 Decision Tree Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Train Time (opt+fit): {train_time:.2f}s")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    plot_confusion_matrix(y_test, y_pred, 'Decision Tree', threshold)

    if feature_names:
        try:
            plot_feature_importance(best_pipeline.named_steps['clf'], 'Decision Tree', feature_names)
        except Exception:
            pass

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'train_time': train_time,
        'cv_best_mean_roc_auc': study.best_value,
        'cv_best_params': study.best_params,
        'cv_n_splits': 5,
        'cv_scoring': 'roc_auc',
    }

    save_model(best_pipeline, 'Decision Tree', threshold, metrics)

    return best_pipeline, threshold, metrics

def train_svm(X_train, y_train, X_val, y_val, X_test, y_test, class_weights=None, feature_names=None, n_trials=20):
    """Train and optimize SVM with Optuna CV and pipeline (no leakage)."""
    print("⚡ Training SVM (Optuna CV)...")
    print("=" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'C': trial.suggest_float('C', 0.1, 100.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'probability': True,
            'random_state': 42,
        }

        estimator = SVC(**params)
        pipeline = build_training_pipeline(estimator, use_scaler=True, use_smote=True)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    start_time = time.time()
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    train_time = time.time() - start_time

    print(f"\n Best Parameters Found:")
    print("-" * 40)
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    print(f"Best CV ROC AUC: {study.best_value:.4f}")

    best_estimator = SVC(**{**study.best_params})
    best_pipeline = build_training_pipeline(best_estimator, use_scaler=True, use_smote=True)
    best_pipeline.fit(X_train, y_train)

    threshold = find_best_threshold(best_pipeline, X_val, y_val)

    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 SVM Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Train Time (opt+fit): {train_time:.2f}s")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    plot_confusion_matrix(y_test, y_pred, 'SVM', threshold)

    print(f"\nSVM does not support feature importance analysis.")

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'train_time': train_time,
        'cv_best_mean_roc_auc': study.best_value,
        'cv_best_params': study.best_params,
        'cv_n_splits': 5,
        'cv_scoring': 'roc_auc',
    }

    save_model(best_pipeline, 'SVM', threshold, metrics)

    return best_pipeline, threshold, metrics

def train_logistic_regression(X_train, y_train, X_val, y_val, X_test, y_test, class_weights=None, feature_names=None, n_trials=20):
    """Train and optimize Logistic Regression"""
    print("📈 Training Logistic Regression...")
    print("=" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        if penalty == 'elasticnet':
            solver = 'saga'
        elif penalty == 'l1':
            solver = 'liblinear'
        else:
            solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])

        params = {
            'C': trial.suggest_float('C', 0.01, 100.0, log=True),
            'penalty': penalty,
            'solver': solver,
            'max_iter': 1000,
            'random_state': 42,
        }
        if penalty == 'elasticnet':
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)

        estimator = LogisticRegression(**params)
        pipeline = build_training_pipeline(estimator, use_scaler=True, use_smote=True)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    start_time = time.time()
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    train_time = time.time() - start_time

    print(f"\n Best Parameters Found:")
    print("-" * 40)
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    print(f"Best CV ROC AUC: {study.best_value:.4f}")

    best_params = {**study.best_params}
    penalty = best_params.get('penalty', 'l2')
    if penalty == 'elasticnet':
        best_params['solver'] = 'saga'
    elif penalty == 'l1':
        best_params['solver'] = 'liblinear'
    else:
        best_params.setdefault('solver', 'lbfgs')
    best_estimator = LogisticRegression(**best_params)
    best_pipeline = build_training_pipeline(best_estimator, use_scaler=True, use_smote=True)
    best_pipeline.fit(X_train, y_train)

    threshold = find_best_threshold(best_pipeline, X_val, y_val)

    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 Logistic Regression Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Train Time (opt+fit): {train_time:.2f}s")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    plot_confusion_matrix(y_test, y_pred, 'Logistic Regression', threshold)

    if feature_names and hasattr(best_pipeline.named_steps['clf'], 'coef_'):
        coef = best_pipeline.named_steps['clf'].coef_[0]
        indices = np.argsort(np.abs(coef))[::-1][:10]
        print(f"\nLogistic Regression - Top 10 Most Important Features:")
        print("-" * 60)
        for i, idx in enumerate(indices):
            print(f"{i+1:2d}. {feature_names[idx]:25s} {coef[idx]:.4f}")

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'train_time': train_time,
        'cv_best_mean_roc_auc': study.best_value,
        'cv_best_params': study.best_params,
        'cv_n_splits': 5,
        'cv_scoring': 'roc_auc',
    }

    save_model(best_pipeline, 'Logistic Regression', threshold, metrics)

    return best_pipeline, threshold, metrics

def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, class_weights=None, feature_names=None, n_trials=20):
    """Train and optimize MLP (Neural Network)"""
    print("🧠 Training MLP (Neural Network)...")
    print("=" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes',
                [(50,), (100,), (50, 50), (100, 50), (100, 100), (200, 100), (150, 100, 50)]),
            'activation': trial.suggest_categorical('activation', ['relu']),
            'solver': trial.suggest_categorical('solver', ['adam']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
            'early_stopping': True,
            'max_iter': 1000,
            'random_state': 42,
        }

        estimator = MLPClassifier(**params)
        pipeline = build_training_pipeline(estimator, use_scaler=True, use_smote=True)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    start_time = time.time()
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    train_time = time.time() - start_time

    print(f"\n Best Parameters Found:")
    print("-" * 40)
    for param, value in study.best_params.items():
        print(f"{param}: {value}")
    print(f"Best CV ROC AUC: {study.best_value:.4f}")

    best_estimator = MLPClassifier(**{**study.best_params})
    best_pipeline = build_training_pipeline(best_estimator, use_scaler=True, use_smote=True)
    best_pipeline.fit(X_train, y_train)

    threshold = find_best_threshold(best_pipeline, X_val, y_val)

    y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 MLP Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Train Time (opt+fit): {train_time:.2f}s")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    plot_confusion_matrix(y_test, y_pred, 'MLP', threshold)

    print(f"\nMLP does not support feature importance analysis.")

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'train_time': train_time,
        'cv_best_mean_roc_auc': study.best_value,
        'cv_best_params': study.best_params,
        'cv_n_splits': 5,
        'cv_scoring': 'roc_auc',
    }

    save_model(best_pipeline, 'MLP', threshold, metrics)

    return best_pipeline, threshold, metrics

def train_naive_bayes(X_train, y_train, X_val, y_val, X_test, y_test, class_weights=None, feature_names=None, n_trials=20):
    """Train Naive Bayes within pipeline (scaled + SMOTE) without hyperparameter tuning."""
    print("📊 Training Naive Bayes (pipeline)...")
    print("=" * 50)

    start_time = time.time()
    estimator = GaussianNB()
    pipeline = build_training_pipeline(estimator, use_scaler=True, use_smote=True)
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time

    threshold = find_best_threshold(pipeline, X_val, y_val)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 Naive Bayes Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Train Time (fit): {train_time:.2f}s")

    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    plot_confusion_matrix(y_test, y_pred, 'Naive Bayes', threshold)

    print(f"\nNaive Bayes does not support feature importance analysis.")

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'train_time': train_time,
    }

    save_model(pipeline, 'Naive Bayes', threshold, metrics)

    return pipeline, threshold, metrics

def compare_all_models(models_dict, title="Model Performance Comparison"):
    """
    Compare performance and training time of all models in a business-ready format
    
    Args:
        models_dict: Dictionary with model names as keys and tuples (model, threshold, metrics_dict) as values
        title: Title for the comparison chart
    """
    print(f"\n{'='*80}")
    print(f"📊 {title}")
    print(f"{'='*80}")
    
    # Extract data for comparison
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    auc_scores = []
    train_times = []
    
    for name, (model, threshold, metrics) in models_dict.items():
        model_names.append(name)
        accuracies.append(metrics['accuracy'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1'])
        auc_scores.append(metrics['auc'])
        train_times.append(metrics['train_time'])
    
    # Create comprehensive comparison table
    print(f"\n📋 PERFORMANCE SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10} {'Time(s)':<10}")
    print(f"{'-'*80}")
    
    for i, name in enumerate(model_names):
        print(f"{name:<20} {accuracies[i]:<10.4f} {precisions[i]:<10.4f} {recalls[i]:<10.4f} "
              f"{f1_scores[i]:<10.4f} {auc_scores[i]:<10.4f} {train_times[i]:<10.2f}")
    
    # Find best performers
    best_recall_idx = np.argmax(recalls)
    best_accuracy_idx = np.argmax(accuracies)
    best_f1_idx = np.argmax(f1_scores)
    fastest_idx = np.argmin(train_times)
    
    print(f"\n🏆 KEY INSIGHTS")
    print(f"{'='*80}")
    print(f"🥇 Best Recall:     {model_names[best_recall_idx]} ({recalls[best_recall_idx]:.4f})")
    print(f"🥇 Best Accuracy:  {model_names[best_accuracy_idx]} ({accuracies[best_accuracy_idx]:.4f})")
    print(f"🥇 Best F1-Score: {model_names[best_f1_idx]} ({f1_scores[best_f1_idx]:.4f})")
    print(f"⚡ Fastest:        {model_names[fastest_idx]} ({train_times[fastest_idx]:.2f}s)")
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance metrics comparison
    x = np.arange(len(model_names))
    width = 0.2
    
    ax1.bar(x - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
    ax1.bar(x - 0.5*width, precisions, width, label='Precision', alpha=0.8, color='lightgreen')
    ax1.bar(x + 0.5*width, recalls, width, label='Recall', alpha=0.8, color='orange')
    ax1.bar(x + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8, color='pink')
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_title('📊 Performance Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # AUC comparison
    colors = ['gold' if i == np.argmax(auc_scores) else 'lightblue' for i in range(len(auc_scores))]
    bars = ax2.bar(model_names, auc_scores, color=colors, alpha=0.8)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('AUC Score')
    ax2.set_title('🎯 AUC Score Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Highlight best AUC
    best_auc_idx = np.argmax(auc_scores)
    bars[best_auc_idx].set_color('gold')
    bars[best_auc_idx].set_edgecolor('black')
    bars[best_auc_idx].set_linewidth(2)
    
    # Training time comparison
    colors = ['lightcoral' if i == np.argmin(train_times) else 'lightblue' for i in range(len(train_times))]
    bars = ax3.bar(model_names, train_times, color=colors, alpha=0.8)
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('⏱️ Training Time Comparison')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Highlight fastest
    fastest_idx = np.argmin(train_times)
    bars[fastest_idx].set_color('lightcoral')
    bars[fastest_idx].set_edgecolor('black')
    bars[fastest_idx].set_linewidth(2)
    
    # Recall vs Training Time scatter plot
    scatter = ax4.scatter(train_times, recalls, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    ax4.set_xlabel('Training Time (seconds)')
    ax4.set_ylabel('Recall Score')
    ax4.set_title('⚖️ Recall vs Training Time Trade-off')
    ax4.grid(True, alpha=0.3)
    
    # Add model labels to scatter plot
    for i, name in enumerate(model_names):
        ax4.annotate(name, (train_times[i], recalls[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Business recommendations
    print(f"\n💼 BUSINESS RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Find balanced performer (good recall + reasonable time)
    balanced_scores = []
    for i in range(len(model_names)):
        # Weight recall more heavily (0.6) and consider time penalty
        time_penalty = min(train_times[i] / max(train_times), 0.3)  # Max 30% penalty
        balanced_score = 0.6 * recalls[i] + 0.4 * f1_scores[i] - time_penalty
        balanced_scores.append(balanced_score)
    
    best_balanced_idx = np.argmax(balanced_scores)
    
    print(f"🎯 RECOMMENDED MODEL: {model_names[best_balanced_idx]}")
    print(f"   • Recall: {recalls[best_balanced_idx]:.4f} (Catches {recalls[best_balanced_idx]*100:.1f}% of churning customers)")
    print(f"   • Training Time: {train_times[best_balanced_idx]:.2f}s (Fast deployment)")
    print(f"   • Overall Performance: {balanced_scores[best_balanced_idx]:.4f}")
    
    if recalls[best_recall_idx] > 0.8:
        print(f"\n✅ EXCELLENT: {model_names[best_recall_idx]} achieves >80% recall!")
    elif recalls[best_recall_idx] > 0.7:
        print(f"\n✅ GOOD: {model_names[best_recall_idx]} achieves >70% recall")
    else:
        print(f"\n⚠️  CONSIDER: Recall could be improved further")
    
    print(f"\n📈 DEPLOYMENT STRATEGY:")
    print(f"   • Use {model_names[best_balanced_idx]} for production")
    print(f"   • Monitor performance monthly")
    print(f"   • Retrain quarterly with new data")
    print(f"   • Set up alerts for recall drops below 70%")
    
    return {
        'best_recall': model_names[best_recall_idx],
        'best_accuracy': model_names[best_accuracy_idx],
        'best_f1': model_names[best_f1_idx],
        'fastest': model_names[fastest_idx],
        'recommended': model_names[best_balanced_idx],
        'metrics': {
            'recalls': recalls,
            'accuracies': accuracies,
            'f1_scores': f1_scores,
            'train_times': train_times
        }
    }