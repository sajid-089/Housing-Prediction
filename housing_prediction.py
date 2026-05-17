# ============================================================
# CALIFORNIA HOUSING PRICE PREDICTION
# Regression Analysis using Multiple ML Algorithms
# ============================================================

"""
This project builds and compares multiple regression models to predict
median house values in California based on census data features like
location coordinates, income levels, room counts, and population density.

Dataset: California Housing Prices (20,640 samples, 10 features)
Models:  Linear Regression, Random Forest, Gradient Boosting
Metrics: MAE, RMSE, R2 Score
"""

# --- standard library imports ---
import os
import sys

# --- data handling ---
import pandas as pd
import numpy as np

# --- visualization ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- machine learning ---
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- configuration ---
import warnings
warnings.filterwarnings('ignore')

# plot styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
FIGURE_DPI = 150


# ============================================================
# DATA LOADING MODULE
# ============================================================

class DataLoader:
    """Handles loading the housing dataset from various sources."""

    def __init__(self, data_dir='data'):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(base_path, data_dir)

    def load(self):
        """
        Attempt to load from local CSV first.
        Falls back to sklearn built-in dataset if file not found.
        Returns a clean pandas DataFrame.
        """
        local_path = os.path.join(self.data_dir, 'housing.csv')


        if os.path.exists(local_path):
            print("[INFO] Loading dataset from local CSV...")
            df = pd.read_csv(local_path, encoding='latin-1')
        else:
            print("[INFO] CSV not found locally, loading from sklearn...")
            from sklearn.datasets import fetch_california_housing
            raw = fetch_california_housing(as_frame=True)
            df = raw.frame

        print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df


# ============================================================
# EXPLORATORY DATA ANALYSIS MODULE
# ============================================================

class ExploratoryAnalysis:
    """Performs and visualizes exploratory data analysis."""

    def __init__(self, df):
        self.df = df.copy()

    def identify_target(self):
        """
        Automatically detect the target column by looking for common
        names like 'value', 'price', or 'MedHouseVal' in column names.
        If none found, defaults to the last column.
        """
        target_keywords = ['value', 'price', 'medhouseval', 'median_house']

        for col in self.df.columns:
            if any(kw in col.lower() for kw in target_keywords):
                return col

        # fallback: last column is usually the target
        return self.df.columns[-1]

    def print_summary(self, target_col):
        """Print a comprehensive text summary of the dataset."""

        print("\n" + "=" * 55)
        print("  DATASET OVERVIEW")
        print("=" * 55)

        print(f"\n  Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns")

        # column details
        print(f"\n  {'Column':<25s} {'Type':<12s} {'Nulls':<8s} {'Unique':<8s}")
        print(f"  {'-'*53}")
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            nulls = str(self.df[col].isnull().sum())
            unique = str(self.df[col].nunique())
            print(f"  {col:<25s} {dtype:<12s} {nulls:<8s} {unique:<8s}")

        # missing value summary
        total_missing = self.df.isnull().sum().sum()
        print(f"\n  Total missing values: {total_missing}")

        # duplicate check
        duplicates = self.df.duplicated().sum()
        print(f"  Duplicate rows: {duplicates}")

        # target variable stats
        print(f"\n  Target Variable: {target_col}")
        target = self.df[target_col]
        print(f"    Minimum:  {target.min():.4f}")
        print(f"    Maximum:  {target.max():.4f}")
        print(f"    Mean:     {target.mean():.4f}")
        print(f"    Median:   {target.median():.4f}")
        print(f"    Std Dev:  {target.std():.4f}")

        # statistical summary
        print(f"\n  Full Statistical Summary:")
        print(self.df.describe().round(3).to_string())

    def plot_target_distribution(self, target_col):
        """Visualize the distribution of the target variable."""

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # histogram
        axes[0].hist(self.df[target_col], bins=50, color='steelblue',
                     alpha=0.75, edgecolor='white')
        axes[0].axvline(self.df[target_col].mean(), color='red',
                        linestyle='--', linewidth=2, label='Mean')
        axes[0].axvline(self.df[target_col].median(), color='orange',
                        linestyle='--', linewidth=2, label='Median')
        axes[0].set_title(f'Distribution of {target_col}', fontsize=13)
        axes[0].set_xlabel(target_col, fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].legend()

        # boxplot
        bp = axes[1].boxplot(self.df[target_col].dropna(), vert=True,
                             patch_artist=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][0].set_alpha(0.7)
        axes[1].set_title(f'Boxplot of {target_col}', fontsize=13)
        axes[1].set_ylabel(target_col, fontsize=11)

        plt.tight_layout()
        plt.savefig('target_distribution.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("[INFO] Saved: target_distribution.png")

    def plot_correlation_matrix(self, target_col):
        """Generate correlation heatmap for all numeric features."""

        numeric = self.df.select_dtypes(include=[np.number])
        corr = numeric.corr()

        # create mask for upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(12, 9))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("[INFO] Saved: correlation_heatmap.png")

        # print top correlations with target
        if target_col in corr.columns:
            target_corr = corr[target_col].drop(target_col).sort_values(ascending=False)
            print(f"\n  Correlation with {target_col}:")
            for feat, val in target_corr.items():
                direction = "+" if val > 0 else "-"
                print(f"    {feat:<25s} {direction}{abs(val):.3f}")

    def plot_feature_distributions(self, target_col):
        """Plot histogram for each numeric feature."""

        numeric = self.df.select_dtypes(include=[np.number])
        features = [c for c in numeric.columns if c != target_col]
        n_features = len(features)

        cols = 3
        rows = (n_features + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten()

        for i, col in enumerate(features):
            axes[i].hist(self.df[col].dropna(), bins=40,
                        color='teal', alpha=0.7, edgecolor='white')
            axes[i].set_title(col, fontsize=11)
            axes[i].set_ylabel('Count')

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Feature Distributions', fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("[INFO] Saved: feature_distributions.png")

    def plot_target_correlations(self, target_col):
        """Horizontal bar chart of feature correlations with target."""

        numeric = self.df.select_dtypes(include=[np.number])
        if target_col not in numeric.columns:
            return

        corr = numeric.corr()[target_col].drop(target_col).sort_values()

        plt.figure(figsize=(10, 6))
        colors = ['#e74c3c' if x < 0 else '#27ae60' for x in corr.values]
        corr.plot(kind='barh', color=colors, edgecolor='white')
        plt.title(f'Feature Correlation with {target_col}', fontsize=14)
        plt.xlabel('Correlation Coefficient', fontsize=11)
        plt.axvline(x=0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.savefig('target_correlations.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("[INFO] Saved: target_correlations.png")

    def run_full_eda(self, target_col):
        """Execute the complete EDA pipeline."""
        self.print_summary(target_col)
        self.plot_target_distribution(target_col)
        self.plot_correlation_matrix(target_col)
        self.plot_feature_distributions(target_col)
        self.plot_target_correlations(target_col)


# ============================================================
# DATA PREPROCESSING MODULE
# ============================================================

class DataPreprocessor:
    """Handles all data cleaning, encoding, scaling, and splitting."""

    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def handle_missing_values(self, df):
        """
        Fill missing values using appropriate strategies:
        - Numeric columns: fill with median (robust to outliers)
        - Categorical columns: fill with mode (most frequent value)
        """
        missing_before = df.isnull().sum().sum()

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype == 'object':
                    fill_value = df[col].mode()[0]
                    df[col].fillna(fill_value, inplace=True)
                    print(f"    Filled '{col}' nulls with mode: '{fill_value}'")
                else:
                    fill_value = df[col].median()
                    df[col].fillna(fill_value, inplace=True)
                    print(f"    Filled '{col}' nulls with median: {fill_value:.2f}")

        missing_after = df.isnull().sum().sum()
        print(f"  Missing values: {missing_before} -> {missing_after}")
        return df

    def encode_categoricals(self, df):
        """Convert text/category columns into numeric values using LabelEncoder."""
        cat_cols = df.select_dtypes(include=['object']).columns

        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            print(f"    Encoded '{col}': {len(le.classes_)} unique categories")

        return df

    def remove_outliers(self, df, target_col, lower=0.05, upper=0.95):
        """
        Remove extreme outliers from the target variable using percentile clipping.
        Keeps data between the 5th and 95th percentile to reduce noise.
        """
        q_low = df[target_col].quantile(lower)
        q_high = df[target_col].quantile(upper)

        before = len(df)
        df = df[(df[target_col] >= q_low) & (df[target_col] <= q_high)]
        after = len(df)

        removed = before - after
        print(f"  Outlier removal: {before} -> {after} rows ({removed} removed)")
        print(f"    Kept range: {q_low:.2f} to {q_high:.2f}")

        return df

    def prepare(self, df, target_col):
        """
        Run the complete preprocessing pipeline:
        1. Handle missing values
        2. Encode categorical features
        3. Remove target outliers
        4. Separate features and target
        5. Split into train/test
        6. Scale features
        """
        print("\n[INFO] Starting data preprocessing...")

        # clean
        df = self.handle_missing_values(df)
        df = self.encode_categoricals(df)
        df = self.remove_outliers(df, target_col)

        # separate
        X = df.drop(columns=[target_col])
        y = df[target_col]
        feature_names = X.columns.tolist()

        print(f"  Features ({len(feature_names)}): {feature_names}")

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=self.random_state
        )
        print(f"  Training: {X_train.shape[0]} samples | Testing: {X_test.shape[0]} samples")

        # scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"  Features scaled using StandardScaler")

        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names


# ============================================================
# MODEL TRAINING MODULE
# ============================================================

class ModelTrainer:
    """Trains multiple regression models and stores them for evaluation."""

    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.models = {}

    def train_all(self, X_train, y_train):
        """Train all three regression models and return them as a dictionary."""

        print("\n[INFO] Training models...")

        # model 1: linear regression
        print("  [1/3] Linear Regression...")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        self.models['Linear Regression'] = lr

        # model 2: random forest
        print("  [2/3] Random Forest Regressor...")
        rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['Random Forest'] = rf

        # model 3: gradient boosting
        print("  [3/3] Gradient Boosting Regressor...")
        gb = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.random_state
        )
        gb.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb

        print(f"  All {len(self.models)} models trained successfully!")
        return self.models


# ============================================================
# MODEL EVALUATION MODULE
# ============================================================

class ModelEvaluator:
    """Evaluates and visualizes regression model performance."""

    def __init__(self):
        self.results = []

    def evaluate_single(self, model, name, X_test, y_test):
        """
        Evaluate one model and return its metrics.
        - MAE: average prediction error in original units
        - RMSE: penalizes larger errors more than MAE
        - R2: proportion of variance explained (1.0 = perfect)
        """
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        result = {
            'name': name,
            'predictions': preds,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        self.results.append(result)

        print(f"\n  {name}:")
        print(f"    MAE:  {mae:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    R2:   {r2:.4f} ({r2*100:.1f}% variance explained)")

        return result

    def evaluate_all(self, models, X_test, y_test):
        """Evaluate all trained models and print comparison."""

        print("\n" + "=" * 55)
        print("  MODEL EVALUATION RESULTS")
        print("=" * 55)

        for name, model in models.items():
            self.evaluate_single(model, name, X_test, y_test)

        return self.results

    def get_best_model(self, models):
        """Identify the model with the highest R2 score."""
        best = max(self.results, key=lambda x: x['r2'])
        best_model = models[best['name']]
        return best, best_model

    def plot_comparison(self):
        """Bar charts comparing all models across three metrics."""

        names = [r['name'] for r in self.results]
        r2_vals = [r['r2'] for r in self.results]
        mae_vals = [r['mae'] for r in self.results]
        rmse_vals = [r['rmse'] for r in self.results]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = ['#27ae60', '#2980b9', '#e67e22']

        # R2 score
        bars = axes[0].bar(names, r2_vals, color=colors, width=0.5)
        for bar, val in zip(bars, r2_vals):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
        axes[0].set_title('R2 Score (higher = better)', fontsize=13)
        axes[0].set_ylim(0, 1.0)
        axes[0].grid(axis='y', alpha=0.3)

        # MAE
        bars = axes[1].bar(names, mae_vals, color=colors, width=0.5)
        for bar, val in zip(bars, mae_vals):
            axes[1].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
        axes[1].set_title('MAE (lower = better)', fontsize=13)
        axes[1].grid(axis='y', alpha=0.3)

        # RMSE
        bars = axes[2].bar(names, rmse_vals, color=colors, width=0.5)
        for bar, val in zip(bars, rmse_vals):
            axes[2].text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005,
                        f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
        axes[2].set_title('RMSE (lower = better)', fontsize=13)
        axes[2].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("[INFO] Saved: model_comparison.png")

    def plot_predictions_scatter(self, y_test, best_result):
        """Scatter plot: predicted vs actual values for the best model."""

        preds = best_result['predictions']
        name = best_result['name']
        r2 = best_result['r2']

        plt.figure(figsize=(8, 7))
        plt.scatter(y_test, preds, alpha=0.25, color='steelblue', s=12, edgecolors='none')

        # perfect prediction line
        bounds = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        plt.plot(bounds, bounds, 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel('Actual House Value', fontsize=12)
        plt.ylabel('Predicted House Value', fontsize=12)
        plt.title(f'{name}: Predicted vs Actual (R2={r2:.3f})', fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('predicted_vs_actual.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("[INFO] Saved: predicted_vs_actual.png")

    def plot_residual_analysis(self, y_test, best_result):
        """
        Residual plots to diagnose model behavior.
        Left: residuals vs predicted (should show no pattern)
        Right: residual histogram (should be roughly bell-shaped around zero)
        """
        preds = best_result['predictions']
        residuals = y_test.values - preds

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # residual scatter
        axes[0].scatter(preds, residuals, alpha=0.25, color='teal', s=12, edgecolors='none')
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values', fontsize=11)
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].set_title(f'{best_result["name"]} - Residual Scatter', fontsize=13)
        axes[0].grid(True, alpha=0.3)

        # residual histogram
        axes[1].hist(residuals, bins=50, color='teal', alpha=0.7, edgecolor='white')
        axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Residual Value', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Residual Distribution', fontsize=13)

        plt.tight_layout()
        plt.savefig('residual_analysis.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("[INFO] Saved: residual_analysis.png")

    def plot_feature_importance(self, model, feature_names):
        """Feature importance chart from tree-based models."""

        if not hasattr(model, 'feature_importances_'):
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices],
                 color='steelblue', edgecolor='white')
        plt.yticks(range(len(indices)),
                   [feature_names[i] for i in indices], fontsize=10)
        plt.xlabel('Importance Score', fontsize=11)
        plt.title('Feature Importance (Best Model)', fontsize=14)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print("[INFO] Saved: feature_importance.png")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """Execute the complete housing price prediction pipeline."""

    print("=" * 60)
    print("  CALIFORNIA HOUSING PRICE PREDICTION")
    print("  Regression Analysis Project")
    print("=" * 60)

    # --- phase 1: data loading ---
    loader = DataLoader()
    df = loader.load()

    # --- phase 2: exploratory analysis ---
    eda = ExploratoryAnalysis(df)
    target_col = eda.identify_target()
    eda.run_full_eda(target_col)

    # --- phase 3: preprocessing ---
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare(df, target_col)

    # --- phase 4: model training ---
    trainer = ModelTrainer()
    models = trainer.train_all(X_train, y_train)

    # --- phase 5: evaluation ---
    evaluator = ModelEvaluator()
    evaluator.evaluate_all(models, X_test, y_test)

    best_result, best_model = evaluator.get_best_model(models)

    # --- phase 6: visualization ---
    evaluator.plot_comparison()
    evaluator.plot_predictions_scatter(y_test, best_result)
    evaluator.plot_residual_analysis(y_test, best_result)
    evaluator.plot_feature_importance(best_model, feature_names)

    # --- final summary ---
    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    print(f"\n  {'Model':<22s} {'MAE':<10s} {'RMSE':<10s} {'R2':<10s}")
    print(f"  {'-'*52}")
    for r in evaluator.results:
        marker = " <-- BEST" if r['name'] == best_result['name'] else ""
        print(f"  {r['name']:<22s} {r['mae']:<10.4f} {r['rmse']:<10.4f} {r['r2']:<10.4f}{marker}")

    print(f"\n  Winner: {best_result['name']}")
    print(f"  R2 Score: {best_result['r2']:.4f} ({best_result['r2']*100:.1f}% variance explained)")

    print(f"\n  Generated Visualizations:")
    files = ['target_distribution.png', 'correlation_heatmap.png',
             'feature_distributions.png', 'target_correlations.png',
             'model_comparison.png', 'predicted_vs_actual.png',
             'residual_analysis.png', 'feature_importance.png']
    for f in files:
        status = "exists" if os.path.exists(f) else "missing"
        print(f"    [{status}] {f}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()