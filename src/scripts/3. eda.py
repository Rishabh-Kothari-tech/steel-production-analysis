import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math
import json
from datetime import datetime

# Set working directory to project root
project_root = r"C:\Users\risha\steel_production_analysis"
os.chdir(project_root)

# Custom styling
sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

class EDAAnalyzer:
    """Comprehensive Exploratory Data Analysis for Steel Production Data"""
    
    def __init__(self, clean_data_path='results/clean_train.csv'):
        """Initialize EDA analyzer with cleaned data"""
        self.data_path = clean_data_path
        self.df = None
        self.data_quality_report = {}
        
    def load_data(self):
        """Load cleaned data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data shape: {self.df.shape}")
        return self
    
    def check_data_quality(self):
        """Analyze data quality after preprocessing"""
        print("Checking data quality...")
        
        quality_report = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.to_dict(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object', 'category']).columns)
        }
        
        self.data_quality_report = quality_report
        
        print(f"Shape: {quality_report['shape']}")
        print(f"Missing values: {quality_report['missing_values']}")
        print(f"Duplicates: {quality_report['duplicates']}")
        print(f"Numeric columns: {len(quality_report['numeric_columns'])}")
        print(f"Categorical columns: {len(quality_report['categorical_columns'])}")
        
        return self
    
    def plot_correlation_matrix(self, save_path='results/figures/01_corr_matrix.png'):
        """Generate correlation heatmap"""
        print("Creating correlation matrix...")
        
        plt.figure(figsize=(14, 11))
        corr = self.df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu_r', square=True, 
                    mask=mask, center=0, vmin=-1, vmax=1, linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix - Steel Production Data', fontsize=16, fontweight='bold', pad=20)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {save_path}")
        return self

    def plot_feature_distributions(self, save_folder='results/figures/02_distributions'):
        """Plot individual histograms for each numerical feature"""
        print("Creating feature distributions...")
        
        os.makedirs(save_folder, exist_ok=True)
        num_cols = self.df.select_dtypes(include='number').columns
        
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(self.df[col], kde=True, bins=30, color='steelblue', edgecolor='black', ax=ax)
            ax.set_title(f'Distribution: {col}', fontsize=12, fontweight='bold')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            
            stats = f"Mean: {self.df[col].mean():.4f}\nStd: {self.df[col].std():.4f}"
            ax.text(0.98, 0.97, stats, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            save_path = f'{save_folder}/{col}_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Saved {len(num_cols)} distribution plots to {save_folder}/")
        return self
        return self

    def plot_boxplots(self, save_folder='results/figures/03_boxplots'):
        """Generate boxplots for outlier detection"""
        print("Creating boxplots...")
        
        os.makedirs(save_folder, exist_ok=True)
        num_cols = self.df.select_dtypes(include='number').columns
        
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(y=self.df[col], color='lightcoral', ax=ax)
            ax.set_title(f'Boxplot: {col}', fontsize=12, fontweight='bold')
            ax.set_ylabel(col, fontsize=10)
            
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            ax.text(0.02, 0.98, f'IQR: {IQR:.4f}', transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            save_path = f'{save_folder}/{col}_boxplot.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Saved {len(num_cols)} boxplots to {save_folder}/")
        return self

    def plot_pairplot(self, sample_frac=0.3, save_path='results/figures/04_pairplot.png'):
        """Generate pairplot with sampled data"""
        print("Creating pairplot...")
        
        df_sample = self.df.sample(frac=min(1.0, sample_frac)) if sample_frac < 1.0 else self.df
        g = sns.pairplot(df_sample, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30},
                         diag_kws={'fill': True, 'linewidth': 1.5})
        g.fig.suptitle(f'Pairplot ({len(df_sample)} rows)', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        g.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved pairplot to {save_path}")
        return self

    def plot_target_distribution(self, target_col='output', save_path='results/figures/05_target_distribution.png'):
        """Analyze target variable distribution"""
        print("Creating target distribution analysis...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.histplot(self.df[target_col], kde=True, bins=40, color='green', edgecolor='black', ax=axes[0])
        axes[0].set_title(f'Target Distribution: {target_col}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(target_col)
        axes[0].set_ylabel('Frequency')
        
        stats_text = (f"Mean: {self.df[target_col].mean():.6f}\n"
                     f"Std: {self.df[target_col].std():.6f}\n"
                     f"Min: {self.df[target_col].min():.6f}\n"
                     f"Max: {self.df[target_col].max():.6f}\n"
                     f"Median: {self.df[target_col].median():.6f}\n"
                     f"Skewness: {self.df[target_col].skew():.6f}\n"
                     f"Kurtosis: {self.df[target_col].kurtosis():.6f}")
        axes[1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), family='monospace')
        axes[1].axis('off')
        axes[1].set_title('Target Statistics', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {save_path}")
        return self

    def plot_all_histograms(self, save_path='results/figures/06_all_features_histograms.png'):
        """Plot histograms for all numerical features"""
        print("Creating all features histogram grid...")
        
        num_cols = self.df.select_dtypes(include='number').columns
        n_features = len(num_cols)
        
        cols = 4
        rows = math.ceil(n_features / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = axes.flatten()

        for i, col in enumerate(num_cols):
            sns.histplot(self.df[col], kde=True, ax=axes[i], color='skyblue', edgecolor='black', bins=25)
            axes[i].set_title(f'{col}', fontsize=10, fontweight='bold')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Freq' if i % cols == 0 else '')

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle('All Features Distribution Overview', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {save_path}")
        return self

    def generate_eda_summary(self, save_path='results/figures/07_eda_summary.txt'):
        """Generate statistical summary report"""
        print("Generating EDA summary report...")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write("EXPLORATORY DATA ANALYSIS - SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Source: {self.data_path}\n\n")
            
            f.write("DATASET SHAPE:\n")
            f.write(f"Rows: {self.df.shape[0]}\n")
            f.write(f"Columns: {self.df.shape[1]}\n\n")
            
            f.write("DATA QUALITY:\n")
            f.write(f"Missing Values: {self.df.isnull().sum().sum()}\n")
            f.write(f"Duplicates: {self.df.duplicated().sum()}\n")
            f.write(f"Numeric Columns: {len(self.data_quality_report['numeric_columns'])}\n")
            f.write(f"Categorical Columns: {len(self.data_quality_report['categorical_columns'])}\n\n")
            
            f.write("DATA TYPES:\n")
            f.write(str(self.df.dtypes) + "\n\n")
            
            f.write("STATISTICAL SUMMARY:\n")
            f.write(str(self.df.describe()) + "\n\n")
            
            f.write("MISSING VALUES:\n")
            f.write(str(self.df.isnull().sum()) + "\n\n")
            
            f.write("SKEWNESS & KURTOSIS:\n")
            skew_kurt = pd.DataFrame({
                'Skewness': self.df.skew(),
                'Kurtosis': self.df.kurtosis()
            })
            f.write(skew_kurt.to_string() + "\n\n")
            
            f.write("CORRELATION WITH TARGET:\n")
            if 'output' in self.df.columns:
                corr_with_target = self.df.corr()['output'].sort_values(ascending=False)
                f.write(corr_with_target.to_string() + "\n")
        
        print(f"Saved to {save_path}")
        return self
    
    def save_quality_report(self, save_path='results/figures/08_data_quality_report.json'):
        """Save data quality report as JSON"""
        print("Saving data quality report...")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_source': self.data_path,
            'shape': self.df.shape,
            'missing_values': int(self.df.isnull().sum().sum()),
            'duplicates': int(self.df.duplicated().sum()),
            'numeric_columns': self.data_quality_report['numeric_columns'],
            'categorical_columns': self.data_quality_report['categorical_columns'],
            'memory_usage_mb': float(self.df.memory_usage(deep=True).sum() / 1024**2)
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Saved to {save_path}")
        return self

    def run_analysis(self):
        """Execute complete EDA pipeline"""
        print("Starting EDA...\n")
        
        self.load_data().check_data_quality()
        self.plot_correlation_matrix()
        self.plot_feature_distributions()
        self.plot_boxplots()
        self.plot_pairplot(sample_frac=0.3)
        self.plot_target_distribution(target_col=self.df.columns[0])
        self.plot_all_histograms()
        self.generate_eda_summary()
        self.save_quality_report()
        
        print("\nEDA analysis complete!")
        print("Output saved to results/figures/")

if __name__ == "__main__":
    try:
        analyzer = EDAAnalyzer(clean_data_path='results/clean_train.csv')
        analyzer.run_analysis()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure clean_train.csv exists in results/ folder.")
        print("Run: python src/scripts/2. data_preprocessing.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
