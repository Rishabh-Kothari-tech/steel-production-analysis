import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set working directory
project_root = r"C:\Users\risha\steel_production_analysis"
os.chdir(project_root)

# Setup directories
os.makedirs('results/analysis', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

class ResultsAnalyzer:
    """Advanced results analysis for steel production models"""
    
    def __init__(self, report_path='results/model_training_report.json'):
        """Initialize analyzer with training report"""
        self.report_path = report_path
        self.report_data = None
        self.best_model = None
        self.best_model_name = None
        self.analysis_results = {}
        
    def load_report(self):
        """Load the model training report"""
        print("Loading training report...")
        try:
            with open(self.report_path, 'r') as f:
                self.report_data = json.load(f)
            print(f"Report loaded successfully")
            return self
        except FileNotFoundError:
            print(f"Error: Report not found at {self.report_path}")
            return None
    
    def load_best_model(self):
        """Load the best trained model"""
        print("\nLoading best model...")
        model_dir = 'results/models'
        self.best_model_name = self.report_data['best_model']
        
        model_filename = self.best_model_name.replace(' ', '_') + '.pkl'
        best_path = os.path.join(model_dir, model_filename)
        
        if os.path.exists(best_path):
            self.best_model = joblib.load(best_path)
            print(f"Best model loaded: {self.best_model_name}")
            return self
        else:
            print(f"Error: Model not found at {best_path}")
            return None
    
    def generate_performance_summary(self):
        """Generate performance summary"""
        print("\nGenerating performance summary...")
        
        summary_data = []
        for model_name, metrics in self.report_data['all_models_results'].items():
            summary_data.append({
                'Model': model_name,
                'R² Score': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'Training_Time': metrics.get('Training_Time', 'N/A'),
                'Inference_Time': metrics.get('Inference_Time', 'N/A')
            })
        
        df_summary = pd.DataFrame(summary_data).sort_values('R² Score', ascending=False)
        
        summary_path = 'results/analysis/01_performance_summary.csv'
        df_summary.to_csv(summary_path, index=False)
        print(f"Performance summary saved to {summary_path}")
        
        return df_summary
    
    def create_performance_rankings(self):
        """Create performance rankings"""
        print("\nCreating performance rankings...")
        
        rankings_data = []
        all_results = self.report_data['all_models_results']
        
        for rank, (model_name, metrics) in enumerate(
            sorted(all_results.items(), key=lambda x: x[1]['R2'], reverse=True), 1
        ):
            rankings_data.append({
                'Rank': rank,
                'Model': model_name,
                'R² Score': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'Training_Time (s)': metrics.get('Training_Time', 'N/A'),
                'Improvement vs Worst': metrics['R2'] - min(m['R2'] for m in all_results.values())
            })
        
        df_rankings = pd.DataFrame(rankings_data)
        rankings_path = 'results/analysis/02_model_rankings.csv'
        df_rankings.to_csv(rankings_path, index=False)
        print(f"Rankings saved to {rankings_path}")
        
        return df_rankings
    
    def analyze_predictions(self):
        """Load and analyze best model predictions"""
        print("\nAnalyzing predictions...")
        
        try:
            df_test = pd.read_csv('results/clean_test.csv')
            X_test = df_test.iloc[:, 1:]
            y_test = df_test.iloc[:, 0]
            
            y_pred = self.best_model.predict(X_test.values)
            
            residuals = y_test.values - y_pred
            
            analysis = {
                'y_true': y_test.values,
                'y_pred': y_pred,
                'residuals': residuals,
                'metrics': {
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R2': r2_score(y_test, y_pred),
                    'MAPE': mean_absolute_percentage_error(y_test, y_pred),
                    'Mean_Residual': np.mean(residuals),
                    'Std_Residual': np.std(residuals),
                    'Min_Residual': np.min(residuals),
                    'Max_Residual': np.max(residuals)
                }
            }
            
            self.analysis_results = analysis
            
            analysis_path = 'results/analysis/03_prediction_analysis.json'
            with open(analysis_path, 'w') as f:
                json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in analysis.items() if k != 'y_true' and k != 'y_pred' and k != 'residuals'}, 
                         f, indent=4)
            
            print(f"Prediction analysis complete")
            print(f"RMSE: {analysis['metrics']['RMSE']:.6f}")
            print(f"MAE: {analysis['metrics']['MAE']:.6f}")
            print(f"R2 Score: {analysis['metrics']['R2']:.6f}")
            print(f"MAPE: {analysis['metrics']['MAPE']:.6f}")
            
            return analysis
            
        except Exception as e:
            print(f"Error in prediction analysis: {e}")
            return None
    
    def plot_model_comparison(self, metric='R2'):
        """Plot model performance comparison"""
        print(f"\nCreating model comparison plot ({metric})...")
        
        models = []
        scores = []
        colors_list = []
        
        for model_name, metrics in sorted(
            self.report_data['all_models_results'].items(), 
            key=lambda x: x[1][metric], 
            reverse=True
        ):
            models.append(model_name.replace('_', ' '))
            scores.append(metrics[metric])
            colors_list.append('#2ecc71' if model_name == self.best_model_name else '#3498db')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.barh(models, scores, color=colors_list, edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel(f'{metric} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Comparison - {metric} Metric', fontsize=14, fontweight='bold', pad=20)
        ax.invert_yaxis()
        
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + max(scores)*0.01, i, f'{score:.4f}', 
                   va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        save_path = f'results/plots/03_model_comparison_{metric}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to {save_path}")
    
    def plot_prediction_accuracy(self):
        """Plot actual vs predicted for best model"""
        print("\nCreating prediction accuracy plot...")
        
        if not self.analysis_results:
            print("Warning: No analysis results available")
            return
        
        y_true = self.analysis_results['y_true']
        y_pred = self.analysis_results['y_pred']
        residuals = self.analysis_results['residuals']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Best Model Analysis - {self.best_model_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=20, edgecolor='black', linewidth=0.5)
        axes[0, 0].plot([y_true.min(), y_true.max()], 
                       [y_true.min(), y_true.max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Values', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals Distribution
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Residual Value', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Residuals vs Predicted
        axes[1, 0].scatter(y_pred, residuals, alpha=0.5, s=20, edgecolor='black', linewidth=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Metrics Summary
        metrics = self.analysis_results['metrics']
        metrics_text = f"""
PERFORMANCE METRICS

R² Score: {metrics['R2']:.6f}
RMSE: {metrics['RMSE']:.6f}
MAE: {metrics['MAE']:.6f}
MAPE: {metrics['MAPE']:.6f}

RESIDUALS STATISTICS
Mean: {metrics['Mean_Residual']:.6f}
Std Dev: {metrics['Std_Residual']:.6f}
Min: {metrics['Min_Residual']:.6f}
Max: {metrics['Max_Residual']:.6f}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       family='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = 'results/plots/04_prediction_accuracy.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Accuracy plot saved to {save_path}")
    
    def generate_report(self):
        """Generate analysis report"""
        print("\nGenerating analysis report...")
        
        report_path = 'results/analysis/COMPREHENSIVE_ANALYSIS_REPORT.txt'
        
        with open(report_path, 'w') as f:
            f.write("STEEL PRODUCTION - ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training Report: {self.report_data['timestamp']}\n\n")
            
            f.write("BEST MODEL SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Model Name: {self.best_model_name}\n")
            f.write(f"Model Type: {type(self.best_model).__name__}\n\n")
            
            best_metrics = self.report_data['best_model_metrics']
            f.write("Performance Metrics:\n")
            f.write(f"R2 Score: {best_metrics['R2']:.6f}\n")
            f.write(f"RMSE: {best_metrics['RMSE']:.6f}\n")
            f.write(f"MAE: {best_metrics['MAE']:.6f}\n")
            f.write(f"Training Time: {best_metrics.get('Training_Time', 'N/A')}s\n")
            f.write(f"Inference Time: {best_metrics.get('Inference_Time', 'N/A')}s\n\n")
            
            f.write("ALL MODELS PERFORMANCE\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Model':<25} {'R2':<12} {'RMSE':<12} {'MAE':<12}\n")
            f.write("-" * 60 + "\n")
            
            for model_name in sorted(self.report_data['all_models_results'].keys(), 
                                    key=lambda x: self.report_data['all_models_results'][x]['R2'], 
                                    reverse=True):
                metrics = self.report_data['all_models_results'][model_name]
                f.write(f"{model_name:<25} {metrics['R2']:<12.6f} {metrics['RMSE']:<12.6f} {metrics['MAE']:<12.6f}\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 60 + "\n")
            f.write(f"Use {self.best_model_name} for production predictions\n")
            f.write(f"Monitor RMSE: {best_metrics['RMSE']:.6f}\n")
            f.write(f"Expected MAE: {best_metrics['MAE']:.6f}\n")
            f.write(f"Model explains {best_metrics['R2']*100:.2f}% of variance\n")
        
        print(f"Report saved to {report_path}")
    
    def run_analysis(self):
        """Execute complete analysis pipeline"""
        print("Starting analysis...\n")
        
        if not self.load_report():
            return
        
        if not self.load_best_model():
            return
        
        self.generate_performance_summary()
        self.create_performance_rankings()
        self.analyze_predictions()
        
        self.plot_model_comparison('R2')
        self.plot_model_comparison('RMSE')
        self.plot_model_comparison('MAE')
        self.plot_prediction_accuracy()
        
        self.generate_report()
        
        print("\nAnalysis complete!")
        print("Output files saved to results/analysis/ and results/plots/")

if __name__ == "__main__":
    try:
        analyzer = ResultsAnalyzer()
        analyzer.run_analysis()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
