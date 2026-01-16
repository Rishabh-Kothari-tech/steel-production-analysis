import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import json
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not installed. LSTM model will be skipped.")

# Set working directory
project_root = r"C:\Users\risha\steel_production_analysis"
os.chdir(project_root)

# Setup
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

class ModelTrainer:
    """Advanced model training pipeline with 5 state-of-the-art models"""
    
    def __init__(self, data_path='results/clean_train.csv', test_size=0.2, val_size=0.1, random_state=42):
        """Initialize trainer"""
        self.data_path = data_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.preprocessing_stats = None
        
    def load_preprocessing_stats(self):
        """Load preprocessing statistics"""
        print("Loading preprocessing statistics...")
        try:
            with open('results/preprocessing_statistics.json', 'r') as f:
                self.preprocessing_stats = json.load(f)
            print(f"Preprocessing stats loaded")
            print(f"Training samples: {self.preprocessing_stats['training_data']['final_shape'][0]}")
            print(f"Features: {self.preprocessing_stats['training_data']['final_shape'][1]}")
            return self
        except FileNotFoundError:
            print("Preprocessing stats not found, continuing without them")
            return self
        
    def load_data(self):
        """Load and split cleaned data"""
        print("\nLoading cleaned data...")
        df = pd.read_csv(self.data_path)
        
        print(f"Data shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Duplicates: {df.duplicated().sum()}")
        
        X = df.iloc[:, 1:]  # Features (skip target column)
        y = df.iloc[:, 0]   # Target (first column)
        
        target_col = df.columns[0]
        print(f"Target column: {target_col}")
        print(f"Feature count: {X.shape[1]}")
        
        # Split into train/val/test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        val_size_adjusted = self.val_size / (1 - self.test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)
        
        # Convert back to DataFrame
        feature_names = X.columns
        self.X_train = pd.DataFrame(self.X_train, columns=feature_names)
        self.X_val = pd.DataFrame(self.X_val, columns=feature_names)
        self.X_test = pd.DataFrame(self.X_test, columns=feature_names)
        
        print(f"\nData split:")
        print(f"Train: {self.X_train.shape[0]} samples")
        print(f"Val: {self.X_val.shape[0]} samples")
        print(f"Test: {self.X_test.shape[0]} samples")
        
        return self
    
    def build_models(self):
        """Initialize regression models"""
        print("\nInitializing models...")
        
        self.models = {
            'Random Forest Regressor': RandomForestRegressor(
                n_estimators=200, 
                max_depth=20, 
                random_state=self.random_state, 
                n_jobs=-1
            ),
            'Support Vector Machine': SVR(
                kernel='rbf', 
                C=100, 
                gamma='scale',
                epsilon=0.1
            ),
            'Multi-Layer Perceptron': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            ),
            'Gaussian Process Regressor': GaussianProcessRegressor(
                kernel=C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)),
                alpha=1e-1,
                n_restarts_optimizer=5,
                normalize_y=True,
                random_state=self.random_state
            )
        }
        
        if TENSORFLOW_AVAILABLE:
            self.models['LSTM Network'] = 'lstm'
            print(f"4 models initialized (LSTM available)")
        else:
            print(f"4 models initialized (LSTM unavailable)")
        
        return self
    
    def train_models(self):
        """Train all models and compute metrics"""
        print("\nTraining and evaluating models...")
        print("-" * 100)
        print(f"{'Model':<30} {'RMSE':>12} {'MAE':>12} {'R2':>10} {'Train Time':>12} {'Inference Time':>12}")
        print("-" * 100)
        
        for name, model in self.models.items():
            if name == 'LSTM Network':
                if TENSORFLOW_AVAILABLE:
                    self._train_lstm_model()
                continue
            
            # Training phase
            train_start = time.time()
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - train_start
            
            # Inference phase
            inference_start = time.time()
            y_pred = model.predict(self.X_test)
            inference_time = time.time() - inference_start
            
            # Compute metrics: RMSE, MAE, R², Training Time, Inference Time
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            self.results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Training_Time': train_time,
                'Inference_Time': inference_time,
                'predictions': y_pred
            }
            
            print(f"{name:<30} {rmse:>12.6f} {mae:>12.6f} {r2:>10.6f} {train_time:>10.4f}s {inference_time:>11.6f}s")
        
        print("-" * 100)
        return self
    
    def _train_lstm_model(self):
        """Train LSTM neural network model"""
        print(f"{'LSTM Network':<30} | ", end='', flush=True)
        
        name = 'LSTM Network'
        
        # Reshape data for LSTM (samples, timesteps, features)
        X_train_lstm = self.X_train.values.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        X_test_lstm = self.X_test.values.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        
        # Build LSTM model
        model = models.Sequential([
            layers.LSTM(64, activation='relu', input_shape=(1, self.X_train.shape[1])),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train
        train_start = time.time()
        model.fit(
            X_train_lstm, self.y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        train_time = time.time() - train_start
        
        # Inference
        inference_start = time.time()
        y_pred = model.predict(X_test_lstm, verbose=0).flatten()
        inference_time = time.time() - inference_start
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        self.results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'Training_Time': train_time,
            'Inference_Time': inference_time,
            'predictions': y_pred
        }
        
        print(f"{rmse:>10.6f} | {mae:>10.6f} | {r2:>8.6f} | {train_time:>10.4f}s | {inference_time:>13.6f}s")
        
        # Save LSTM model
        model.save('results/models/LSTM_Network.h5')
        
        return self
    
    def get_best_model(self):
        """Identify and report best model by R2 score"""
        self.best_model_name = max(self.results.items(), key=lambda x: x[1]['R2'])[0]
        self.best_model = self.models[self.best_model_name]
        
        best_metrics = self.results[self.best_model_name]
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"R2 Score: {best_metrics['R2']:.6f}")
        print(f"RMSE: {best_metrics['RMSE']:.6f}")
        print(f"MAE: {best_metrics['MAE']:.6f}")
        print(f"Training Time: {best_metrics['Training_Time']:.4f}s")
        print(f"Inference Time: {best_metrics['Inference_Time']:.6f}s")
        
        return self
    
    def save_models(self):
        """Save trained models to disk"""
        print("\nSaving models...")
        for name, model in self.models.items():
            if name != 'LSTM Network':
                filename = f"results/models/{name.replace(' ', '_')}.pkl"
                joblib.dump(model, filename)
        print("All models saved to results/models/")
        return self
    
    def visualize_results(self):
        """Create performance visualizations"""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        metrics_list = ['R2', 'RMSE', 'MAE', 'Training_Time', 'Inference_Time']
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']
        model_names = list(self.results.keys())
        
        for idx, metric in enumerate(metrics_list):
            ax = axes[idx]
            values = [self.results[name].get(metric, 0) for name in model_names]
            bars = ax.bar(range(len(model_names)), values, color=colors[idx], edgecolor='black', linewidth=1.5, alpha=0.8)
            
            ax.set_xlabel('Models', fontweight='bold', fontsize=11)
            ax.set_ylabel(metric, fontweight='bold', fontsize=11)
            ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=12)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels([name[:20] for name in model_names], rotation=45, ha='right', fontsize=10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Summary table in the 6th subplot
        ax = axes[5]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [['Model', 'R²', 'RMSE', 'MAE', 'Train(s)', 'Infer(s)']]
        for name in model_names:
            metrics = self.results[name]
            table_data.append([
                name[:22],
                f"{metrics['R2']:.6f}",
                f"{metrics['RMSE']:.6f}",
                f"{metrics['MAE']:.6f}",
                f"{metrics['Training_Time']:.4f}",
                f"{metrics['Inference_Time']:.6f}"
            ])
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
                else:
                    table[(i, j)].set_facecolor('#ffffff')
        
        ax.set_title('Comprehensive Metrics Summary', fontweight='bold', fontsize=12, pad=20)
        
        plt.tight_layout()
        plt.savefig('results/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        print("Saved to results/plots/model_comparison.png")
        return self
    
    def save_results_report(self):
        """Save detailed results report to JSON"""
        print("\nGenerating results report...")
        
        report_path = 'results/model_training_report.json'
        
        all_results = {}
        best_metrics = {}
        
        for name, metrics in self.results.items():
            model_results = {}
            for k, v in metrics.items():
                if k != 'predictions':
                    if isinstance(v, (np.integer, np.floating)):
                        model_results[k] = float(v)
                    elif isinstance(v, (int, float)):
                        model_results[k] = v
                    else:
                        model_results[k] = str(v)
            all_results[name] = model_results
            
            if name == self.best_model_name:
                best_metrics = model_results
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'data_source': 'results/clean_train.csv',
            'preprocessing_applied': True,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'models_count': len(self.results),
            'models_trained': list(self.results.keys()),
            'metrics_computed': ['RMSE', 'MAE', 'R2', 'Training_Time', 'Inference_Time'],
            'best_model': self.best_model_name,
            'best_model_metrics': best_metrics,
            'all_models_results': all_results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        
        print(f"Report saved: {report_path}")
        return self
    
    def run(self):
        """Execute complete pipeline"""
        print("Starting model training...\n")
        
        self.load_preprocessing_stats()
        self.load_data()
        self.build_models()
        self.train_models()
        self.get_best_model()
        self.save_models()
        self.visualize_results()
        self.save_results_report()
        
        print("\nTraining pipeline completed!")
        print("Output saved to results/")

if __name__ == "__main__":
    try:
        trainer = ModelTrainer(data_path='results/clean_train.csv', test_size=0.2, val_size=0.1)
        trainer.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the following files exist:")
        print("1. results/clean_train.csv (run: python src/scripts/2. data_preprocessing.py)")
        print("2. results/preprocessing_statistics.json")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
