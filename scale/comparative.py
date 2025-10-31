import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

class FLResultsAnalyzer:
    """Analyze and visualize Federated Learning experiment results"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.experiments = {}
        self.eval_metrics = {}
        self.fit_metrics = {}
        
    def discover_experiments(self) -> Dict[str, Path]:
        """Discover all experiment directories"""
        print(f"🔍 Scanning: {self.base_path}")
        
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")
        
        # Find all experiment directories (e.g., nasa_25c_alpha_0.02_fedavg)
        experiment_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]
        
        for exp_dir in experiment_dirs:
            exp_name = exp_dir.name
            
            # Look for nested structure: exp_dir/exp_name/metrics/
            metrics_path = exp_dir / exp_name / "metrics"
            
            if metrics_path.exists():
                self.experiments[exp_name] = metrics_path
                print(f"✅ Found: {exp_name}")
            else:
                print(f"⚠️  Skipping {exp_name} (no metrics folder)")
        
        print(f"\n📊 Total experiments found: {len(self.experiments)}")
        return self.experiments
    
    def load_metrics(self) -> Tuple[Dict, Dict]:
        """Load eval and fit metrics for all experiments"""
        for exp_name, metrics_path in self.experiments.items():
            # Load evaluation metrics (global model)
            eval_file = metrics_path / f"{exp_name}_eval_metrics.csv"
            if eval_file.exists():
                self.eval_metrics[exp_name] = pd.read_csv(eval_file)
                print(f"📈 Loaded eval metrics: {exp_name} ({len(self.eval_metrics[exp_name])} rounds)")
            
            # Load fit metrics (client training)
            fit_file = metrics_path / f"{exp_name}_fit_metrics.csv"
            if fit_file.exists():
                self.fit_metrics[exp_name] = pd.read_csv(fit_file)
                print(f"📊 Loaded fit metrics: {exp_name} ({len(self.fit_metrics[exp_name])} records)")
        
        return self.eval_metrics, self.fit_metrics
    
    def extract_config(self, exp_name: str) -> Dict:
        """Extract configuration from experiment name"""
        # Example: nasa_25c_alpha_0.02_fedavg
        parts = exp_name.split('_')
        config = {
            'dataset': parts[0] if parts else 'unknown',
            'num_clients': int(parts[1].replace('c', '')) if len(parts) > 1 else None,
            'alpha': float(parts[3]) if len(parts) > 3 else None,
            'algorithm': parts[4] if len(parts) > 4 else 'unknown'
        }
        return config
    
    def plot_global_convergence(self, save_path: str = None):
        """Plot global model convergence across experiments"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Global Model Performance Comparison (Eval Metrics)', fontsize=16, fontweight='bold')
        
        metrics = ['rmse', 'mae', 'r2', 'loss']
        titles = ['RMSE (Lower is Better)', 'MAE (Lower is Better)', 
                  'R² Score (Higher is Better)', 'Loss (Lower is Better)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]
            
            for exp_name, df in self.eval_metrics.items():
                if metric in df.columns:
                    config = self.extract_config(exp_name)
                    label = f"α={config['alpha']} ({config['algorithm'].upper()})"
                    ax.plot(df['round'], df[metric], marker='o', linewidth=2, 
                           markersize=4, label=label, alpha=0.8)
            
            ax.set_xlabel('Round', fontsize=11, fontweight='bold')
            ax.set_ylabel(metric.upper(), fontsize=11, fontweight='bold')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")
        plt.show()
    
    def plot_client_heterogeneity(self, experiment_name: str = None, save_path: str = None):
        """Plot client training heterogeneity"""
        if experiment_name is None:
            experiment_name = list(self.fit_metrics.keys())[0]
        
        df = self.fit_metrics.get(experiment_name)
        if df is None:
            print(f"❌ No fit metrics for {experiment_name}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Client Training Heterogeneity: {experiment_name}', 
                     fontsize=16, fontweight='bold')
        
        # Validation RMSE distribution per round
        ax = axes[0, 0]
        round_groups = df.groupby('round')['val_rmse'].apply(list)
        positions = list(round_groups.index)
        ax.boxplot(round_groups.values, positions=positions, widths=0.6)
        ax.set_xlabel('Round', fontsize=11, fontweight='bold')
        ax.set_ylabel('Validation RMSE', fontsize=11, fontweight='bold')
        ax.set_title('Client Val RMSE Distribution per Round', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Client performance variance
        ax = axes[0, 1]
        client_avg_rmse = df.groupby('client_id')['val_rmse'].mean().sort_values()
        ax.barh(range(len(client_avg_rmse)), client_avg_rmse.values, color='steelblue')
        ax.set_yticks(range(len(client_avg_rmse)))
        ax.set_yticklabels([f"C{c.split('_')[1]}" for c in client_avg_rmse.index], fontsize=8)
        ax.set_xlabel('Average Val RMSE', fontsize=11, fontweight='bold')
        ax.set_title('Client Performance Ranking', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Training vs Validation Loss
        ax = axes[1, 0]
        for client in df['client_id'].unique()[:10]:  # Plot first 10 clients
            client_data = df[df['client_id'] == client]
            ax.scatter(client_data['train_loss'], client_data['val_loss'], 
                      alpha=0.6, s=50, label=f"C{client.split('_')[1]}")
        ax.plot([0, df['train_loss'].max()], [0, df['train_loss'].max()], 
               'k--', alpha=0.3, label='Perfect fit')
        ax.set_xlabel('Training Loss', fontsize=11, fontweight='bold')
        ax.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
        ax.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        # Data distribution (num_examples)
        ax = axes[1, 1]
        client_data_size = df.groupby('client_id')['num_examples'].first().sort_values()
        ax.barh(range(len(client_data_size)), client_data_size.values, color='coral')
        ax.set_yticks(range(len(client_data_size)))
        ax.set_yticklabels([f"C{c.split('_')[1]}" for c in client_data_size.index], fontsize=8)
        ax.set_xlabel('Number of Examples', fontsize=11, fontweight='bold')
        ax.set_title('Data Distribution per Client', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")
        plt.show()
    
    def compare_algorithms(self, save_path: str = None):
        """Compare different algorithms (FedAvg, FedProx, etc.)"""
        algorithm_groups = {}
        
        for exp_name, df in self.eval_metrics.items():
            config = self.extract_config(exp_name)
            algo = config['algorithm']
            alpha = config['alpha']
            
            key = f"{algo}_alpha{alpha}"
            algorithm_groups[key] = df
        
        if len(algorithm_groups) < 2:
            print("⚠️  Need at least 2 experiments to compare")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['rmse', 'mae', 'r2']
        titles = ['RMSE', 'MAE', 'R² Score']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            
            for key, df in algorithm_groups.items():
                if metric in df.columns:
                    ax.plot(df['round'], df[metric], marker='o', linewidth=2.5,
                           markersize=5, label=key, alpha=0.8)
            
            ax.set_xlabel('Round', fontsize=11, fontweight='bold')
            ax.set_ylabel(title, fontsize=11, fontweight='bold')
            ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved: {save_path}")
        plt.show()
    
    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary statistics table"""
        summary_data = []
        
        for exp_name, df in self.eval_metrics.items():
            config = self.extract_config(exp_name)
            
            # Get final round metrics
            final = df.iloc[-1]
            
            # Get best metrics across all rounds
            best_rmse = df['rmse'].min()
            best_mae = df['mae'].min()
            best_r2 = df['r2'].max()
            
            summary_data.append({
                'Experiment': exp_name,
                'Algorithm': config['algorithm'].upper(),
                'Alpha': config['alpha'],
                'Num Clients': config['num_clients'],
                'Total Rounds': len(df),
                'Final RMSE': final['rmse'],
                'Final MAE': final['mae'],
                'Final R²': final['r2'],
                'Best RMSE': best_rmse,
                'Best MAE': best_mae,
                'Best R²': best_r2,
                'RMSE StdDev': df['rmse'].std(),
                'Convergence': 'Yes' if df['rmse'].iloc[-1] < df['rmse'].iloc[0] else 'No'
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Final RMSE')
        
        return summary_df
    
    def print_summary(self):
        """Print formatted summary"""
        print("\n" + "="*80)
        print("📊 FEDERATED LEARNING RESULTS SUMMARY")
        print("="*80)
        
        summary = self.generate_summary_table()
        
        print("\n🏆 FINAL PERFORMANCE RANKING (by RMSE):")
        print("-" * 80)
        print(summary[['Experiment', 'Algorithm', 'Alpha', 'Final RMSE', 'Final MAE', 'Final R²']].to_string(index=False))
        
        print("\n\n📈 BEST PERFORMANCE ACHIEVED:")
        print("-" * 80)
        print(summary[['Experiment', 'Best RMSE', 'Best MAE', 'Best R²']].to_string(index=False))
        
        print("\n\n📉 CONVERGENCE ANALYSIS:")
        print("-" * 80)
        print(summary[['Experiment', 'Total Rounds', 'RMSE StdDev', 'Convergence']].to_string(index=False))
        
        # Save summary
        summary_path = self.base_path / "summary_table.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n💾 Summary saved to: {summary_path}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting FL Results Analysis...\n")
        
        # Discover and load
        self.discover_experiments()
        self.load_metrics()
        
        if not self.eval_metrics:
            print("❌ No evaluation metrics found!")
            return
        
        # Create output directory
        output_dir = self.base_path / "analysis_plots"
        output_dir.mkdir(exist_ok=True)
        
        # Generate plots
        print("\n📊 Generating plots...\n")
        
        self.plot_global_convergence(save_path=output_dir / "global_convergence.png")
        
        if self.fit_metrics:
            for exp_name in list(self.fit_metrics.keys())[:3]:  # Plot first 3
                self.plot_client_heterogeneity(
                    experiment_name=exp_name,
                    save_path=output_dir / f"client_heterogeneity_{exp_name}.png"
                )
        
        self.compare_algorithms(save_path=output_dir / "algorithm_comparison.png")
        
        # Print summary
        self.print_summary()
        
        print(f"\n✅ Analysis complete! Plots saved to: {output_dir}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set your results path
    BASE_PATH = "/mnt/ceph_drive/FL_IoT_Network/scale/results/run_20251031_104443"
    
    # Run analysis
    analyzer = FLResultsAnalyzer(BASE_PATH)
    analyzer.run_full_analysis()
    
    # Optional: Interactive analysis
    print("\n" + "="*80)
    print("💡 TIP: You can also use the analyzer interactively:")
    print("="*80)
    print("""
    analyzer = FLResultsAnalyzer(BASE_PATH)
    analyzer.discover_experiments()
    analyzer.load_metrics()
    
    # Custom plots
    analyzer.plot_global_convergence()
    analyzer.plot_client_heterogeneity('nasa_25c_alpha_0.02_fedavg')
    
    # Get data
    summary = analyzer.generate_summary_table()
    print(summary)
    """)