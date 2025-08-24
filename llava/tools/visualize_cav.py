import argparse
import numpy as np
import torch
import json
import pickle
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class CAVVisualizer:
    """Visualize CAV learned concepts and activations"""
    
    def __init__(self, activation_dir: str, cav_path: str, output_dir: str = "visualization_results"):
        self.activation_dir = Path(activation_dir)
        self.cav_path = Path(cav_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.activations = self.load_activations()
        self.cavs = self.load_cavs()
        
    def load_activations(self) -> Dict:
        """Load activation data"""
        print(f"Loading activations from {self.activation_dir}")
        
        activations = {}
        
        # Find all layer files
        pos_files = list(self.activation_dir.glob("layer_*_positive.npy"))
        
        for pos_file in pos_files:
            layer_num = int(pos_file.stem.split("_")[1])
            neg_file = self.activation_dir / f"layer_{layer_num}_negative.npy"
            
            if neg_file.exists():
                pos_acts = np.load(pos_file)
                neg_acts = np.load(neg_file)
                
                activations[layer_num] = {
                    'positive': pos_acts,
                    'negative': neg_acts
                }
                print(f"Loaded layer {layer_num}: pos={pos_acts.shape}, neg={neg_acts.shape}")
        
        return activations
    
    def load_cavs(self) -> Dict:
        """Load trained CAVs"""
        print(f"Loading CAVs from {self.cav_path}")
        
        if self.cav_path.suffix == '.pkl':
            with open(self.cav_path, 'rb') as f:
                cavs = pickle.load(f)
        elif self.cav_path.suffix == '.pt':
            cavs = torch.load(self.cav_path, map_location='cpu')
        else:
            raise ValueError(f"Unsupported CAV file format: {self.cav_path.suffix}")
        
        print(f"Loaded CAVs for {len(cavs)} layers")
        return cavs
    
    def compute_concept_probabilities(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute concept probabilities for positive and negative samples"""
        
        if layer not in self.activations or layer not in self.cavs:
            raise ValueError(f"Layer {layer} not found in activations or CAVs")
        
        # Get activations and CAV
        pos_acts = self.activations[layer]['positive']
        neg_acts = self.activations[layer]['negative']
        
        cav = self.cavs[layer]
        
        # Handle different data types for weights and bias
        if isinstance(cav['w'], np.ndarray):
            w = cav['w']
        elif hasattr(cav['w'], 'numpy'):  # torch tensor
            w = cav['w'].numpy()
        else:  # scalar or other
            w = np.array(cav['w'])
        
        if isinstance(cav['b'], np.ndarray):
            b = cav['b'].item() if cav['b'].size == 1 else cav['b']
        elif hasattr(cav['b'], 'numpy'):  # torch tensor
            b = cav['b'].numpy().item() if cav['b'].numel() == 1 else cav['b'].numpy()
        else:  # scalar (numpy.float64, float, int, etc.)
            b = float(cav['b'])
        
        # Reshape activations for dot product
        pos_flat = pos_acts.reshape(pos_acts.shape[0], -1)
        neg_flat = neg_acts.reshape(neg_acts.shape[0], -1)
        
        # Compute probabilities using sigmoid(w^T * x + b)
        pos_logits = np.dot(pos_flat, w) + b
        neg_logits = np.dot(neg_flat, w) + b
        
        pos_probs = 1 / (1 + np.exp(-pos_logits))
        neg_probs = 1 / (1 + np.exp(-neg_logits))
        
        return pos_probs, neg_probs
    
    def visualize_concept_separation(self, layer: int, method: str = 'tsne'):
        """Visualize concept separation in 2D using t-SNE or PCA"""
        
        print(f"Creating concept separation visualization for layer {layer}")
        
        # Get activations
        pos_acts = self.activations[layer]['positive']
        neg_acts = self.activations[layer]['negative']
        
        # Combine and flatten activations
        all_acts = np.vstack([pos_acts.reshape(pos_acts.shape[0], -1),
                             neg_acts.reshape(neg_acts.shape[0], -1)])
        
        # Labels (1 for positive/hateful, 0 for negative/benign)
        labels = np.hstack([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])
        
        # Dimensionality reduction
        if method.lower() == 'tsne':
            print("Applying t-SNE dimensionality reduction...")
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_acts)//4))
            reduced_acts = reducer.fit_transform(all_acts)
        elif method.lower() == 'pca':
            print("Applying PCA dimensionality reduction...")
            reducer = PCA(n_components=2, random_state=42)
            reduced_acts = reducer.fit_transform(all_acts)
            explained_var = reducer.explained_variance_ratio_
            print(f"Explained variance ratio: {explained_var}")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        colors = ['red' if l == 1 else 'blue' for l in labels]
        labels_text = ['Hateful' if l == 1 else 'Benign' for l in labels]
        
        scatter = plt.scatter(reduced_acts[:, 0], reduced_acts[:, 1], 
                            c=colors, alpha=0.6, s=50)
        
        # Add legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Hateful')
        blue_patch = mpatches.Patch(color='blue', label='Benign')
        plt.legend(handles=[red_patch, blue_patch])
        
        plt.title(f'Concept Separation Visualization - Layer {layer} ({method.upper()})', fontsize=14)
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = self.output_dir / f"concept_separation_layer{layer}_{method}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved concept separation plot to {output_path}")
        
        return reduced_acts, labels
    
    def visualize_concept_separation_with_probabilities(self, layer: int, method: str = 'tsne'):
        """Enhanced visualization with CAV probability color mapping and iso-probability lines"""
        
        print(f"Creating enhanced concept separation visualization for layer {layer}")
        
        # Get activations and compute probabilities
        pos_acts = self.activations[layer]['positive']
        neg_acts = self.activations[layer]['negative']
        pos_probs, neg_probs = self.compute_concept_probabilities(layer)
        
        # Combine and flatten activations
        all_acts = np.vstack([pos_acts.reshape(pos_acts.shape[0], -1),
                             neg_acts.reshape(neg_acts.shape[0], -1)])
        
        # Combine probabilities
        all_probs = np.hstack([pos_probs, neg_probs])
        labels = np.hstack([np.ones(len(pos_acts)), np.zeros(len(neg_acts))])
        
        # Dimensionality reduction
        if method.lower() == 'tsne':
            print("Applying t-SNE dimensionality reduction...")
            perplexity = min(30, max(5, len(all_acts)//4))
            print(f"🎯 Using fixed t-SNE parameters: perplexity={perplexity}, random_state=42")
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                          learning_rate=200, max_iter=1000)
            reduced_acts = reducer.fit_transform(all_acts)
        elif method.lower() == 'pca':
            print("Applying PCA dimensionality reduction...")
            reducer = PCA(n_components=2, random_state=42)
            reduced_acts = reducer.fit_transform(all_acts)
            explained_var = reducer.explained_variance_ratio_
            print(f"Explained variance ratio: {explained_var}")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create enhanced visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        ### Plot 1: Probability-based color mapping
        scatter = ax1.scatter(reduced_acts[:, 0], reduced_acts[:, 1], 
                            c=all_probs, cmap='RdYlBu_r', 
                            alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('CAV Probability (Higher = More Harmful)', fontsize=12)
        
        # Add iso-probability contours (approximate)
        self._add_approximate_contours(ax1, reduced_acts, all_probs, levels=[0.3, 0.5, 0.7])
        
        ax1.set_title(f'CAV Probability Color Mapping - Layer {layer} ({method.upper()})', fontsize=14)
        ax1.set_xlabel(f'{method.upper()} Component 1')
        ax1.set_ylabel(f'{method.upper()} Component 2')
        ax1.grid(True, alpha=0.3)
        
        ### Plot 2: Traditional view with CAV direction (if PCA)
        colors = ['red' if l == 1 else 'blue' for l in labels]
        ax2.scatter(reduced_acts[:, 0], reduced_acts[:, 1], 
                   c=colors, alpha=0.6, s=50)
        
        # Add CAV direction vector (only for PCA)
        if method.lower() == 'pca' and hasattr(reducer, 'components_'):
            self._add_cav_direction_arrow(ax2, reducer, layer)
        
        # Add legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Hateful (Training)')
        blue_patch = mpatches.Patch(color='blue', label='Benign (Training)')
        ax2.legend(handles=[red_patch, blue_patch])
        
        ax2.set_title(f'Traditional View + CAV Direction - Layer {layer} ({method.upper()})', fontsize=14)
        ax2.set_xlabel(f'{method.upper()} Component 1')
        ax2.set_ylabel(f'{method.upper()} Component 2')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / f"enhanced_concept_separation_layer{layer}_{method}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved enhanced concept separation plot to {output_path}")
        
        return reduced_acts, all_probs
    
    def _add_approximate_contours(self, ax, reduced_acts, probs, levels):
        """Add approximate iso-probability contours using interpolation"""
        try:
            from scipy.interpolate import griddata
            
            # Create grid
            x_min, x_max = reduced_acts[:, 0].min(), reduced_acts[:, 0].max()
            y_min, y_max = reduced_acts[:, 1].min(), reduced_acts[:, 1].max()
            
            # Expand boundaries slightly
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_min -= 0.1 * x_range
            x_max += 0.1 * x_range
            y_min -= 0.1 * y_range
            y_max += 0.1 * y_range
            
            # Create interpolation grid
            grid_x, grid_y = np.meshgrid(
                np.linspace(x_min, x_max, 50),
                np.linspace(y_min, y_max, 50)
            )
            
            # Interpolate probabilities onto grid
            grid_probs = griddata(
                reduced_acts, probs, 
                (grid_x, grid_y), 
                method='linear', 
                fill_value=0.5
            )
            
            # Add contour lines
            contour = ax.contour(grid_x, grid_y, grid_probs, 
                               levels=levels, colors=['green', 'orange', 'purple'], 
                               linewidths=2, alpha=0.8)
            
            # Add labels
            ax.clabel(contour, inline=True, fontsize=10, fmt='%.1f')
            
        except ImportError:
            print("SciPy not available for contour plotting")
        except Exception as e:
            print(f"Could not add contours: {e}")
    
    def _add_cav_direction_arrow(self, ax, pca_reducer, layer):
        """Add CAV direction vector as arrow (only works with PCA)"""
        try:
            # Get CAV weights
            cav = self.cavs[layer]
            if isinstance(cav['w'], np.ndarray):
                w = cav['w']
            elif hasattr(cav['w'], 'numpy'):
                w = cav['w'].numpy()
            else:
                w = np.array(cav['w'])
            
            # Project CAV vector onto PCA space
            # CAV direction in original space: w
            # Project onto first 2 PCA components
            cav_proj = np.dot(w.reshape(1, -1), pca_reducer.components_.T).flatten()
            
            # Normalize for visualization
            cav_proj = cav_proj / np.linalg.norm(cav_proj) * 2
            
            # Get center of plot for arrow origin
            x_center = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
            y_center = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
            
            # Add arrow
            ax.arrow(x_center, y_center, cav_proj[0], cav_proj[1],
                    head_width=0.3, head_length=0.2, fc='black', ec='black',
                    linewidth=3, alpha=0.8)
            
            # Add label
            ax.text(x_center + cav_proj[0] * 1.2, y_center + cav_proj[1] * 1.2,
                   'CAV Direction\n(→ More Harmful)', 
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
        except Exception as e:
            print(f"Could not add CAV direction arrow: {e}")
    
    def visualize_concept_probabilities(self, layer: int):
        """Visualize concept probability distributions"""
        
        print(f"Creating concept probability visualization for layer {layer}")
        
        pos_probs, neg_probs = self.compute_concept_probabilities(layer)
        
        plt.figure(figsize=(12, 6))
        
        # Plot histograms
        plt.subplot(1, 2, 1)
        plt.hist(pos_probs, bins=20, alpha=0.7, color='red', label='Hateful', density=True)
        plt.hist(neg_probs, bins=20, alpha=0.7, color='blue', label='Benign', density=True)
        plt.xlabel('Concept Probability')
        plt.ylabel('Density')
        plt.title(f'Concept Probability Distribution - Layer {layer}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot box plot
        plt.subplot(1, 2, 2)
        data_to_plot = [pos_probs, neg_probs]
        labels = ['Hateful', 'Benign']
        colors = ['red', 'blue']
        
        box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.ylabel('Concept Probability')
        plt.title(f'Concept Probability Distribution - Layer {layer}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / f"concept_probabilities_layer{layer}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved concept probability plot to {output_path}")
        
        return pos_probs, neg_probs
    
    def compute_separation_metrics(self, layer: int) -> Dict:
        """Compute quantitative separation metrics"""
        
        pos_probs, neg_probs = self.compute_concept_probabilities(layer)
        
        # Basic statistics
        pos_mean, pos_std = np.mean(pos_probs), np.std(pos_probs)
        neg_mean, neg_std = np.mean(neg_probs), np.std(neg_probs)
        
        # Separation metrics
        cohen_d = abs(pos_mean - neg_mean) / np.sqrt((pos_std**2 + neg_std**2) / 2)
        
        # Classification metrics at threshold 0.5
        pos_correct = np.sum(pos_probs > 0.5) / len(pos_probs)
        neg_correct = np.sum(neg_probs <= 0.5) / len(neg_probs)
        overall_accuracy = (np.sum(pos_probs > 0.5) + np.sum(neg_probs <= 0.5)) / (len(pos_probs) + len(neg_probs))
        
        # Area Under ROC Curve
        from sklearn.metrics import roc_auc_score
        true_labels = np.hstack([np.ones(len(pos_probs)), np.zeros(len(neg_probs))])
        pred_probs = np.hstack([pos_probs, neg_probs])
        auc = roc_auc_score(true_labels, pred_probs)
        
        metrics = {
            'layer': layer,
            'positive_mean': float(pos_mean),
            'positive_std': float(pos_std),
            'negative_mean': float(neg_mean),
            'negative_std': float(neg_std),
            'cohen_d': float(cohen_d),
            'positive_accuracy': float(pos_correct),
            'negative_accuracy': float(neg_correct),
            'overall_accuracy': float(overall_accuracy),
            'auc': float(auc)
        }
        
        return metrics
    
    def create_comprehensive_report(self):
        """Create a comprehensive CAV analysis report"""
        
        print("Creating comprehensive CAV analysis report...")
        
        report = {
            'dataset_info': {
                'activation_dir': str(self.activation_dir),
                'cav_path': str(self.cav_path),
                'layers_analyzed': list(self.activations.keys())
            },
            'layer_metrics': {}
        }
        
        plt.style.use('default')
        
        # Analyze each layer
        for layer in sorted(self.activations.keys()):
            print(f"Analyzing layer {layer}...")
            
            # Compute metrics
            metrics = self.compute_separation_metrics(layer)
            report['layer_metrics'][layer] = metrics
            
            # Create visualizations
            self.visualize_concept_separation(layer, 'tsne')
            self.visualize_concept_separation(layer, 'pca')
            self.visualize_concept_probabilities(layer)
            
            # Create enhanced visualizations with probability mapping
            self.visualize_concept_separation_with_probabilities(layer, 'tsne')
            self.visualize_concept_separation_with_probabilities(layer, 'pca')
        
        # Create summary comparison plot
        self.create_layer_comparison_plot(report['layer_metrics'])
        
        # Save report
        report_path = self.output_dir / "cav_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comprehensive report saved to {report_path}")
        print(f"All visualizations saved to {self.output_dir}")
        
        return report
    
    def create_layer_comparison_plot(self, layer_metrics: Dict):
        """Create comparison plot across layers"""
        
        print("Creating layer comparison plot...")
        
        layers = sorted(layer_metrics.keys())
        metrics_to_plot = ['overall_accuracy', 'auc', 'cohen_d']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics_to_plot):
            values = [layer_metrics[layer][metric] for layer in layers]
            
            axes[i].bar(layers, values, alpha=0.7, color=['blue', 'green', 'orange'][:len(layers)])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Layer')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(layers[j], v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "layer_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Layer comparison plot saved to {output_path}")
    
    def print_summary(self, report: Dict):
        """Print analysis summary"""
        
        print("\n" + "="*60)
        print("CAV ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Dataset: {report['dataset_info']['activation_dir']}")
        print(f"CAV Model: {report['dataset_info']['cav_path']}")
        print(f"Layers Analyzed: {report['dataset_info']['layers_analyzed']}")
        
        print(f"\nPer-Layer Analysis:")
        print("-" * 40)
        for layer in sorted(report['layer_metrics'].keys()):
            metrics = report['layer_metrics'][layer]
            print(f"Layer {layer}:")
            print(f"  Overall Accuracy: {metrics['overall_accuracy']:.3f}")
            print(f"  AUC: {metrics['auc']:.3f}")
            print(f"  Cohen's D: {metrics['cohen_d']:.3f}")
            print(f"  Positive Mean: {metrics['positive_mean']:.3f} ± {metrics['positive_std']:.3f}")
            print(f"  Negative Mean: {metrics['negative_mean']:.3f} ± {metrics['negative_std']:.3f}")
            print()
        
        # Find best layer
        best_layer = max(report['layer_metrics'].keys(), 
                        key=lambda l: report['layer_metrics'][l]['overall_accuracy'])
        print(f"Best Performing Layer: {best_layer}")
        print(f"Best Accuracy: {report['layer_metrics'][best_layer]['overall_accuracy']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Visualize CAV concept learning")
    parser.add_argument("--activation_dir", required=True, help="Directory containing activation files")
    parser.add_argument("--cav_path", required=True, help="Path to trained CAV file (.pkl or .pt)")
    parser.add_argument("--output_dir", default="cav_visualization", help="Output directory for visualizations")
    parser.add_argument("--concept", default="harmful_symbols", help="Concept name for labeling")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = CAVVisualizer(args.activation_dir, args.cav_path, args.output_dir)
    
    # Create comprehensive report
    report = visualizer.create_comprehensive_report()
    
    # Print summary
    visualizer.print_summary(report)
    
    print(f"\n✅ CAV visualization complete!")
    print(f"📊 Results saved to: {args.output_dir}")
    print(f"📈 Check the following files:")
    print(f"   - concept_separation_*.png: 2D visualization of concept clusters")
    print(f"   - concept_probabilities_*.png: Probability distributions")
    print(f"   - layer_comparison.png: Performance across layers")
    print(f"   - cav_analysis_report.json: Detailed metrics")


if __name__ == "__main__":
    main()
