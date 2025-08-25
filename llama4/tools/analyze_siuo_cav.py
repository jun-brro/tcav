import argparse
import json
import numpy as np
import torch
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')


class SIUOCAVAnalyzer:
    
    def __init__(self, siuo_data_path: str, cav_path: str, model_name: str = "meta-llama/Llama-Guard-4-12B", device: str = "cuda:0"):
        self.siuo_data_path = Path(siuo_data_path)
        self.cav_path = Path(cav_path)
        self.device = device
        self.model_name = model_name
        
        # Load model and processor from local path
        local_path = '/scratch2/pljh0906/models/Llama-Guard-4-12B'
        print(f"Loading Llama Guard 4 from {local_path} on {device}...")
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            local_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            local_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Shim: ensure config objects support dict-like get() to avoid '... has no attribute get' errors
        def _ensure_config_get(cfg):
            if cfg is None:
                return
            if not hasattr(cfg, 'get'):
                import types
                def _get(self, key, default=None):
                    return getattr(self, key, default)
                cfg.get = types.MethodType(_get, cfg)
        
        _ensure_config_get(getattr(self.model, 'config', None))
        _ensure_config_get(getattr(self.model.config, 'text_config', None))
        _ensure_config_get(getattr(self.model.config, 'vision_config', None))
        
        # Load pre-trained CAV
        self.cavs = self.load_cavs()
        
        print(f"SIUO CAV Analyzer initialized!")
        
    def load_cavs(self):
        """Load pre-trained CAVs"""
        print(f"Loading pre-trained CAVs from {self.cav_path}")
        
        if self.cav_path.suffix == '.pkl':
            with open(self.cav_path, 'rb') as f:
                cavs = pickle.load(f)
        elif self.cav_path.suffix == '.pt':
            cavs = torch.load(self.cav_path, map_location='cpu')
        else:
            raise ValueError(f"Unsupported CAV file format: {self.cav_path.suffix}")
        
        print(f"Loaded CAVs for layers: {list(cavs.keys())}")
        return cavs
    
    def load_siuo_data(self, data_type: str = "gen", max_samples: int = None):
        """Load SIUO dataset"""
        data_file = self.siuo_data_path / f"siuo_{data_type}.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"SIUO data file not found: {data_file}")
        
        print(f"Loading SIUO data from {data_file}")
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
            
        print(f"Loaded {len(data)} SIUO samples")
        
        # Process data
        processed_data = []
        for item in data:
            image_path = self.siuo_data_path / "images" / item["image"]
            processed_data.append({
                "id": item["question_id"],
                "image_path": str(image_path),
                "text": item["question"],
                "category": item.get("category", "unknown")
            })
        
        return processed_data
    
    def extract_siuo_activations(self, siuo_data, target_layers):
        """Extract activations for SIUO data using the pre-trained model"""
        
        activations = {layer: [] for layer in target_layers}
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                try:
                    # Some layers return tuples (e.g., (hidden_states, ...))
                    if isinstance(output, (tuple, list)):
                        output = output[0]
                    # Apply same preprocessing as training: mean across sequence dimension
                    if output.dim() > 2:
                        pooled_output = output.mean(dim=1)  # [batch, features]
                    else:
                        pooled_output = output
                    activations[layer_idx].append(pooled_output.detach().cpu())
                except Exception as e:
                    print(f"Error in hook for layer {layer_idx}: {e}")
            return hook
        
        # Register hooks
        hooks = []
        for layer in target_layers:
            if hasattr(self.model.language_model, 'layers') and layer < len(self.model.language_model.layers):
                # Direct access for Qwen2-based models
                hook = self.model.language_model.layers[layer].register_forward_hook(make_hook(layer))
                hooks.append(hook)
            elif hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers') and layer < len(self.model.language_model.model.layers):
                # Nested access for other architectures
                hook = self.model.language_model.model.layers[layer].register_forward_hook(make_hook(layer))
                hooks.append(hook)
        
        print(f"Registered hooks for {len(hooks)} layers")
        
        self.model.eval()
        with torch.no_grad():
            for item in tqdm(siuo_data, desc="Extracting SIUO activations"):
                try:
                    # Load image
                    if Path(item['image_path']).exists():
                        image = Image.open(item['image_path'])
                    else:
                        print(f"Image not found: {item['image_path']}")
                        continue
                    
                    # Prepare input with proper image placeholder for Llama Guard 4
                    conversation = [{
                        "role": "user", 
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": item['text']}
                        ]
                    }]
                    
                    try:
                        text_prompt = self.processor.apply_chat_template(
                            conversation, 
                            add_generation_prompt=True, 
                            tokenize=False
                        )
                    except Exception as e:
                        # Fallback: manually add image placeholder
                        print(f"Chat template failed: {e}")
                        text_prompt = f"<image>\n{item['text']}"
                        
                    inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Forward pass to trigger hooks (disable KV cache to avoid DynamicCache errors)
                    _ = self.model(**inputs, use_cache=False)
                    
                except Exception as e:
                    print(f"Error processing {item['id']}: {e}")
                    continue
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Convert to numpy
        processed_activations = {}
        for layer in target_layers:
            if activations[layer]:
                try:
                    # Convert bfloat16 to float32 before numpy conversion
                    layer_acts = [act.cpu().float().numpy() for act in activations[layer]]
                    processed_activations[layer] = np.vstack(layer_acts)
                    print(f"SIUO Layer {layer}: {processed_activations[layer].shape}")
                except Exception as e:
                    print(f"Error processing activations for layer {layer}: {e}")
                    processed_activations[layer] = np.array([])
            else:
                print(f"No activations captured for layer {layer}")
                processed_activations[layer] = np.array([])
        
        return processed_activations
    
    def compute_siuo_concept_probabilities(self, siuo_activations, layer):
        """Compute concept probabilities for SIUO data"""
        
        if layer not in self.cavs or layer not in siuo_activations:
            return np.array([])
        
        # Get CAV parameters
        cav = self.cavs[layer]
        w = cav['w'] if isinstance(cav['w'], np.ndarray) else cav['w'].numpy()
        b = cav['b'] if isinstance(cav['b'], np.ndarray) else float(cav['b'])
        
        # Get SIUO activations
        activations = siuo_activations[layer]
        if len(activations) == 0:
            return np.array([])
        
        # Reshape activations for dot product
        acts_flat = activations.reshape(activations.shape[0], -1)
        
        # Compute probabilities using sigmoid(w^T * x + b)
        logits = np.dot(acts_flat, w) + b
        probs = 1 / (1 + np.exp(-logits))
        
        return probs
    
    def create_combined_visualization(self, siuo_data, siuo_activations, training_activations_path, layer, output_dir):
        """Create visualization combining training data and SIUO data"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training activations
        training_path = Path(training_activations_path)
        train_pos = np.load(training_path / f"layer_{layer}_positive.npy")
        train_neg = np.load(training_path / f"layer_{layer}_negative.npy")
        
        # Get SIUO activations
        siuo_acts = siuo_activations[layer]
        
        if len(siuo_acts) == 0:
            print(f"No SIUO activations found for layer {layer}")
            return None, None
        
        # Combine all activations
        all_activations = np.vstack([
            train_pos.reshape(train_pos.shape[0], -1),
            train_neg.reshape(train_neg.shape[0], -1), 
            siuo_acts.reshape(siuo_acts.shape[0], -1)
        ])
        
        # Create labels
        labels = (
            ['Hateful (Training)'] * len(train_pos) +
            ['Benign (Training)'] * len(train_neg) +
            [f"SIUO-{item['category']}" for item in siuo_data[:len(siuo_acts)]]
        )
        
        # Apply dimensionality reduction with FIXED seed for consistency  
        print(f"Applying t-SNE to {all_activations.shape[0]} samples...")
        perplexity = min(30, max(5, len(all_activations)//4))
        print(f"🎯 Using fixed t-SNE parameters: perplexity={perplexity}, random_state=42")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                   learning_rate=200, max_iter=1000)
        reduced_acts = tsne.fit_transform(all_activations)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Define colors for different types
        color_map = {
            'Hateful (Training)': 'red',
            'Benign (Training)': 'blue',
        }
        
        # Assign colors to SIUO categories
        siuo_categories = list(set([item['category'] for item in siuo_data]))
        siuo_colors = plt.cm.Set3(np.linspace(0, 1, len(siuo_categories)))
        for i, cat in enumerate(siuo_categories):
            color_map[f'SIUO-{cat}'] = siuo_colors[i]
        
        # Plot each group
        for label_type in set(labels):
            mask = [l == label_type for l in labels]
            indices = np.where(mask)[0]
            
            if 'Training' in label_type:
                plt.scatter(reduced_acts[indices, 0], reduced_acts[indices, 1], 
                           c=color_map[label_type], label=label_type, alpha=0.7, s=50)
            else:
                plt.scatter(reduced_acts[indices, 0], reduced_acts[indices, 1], 
                           c=color_map[label_type], label=label_type, alpha=0.8, s=60, marker='^')
        
        plt.title(f'CAV Space Visualization - Layer {layer}\n(Training Data + SIUO Dataset)', fontsize=14)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"siuo_cav_analysis_layer{layer}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved combined visualization to {output_path}")
        
        return reduced_acts, labels
    
    def create_enhanced_combined_visualization(self, siuo_data, siuo_activations, training_activations_path, layer, output_dir):
        """Create enhanced visualization with CAV probability color mapping"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training activations
        training_path = Path(training_activations_path)
        train_pos = np.load(training_path / f"layer_{layer}_positive.npy")
        train_neg = np.load(training_path / f"layer_{layer}_negative.npy")
        
                # Get SIUO activations
        siuo_acts = siuo_activations[layer]
        
        if len(siuo_acts) == 0:
            print(f"No SIUO activations found for layer {layer}")
            return None, None, None

        # Compute probabilities for all data
        # Training probabilities
        cav = self.cavs[layer]
        w = cav['w'] if isinstance(cav['w'], np.ndarray) else cav['w'].numpy()
        b = cav['b'] if isinstance(cav['b'], np.ndarray) else float(cav['b'])
        
        train_pos_flat = train_pos.reshape(train_pos.shape[0], -1)
        train_neg_flat = train_neg.reshape(train_neg.shape[0], -1)
        siuo_flat = siuo_acts.reshape(siuo_acts.shape[0], -1)
        
        train_pos_probs = 1 / (1 + np.exp(-(np.dot(train_pos_flat, w) + b)))
        train_neg_probs = 1 / (1 + np.exp(-(np.dot(train_neg_flat, w) + b)))
        siuo_probs = 1 / (1 + np.exp(-(np.dot(siuo_flat, w) + b)))
        
        # Combine all activations and probabilities
        all_activations = np.vstack([train_pos_flat, train_neg_flat, siuo_flat])
        all_probs = np.hstack([train_pos_probs, train_neg_probs, siuo_probs])
        
        # Create labels for data types
        type_labels = (
            ['Training-Hateful'] * len(train_pos) +
            ['Training-Benign'] * len(train_neg) +
            ['SIUO'] * len(siuo_acts)
        )
        
        # Apply dimensionality reduction with FIXED seed for consistency
        print(f"Applying t-SNE to {all_activations.shape[0]} samples...")
        # Use fixed parameters to ensure reproducible results when SIUO data is added
        perplexity = min(30, max(5, len(all_activations)//4))
        print(f"🎯 Using fixed t-SNE parameters: perplexity={perplexity}, random_state=42")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                   learning_rate=200, max_iter=1000)
        reduced_acts = tsne.fit_transform(all_activations)
        
        # Create enhanced visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        ### Plot 1: Probability-based color mapping (all points)
        scatter1 = ax1.scatter(reduced_acts[:, 0], reduced_acts[:, 1], 
                             c=all_probs, cmap='RdYlBu_r', 
                             alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('CAV Probability (Higher = More Harmful)', fontsize=12)
        
        # Add iso-probability contours
        self._add_approximate_contours(ax1, reduced_acts, all_probs, levels=[0.3, 0.5, 0.7])
        
        ax1.set_title(f'Enhanced CAV Space - All Data with Probability Mapping\nLayer {layer}', fontsize=14)
        ax1.set_xlabel('t-SNE Component 1')
        ax1.set_ylabel('t-SNE Component 2')
        ax1.grid(True, alpha=0.3)
        
        ### Plot 2: Traditional view with distinct markers for SIUO
        # Plot training data
        train_indices = np.arange(len(train_pos) + len(train_neg))
        ax2.scatter(reduced_acts[train_indices, 0], reduced_acts[train_indices, 1], 
                   c=['red' if i < len(train_pos) else 'blue' for i in range(len(train_indices))], 
                   alpha=0.6, s=40, label='Training Data')
        
        # Plot SIUO data with probability-based colors
        siuo_indices = np.arange(len(train_pos) + len(train_neg), len(all_activations))
        siuo_colors = plt.cm.RdYlBu_r(siuo_probs)
        
        scatter2 = ax2.scatter(reduced_acts[siuo_indices, 0], reduced_acts[siuo_indices, 1], 
                             c=siuo_colors, s=80, marker='^', 
                             edgecolors='black', linewidth=1, alpha=0.8, label='SIUO Data')
        
        # Add legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Hateful (Training)')
        blue_patch = mpatches.Patch(color='blue', label='Benign (Training)')
        triangle_patch = mpatches.Patch(color='gray', label='SIUO (Color = Probability)')
        ax2.legend(handles=[red_patch, blue_patch, triangle_patch])
        
        ax2.set_title(f'Traditional View + SIUO Data\nLayer {layer}', fontsize=14)
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / f"enhanced_siuo_cav_analysis_layer{layer}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved enhanced combined visualization to {output_path}")
        
        return reduced_acts, all_probs, type_labels
    
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
                               linewidths=2, alpha=0.8, linestyles='--')
            
            # Add labels
            ax.clabel(contour, inline=True, fontsize=9, fmt='%.1f')
            
        except ImportError:
            print("SciPy not available for contour plotting")
        except Exception as e:
            print(f"Could not add contours: {e}")
    
    def create_concept_probability_analysis(self, siuo_data, siuo_activations, layer, output_dir):
        """Analyze SIUO concept probabilities by category"""
        
        output_dir = Path(output_dir)
        
        # Compute concept probabilities
        probs = self.compute_siuo_concept_probabilities(siuo_activations, layer)
        if len(probs) == 0:
            return
        
        # Group by category
        categories = [item['category'] for item in siuo_data[:len(probs)]]
        
        # Create probability analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot by category
        category_probs = {}
        for cat, prob in zip(categories, probs):
            if cat not in category_probs:
                category_probs[cat] = []
            category_probs[cat].append(prob)
        
        ax1.boxplot(category_probs.values(), labels=category_probs.keys())
        ax1.set_title(f'Concept Probability by Category - Layer {layer}')
        ax1.set_ylabel('Harmful Concept Probability')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Histogram of all probabilities
        ax2.hist(probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title(f'Distribution of Concept Probabilities - Layer {layer}')
        ax2.set_xlabel('Harmful Concept Probability')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = output_dir / f"siuo_concept_probs_layer{layer}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved concept probability analysis to {output_path}")
        
        return category_probs
    
    def analyze_samples_with_probabilities(self, siuo_data, probs):
        """Analyze individual samples with their concept probabilities and group them"""
        
        if len(probs) == 0:
            return {}
        
        # Ensure we have matching data
        num_samples = min(len(siuo_data), len(probs))
        
        # Define thresholds for grouping
        HIGH_THRESHOLD = 0.7  # Group A: clear/obvious danger
        LOW_THRESHOLD = 0.15  # Group B: surface safety
        # Group C: ambiguous area is between LOW and HIGH
        
        detailed_samples = {
            'group_A_high_risk': [],      # prob >= 0.7
            'group_B_low_risk': [],       # prob <= 0.15  
            'group_C_medium_risk': [],    # 0.15 < prob < 0.7
            'all_samples': []
        }
        
        # Analyze each sample
        for i in range(num_samples):
            sample_info = {
                'index': i,
                'id': siuo_data[i]['id'],
                'text': siuo_data[i]['text'],
                'image_path': siuo_data[i]['image_path'],
                'category': siuo_data[i]['category'],
                'harmful_probability': float(probs[i])
            }
            
            # Add to appropriate group
            if probs[i] >= HIGH_THRESHOLD:
                detailed_samples['group_A_high_risk'].append(sample_info)
            elif probs[i] <= LOW_THRESHOLD:
                detailed_samples['group_B_low_risk'].append(sample_info)
            else:
                detailed_samples['group_C_medium_risk'].append(sample_info)
            
            # Add to all samples
            detailed_samples['all_samples'].append(sample_info)
        
        # Add group statistics
        detailed_samples['group_statistics'] = {
            'group_A_count': len(detailed_samples['group_A_high_risk']),
            'group_B_count': len(detailed_samples['group_B_low_risk']),
            'group_C_count': len(detailed_samples['group_C_medium_risk']),
            'total_samples': num_samples,
            'group_A_mean_prob': np.mean([s['harmful_probability'] for s in detailed_samples['group_A_high_risk']]) if detailed_samples['group_A_high_risk'] else 0.0,
            'group_B_mean_prob': np.mean([s['harmful_probability'] for s in detailed_samples['group_B_low_risk']]) if detailed_samples['group_B_low_risk'] else 0.0,
            'group_C_mean_prob': np.mean([s['harmful_probability'] for s in detailed_samples['group_C_medium_risk']]) if detailed_samples['group_C_medium_risk'] else 0.0,
            'thresholds': {
                'high_threshold': HIGH_THRESHOLD,
                'low_threshold': LOW_THRESHOLD
            }
        }
        
        return detailed_samples
    
    def analyze_siuo_dataset(self, data_type="gen", max_samples=100, target_layers=[10, 14, 18], 
                           training_activations_path="../artifacts/activations/harmful_symbols",
                           output_dir="siuo_cav_analysis"):
        """Complete SIUO analysis pipeline"""
        
        print("="*60)
        print("SIUO CAV ANALYSIS PIPELINE")
        print("="*60)
        
        # Load SIUO data
        siuo_data = self.load_siuo_data(data_type, max_samples)
        
        # Extract activations
        print("Extracting SIUO activations...")
        siuo_activations = self.extract_siuo_activations(siuo_data, target_layers)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze each layer
        results = {}
        for layer in target_layers:
            print(f"\nAnalyzing layer {layer}...")
            
            # Check if we have activations for this layer
            if layer not in siuo_activations or len(siuo_activations[layer]) == 0:
                print(f"Skipping layer {layer} - no activations found")
                continue
                
            # Combined visualization (original)
            reduced_acts, labels = self.create_combined_visualization(
                siuo_data, siuo_activations, training_activations_path, layer, output_dir
            )
            
            # Skip enhanced visualization if no activations
            if reduced_acts is None:
                print(f"Skipping enhanced visualization for layer {layer} - no data")
                continue
                
            # Enhanced visualization with probability mapping
            enhanced_acts, enhanced_probs, type_labels = self.create_enhanced_combined_visualization(
                siuo_data, siuo_activations, training_activations_path, layer, output_dir
            )
            
            # Concept probability analysis
            category_probs = self.create_concept_probability_analysis(
                siuo_data, siuo_activations, layer, output_dir
            )
            
            # Compute detailed sample analysis with probabilities
            probs = self.compute_siuo_concept_probabilities(siuo_activations, layer)
            detailed_samples = self.analyze_samples_with_probabilities(siuo_data, probs)
            
            results[layer] = {
                'concept_probabilities': category_probs if category_probs else {},
                'num_samples': len(siuo_activations[layer]) if layer in siuo_activations else 0,
                'detailed_samples': detailed_samples
            }
        
        # Save analysis results
        results_file = output_dir / "siuo_analysis_results.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for layer, data in results.items():
                serializable_results[str(layer)] = {
                    'concept_probabilities': {cat: [float(p) for p in probs] 
                                            for cat, probs in data['concept_probabilities'].items()},
                    'num_samples': data['num_samples'],
                    'detailed_samples': data['detailed_samples']  # Already contains serializable data
                }
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nAnalysis complete! Results saved to {output_dir}")
        print(f"Summary statistics saved to {results_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Analyze SIUO dataset using pre-trained CAV")
    parser.add_argument("--siuo_data_path", default="/scratch2/pljh0906/tcav/datasets/SIUO/data", 
                       help="Path to SIUO dataset")
    parser.add_argument("--cav_path", default="artifacts/cavs/harmful_symbols_cavs.pkl",
                       help="Path to pre-trained CAV file")
    parser.add_argument("--training_activations", default="artifacts/activations/harmful_symbols",
                       help="Path to training activations for comparison")
    parser.add_argument("--data_type", default="gen", choices=["gen", "mcqa"],
                       help="SIUO data type to analyze")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum number of SIUO samples to analyze")
    parser.add_argument("--layers", default="10,14,18", 
                       help="Comma-separated layer indices")
    parser.add_argument("--output_dir", default="siuo_cav_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    # Parse layers
    target_layers = [int(x.strip()) for x in args.layers.split(",")]
    
    # Create analyzer
    analyzer = SIUOCAVAnalyzer(
        args.siuo_data_path, 
        args.cav_path,
        device=args.device
    )
    
    # Run analysis
    results = analyzer.analyze_siuo_dataset(
        data_type=args.data_type,
        max_samples=args.max_samples,
        target_layers=target_layers,
        training_activations_path=args.training_activations,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    for layer in target_layers:
        if layer in results:
            # Category-based analysis (original)
            category_probs = results[layer]['concept_probabilities']
            print(f"\nLayer {layer} - Category Analysis:")
            for category, probs in category_probs.items():
                mean_prob = np.mean(probs)
                std_prob = np.std(probs)
                print(f"  {category}: {mean_prob:.3f} ± {std_prob:.3f} (n={len(probs)})")
            
            # Group-based analysis (new)
            if 'detailed_samples' in results[layer]:
                group_stats = results[layer]['detailed_samples']['group_statistics']
                print(f"\nLayer {layer} - Risk Group Analysis:")
                print(f"  🚨 Group A (High Risk, ≥{group_stats['thresholds']['high_threshold']}): {group_stats['group_A_count']} samples, avg={group_stats['group_A_mean_prob']:.3f}")
                print(f"  ✅ Group B (Low Risk, ≤{group_stats['thresholds']['low_threshold']}): {group_stats['group_B_count']} samples, avg={group_stats['group_B_mean_prob']:.3f}")
                print(f"  ⚠️ Group C (Medium Risk): {group_stats['group_C_count']} samples, avg={group_stats['group_C_mean_prob']:.3f}")
                print(f"  📊 Total: {group_stats['total_samples']} samples")


if __name__ == "__main__":
    main()
