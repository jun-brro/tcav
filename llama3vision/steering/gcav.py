import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle


class GCAVSteering:
    """Implements GCAV steering mechanism with closed-form intervention"""
    
    def __init__(self, cav_path: str, target_layers: List[int], device: str = "cuda:0"):
        self.device = device
        self.target_layers = target_layers
        self.cavs = self.load_cavs(cav_path)
        self.hooks = []
        self.intervention_stats = []  # Track interventions for analysis
        
    def load_cavs(self, cav_path: str) -> Dict:
        """Load pre-trained CAVs"""
        cav_path = Path(cav_path)
        
        if cav_path.suffix == '.pt':
            cavs = torch.load(cav_path, map_location=self.device)
        else:
            with open(cav_path, 'rb') as f:
                cavs = pickle.load(f)
            
    
            for layer in cavs:
                for key in ['w', 'b', 'v']:
                    if not isinstance(cavs[layer][key], torch.Tensor):
                        cavs[layer][key] = torch.tensor(
                            cavs[layer][key], dtype=torch.float32, device=self.device
                        )
        return cavs
    
    def compute_concept_probability(self, activation: torch.Tensor, layer: int) -> torch.Tensor:
        """Compute P_d(e) = σ(w^T * e + b)"""
        if layer not in self.cavs:
            return torch.tensor(0.5, device=self.device, dtype=activation.dtype)

        cav = self.cavs[layer]
        # Ensure CAV tensors match activation dtype (e.g., float16 for half-precision models)
        activation_dtype = activation.dtype
        w = cav['w'].to(device=self.device, dtype=activation_dtype)
        b = cav['b'].to(device=self.device, dtype=activation_dtype)
        
        # Apply same preprocessing as during training: mean across sequence dimension
        if activation.dim() > 2:
            # Take mean across sequence dimension (dim=1), same as in dump_activations.py
            activation_pooled = activation.mean(dim=1)  # [batch, features]
        else:
            activation_pooled = activation
        
        # Flatten activation for dot product
        e_flat = activation_pooled.view(activation_pooled.size(0), -1)  # [batch, features]
        
        # Compute logits: w^T * e + b
        logits = torch.matmul(e_flat, w) + b  # [batch]
        
        # Convert to probability
        prob = torch.sigmoid(logits)
        return prob
    
    def compute_closed_form_epsilon(
        self, 
        activation: torch.Tensor, 
        layer: int, 
        target_prob: float = 0.7,
        direction: str = "suppress"
    ) -> torch.Tensor:
        """
        Compute closed-form intervention strength ε
        
        For amplification: ε = max(0, (s_0 - b - w^T*e) / ||w||)
        For suppression: ε = max(0, (b + w^T*e - s_0) / ||w||)
        
        where s_0 = σ^(-1)(p_0) = logit(p_0)
        """
        if layer not in self.cavs:
            return torch.zeros(activation.size(0), device=self.device, dtype=activation.dtype)
        
        cav = self.cavs[layer]
        # Ensure CAV tensors match activation dtype (e.g., float16 for half-precision models)
        activation_dtype = activation.dtype
        w = cav['w'].to(device=self.device, dtype=activation_dtype)
        b = cav['b'].to(device=self.device, dtype=activation_dtype)
        
        # Apply same preprocessing as during training: mean across sequence dimension
        if activation.dim() > 2:
            # Take mean across sequence dimension (dim=1), same as in dump_activations.py
            activation_pooled = activation.mean(dim=1)  # [batch, features]
        else:
            activation_pooled = activation
        
        # Flatten activation
        e_flat = activation_pooled.view(activation_pooled.size(0), -1)  # [batch, features]
        
        # Compute current logit: w^T * e + b
        current_logit = torch.matmul(e_flat, w) + b  # [batch]
        
        # Target logit: s_0 = logit(p_0)
        target_logit = torch.logit(torch.tensor(target_prob, device=self.device, dtype=activation_dtype))
        
        # Compute epsilon based on direction
        w_norm = torch.norm(w)
        
        if direction == "amplify":
            # Want to increase concept probability
            epsilon = torch.clamp(
                (target_logit - current_logit) / w_norm, 
                min=0.0
            )
        elif direction == "suppress":
            # Want to decrease concept probability  
            epsilon = torch.clamp(
                (current_logit - target_logit) / w_norm,
                min=0.0
            )
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        return epsilon
    
    def apply_intervention(
        self,
        activation: torch.Tensor,
        layer: int,
        target_prob: float = 0.7,
        direction: str = "suppress"
    ) -> torch.Tensor:
        """Apply GCAV intervention: e' = e + ε * v"""
        
        if layer not in self.cavs:
            return activation  # No intervention
        
        # Compute intervention strength
        epsilon = self.compute_closed_form_epsilon(
            activation, layer, target_prob, direction
        )
        
        # Get normalized CAV direction
        # Ensure v matches activation dtype (e.g., float16 for half-precision models)
        activation_dtype = activation.dtype
        v = self.cavs[layer]['v'].to(device=self.device, dtype=activation_dtype)
        
        # Reshape v to match activation dimensions
        # activation: [batch, seq_len, hidden_dim]
        # v: [hidden_dim] -> need to broadcast properly
        
        original_shape = activation.shape
        
        # Apply same preprocessing as during training: mean across sequence dimension
        if activation.dim() > 2:
            # Take mean across sequence dimension (dim=1), same as in dump_activations.py
            activation_pooled = activation.mean(dim=1)  # [batch, features]
        else:
            activation_pooled = activation
            
        e_flat = activation_pooled.view(activation_pooled.size(0), -1)  # [batch, features]
        
        # Ensure v matches flattened feature dimension
        if v.size(0) != e_flat.size(1):
            # Handle dimension mismatch (shouldn't happen with proper training)
            print(f"Warning: CAV dimension mismatch. Expected {e_flat.size(1)}, got {v.size(0)}")
            return activation
        
        # Apply intervention
        if direction == "suppress":
            epsilon = -epsilon  # Negative direction for suppression
        
        # Broadcast epsilon and v for batch computation
        epsilon_expanded = epsilon.unsqueeze(1)  # [batch, 1]
        v_expanded = v.unsqueeze(0)  # [1, features]
        
        intervention = epsilon_expanded * v_expanded  # [batch, features]
        
        # If original activation was 3D (has sequence dimension), broadcast intervention
        if len(original_shape) == 3:
            batch_size, seq_len, hidden_dim = original_shape
            # Reshape intervention to match hidden dimension
            intervention_reshaped = intervention.view(batch_size, 1, hidden_dim)
            # Broadcast across sequence dimension
            intervention_broadcasted = intervention_reshaped.expand(batch_size, seq_len, hidden_dim)
            # Apply intervention to original activation
            activation_modified = activation + intervention_broadcasted
        else:
            # For 2D activations, apply directly
            e_modified = e_flat + intervention
            activation_modified = e_modified.view(original_shape)
        
        # Log intervention statistics
        self.intervention_stats.append({
            'layer': layer,
            'epsilon_mean': epsilon.mean().item(),
            'epsilon_std': epsilon.std().item() if epsilon.numel() > 1 else 0.0,
            'direction': direction,
            'target_prob': target_prob,
            'batch_size': activation.size(0)
        })
        
        return activation_modified
    
    def create_steering_hook(
        self, 
        layer: int, 
        target_prob: float = 0.7,
        direction: str = "suppress"
    ):
        """Create forward hook for steering"""
        
        def hook_fn(module, input, output):
            # Handle tuple outputs (hidden_states, attention_weights, ...)
            if isinstance(output, tuple):
                # Apply intervention to the first element (hidden states)
                modified_hidden_states = self.apply_intervention(
                    output[0], layer, target_prob, direction
                )
                # Return tuple with modified hidden states and other unchanged elements
                return (modified_hidden_states,) + output[1:]
            else:
                # Apply intervention to tensor output directly
                modified_output = self.apply_intervention(
                    output, layer, target_prob, direction
                )
                return modified_output
        
        return hook_fn
    
    def register_hooks(self, model, config: Dict):
        """Register steering hooks on target layers"""
        
        # Clear existing hooks
        self.remove_hooks()
        
        for layer_idx in self.target_layers:
            if layer_idx in self.cavs:
                
                # Get concept configuration
                direction = config.get('direction', 'suppress')
                strength = config.get('strength', 0.7)
                
                # Create and register hook
                hook_fn = self.create_steering_hook(layer_idx, strength, direction)
                
                # Register on the appropriate layer
                # Handle different model architectures (MllamaForConditionalGeneration vs others)
                layer_module = None
                
                # For Llama3Vision (MllamaForConditionalGeneration)
                if hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers') and layer_idx < len(model.language_model.model.layers):
                    # Access: model.language_model.model.layers[idx]
                    layer_module = model.language_model.model.layers[layer_idx]
                elif hasattr(model.language_model, 'layers') and layer_idx < len(model.language_model.layers):
                    # Direct access for some architectures
                    layer_module = model.language_model.layers[layer_idx]
                elif hasattr(model, 'model') and hasattr(model.model, 'layers') and layer_idx < len(model.model.layers):
                    # Alternative access pattern
                    layer_module = model.model.layers[layer_idx]
                
                if layer_module is not None:
                    hook = layer_module.register_forward_hook(hook_fn)
                    self.hooks.append(hook)
                    print(f"Registered GCAV hook on layer {layer_idx} ({direction}, {strength})")
                else:
                    print(f"Warning: Layer {layer_idx} not found in model architecture")
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_intervention_stats(self) -> Dict:
        """Get statistics about recent interventions"""
        if not self.intervention_stats:
            return {}
        
        stats = {}
        for stat in self.intervention_stats[-10:]:  # Last 10 interventions
            layer = stat['layer']
            if layer not in stats:
                stats[layer] = []
            stats[layer].append(stat)
        
        # Compute aggregated statistics
        summary = {}
        for layer, layer_stats in stats.items():
            epsilon_values = [s['epsilon_mean'] for s in layer_stats]
            summary[layer] = {
                'avg_epsilon': np.mean(epsilon_values),
                'max_epsilon': np.max(epsilon_values),
                'interventions_count': len(layer_stats)
            }
        
        return summary
    
    def reset_stats(self):
        """Reset intervention statistics"""
        self.intervention_stats = []


class MultiConceptSteering(GCAVSteering):
    """Handle simultaneous steering of multiple concepts"""
    
    def __init__(self, concept_cavs: Dict[str, str], target_layers: List[int], device: str = "cuda:0"):
        self.device = device
        self.target_layers = target_layers
        self.concept_cavs = {}
        
        # Load CAVs for each concept
        for concept_name, cav_path in concept_cavs.items():
            self.concept_cavs[concept_name] = self.load_cavs(cav_path)
        
        self.hooks = []
        self.intervention_stats = []
    
    def apply_multi_concept_intervention(
        self,
        activation: torch.Tensor, 
        layer: int,
        concept_configs: Dict[str, Dict]
    ) -> torch.Tensor:
        """Apply simultaneous multi-concept steering"""
        
        # This would implement the linear constraint optimization from the paper
        # For simplicity, we'll apply sequential interventions here
        # In practice, you'd want to solve the optimization problem:
        # min Σ|ε_i| + Σ|δ_j| subject to probability constraints
        
        modified_activation = activation
        
        for concept_name, config in concept_configs.items():
            if concept_name in self.concept_cavs and layer in self.concept_cavs[concept_name]:
                # Apply intervention for this concept
                direction = config.get('direction', 'suppress')
                target_prob = config.get('strength', 0.7)
                
                # Use the concept-specific CAV
                temp_steerer = GCAVSteering.__new__(GCAVSteering)
                temp_steerer.cavs = self.concept_cavs[concept_name] 
                temp_steerer.device = self.device
                temp_steerer.intervention_stats = []
                
                modified_activation = temp_steerer.apply_intervention(
                    modified_activation, layer, target_prob, direction
                )
        
        return modified_activation


def create_gcav_steerer(config_path: str, device: str = "cuda:0") -> GCAVSteering:
    """Factory function to create GCAV steerer from config"""
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    gcav_config = config.get('gcav', {})
    if not gcav_config.get('enabled', False):
        print("GCAV not enabled in config")
        return None
    
    # Find CAV files (simplified path resolution)
    cav_dir = Path(config_path).parent.parent / "artifacts" / "cavs"
    cav_files = list(cav_dir.glob("*_cavs.pt"))
    
    if not cav_files:
        print(f"No CAV files found in {cav_dir}")
        return None
    
    # Use first CAV file found (in practice, you'd select based on concept)
    cav_path = str(cav_files[0])
    target_layers = gcav_config.get('target_layers', [10, 14, 18])
    
    steerer = GCAVSteering(cav_path, target_layers, device)
    print(f"Created GCAV steerer with {len(target_layers)} target layers")
    
    return steerer
