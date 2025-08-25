import os
import argparse
import numpy as np
import torch
import json
import yaml
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm


class GCAVTrainer:
    """Trains CAV classifiers and computes concept direction vectors"""
    
    def __init__(self, regularization=0.01, test_size=0.33, random_state=42):
        self.regularization = regularization
        self.test_size = test_size
        self.random_state = random_state
        self.cavs = {}  # layer -> {w, b, v, accuracy, auc}
    
    def load_activations(self, activation_dir):
        """Load positive and negative activations from directory"""
        activation_dir = Path(activation_dir)
        
        if not activation_dir.exists():
            raise FileNotFoundError(f"Activation directory not found: {activation_dir}")
        
        # Find all layer files
        pos_files = list(activation_dir.glob("layer_*_positive.npy"))
        neg_files = list(activation_dir.glob("layer_*_negative.npy"))
        
        if not pos_files or not neg_files:
            raise ValueError(f"No activation files found in {activation_dir}")
        
        activations = {}
        
        for pos_file in pos_files:
            # Extract layer number
            layer_num = int(pos_file.stem.split("_")[1])
            
            # Find corresponding negative file
            neg_file = activation_dir / f"layer_{layer_num}_negative.npy"
            
            if not neg_file.exists():
                continue
            
            # Load activations
            try:
                pos_acts = np.load(pos_file)
                neg_acts = np.load(neg_file)
                
                if pos_acts.shape[0] == 0 or neg_acts.shape[0] == 0:
                    continue
                
                activations[layer_num] = {
                    'positive': pos_acts,
                    'negative': neg_acts
                }
                
                print(f"Loaded layer {layer_num}: pos={pos_acts.shape}, neg={neg_acts.shape}")
                
            except Exception as e:
                continue
        
        return activations
    
    def train_cav_for_layer(self, layer_num, pos_acts, neg_acts):
        """Train CAV classifier for a specific layer"""
        
        X_pos = pos_acts.reshape(pos_acts.shape[0], -1)
        X_neg = neg_acts.reshape(neg_acts.shape[0], -1)
        
        # Balance dataset (use minimum size)
        min_size = min(X_pos.shape[0], X_neg.shape[0])
        X_pos = X_pos[:min_size]
        X_neg = X_neg[:min_size]

        X = np.vstack([X_pos, X_neg])
        y = np.hstack([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])
        
        print(f"Layer {layer_num}: Training on {X.shape[0]} samples, {X.shape[1]} features")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        clf = LogisticRegression(
            C=1.0/self.regularization,  # C = 1/alpha in sklearn
            random_state=self.random_state,
            max_iter=1000,
            solver='liblinear'
        )
        
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        w = clf.coef_[0]
        b = clf.intercept_[0]
        v = w / np.linalg.norm(w)
        
        cav_info = {
            'w': w,
            'b': b, 
            'v': v,
            'accuracy': accuracy,
            'auc': auc,
            'n_features': X.shape[1],
            'n_train': X_train.shape[0],
            'n_test': X_test.shape[0]
        }
        
        print(f"Layer {layer_num}: Accuracy={accuracy:.3f}, AUC={auc:.3f}")

        return cav_info

    def train_all_layers(self, activations):
        """Train CAV for all layers"""

        print(f"Training CAVs for {len(activations)} layers...")

        for layer_num in sorted(activations.keys()):
            
            pos_acts = activations[layer_num]['positive']
            neg_acts = activations[layer_num]['negative']
            
            try:
                cav_info = self.train_cav_for_layer(layer_num, pos_acts, neg_acts)
                self.cavs[layer_num] = cav_info
            except Exception as e:
                continue
        
        return self.cavs
    
    def save_cavs(self, output_path):
        """Save trained CAVs to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as both pickle (for Python) and torch (for inference)
        with open(output_path, 'wb') as f:
            pickle.dump(self.cavs, f)
        
        # Also save as torch tensors for faster loading
        torch_cavs = {}
        for layer, cav_info in self.cavs.items():
            torch_cavs[layer] = {
                'w': torch.tensor(cav_info['w'], dtype=torch.float32),
                'b': torch.tensor(cav_info['b'], dtype=torch.float32),
                'v': torch.tensor(cav_info['v'], dtype=torch.float32),
                'accuracy': cav_info['accuracy'],
                'auc': cav_info['auc']
            }
        
        torch_path = output_path.with_suffix('.pt')
        torch.save(torch_cavs, torch_path)

        print(f"CAVs saved to {output_path} and {torch_path}")

        return output_path

    def print_summary(self):
        """Print training summary"""
        if not self.cavs:
            return
        
        print("\n" + "="*50)
        print("CAV Training Summary")
        print("="*50)
        
        for layer in sorted(self.cavs.keys()):
            cav = self.cavs[layer]
            print(f"Layer {layer:2d}: Acc={cav['accuracy']:.3f}, AUC={cav['auc']:.3f}, "
                  f"Features={cav['n_features']}")
        
        # Find best layer
        best_layer = max(self.cavs.keys(), key=lambda l: self.cavs[l]['accuracy'])
        best_acc = self.cavs[best_layer]['accuracy']
        print(f"\nBest layer: {best_layer} (accuracy={best_acc:.3f})")


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train CAV classifiers")
    parser.add_argument("--config", help="Path to YAML config file")
    # Keep individual arguments for backward compatibility
    parser.add_argument("--input_dir", help="Directory with activation files")
    parser.add_argument("--output_dir", help="Output directory for CAVs")
    parser.add_argument("--concept", help="Concept name (for output filename)")
    parser.add_argument("--regularization", type=float, help="L2 regularization")
    parser.add_argument("--test_size", type=float, help="Test split ratio")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Use config values, with args overriding if provided
        input_dir = args.input_dir or f"{config['output']['activations_dir']}/{config['data']['concept']}"
        output_dir = args.output_dir or config['output']['cavs_dir']
        concept = args.concept or config['data']['concept']
        regularization = args.regularization or config['training']['regularization']
        test_size = args.test_size or config['training']['test_size']
    else:
        # Use defaults if no config
        input_dir = args.input_dir
        output_dir = args.output_dir or "../artifacts/cavs"
        concept = args.concept or ""
        regularization = args.regularization or 0.01
        test_size = args.test_size or 0.33

    trainer = GCAVTrainer(
        regularization=regularization,
        test_size=test_size
    )

    print(f"Loading activations from {input_dir}...")
    try:
        activations = trainer.load_activations(input_dir)
    except Exception as e:
        return

    if not activations:
        return

    cavs = trainer.train_all_layers(activations)

    if not cavs:
        return

    trainer.print_summary()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    concept_name = concept if concept else Path(input_dir).name
    output_file = output_path / f"{concept_name}_cavs.pkl"
    
    trainer.save_cavs(output_file)

    metadata = {
        "concept": concept_name,
        "input_dir": str(args.input_dir),
        "layers": list(cavs.keys()),
        "best_layer": max(cavs.keys(), key=lambda l: cavs[l]['accuracy']),
        "best_accuracy": max(cav['accuracy'] for cav in cavs.values()),
        "regularization": args.regularization,
        "test_size": args.test_size
    }
    
    with open(output_path / f"{concept_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTraining complete! Best layer: {metadata['best_layer']} "
          f"(accuracy: {metadata['best_accuracy']:.3f})")


if __name__ == "__main__":
    main()
