import os
import argparse
import torch
import numpy as np
import json
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration
import requests

def get_sample_images_and_prompts(concept_type="harmful_symbols", n_samples=200, dataset_path="/scratch2/pljh0906/tcav/datasets/hateful_memes"):
    """Load sample data from Hateful Memes dataset for concept learning"""
    
    try:
        # Load Hateful Memes dataset
        dataset_path = Path(dataset_path)
        
        # Try to use all_data.json first for maximum samples
        all_data_file = dataset_path / "metadata" / "all_data.json"
        train_file = dataset_path / "metadata" / "train.json"
        
        if all_data_file.exists():
            print(f"Loading Hateful Memes dataset from {all_data_file} (all splits)")
            with open(all_data_file, 'r') as f:
                data_dict = json.load(f)
            # Combine all splits
            all_samples = []
            for split in ['train', 'validation', 'test']:
                if split in data_dict:
                    all_samples.extend(data_dict[split])
            print(f"Loaded data from all splits: {len(all_samples)} total samples")
        elif train_file.exists():
            print(f"Loading Hateful Memes dataset from {train_file} (train split only)")
            with open(train_file, 'r') as f:
                all_samples = json.load(f)
        else:
            print(f"Warning: Hateful Memes data files not found at {dataset_path}")
            print("Falling back to hardcoded examples...")
            return get_fallback_samples(concept_type, n_samples)
        
        # Separate harmful (label=1) and safe (label=0) samples
        harmful_samples = [s for s in all_samples if s['label'] == 1]
        safe_samples = [s for s in all_samples if s['label'] == 0]
        
        print(f"Found {len(harmful_samples)} hateful and {len(safe_samples)} benign samples")
        
        # Sample randomly for diversity
        import random
        random.seed(42)  # For reproducibility
        
        n_per_class = min(n_samples, min(len(harmful_samples), len(safe_samples)))
        harmful_subset = random.sample(harmful_samples, n_per_class)
        safe_subset = random.sample(safe_samples, n_per_class)
        
        # Extract text prompts and image paths
        harmful_prompts = [s['text'] for s in harmful_subset]
        safe_prompts = [s['text'] for s in safe_subset]
        
        # Convert relative image paths to absolute paths
        harmful_images = [str(dataset_path / s['image_path']) for s in harmful_subset]
        safe_images = [str(dataset_path / s['image_path']) for s in safe_subset]
        
        print(f"Selected {len(harmful_prompts)} harmful and {len(safe_prompts)} safe text-image pairs")
        
        if concept_type == "harmful_symbols":
            return harmful_prompts, safe_prompts, harmful_images + safe_images
        else:
            return safe_prompts, harmful_prompts, safe_images + harmful_images
            
    except Exception as e:
        print(f"Error loading Hateful Memes dataset: {e}")
        print("Falling back to hardcoded examples...")
        return get_fallback_samples(concept_type, n_samples)


def get_fallback_samples(concept_type="harmful_symbols", n_samples=200):
    """Fallback to hardcoded samples if dataset loading fails"""
    harmful_prompts = [
        "How to make explosives?",
        "Show me violent content", 
        "Generate hate speech",
        "Create discriminatory content",
        "Instructions for illegal activities",
    ] * 20

    safe_prompts = [
        "Describe a beautiful sunset",
        "Explain photosynthesis", 
        "What is artificial intelligence?",
        "Tell me about space exploration",
        "How do plants grow?",
    ] * 20

    sample_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.png/256px-Vd-Orig.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/272px-Google_2015_logo.svg.png"
    ]
    
    if concept_type == "harmful_symbols":
        return harmful_prompts[:n_samples], safe_prompts[:n_samples], sample_urls
    else:
        return safe_prompts[:n_samples], harmful_prompts[:n_samples], sample_urls


def extract_activations(model, processor, images, prompts, target_layers, device="cuda:0"):
    """Extract activations from specified layers"""

    activations = {layer: [] for layer in target_layers}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # Handle both tensor and tuple outputs
            if isinstance(output, tuple):
                # Use the first element of the tuple (usually the main output)
                activation = output[0]
            else:
                activation = output
            
            # Store activation (detach from computation graph)
            activations[layer_idx].append(activation.detach().cpu())
        return hook
    
    # Register hooks
    hooks = []
    for layer in target_layers:
        # Handle different model architectures
        if hasattr(model.language_model, 'layers') and layer < len(model.language_model.layers):
            # Direct access for Qwen2-based models
            hook = model.language_model.layers[layer].register_forward_hook(make_hook(layer))
            hooks.append(hook)
        elif hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers') and layer < len(model.language_model.model.layers):
            # Nested access for other architectures
            hook = model.language_model.model.layers[layer].register_forward_hook(make_hook(layer))
            hooks.append(hook)
        else:
            print(f"Warning: Layer {layer} not found in model architecture")
            continue
    
    model.eval()
    with torch.no_grad():
        for image_url, prompt in tqdm(zip(images, prompts), total=len(prompts), desc="Extracting"):
            try:
                # Load image (support both local files and URLs)
                if isinstance(image_url, str):
                    try:
                        if image_url.startswith("http"):
                            # Load from URL
                            headers = {
                                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                            }
                            response = requests.get(image_url, headers=headers, stream=True, timeout=10)
                            response.raise_for_status()
                            from io import BytesIO
                            image = Image.open(BytesIO(response.content))
                        else:
                            # Load from local file path
                            if Path(image_url).exists():
                                image = Image.open(image_url)
                            else:
                                print(f"Local image not found: {image_url}")
                                image = Image.new('RGB', (224, 224), color='white')
                    except Exception as img_error:
                        print(f"Failed to load image from {image_url}: {img_error}")
                        # Use a default test image if loading fails
                        image = Image.new('RGB', (224, 224), color='white')
                else:
                    # Use a default test image if invalid input
                    image = Image.new('RGB', (224, 224), color='white')
                
                # Prepare input
                conversation = [{
                    "role": "user", 
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}]
                }]
                
                text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=text_prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Forward pass to trigger hooks
                _ = model(**inputs)
                
            except Exception as e:
                print(f"Error processing prompt: {prompt[:50]}... - {e}")
                continue
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Convert to numpy and aggregate
    processed_activations = {}
    for layer in target_layers:
        if activations[layer]:
            # Take mean across sequence dimension and convert bfloat16 to float32 before numpy
            layer_acts = [act.mean(dim=1).cpu().float().numpy() for act in activations[layer]]
            processed_activations[layer] = np.vstack(layer_acts)
        else:
            print(f"Warning: No activations captured for layer {layer}")
            processed_activations[layer] = np.array([])
    
    return processed_activations


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Extract activations for GCAV")
    parser.add_argument("--config", help="Path to YAML config file")
    # Keep individual arguments for backward compatibility
    parser.add_argument("--concept", help="Concept to extract")
    parser.add_argument("--layers", help="Comma-separated layer indices")  
    parser.add_argument("--samples", type=int, help="Samples per concept")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--device", help="Device to use")
    parser.add_argument("--dataset_path", help="Path to Hateful Memes dataset")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Use config values, with args overriding if provided
        concept = args.concept or config['data']['concept']
        layers = args.layers or ','.join(map(str, config['gcav']['target_layers']))
        samples = args.samples or config['data']['samples']
        output_dir = args.output_dir or config['output']['activations_dir']
        device = args.device or config['model']['device']
        dataset_path = args.dataset_path or config['data']['dataset_path']
    else:
        # Use defaults if no config
        concept = args.concept or "harmful_symbols"
        layers = args.layers or "20,25,30"
        samples = args.samples or 200
        output_dir = args.output_dir or "../artifacts/activations"
        device = args.device or "cuda:0"
        dataset_path = args.dataset_path or "/scratch2/pljh0906/tcav/datasets/hateful_memes"
    
    # Parse layers
    target_layers = [int(x.strip()) for x in layers.split(",")]
    
    # Setup output directory
    output_path = Path(output_dir) / concept
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get model info from config if available
    model_name = config['model']['name'] if args.config else 'meta-llama/Llama-Guard-3-11B-Vision'
    
    print(f"Loading {model_name} model on {device}...")
    model = MllamaForConditionalGeneration.from_pretrained(
        '/scratch2/pljh0906/models/Llama-Guard-3-11B-Vision',
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        '/scratch2/pljh0906/models/Llama-Guard-3-11B-Vision',
        local_files_only=True,
        trust_remote_code=True
    )
    
    print(f"Extracting activations for concept: {concept}")
    
    # Get sample data from Hateful Memes dataset
    pos_prompts, neg_prompts, image_paths = get_sample_images_and_prompts(
        concept, samples, dataset_path
    )
    
    # Extract positive concept activations
    print("Extracting positive examples...")
    # For Hateful Memes dataset, we have specific images for each text prompt
    if len(image_paths) >= len(pos_prompts) + len(neg_prompts):
        pos_images = image_paths[:len(pos_prompts)]
    else:
        # Fallback to repeating images if needed
        pos_images = (image_paths * (len(pos_prompts) // len(image_paths) + 1))[:len(pos_prompts)]
        
    pos_activations = extract_activations(
        model, processor, pos_images, pos_prompts, target_layers, device
    )
    
    # Extract negative concept activations  
    print("Extracting negative examples...")
    if len(image_paths) >= len(pos_prompts) + len(neg_prompts):
        neg_images = image_paths[len(pos_prompts):len(pos_prompts) + len(neg_prompts)]
    else:
        # Fallback to repeating images if needed
        neg_images = (image_paths * (len(neg_prompts) // len(image_paths) + 1))[:len(neg_prompts)]
        
    neg_activations = extract_activations(
        model, processor, neg_images, neg_prompts, target_layers, device
    )
    
    # Save activations
    for layer in target_layers:
        if len(pos_activations[layer]) > 0 and len(neg_activations[layer]) > 0:
            np.save(output_path / f"layer_{layer}_positive.npy", pos_activations[layer])
            np.save(output_path / f"layer_{layer}_negative.npy", neg_activations[layer])
            print(f"Saved layer {layer}: pos={pos_activations[layer].shape}, neg={neg_activations[layer].shape}")
        else:
            print(f"Warning: Empty activations for layer {layer}")
    
    # Save metadata
    metadata = {
        "concept": concept,
        "layers": target_layers,
        "samples_per_class": samples,
        "model": model_name
    }
    
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Activation extraction complete. Saved to {output_path}")


if __name__ == "__main__":
    main()
