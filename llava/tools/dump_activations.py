import os
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import requests

def get_sample_images_and_prompts(concept_type="harmful_symbols", n_samples=100):
    """Generate sample data for concept learning"""
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
            # Store activation (detach from computation graph)
            activations[layer_idx].append(output.detach().cpu())
        return hook
    
    # Register hooks
    hooks = []
    for layer in target_layers:
        if hasattr(model.language_model.model.layers, f'{layer}'):
            hook = model.language_model.model.layers[layer].register_forward_hook(make_hook(layer))
            hooks.append(hook)
    
    model.eval()
    with torch.no_grad():
        for image_url, prompt in tqdm(zip(images, prompts), total=len(prompts), desc="Extracting"):
            try:
                # Load image
                if isinstance(image_url, str) and image_url.startswith("http"):
                    image = Image.open(requests.get(image_url, stream=True).raw)
                else:
                    # Use a default test image if URL fails
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
            # Take mean across sequence dimension
            layer_acts = [act.mean(dim=1).numpy() for act in activations[layer]]
            processed_activations[layer] = np.vstack(layer_acts)
        else:
            print(f"Warning: No activations captured for layer {layer}")
            processed_activations[layer] = np.array([])
    
    return processed_activations


def main():
    parser = argparse.ArgumentParser(description="Extract activations for GCAV")
    parser.add_argument("--concept", default="harmful_symbols", help="Concept to extract")
    parser.add_argument("--layers", default="10,14,18", help="Comma-separated layer indices")  
    parser.add_argument("--samples", type=int, default=50, help="Samples per concept")
    parser.add_argument("--output_dir", default="../artifacts/activations", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    
    args = parser.parse_args()
    
    # Parse layers
    target_layers = [int(x.strip()) for x in args.layers.split(",")]
    
    # Setup output directory
    output_dir = Path(args.output_dir) / args.concept
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading LlavaGuard model on {args.device}...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        'AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf',
        torch_dtype=torch.float16,
        device_map=args.device
    )
    processor = AutoProcessor.from_pretrained('AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf')
    
    print(f"Extracting activations for concept: {args.concept}")
    
    # Get sample data
    pos_prompts, neg_prompts, image_urls = get_sample_images_and_prompts(
        args.concept, args.samples
    )
    
    # Extract positive concept activations
    print("Extracting positive examples...")
    pos_images = image_urls * (len(pos_prompts) // len(image_urls) + 1)
    pos_activations = extract_activations(
        model, processor, pos_images[:len(pos_prompts)], pos_prompts, target_layers, args.device
    )
    
    # Extract negative concept activations  
    print("Extracting negative examples...")
    neg_images = image_urls * (len(neg_prompts) // len(image_urls) + 1)
    neg_activations = extract_activations(
        model, processor, neg_images[:len(neg_prompts)], neg_prompts, target_layers, args.device
    )
    
    # Save activations
    for layer in target_layers:
        if len(pos_activations[layer]) > 0 and len(neg_activations[layer]) > 0:
            np.save(output_dir / f"layer_{layer}_positive.npy", pos_activations[layer])
            np.save(output_dir / f"layer_{layer}_negative.npy", neg_activations[layer])
            print(f"Saved layer {layer}: pos={pos_activations[layer].shape}, neg={neg_activations[layer].shape}")
        else:
            print(f"Warning: Empty activations for layer {layer}")
    
    # Save metadata
    metadata = {
        "concept": args.concept,
        "layers": target_layers,
        "samples_per_class": args.samples,
        "model": "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf"
    }
    
    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Activation extraction complete. Saved to {output_dir}")


if __name__ == "__main__":
    main()
