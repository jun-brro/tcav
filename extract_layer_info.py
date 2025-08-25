#!/usr/bin/env python3
"""
Extract layer information from the 3 models: LLaVA, Llama3Vision, Llama4
"""

import os
import sys
import torch
from pathlib import Path

# Function to safely import and analyze a model
def analyze_model(model_name, model_class_name, transformers_import):
    print(f"\n{'='*60}")
    print(f"🔍 ANALYZING {model_name.upper()}")
    print(f"{'='*60}")
    
    try:
        # Dynamic import
        exec(f"from transformers import {transformers_import}")
        model_class = eval(transformers_import.split(',')[1].strip())
        
        # Use a small model or config-only loading for analysis
        if "llava" in model_name.lower():
            model_path = "AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf"
        elif "llama3" in model_name.lower():
            model_path = "/scratch2/pljh0906/models/Llama-Guard-3-11B-Vision"
        elif "llama4" in model_name.lower():
            model_path = "/scratch2/pljh0906/models/Llama-Guard-4-12B"
        else:
            model_path = "meta-llama/Llama-2-7b-hf"
        
        print(f"📂 Loading config from: {model_path}")
        
        # Load config only (faster)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        print(f"📋 Model Architecture: {config.architectures}")
        print(f"🏗️  Model Type: {type(config).__name__}")
        
        # Extract layer information
        layer_info = {}
        
        # Common attributes to check
        layer_attrs = [
            'num_hidden_layers', 'num_layers', 'n_layer', 'n_layers',
            'num_attention_heads', 'num_key_value_heads', 
            'hidden_size', 'intermediate_size',
            'max_position_embeddings', 'vocab_size'
        ]
        
        for attr in layer_attrs:
            if hasattr(config, attr):
                layer_info[attr] = getattr(config, attr)
                print(f"  🔹 {attr}: {getattr(config, attr)}")
        
        # Check for vision-related configs
        if hasattr(config, 'vision_config'):
            print(f"👁️  Vision Config Found:")
            vision_config = config.vision_config
            vision_attrs = ['num_hidden_layers', 'hidden_size', 'image_size', 'patch_size']
            for attr in vision_attrs:
                if hasattr(vision_config, attr):
                    print(f"    🔹 vision_{attr}: {getattr(vision_config, attr)}")
        
        # Check for text config
        if hasattr(config, 'text_config'):
            print(f"📝 Text Config Found:")
            text_config = config.text_config
            text_attrs = ['num_hidden_layers', 'hidden_size', 'vocab_size']
            for attr in text_attrs:
                if hasattr(text_config, attr):
                    print(f"    🔹 text_{attr}: {getattr(text_config, attr)}")
        
        # Try to get layer names by looking at a sample model structure
        print(f"\n🧱 LAYER STRUCTURE ANALYSIS:")
        try:
            # Load actual model (very memory intensive, so we'll be careful)
            print("  ⚠️  Loading minimal model for structure analysis...")
            model = model_class.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cpu",  # Keep on CPU to save GPU memory
                local_files_only=True if "scratch2" in model_path else False,
                low_cpu_mem_usage=True
            )
            
            # Get layer names
            layer_names = []
            for name, module in model.named_modules():
                if any(keyword in name for keyword in ['layer', 'block', 'decoder', 'encoder']):
                    if not any(skip in name for skip in ['embed', 'norm', 'head']):
                        layer_names.append(name)
            
            print(f"  📊 Found {len(layer_names)} relevant layers")
            
            # Show first few and last few layer names
            if layer_names:
                print(f"  🔍 Sample layer names:")
                for i, name in enumerate(layer_names[:5]):
                    print(f"    {i}: {name}")
                if len(layer_names) > 10:
                    print(f"    ...")
                    for i, name in enumerate(layer_names[-5:], len(layer_names)-5):
                        print(f"    {i}: {name}")
            
            # Extract layer indices that are commonly used
            print(f"\n  🎯 RECOMMENDED LAYER INDICES:")
            if 'num_hidden_layers' in layer_info:
                total_layers = layer_info['num_hidden_layers']
            elif 'num_layers' in layer_info:
                total_layers = layer_info['num_layers']
            else:
                total_layers = len([n for n in layer_names if 'decoder' in n or 'layer' in n])
            
            if total_layers > 0:
                # Common layer selection strategy
                early = max(1, total_layers // 4)
                mid_early = max(1, total_layers // 3)
                mid_late = max(1, 2 * total_layers // 3)
                late = max(1, 3 * total_layers // 4)
                
                recommended = [early, mid_early, mid_late, late]
                print(f"    Total layers: {total_layers}")
                print(f"    Recommended: {recommended}")
            
            # Clean up model to free memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ⚠️  Could not load full model: {str(e)[:100]}...")
            print(f"  💡 Using config-based estimation")
            
        return layer_info
            
    except Exception as e:
        print(f"❌ Error analyzing {model_name}: {str(e)[:200]}...")
        return {}

def main():
    print("🚀 EXTRACTING LAYER INFORMATION FOR 3 MODELS")
    print("=" * 80)
    
    models_info = [
        ("LLaVA", "LlavaOnevisionForConditionalGeneration", "AutoProcessor, LlavaOnevisionForConditionalGeneration"),
        ("Llama3Vision", "MllamaForConditionalGeneration", "AutoProcessor, MllamaForConditionalGeneration"), 
        ("Llama4", "Llama4ForConditionalGeneration", "AutoProcessor, Llama4ForConditionalGeneration")
    ]
    
    results = {}
    
    for model_name, class_name, import_str in models_info:
        try:
            layer_info = analyze_model(model_name, class_name, import_str)
            results[model_name] = layer_info
        except Exception as e:
            print(f"❌ Failed to analyze {model_name}: {e}")
            results[model_name] = {}
    
    print(f"\n{'='*80}")
    print("📊 SUMMARY OF LAYER INFORMATION")
    print(f"{'='*80}")
    
    for model_name, info in results.items():
        print(f"\n🤖 {model_name}:")
        if info:
            for key, value in info.items():
                print(f"  • {key}: {value}")
        else:
            print(f"  ❌ No information extracted")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
