import argparse
import torch
import yaml
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Llama4ForConditionalGeneration
import requests
from steering.gcav import create_gcav_steerer

class GCAVInference:
    """Llama Guard 4 inference with GCAV steering capabilities"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-Guard-4-12B", device: str = "cuda:0"):
        self.device = device
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.gcav_steerer = None
        
    def load_model(self):
        """Load Llama Guard 4 model and processor"""
        print(f"Loading {self.model_name} on {self.device}...")
        
        # Use local path for faster loading
        local_path = '/scratch2/pljh0906/models/Llama-Guard-4-12B'
        
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
        
        print("Model loaded successfully!")
    
    def load_gcav(self, config_path: str):
        """Load GCAV steerer from configuration"""
        try:
            self.gcav_steerer = create_gcav_steerer(config_path, self.device)
            if self.gcav_steerer:
                print("GCAV steerer loaded successfully!")
                return True
            else:
                print("Failed to load GCAV steerer")
                return False
        except Exception as e:
            print(f"Error loading GCAV: {e}")
            return False
    
    def enable_gcav_steering(self, concept_configs: dict):
        """Enable GCAV steering with specified concept configurations"""
        if not self.gcav_steerer:
            print("GCAV steerer not loaded")
            return False
        
        try:
            self.gcav_steerer.register_hooks(self.model, concept_configs)
            print("GCAV steering enabled")
            return True
        except Exception as e:
            print(f"Error enabling GCAV: {e}")
            return False
    
    def disable_gcav_steering(self):
        """Disable GCAV steering"""
        if self.gcav_steerer:
            self.gcav_steerer.remove_hooks()
            print("GCAV steering disabled")
    
    def prepare_input(self, image_path: str, prompt: str, safety_prompt: str = None):
        """Prepare input for LlavaGuard"""
        
        # Load image
        if image_path.startswith("http"):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(image_path, headers=headers, stream=True, timeout=10)
                response.raise_for_status()
                from io import BytesIO
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                print(f"Failed to load image from {image_path}: {e}")
                # Use a default test image if URL fails
                image = Image.new('RGB', (224, 224), color='white')
        else:
            image = Image.open(image_path)
        
        # Default safety assessment prompt (from LlavaGuard paper)
        if safety_prompt is None:
            safety_prompt = """
Provide a safety assessment for the content provided by the user. Determine compliance or violation against our safety policy by reviewing the following policy categories:

O1: Hate, Humiliation, Harassment
O2: Violence, Harm, or Cruelty  
O3: Sexual Content
O4: Nudity Content
O5: Criminal Planning
O6: Weapons or Substance Abuse
O7: Self-Harm
O8: Animal Cruelty
O9: Disasters or Emergencies

Assessment Steps:
1. Select a safety "rating". This should be "Unsafe" if the content violates the safety policy; otherwise, it should be "Safe".
2. Assign a policy "category". If none of the policy categories are applicable, select "NA: None applying".
3. Provide a "rationale". Describe the user content and justify why it is considered safe or unsafe.

To provide your assessment use the following json template:
{
    "rating": str<"Unsafe"|"Safe">,
    "category": str<"O1"|"O2"|"O3"|"O4"|"O5"|"O6"|"O7"|"O8"|"O9"|"NA">,
    "rationale": str,
}
"""
        
        # Combine user prompt with safety assessment
        full_prompt = f"{safety_prompt}\n\nUser content: {prompt}"
        
        # Prepare conversation
        # Process input with proper image placeholder for Llama Guard 4
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": full_prompt}
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
            text_prompt = f"<image>\n{full_prompt}"
            
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def generate(self, inputs: dict, **generation_kwargs):
        """Generate response with optional GCAV steering"""
        
        # Default generation parameters (warning-free safe settings)
        default_kwargs = {
            "max_new_tokens": 200,
            "do_sample": False,  # Greedy decoding for stability
            "num_beams": 1,
            # Disable KV cache to avoid DynamicCache version mismatches
            "use_cache": False,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            # Additional safety parameters
            "repetition_penalty": 1.0,  # No repetition penalty
            "length_penalty": 1.0,      # No length penalty
            "output_scores": False,     # Don't compute scores
            "return_dict_in_generate": False  # Simple output format
        }
        

        default_kwargs.update(generation_kwargs)
        

        if self.gcav_steerer:
            self.gcav_steerer.reset_stats()

        # CUDA generation with minimal safety checks
        with torch.no_grad():
            try:
                outputs = self.model.generate(**inputs, **default_kwargs)
            except RuntimeError as e:
                if "CUDA error" in str(e) or "assertion" in str(e).lower():
                    print(f"⚠️ CUDA generation error: {str(e)[:100]}...")
                    print("🔄 Retrying with different parameters...")
                    # Retry with more conservative parameters
                    safe_kwargs = default_kwargs.copy()
                    safe_kwargs.update({
                        "max_new_tokens": min(100, default_kwargs["max_new_tokens"]),
                        "do_sample": False,  # Ensure deterministic
                    })
                    outputs = self.model.generate(**inputs, **safe_kwargs)
                else:
                    raise e

        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated part (remove input prompt)
        input_length = len(self.processor.decode(inputs['input_ids'][0], skip_special_tokens=True))
        generated_text = response[input_length:].strip()

        gcav_stats = {}
        if self.gcav_steerer:
            gcav_stats = self.gcav_steerer.get_intervention_stats()
        
        return {
            'response': generated_text,
            'full_response': response,
            'gcav_stats': gcav_stats
        }
    
    def run_inference(
        self, 
        image_path: str, 
        prompt: str, 
        enable_gcav: bool = False,
        gcav_config: dict = None,
        **generation_kwargs
    ):
        """Run complete inference pipeline"""
        inputs = self.prepare_input(image_path, prompt)

        if enable_gcav and gcav_config and self.gcav_steerer:
            concept_configs = gcav_config.get('concepts', {})
            self.enable_gcav_steering(concept_configs)
        else:
            self.disable_gcav_steering()

        result = self.generate(inputs, **generation_kwargs)

        result['gcav_enabled'] = enable_gcav and self.gcav_steerer is not None
        result['model'] = self.model_name
        
        return result


def main():
    parser = argparse.ArgumentParser(description="GCAV-enabled Llama Guard 4 Inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Input prompt/question")
    parser.add_argument("--model", default="meta-llama/Llama-Guard-4-12B", help="Model name")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    
    # GCAV arguments
    parser.add_argument("--enable-gcav", action="store_true", help="Enable GCAV steering")
    parser.add_argument("--gcav-config", default="config/gcav_config.yaml", help="GCAV config file")
    
    # Generation arguments
    parser.add_argument("--max-tokens", type=int, default=200, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    
    # Output arguments
    parser.add_argument("--output", help="Output file (JSON format)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()

    inference = GCAVInference(args.model, args.device)
    inference.load_model()

    gcav_config = {}
    if args.enable_gcav:
        if Path(args.gcav_config).exists():
            with open(args.gcav_config, 'r') as f:
                gcav_config = yaml.safe_load(f)

            success = inference.load_gcav(args.gcav_config)
            if not success:
                print("Warning: GCAV loading failed, proceeding without steering")
                args.enable_gcav = False
        else:
            print(f"Warning: GCAV config not found: {args.gcav_config}")
            args.enable_gcav = False

    generation_kwargs = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p
    }

    print(f"\n{'='*60}")
    print(f"Input: {args.prompt}")
    print(f"Image: {args.image}")
    print(f"GCAV: {'Enabled' if args.enable_gcav else 'Disabled'}")
    print(f"{'='*60}")

    try:
        result = inference.run_inference(
            args.image,
            args.prompt,
            args.enable_gcav,
            gcav_config,
            **generation_kwargs
        )

        print("\n" + "="*40 + " RESULT " + "="*40)
        print(result['response'])
        
        if args.verbose:
            print(f"\n{'='*20} METADATA {'='*20}")
            print(f"GCAV Enabled: {result['gcav_enabled']}")
            print(f"Model: {result['model']}")
            
            if result['gcav_stats']:
                print(f"GCAV Interventions:")
                for layer, stats in result['gcav_stats'].items():
                    print(f"  Layer {layer}: avg_ε={stats['avg_epsilon']:.4f}, "
                          f"max_ε={stats['max_epsilon']:.4f}, count={stats['interventions_count']}")

        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()
