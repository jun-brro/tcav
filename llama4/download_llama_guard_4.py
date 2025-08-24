import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, login
import argparse
import torch
import subprocess

def clean_incomplete_download(output_dir: str):
    """Clean up incomplete downloads"""
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"🧹 Cleaning up existing incomplete download at {output_path}")
        try:
            shutil.rmtree(output_path)
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Cleanup warning: {e}")
    
def install_llama4_transformers():
    """Install the special transformers version for Llama Guard 4"""
    print("🔧 Installing Llama Guard 4 compatible transformers...")
    try:
        # Install specific transformers version for Llama Guard 4
        cmd = ["pip", "install", "git+https://github.com/huggingface/transformers@v4.51.3-LlamaGuard-preview", "hf_xet"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Transformers installation successful!")
        else:
            print(f"⚠️ Transformers installation warning: {result.stderr}")
    except Exception as e:
        print(f"❌ Failed to install transformers: {e}")
        print("Please run manually: pip install git+https://github.com/huggingface/transformers@v4.51.3-LlamaGuard-preview hf_xet")

def download_llama_guard_4(output_dir: str = "/scratch2/pljh0906/models/Llama-Guard-4-12B", 
                          token: str = None, 
                          revision: str = "main",
                          clean_cache: bool = False):
    """Download Llama Guard 4 model"""
    
    model_name = "meta-llama/Llama-Guard-4-12B"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🦙 Downloading {model_name}...")
    print(f"📁 Output directory: {output_path}")
    
    # Clean up incomplete downloads if requested
    if clean_cache:
        clean_incomplete_download(output_dir)
    
    # Get token from parameter or environment variable
    if not token:
        token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    # Login with HF token if available
    if token:
        print("🔑 Logging in with Hugging Face token...")
        login(token=token)
    else:
        print("⚠️ No HF token provided. You may need to accept the model license on HuggingFace first.")
        print("   Visit: https://huggingface.co/meta-llama/Llama-Guard-4-12B")
        print("   Set token: export HUGGING_FACE_HUB_TOKEN=your_token")
    
    try:
        # Download the model
        print("📥 Starting download... This may take 20-30 minutes for ~25GB model")
        snapshot_download(
            repo_id=model_name,
            local_dir=str(output_path),
            revision=revision,
            resume_download=True,
            local_dir_use_symlinks=False,  # Ensure actual files are copied, not symlinks
            token=token,  # Pass token explicitly
            # Download all files including safetensors, pytorch_model.bin, etc.
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Skip unnecessary formats
        )
        
        print("✅ Download completed successfully!")
        print(f"📊 Model info:")
        
        # Show downloaded files
        total_size = 0
        for file_path in output_path.rglob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"   📄 {file_path.name}: {size_mb:.1f} MB")
        
        print(f"   📦 Total size: {total_size:.1f} MB")
        
        # Test model loading
        print("\n🧪 Testing model loading...")
        try:
            from transformers import AutoProcessor, Llama4ForConditionalGeneration
            
            model = Llama4ForConditionalGeneration.from_pretrained(
                str(output_path),
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            processor = AutoProcessor.from_pretrained(str(output_path))
            
            print(f"✅ Model loaded successfully!")
            print(f"   🎯 Model type: {type(model).__name__}")
            print(f"   📊 Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
            print(f"   🧠 Architecture: {model.config.architectures}")
            print(f"   🌍 Vocabulary size: {model.config.vocab_size}")
            
            # Clean up memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"⚠️ Model loading test failed: {e}")
            print("   This might be normal if CUDA is not available or transformers version is outdated")
            print("   Make sure you have the correct transformers version:")
            print("   pip install git+https://github.com/huggingface/transformers@v4.51.3-LlamaGuard-preview hf_xet")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection")
        print("2. Make sure you have accepted the model license on HuggingFace")
        print("3. Provide a valid HuggingFace token with --token")
        print("4. Check available disk space (~25GB needed)")
        print("5. Install required transformers version")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download Llama Guard 4 model")
    parser.add_argument("--output_dir", default="/scratch2/pljh0906/models/Llama-Guard-4-12B",
                       help="Directory to save the model")
    parser.add_argument("--token", help="HuggingFace access token")
    parser.add_argument("--revision", default="main", help="Model revision to download")
    parser.add_argument("--install-transformers", action="store_true", 
                       help="Install Llama Guard 4 compatible transformers")
    parser.add_argument("--clean-cache", action="store_true",
                       help="Clean up existing incomplete downloads")
    
    args = parser.parse_args()
    
    print("🦙 Llama Guard 4 Downloader")
    print("=" * 50)
    
    # Install transformers if requested
    if args.install_transformers:
        install_llama4_transformers()
    
    download_llama_guard_4(
        output_dir=args.output_dir,
        token=args.token,
        revision=args.revision,
        clean_cache=args.clean_cache
    )
    
    print("\n🎉 Setup completed!")
    print("📋 Next steps:")
    print("1. cd /scratch2/pljh0906/tcav/llama4")
    print("2. ./run_example.sh")

if __name__ == "__main__":
    main()
