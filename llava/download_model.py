import os
import argparse
import sys
from pathlib import Path
import time
import psutil
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from huggingface_hub import snapshot_download, HfApi


def check_system_requirements():
    print("🔍 Checking system requirements...")

    # Check system memory
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    
    print(f"   💾 System RAM: {memory_gb:.1f}GB")
    if memory_gb < 16:
        print("   ⚠️  Recommended RAM: 16GB+")
    else:
        print("   ✅ Sufficient RAM")

    # Check GPU availability
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory_gb = gpu_props.total_memory / (1024**3)
            print(f"   🚀 GPU {i}: {gpu_props.name} ({gpu_memory_gb:.1f}GB)")
            
            if gpu_memory_gb < 12:
                print(f"      ⚠️  Recommended GPU memory: 12GB+")
            else:
                print(f"      ✅ Sufficient GPU memory")
    else:
        print("   ❌ No CUDA GPU found")
        print("   ⚠️  Can run in CPU mode but will be very slow")

    # Check disk space
    disk_usage = psutil.disk_usage('/')
    free_gb = disk_usage.free / (1024**3)
    print(f"   💿 Available disk space: {free_gb:.1f}GB")

    if free_gb < 20:
        print("   ⚠️  Recommended free space: 20GB+")
        return False
    else:
        print("   ✅ Sufficient disk space")
    
    return True


def check_model_info(model_name):
    """Check model information"""
    print(f"📋 Checking model info: {model_name}")

    try:
        api = HfApi()
        model_info = api.model_info(model_name)

        print(f"   📦 Model: {model_info.modelId}")
        print(f"   👤 Author: {model_info.author or 'N/A'}")
        print(f"   📅 Last modified: {model_info.lastModified}")

        # Estimate model size
        if hasattr(model_info, 'siblings') and model_info.siblings:
            total_size = sum(getattr(file, 'size', 0) for file in model_info.siblings if hasattr(file, 'size'))
            if total_size > 0:
                size_gb = total_size / (1024**3)
                print(f"   📏 Estimated size: {size_gb:.1f}GB")
            else:
                print(f"   📏 Estimated size: ~14GB (7B model baseline)")
        else:
            print(f"   📏 Estimated size: ~14GB (7B model baseline)")

        return True

    except Exception as e:
        print(f"   ❌ Failed to check model info: {e}")
        return False


def download_with_progress(model_name, cache_dir=None, force_download=False):
    """Download model with progress tracking"""

    print(f"📥 Starting model download: {model_name}")
    start_time = time.time()

    try:
        # Set cache directory
        if cache_dir:
            cache_dir = Path(cache_dir).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
            os.environ['HF_HOME'] = str(cache_dir)
            print(f"   📁 Cache directory: {cache_dir}")

        # Step 1: Download model files (using Hugging Face Hub)
        print("   🔄 Step 1/2: Downloading model files...")
        local_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=True
        )
        print(f"   ✅ Model files downloaded: {local_dir}")

        # Step 2: Verify model with Transformers
        print("   🔄 Step 2/2: Verifying model loading...")

        # Test processor loading
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print("   ✅ Processor loaded successfully")

        # Test model loading (metadata only, skip actual weights)
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="cpu",  # CPU for verification only
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("   ✅ Model loading verification successful")

        # Clean up memory
        del model
        del processor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        elapsed_time = time.time() - start_time
        print(f"   ⏱️  Download time: {elapsed_time/60:.1f}min")

        return True, local_dir

    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return False, None


def verify_download(model_name, cache_dir=None):
    """Verify downloaded model"""
    print("🔍 Verifying download...")

    try:
        # Simple loading test
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            local_files_only=True  # Use local files only
        )
        
        # Check model configuration
        config = processor.tokenizer.get_vocab()
        print(f"   ✅ Tokenizer vocabulary size: {len(config)}")

        # Test processor functionality
        test_text = "Test prompt"
        inputs = processor(text=test_text, return_tensors="pt")
        print(f"   ✅ Text processing test passed")

        print("   ✅ Model verification complete")
        return True

    except Exception as e:
        print(f"   ❌ Verification failed: {e}")
        return False


def get_cache_info(cache_dir=None):
    """Display cache directory information"""
    if cache_dir:
        cache_path = Path(cache_dir)
    else:
        # Default Hugging Face cache location
        cache_path = Path.home() / ".cache" / "huggingface"
    
    if cache_path.exists():
        total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        print(f"📁 Cache directory: {cache_path}")
        print(f"📏 Cache size: {size_gb:.2f}GB")
    else:
        print(f"📁 Cache directory: {cache_path} (will be created)")


def main():
    parser = argparse.ArgumentParser(description="Download LlavaGuard model")
    parser.add_argument(
        "--model",
        default="AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf",
        help="Model name to download"
    )
    parser.add_argument(
        "--cache-dir",
        help="Model cache directory (default: ~/.cache/huggingface)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing model without downloading"
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip system requirements check"
    )
    
    args = parser.parse_args()
    
    print("🤖 LlavaGuard Model Downloader")
    print("=" * 50)

    # Check system requirements
    if not args.skip_checks:
        if not check_system_requirements():
            print("\n⚠️  Please check system requirements.")
            print("Continue anyway? (y/N): ", end="")
            if input().lower() != 'y':
                sys.exit(1)
        print()

    # Display cache information
    get_cache_info(args.cache_dir)
    print()

    # Check model information
    if not check_model_info(args.model):
        print("Cannot verify model information. Continue anyway? (y/N): ", end="")
        if input().lower() != 'y':
            sys.exit(1)
    print()

    # Perform verification only
    if args.verify_only:
        success = verify_download(args.model, args.cache_dir)
        if success:
            print("✅ Model is already downloaded and working!")
        else:
            print("❌ Model verification failed. Re-download required.")
            sys.exit(1)
        return

    # Confirm download start
    print(f"📥 Starting download of {args.model}")
    print("Estimated time: 5-30 minutes (depends on network speed)")
    print("Continue? (Y/n): ", end="")
    response = input().lower()
    if response and response != 'y':
        print("Download cancelled.")
        sys.exit(0)

    print()

    # Download model
    success, local_dir = download_with_progress(
        args.model,
        args.cache_dir,
        args.force_download
    )
    
    if not success:
        print("❌ Download failed!")
        sys.exit(1)

    # Verify download
    if verify_download(args.model, args.cache_dir):
        print()
        print("🎉 Model download and verification complete!")
        print()
        print("📋 Usage:")
        print("   cd llava/")
        print("   python inference.py --image sample.jpg --prompt 'Test prompt'")
        print()

        # Display cache information again
        get_cache_info(args.cache_dir)
        
    else:
        print("❌ Download completed but verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
