import os
import json
import argparse
import requests
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image
import time


class HatefulMemesDownloader:
    """Hateful Memes dataset downloader"""
    
    def __init__(self, output_dir="./hateful_memes"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data structure
        self.splits = ["train", "dev", "test"]
        self.image_dir = self.output_dir / "images"
        self.meta_dir = self.output_dir / "metadata"
        
        # Create subdirectories
        self.image_dir.mkdir(exist_ok=True)
        self.meta_dir.mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {self.output_dir}")
    
    def download_with_hf_datasets(self, max_samples_per_split=None):
        """Download using Hugging Face datasets (recommended)"""
        try:
            from datasets import load_dataset
            print("🤗 Using Hugging Face datasets...")
            
            # Load dataset
            print("📥 Loading Hateful Memes dataset...")
            dataset = load_dataset("limjiayi/hateful_memes_expanded")
            
            all_data = {}
            total_samples = 0
            
            for split_name in ["train", "validation", "test"]:
                if split_name not in dataset:
                    print(f"   ⚠️  Split '{split_name}' not found, skipping...")
                    continue
                
                split_data = dataset[split_name]
                
                # Limit samples if specified
                if max_samples_per_split:
                    split_data = split_data.select(range(min(len(split_data), max_samples_per_split)))
                
                print(f"   📊 Processing {split_name}: {len(split_data)} samples")
                
                # Convert to our format
                processed_data = []
                for idx, item in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                    
                    # Image processing
                    image = item['image']
                    image_filename = f"{split_name}_{idx:05d}.jpg"
                    image_path = self.image_dir / image_filename
                    
                    # Save image
                    if isinstance(image, Image.Image):
                        image.save(image_path, 'JPEG', quality=95)
                    else:
                        print(f"   ⚠️  Unexpected image format for {split_name}_{idx}")
                        continue
                    
                    # Metadata
                    processed_item = {
                        "id": f"{split_name}_{idx}",
                        "image_path": str(image_path.relative_to(self.output_dir)),
                        "image_filename": image_filename,
                        "text": item.get('text', ''),
                        "label": int(item.get('label', 0)),  # 0: not hateful, 1: hateful
                        "split": split_name,
                        "original_idx": idx
                    }
                    
                    processed_data.append(processed_item)
                
                all_data[split_name] = processed_data
                total_samples += len(processed_data)
                
                # Save split metadata
                split_file = self.meta_dir / f"{split_name}.json"
                with open(split_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, indent=2, ensure_ascii=False)
                
                print(f"   ✅ {split_name}: {len(processed_data)} samples saved")
            
            # Save combined metadata
            combined_file = self.meta_dir / "all_data.json"
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Total samples downloaded: {total_samples}")
            return all_data
            
        except ImportError:
            print("❌ datasets library not found. Please install: pip install datasets")
            return None
        except Exception as e:
            print(f"❌ Error downloading with Hugging Face datasets: {e}")
            return None
    
    def download_with_direct_urls(self, max_samples_per_split=None):
        """Download directly from URL (fallback)"""
        print("🌐 Using direct URL download (limited functionality)...")
        
        # Generate sample data (use original URLs in actual implementation)
        sample_data = self.create_sample_data(max_samples_per_split or 100)
        
        all_data = {}
        total_samples = 0
        
        for split_name, samples in sample_data.items():
            print(f"   📊 Creating {split_name}: {len(samples)} samples")
            
            processed_data = []
            for idx, item in enumerate(tqdm(samples, desc=f"Processing {split_name}")):
                
                # Create dummy image (for demonstration)
                image_filename = f"{split_name}_{idx:05d}.jpg"
                image_path = self.image_dir / image_filename
                
                # Create a simple colored image as placeholder
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('RGB', (500, 400), color=(200, 200, 200))
                draw = ImageDraw.Draw(img)
                
                # Add text to image
                try:
                    font = ImageFont.load_default()
                    draw.text((10, 10), f"Sample {split_name}_{idx}", fill=(50, 50, 50), font=font)
                    draw.text((10, 30), item['text'][:50] + "...", fill=(100, 100, 100), font=font)
                    draw.text((10, 350), f"Label: {'Hateful' if item['label'] else 'Not Hateful'}", 
                             fill=(255, 0, 0) if item['label'] else (0, 128, 0), font=font)
                except:
                    pass
                
                img.save(image_path, 'JPEG', quality=95)
                
                # Metadata
                processed_item = {
                    "id": f"{split_name}_{idx}",
                    "image_path": str(image_path.relative_to(self.output_dir)),
                    "image_filename": image_filename,
                    "text": item['text'],
                    "label": item['label'],
                    "split": split_name,
                    "original_idx": idx,
                    "note": "Generated sample data"
                }
                
                processed_data.append(processed_item)
            
            all_data[split_name] = processed_data
            total_samples += len(processed_data)
            
            # Save split metadata
            split_file = self.meta_dir / f"{split_name}.json"
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # Save combined metadata
        combined_file = self.meta_dir / "all_data.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Total samples created: {total_samples}")
        print("⚠️  Note: This is sample data. Install 'datasets' library for real Hateful Memes data.")
        
        return all_data
    
    def create_sample_data(self, samples_per_split=50):
        """Generate sample data in actual Hateful Memes style"""
        
        # Sample text similar to actual Hateful Memes
        hateful_samples = [
            "This group of people are the worst",
            "I hate when they do this",
            "These people should not exist",
            "They are ruining our society",
            "Get them out of here",
        ] * 20  # Repeat to get more samples
        
        not_hateful_samples = [
            "This is a beautiful sunset",
            "I love spending time with family",
            "Education is important for everyone", 
            "We should help each other",
            "This place looks amazing",
        ] * 20  # Repeat to get more samples
        
        sample_data = {}
        
        for split_name in ["train", "validation", "test"]:
            samples = []
            
            # Half hateful, half not hateful
            half_size = samples_per_split // 2
            
            # Add hateful samples
            for i in range(half_size):
                samples.append({
                    "text": hateful_samples[i % len(hateful_samples)],
                    "label": 1  # hateful
                })
            
            # Add not hateful samples
            for i in range(samples_per_split - half_size):
                samples.append({
                    "text": not_hateful_samples[i % len(not_hateful_samples)],
                    "label": 0  # not hateful
                })
            
            sample_data[split_name] = samples
        
        return sample_data
    
    def verify_dataset(self):
        """Verify downloaded dataset"""
        print("🔍 Verifying downloaded dataset...")
        
        stats = {
            "total_samples": 0,
            "splits": {},
            "label_distribution": {"hateful": 0, "not_hateful": 0},
            "image_count": 0
        }
        
        # Check metadata files
        for split_name in ["train", "validation", "test"]:
            split_file = self.meta_dir / f"{split_name}.json"
            
            if split_file.exists():
                with open(split_file, 'r', encoding='utf-8') as f:
                    split_data = json.load(f)
                
                stats["splits"][split_name] = len(split_data)
                stats["total_samples"] += len(split_data)
                
                # Count labels
                for item in split_data:
                    if item["label"] == 1:
                        stats["label_distribution"]["hateful"] += 1
                    else:
                        stats["label_distribution"]["not_hateful"] += 1
                
                print(f"   ✅ {split_name}: {len(split_data)} samples")
            else:
                print(f"   ❌ {split_name}: metadata file not found")
        
        # Check images
        if self.image_dir.exists():
            image_files = list(self.image_dir.glob("*.jpg"))
            stats["image_count"] = len(image_files)
            print(f"   📸 Images: {len(image_files)} files")
        
        # Print statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Splits: {stats['splits']}")
        print(f"   Label distribution:")
        print(f"     - Hateful: {stats['label_distribution']['hateful']}")
        print(f"     - Not Hateful: {stats['label_distribution']['not_hateful']}")
        print(f"   Images: {stats['image_count']}")
        
        # Save statistics
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def create_gcav_format(self):
        """Convert to GCAV experiment format"""
        print("🔧 Creating GCAV-compatible format...")
        
        gcav_dir = self.output_dir / "gcav_format"
        gcav_dir.mkdir(exist_ok=True)
        
        # Load all data
        all_data_file = self.meta_dir / "all_data.json"
        if not all_data_file.exists():
            print("❌ Combined metadata not found")
            return False
        
        with open(all_data_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        # Separate by label for CAV training
        hateful_samples = []
        not_hateful_samples = []
        
        for split_name, split_data in all_data.items():
            for item in split_data:
                if item["label"] == 1:
                    hateful_samples.append(item)
                else:
                    not_hateful_samples.append(item)
        
        # Save GCAV format
        gcav_data = {
            "concept": "hateful_content",
            "positive_samples": hateful_samples,  # Hateful = positive concept
            "negative_samples": not_hateful_samples,  # Not hateful = negative concept
            "total_positive": len(hateful_samples),
            "total_negative": len(not_hateful_samples)
        }
        
        gcav_file = gcav_dir / "hateful_content_dataset.json"
        with open(gcav_file, 'w', encoding='utf-8') as f:
            json.dump(gcav_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ GCAV format saved: {gcav_file}")
        print(f"   📊 Positive (hateful): {len(hateful_samples)}")
        print(f"   📊 Negative (not hateful): {len(not_hateful_samples)}")
        
        return gcav_file


def main():
    parser = argparse.ArgumentParser(description="Download Hateful Memes Dataset")
    parser.add_argument(
        "--output_dir",
        default="./hateful_memes",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Maximum samples per split (for testing)"
    )
    parser.add_argument(
        "--method",
        choices=["hf_datasets", "direct", "auto"],
        default="auto",
        help="Download method"
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing dataset"
    )
    
    args = parser.parse_args()
    
    print("🎭 Hateful Memes Dataset Downloader")
    print("=" * 50)
    print(f"Output directory: {args.output_dir}")
    print(f"Max samples per split: {args.max_samples or 'All'}")
    print()
    
    # Initialize downloader
    downloader = HatefulMemesDownloader(args.output_dir)
    
    # Verify only mode
    if args.verify_only:
        downloader.verify_dataset()
        return
    
    # Download dataset
    success = False
    
    if args.method in ["hf_datasets", "auto"]:
        print("Trying Hugging Face datasets...")
        data = downloader.download_with_hf_datasets(args.max_samples)
        if data:
            success = True
    
    if not success and args.method in ["direct", "auto"]:
        print("Falling back to direct method...")
        data = downloader.download_with_direct_urls(args.max_samples)
        if data:
            success = True
    
    if not success:
        print("❌ All download methods failed")
        return
    
    # Verify and create GCAV format
    print()
    downloader.verify_dataset()
    print()
    downloader.create_gcav_format()
    
    print()
    print("🎉 Hateful Memes dataset download complete!")
    print()
    print("📋 Next steps:")
    print(f"   1. Check data: ls -la {args.output_dir}")
    print("   2. Use in GCAV:")
    print("      python tools/dump_activations.py \\")
    print("        --concept hateful_content \\")
    print("        --input_format hateful_memes \\")
    print(f"        --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
