import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import seaborn as sns
import argparse

def load_sample_data(n_train_pos=50, n_train_neg=50, n_siuo=20, seed=42):
    """Generate sample high-dimensional data to simulate the scenario"""
    np.random.seed(seed)
    
    # Simulate training data (fixed)
    train_pos = np.random.randn(n_train_pos, 512) + np.array([2, 1, 0.5] + [0]*509)  # Biased toward "harmful"
    train_neg = np.random.randn(n_train_neg, 512) + np.array([-2, -1, -0.5] + [0]*509)  # Biased toward "benign"
    
    # Simulate SIUO data (mixed)
    siuo = np.random.randn(n_siuo, 512) + np.array([0, 0, 0] + [0]*509)  # Neutral
    
    return train_pos, train_neg, siuo

def test_feature_stability():
    """Test if adding SIUO changes the apparent position of training data"""
    
    print("🔍 Testing Feature Stability with t-SNE vs PCA")
    print("="*60)
    
    # Load sample data
    train_pos, train_neg, siuo = load_sample_data()
    
    # Scenario 1: Training data only
    train_only = np.vstack([train_pos, train_neg])
    train_labels = ['Hateful']*len(train_pos) + ['Benign']*len(train_neg)
    
    # Scenario 2: Training + SIUO data
    all_data = np.vstack([train_pos, train_neg, siuo])
    all_labels = ['Hateful']*len(train_pos) + ['Benign']*len(train_neg) + ['SIUO']*len(siuo)
    
    # Test 1: Are the original features identical?
    print("📊 Test 1: Original High-dimensional Features")
    print(f"Train_pos mean before: {train_pos[:5, :3].mean(axis=0)}")
    print(f"Train_pos mean after:  {all_data[:len(train_pos), :3][:5].mean(axis=0)}")
    print(f"Identical? {np.allclose(train_pos, all_data[:len(train_pos)])}")
    print()
    
    # Test 2: t-SNE stability
    print("🎲 Test 2: t-SNE Stability (Multiple Runs)")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for run in range(3):
        # Training only
        tsne1 = TSNE(n_components=2, random_state=run, perplexity=15)
        reduced1 = tsne1.fit_transform(train_only)
        
        axes[0, run].scatter(reduced1[:len(train_pos), 0], reduced1[:len(train_pos), 1], 
                           c='red', alpha=0.7, s=50, label='Hateful')
        axes[0, run].scatter(reduced1[len(train_pos):, 0], reduced1[len(train_pos):, 1], 
                           c='blue', alpha=0.7, s=50, label='Benign')
        axes[0, run].set_title(f't-SNE Run {run+1}: Training Only')
        axes[0, run].legend()
        axes[0, run].grid(True, alpha=0.3)
        
        # Training + SIUO
        tsne2 = TSNE(n_components=2, random_state=run, perplexity=min(30, len(all_data)//4))
        reduced2 = tsne2.fit_transform(all_data)
        
        axes[1, run].scatter(reduced2[:len(train_pos), 0], reduced2[:len(train_pos), 1], 
                           c='red', alpha=0.7, s=50, label='Hateful')
        axes[1, run].scatter(reduced2[len(train_pos):len(train_pos)+len(train_neg), 0], 
                           reduced2[len(train_pos):len(train_pos)+len(train_neg), 1], 
                           c='blue', alpha=0.7, s=50, label='Benign')
        axes[1, run].scatter(reduced2[len(train_pos)+len(train_neg):, 0], 
                           reduced2[len(train_pos)+len(train_neg):, 1], 
                           c='gray', alpha=0.7, s=60, marker='^', label='SIUO')
        axes[1, run].set_title(f't-SNE Run {run+1}: Training + SIUO')
        axes[1, run].legend()
        axes[1, run].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/scratch2/pljh0906/tcav/llama4/debug_tsne_stability.png', dpi=300)
    plt.close()
    print("✅ Saved t-SNE stability comparison to debug_tsne_stability.png")
    
    # Test 3: PCA stability (should be much more stable)
    print("📐 Test 3: PCA Stability")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCA on training only
    pca1 = PCA(n_components=2, random_state=42)
    pca_reduced1 = pca1.fit_transform(train_only)
    
    ax1.scatter(pca_reduced1[:len(train_pos), 0], pca_reduced1[:len(train_pos), 1], 
               c='red', alpha=0.7, s=50, label='Hateful')
    ax1.scatter(pca_reduced1[len(train_pos):, 0], pca_reduced1[len(train_pos):, 1], 
               c='blue', alpha=0.7, s=50, label='Benign')
    ax1.set_title('PCA: Training Only')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PCA on training + SIUO
    pca2 = PCA(n_components=2, random_state=42)
    pca_reduced2 = pca2.fit_transform(all_data)
    
    ax2.scatter(pca_reduced2[:len(train_pos), 0], pca_reduced2[:len(train_pos), 1], 
               c='red', alpha=0.7, s=50, label='Hateful')
    ax2.scatter(pca_reduced2[len(train_pos):len(train_pos)+len(train_neg), 0], 
               pca_reduced2[len(train_pos):len(train_pos)+len(train_neg), 1], 
               c='blue', alpha=0.7, s=50, label='Benign')
    ax2.scatter(pca_reduced2[len(train_pos)+len(train_neg):, 0], 
               pca_reduced2[len(train_pos)+len(train_neg):, 1], 
               c='gray', alpha=0.7, s=60, marker='^', label='SIUO')
    ax2.set_title('PCA: Training + SIUO')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/scratch2/pljh0906/tcav/llama4/debug_pca_stability.png', dpi=300)
    plt.close()
    print("✅ Saved PCA stability comparison to debug_pca_stability.png")
    
    # Test 4: Distance analysis
    print("📏 Test 4: High-dimensional Distance Analysis")
    
    # Compute pairwise distances in original space
    train_distances = euclidean_distances(train_only)
    all_distances = euclidean_distances(all_data)
    
    # Extract the training-training block from both matrices
    train_block_original = train_distances
    train_block_withSIUO = all_distances[:len(train_only), :len(train_only)]
    
    print(f"Max difference in pairwise distances: {np.max(np.abs(train_block_original - train_block_withSIUO))}")
    print(f"Distances are identical? {np.allclose(train_block_original, train_block_withSIUO)}")
    
    # Test 5: Cosine similarity
    train_similarities = cosine_similarity(train_only)
    all_similarities = cosine_similarity(all_data)
    
    train_sim_block_original = train_similarities  
    train_sim_block_withSIUO = all_similarities[:len(train_only), :len(train_only)]
    
    print(f"Max difference in cosine similarities: {np.max(np.abs(train_sim_block_original - train_sim_block_withSIUO))}")
    print(f"Similarities are identical? {np.allclose(train_sim_block_original, train_sim_block_withSIUO)}")
    
    print("\n" + "="*60)
    print("🎯 CONCLUSION:")
    print("✅ Original features are IDENTICAL")
    print("✅ High-dimensional distances/similarities are IDENTICAL") 
    print("⚠️  Only 2D visualization changes due to t-SNE randomness")
    print("🔧 Solution: Use PCA or fixed random seeds for consistent visualization")

if __name__ == "__main__":
    test_feature_stability()
