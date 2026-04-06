import pickle
import numpy as np
import hashlib
import os

# Mimic main.py logic
CATEGORY_MAP = {
    "CBIR": ["ORB", "SIFT", "AKAZE", "HOG"],
    "DL": ["CNN", "SVM", "K-MEANS", "PCA", "MEANSHIFT", "CLIP"],
    "TBIR": ["TBIR_BASIC"]
}

def get_algo_salt(algo):
    return hashlib.md5(algo.upper().encode()).hexdigest()[:4]

def get_vector(text_or_hash, algo, length=128):
    algo_salt = get_algo_salt(algo)
    seed_text = f"{text_or_hash}_{algo_salt}"
    np.random.seed(int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16) % 2**32)
    v = np.random.randn(length).astype(np.float32)
    v /= (np.linalg.norm(v) + 1e-8)
    return v

def diagnostic():
    INDEX_FILE = "index.pkl"
    if not os.path.exists(INDEX_FILE):
        print("index.pkl not found")
        return

    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} items")
    
    # Test CNN search
    algo = "CNN"
    print(f"\n--- Testing {algo} Search ---")
    
    for i, item in enumerate(data[:3]):
        desc = item.get('description', '')
        filename = item.get('filename', '')
        stored_map = item.get('descriptors_map', {})
        stored_vec = stored_map.get(algo)
        
        if stored_vec is None:
            print(f"Item {i} ({filename}): No {algo} descriptor found")
            continue
            
        stored_vec = np.array(stored_vec)
        print(f"Item {i} ({filename}): Found {algo} vec, shape {stored_vec.shape}")
        
        # Scenario 1: Search with exact description
        if desc:
            query_vec = get_vector(desc.lower().strip(), algo, length=4096)
            sim = np.dot(query_vec, stored_vec) * 100
            print(f"  Exact desc match score: {sim:.1f}%")
        
        # Scenario 2: Search with slightly different text
        if desc:
            query_text = desc.split()[0] if desc.split() else "car"
            query_vec = get_vector(query_text.lower().strip(), algo, length=4096)
            sim = np.dot(query_vec, stored_vec) * 100
            print(f"  Partial desc ('{query_text}') match score: {sim:.1f}% (Expected: VERY LOW)")

        # Scenario 3: Text search (which uses 128 dims in main.py)
        if desc:
            query_text = desc.lower().strip()
            # main.py search_text logic:
            q_vec_128 = get_vector(query_text, algo, length=128)
            i_vec_128 = stored_vec.flatten()[:128] # No re-normalization in main.py
            sim_128 = np.dot(q_vec_128, i_vec_128) * 100
            
            # Re-normalized version for comparison
            i_vec_128_norm = i_vec_128 / (np.linalg.norm(i_vec_128) + 1e-8)
            sim_128_norm = np.dot(q_vec_128, i_vec_128_norm) * 100
            
            print(f"  Text Search sim (128 dims, no re-norm): {sim_128:.1f}%")
            print(f"  Text Search sim (128 dims, WITH re-norm): {sim_128_norm:.1f}%")

if __name__ == "__main__":
    diagnostic()
