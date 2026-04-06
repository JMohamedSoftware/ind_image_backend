import pickle
import numpy as np
import hashlib
import os

def get_algo_salt(algo):
    return hashlib.md5(algo.upper().encode()).hexdigest()[:4]

def get_vector_from_seed_text(seed_text, length=128):
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
    
    algo = "CNN"
    print(f"\n--- Detailed {algo} Analysis ---")
    
    for i, item in enumerate(data[:5]):
        desc = item.get('description')
        img_hash = item.get('hash')
        filename = item.get('filename', '')
        stored_map = item.get('descriptors_map', {})
        stored_vec = stored_map.get(algo)
        
        if stored_vec is None:
            continue
            
        stored_vec = np.array(stored_vec)
        algo_salt = get_algo_salt(algo)
        
        # Determine what seed_text SHOULD have been used based on main.py logic
        if desc:
            expected_seed = f"{desc.lower().strip()}_{algo_salt}"
            source = "description"
        else:
            expected_seed = f"{img_hash[:8]}_{algo_salt}"
            source = "hash"
            
        # Verify if stored_vec matches expected_seed
        repro_vec = get_vector_from_seed_text(expected_seed, length=len(stored_vec))
        match_score = np.dot(repro_vec, stored_vec) * 100
        
        print(f"Item {i} ({filename}):")
        print(f"  Source: {source}")
        print(f"  Seed: '{expected_seed}'")
        print(f"  Match Score with Repro: {match_score:.1f}%")
        
        if match_score < 99:
            # Try the OTHER source just in case
            alt_seed = f"{img_hash[:8]}_{algo_salt}" if source == "description" else f"{desc.lower().strip() if desc else 'None'}_{algo_salt}"
            alt_repro = get_vector_from_seed_text(alt_seed, length=len(stored_vec))
            alt_match = np.dot(alt_repro, stored_vec) * 100
            print(f"  ALT Seed Match Score: {alt_match:.1f}%")

if __name__ == "__main__":
    diagnostic()
