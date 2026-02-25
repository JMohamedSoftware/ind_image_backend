import pickle
import numpy as np
import hashlib
import os

# Logic matched to main.py
def get_algo_salt(algo):
    return hashlib.md5(algo.upper().encode()).hexdigest()[:4]

def get_algo_vector_length(algo):
    algo = algo.upper()
    if algo == "CNN": return 4096
    if algo == "CLIP": return 512
    if algo == "SVM": return 64
    if algo == "MEANSHIFT": return 16
    return 32

def get_semantic_features(text, algo, length=32):
    algo_salt = get_algo_salt(algo)
    tokens = [t.strip().lower() for t in text.replace(',', ' ').replace('.', ' ').split() if len(t.strip()) > 1]
    if not tokens: tokens = ["dummy"]
    vectors = []
    for token in tokens:
        seed = f"{token}_{algo_salt}"
        np.random.seed(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16) % 2**32)
        v = np.random.randn(length).astype(np.float32)
        vectors.append(v)
    avg_vector = np.mean(vectors, axis=0)
    avg_vector /= (np.linalg.norm(avg_vector) + 1e-8)
    return avg_vector

def migrate():
    INDEX_FILE = "index.pkl"
    if not os.path.exists(INDEX_FILE):
        print("No index.pkl found")
        return

    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)

    print(f"Migrating {len(data)} items to new dimensions...")
    
    dl_algos = ["CNN", "SVM", "K-MEANS", "PCA", "MEANSHIFT", "CLIP"]

    for item in data:
        desc = item.get("description")
        img_hash = item.get("hash", "00000000")
        descriptors_map = item.get("descriptors_map", {})
        
        for algo in dl_algos:
            length = get_algo_vector_length(algo)
            
            if desc:
                new_vec = get_semantic_features(desc, algo, length=length)
            else:
                algo_salt = get_algo_salt(algo)
                seed_text = f"{img_hash[:8]}_{algo_salt}"
                np.random.seed(int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16) % 2**32)
                new_vec = np.random.randn(length).astype(np.float32)
                new_vec /= (np.linalg.norm(new_vec) + 1e-8)
            
            descriptors_map[algo] = new_vec

    # Save
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(data, f)
        
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
