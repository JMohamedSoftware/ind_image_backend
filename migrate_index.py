import pickle
import numpy as np
import hashlib
import os

# New logic from main.py
CATEGORY_MAP = {
    "CBIR": ["ORB", "SIFT", "AKAZE", "HOG"],
    "DL": ["CNN", "SVM", "K-MEANS", "PCA", "MEANSHIFT", "CLIP"],
    "TBIR": ["TBIR_BASIC"]
}

def get_algo_salt(algo):
    return hashlib.md5(algo.upper().encode()).hexdigest()[:4]

def get_semantic_features(text, algo, length=128):
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

    print(f"Migrating {len(data)} items...")
    
    all_algos = [algo for group in CATEGORY_MAP.values() for algo in group]
    dl_algos = CATEGORY_MAP["DL"]

    for item in data:
        desc = item.get("description")
        img_hash = item.get("hash")
        
        # We only need to migrate DL algorithms that used the old string-seeding logic
        # If an image has no description, the old logic used seed(hash[:8]_salt).
        # We'll keep that as a fallback in get_semantic_features (wait, main.py has fallback)

        descriptors_map = item.get("descriptors_map", {})
        
        for algo in dl_algos:
            length = 4096 if algo == "CNN" else 512 if algo == "CLIP" else 64 if algo == "SVM" else 16 if algo == "MEANSHIFT" else 32
            
            if desc:
                new_vec = get_semantic_features(desc, algo, length=length)
            else:
                # Fallback matching main.py's new fallback logic
                algo_salt = get_algo_salt(algo)
                seed_text = f"{img_hash[:8]}_{algo_salt}"
                np.random.seed(int(hashlib.md5(seed_text.encode()).hexdigest()[:8], 16) % 2**32)
                new_vec = np.random.randn(length).astype(np.float32)
                new_vec /= (np.linalg.norm(new_vec) + 1e-8)
            
            descriptors_map[algo] = new_vec

    # Save backup
    with open("index_backup.pkl", "wb") as f:
        pickle.dump(data, f)
    
    # Save migrated
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(data, f)
        
    print("Migration complete. Backup saved to index_backup.pkl")

if __name__ == "__main__":
    migrate()
